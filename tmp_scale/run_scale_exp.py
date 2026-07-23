"""
Webshop scaling experiment for credit-assignment anchors.

Goal: show that as the GRPO group size grows (4 -> 8 -> 16), the belief-bin
anchor's COVERAGE (fraction of oracle-labeled steps that receive non-zero,
outcome-relative credit) rises, while its sign-agreement with the oracle stays
stable -- and that it dominates GiGPO (exact-obs grouping), which stays starved
because identical observations across rollouts are rare.

Candidate set for the belief probe = gold title + HARD NEGATIVES:
  hard negatives are the titles of the OTHER products that actually show up in
  the search-result pages the group visits (same query neighborhood => hard),
  plus the group's own (wrong) final purchases.

Per step we store lp_pre = [logπ(cand | instruction, history, obs_t)] over the
fixed candidate set, so the downstream analyzer can bin beliefs toward gold.

Usage:  python run_scale_exp.py <N_ROLLOUT> [N_TASKS] [GPU]
Output: tmp_scale/records_g{N_ROLLOUT}.json   (same schema as event_records.json)
"""
import os, sys, json, re
import numpy as np
import torch

N_ROLLOUT = int(sys.argv[1]) if len(sys.argv) > 1 else 8
N_TASKS   = int(sys.argv[2]) if len(sys.argv) > 2 else 6
# optional task shard:  argv[3]=lo argv[4]=hi argv[5]=tag  (process tasks [lo,hi))
TASK_LO   = int(sys.argv[3]) if len(sys.argv) > 3 else 0
TASK_HI   = int(sys.argv[4]) if len(sys.argv) > 4 else N_TASKS
TAG       = sys.argv[5] if len(sys.argv) > 5 else ""
MAX_STEPS = 8
TEMP = 0.7
MAX_NEW = 320         # keep identical to the original event-anchor setup (forced reasoning)
SEED = int(os.environ.get("SCALE_SEED", "0"))  # configurable for fresh independent rollouts
MAX_NEG = 7          # hard negatives kept per task (candidate set = gold + up to MAX_NEG)

WS = "/home/test/yyy/SDAR/agent_system/environments/env_package/webshop/webshop"
sys.path.insert(0, WS)
OUT = "/home/test/yyy/SDAR/tmp_scale"
os.makedirs(OUT, exist_ok=True)
MODEL = "/home/test/models/Qwen2.5-3B-Instruct"

torch.manual_seed(SEED); np.random.seed(SEED)

from transformers import AutoModelForCausalLM, AutoTokenizer
print(f"[g{N_ROLLOUT}] loading model ...", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cuda:0")
model.eval()
DEV = model.device
print("model ready", flush=True)

from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv
print("building webshop server (num_products=1000) ...", flush=True)
base_env = WebAgentTextEnv(observation_mode="text", num_products=1000, human_goals=False,
                           session_prefix="s0_")
server = base_env.server
envs = [base_env] + [
    WebAgentTextEnv(observation_mode="text", server=server, session_prefix=f"s{k}_")
    for k in range(1, N_ROLLOUT)
]
print("num goals", len(server.goals), flush=True)

ASIN_RE = re.compile(r"^[A-Z0-9]{10}$")
ASIN_SCAN = re.compile(r"\b([A-Z0-9]{10})\b")
ACTION_RE = re.compile(r"<action>(.*?)</action>", re.S | re.I)
THINK_RE = re.compile(r"<think>(.*?)</think>", re.S | re.I)

def title_of(asin):
    p = server.product_item_dict.get(asin.upper())
    if p is None:
        return None
    return (p.get("Title") or p.get("name") or "")[:120]

def parse_gen(text):
    a = ACTION_RE.search(text)
    z = THINK_RE.search(text)
    if z:
        think = z.group(1).strip()
    elif a:
        think = text[:a.start()].strip()
        think = re.sub(r"</?think>", "", think, flags=re.I).strip()
    else:
        think = re.sub(r"</?think>", "", text, flags=re.I).strip()
    return (a.group(1).strip() if a else None), think[:1500]

def build_agent_prompt(instruction, history, obs, clickables):
    hist = ""
    for (a, o) in history:
        hist += f"Action: {a}\nObservation: {o}\n"
    user = (
        "You are an expert shopping agent in the WebShop text environment.\n"
        f"Instruction: {instruction}\n\n"
        + (f"History so far:\n{hist}\n" if hist else "")
        + f"Current observation:\n{obs}\n\n"
        f"Available clickable actions: {clickables}\n\n"
        "First reason carefully inside <think> </think>: state what the instruction "
        "requires, what the current observation tells you, which candidate looks best "
        "and why, and what to do next (at least 2-3 sentences). Then output exactly one "
        "action inside <action> </action>. An action is either search[keywords] "
        "(only on the search page) or click[value] where value is one of the "
        "available clickables (e.g. a product id, 'Buy Now', 'Next >', 'Back to Search')."
    )
    msgs = [{"role": "user", "content": user}]
    p = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    return p + "<think>\n"

@torch.no_grad()
def generate(prompt):
    ids = tok(prompt, return_tensors="pt").to(DEV)
    out = model.generate(**ids, do_sample=True, temperature=TEMP, top_p=0.9,
                         max_new_tokens=MAX_NEW, pad_token_id=tok.eos_token_id)
    gen = out[0][ids.input_ids.shape[1]:]
    return "<think>\n" + tok.decode(gen, skip_special_tokens=True)

@torch.no_grad()
def generate_batch(prompts):
    """Batched generation over the group's rollouts (same setup, just parallel)."""
    tok.padding_side = "left"
    enc = tok(prompts, return_tensors="pt", padding=True).to(DEV)
    out = model.generate(**enc, do_sample=True, temperature=TEMP, top_p=0.9,
                         max_new_tokens=MAX_NEW, pad_token_id=tok.eos_token_id)
    gen = out[:, enc.input_ids.shape[1]:]
    return ["<think>\n" + tok.decode(g, skip_special_tokens=True) for g in gen]

PROBE = "\n\nBased on all the information above, the exact title of the single best product to buy is:"

def belief_context(instruction, history, obs):
    hist = ""
    for (a, o) in history:
        hist += f"Action: {a}\nObservation: {o}\n"
    txt = f"Instruction: {instruction}\n"
    if hist:
        txt += f"\n{hist}"
    txt += f"\nCurrent observation:\n{obs}\n"
    return txt

@torch.no_grad()
def score_candidates(context, candidates):
    msgs = [{"role": "user", "content": context + PROBE}]
    pre = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False) + " "
    pre_ids = tok(pre, add_special_tokens=False).input_ids
    seqs, tlens = [], []
    for c in candidates:
        t = tok(c, add_special_tokens=False).input_ids
        seqs.append(pre_ids + t); tlens.append(len(t))
    maxlen = max(len(s) for s in seqs)
    pad = tok.eos_token_id
    input_ids = torch.full((len(seqs), maxlen), pad, dtype=torch.long)
    attn = torch.zeros((len(seqs), maxlen), dtype=torch.long)
    for i, s in enumerate(seqs):
        input_ids[i, :len(s)] = torch.tensor(s); attn[i, :len(s)] = 1
    input_ids, attn = input_ids.to(DEV), attn.to(DEV)
    logits = model(input_ids=input_ids, attention_mask=attn).logits.float()
    out, L0 = [], len(pre_ids)
    for i, s in enumerate(seqs):
        tl = tlens[i]
        lp = torch.log_softmax(logits[i, L0 - 1:L0 - 1 + tl, :], dim=-1)
        tgt = torch.tensor(s[L0:L0 + tl], device=DEV)
        out.append(float(lp[torch.arange(tl, device=DEV), tgt].mean().item()))
    return out

rng = np.random.RandomState(SEED)
goal_ids = rng.choice(range(len(server.goals)), size=N_TASKS, replace=False).tolist()
print("task goal ids:", goal_ids, flush=True)

all_tasks = []
for ti, gid in enumerate(goal_ids):
    if not (TASK_LO <= ti < TASK_HI):
        continue
    goal = server.goals[gid]
    instruction = goal["instruction_text"]
    gold_asin = goal["asin"].upper()
    gold_title = (goal["name"] or "")[:120]
    goal_options = [str(o).lower() for o in (goal.get("goal_options") or [])]
    print(f"\n=== [g{N_ROLLOUT}] task {ti} gid={gid} gold={gold_asin} '{gold_title[:50]}' ===", flush=True)

    seen_asins = set()          # hard-negative pool: ASINs seen in observations
    st = []
    for k in range(N_ROLLOUT):
        env = envs[k]
        obs, _ = env.reset(session=gid)
        st.append(dict(k=k, env=env, obs=obs, history=[], steps=[],
                       cur_asin=None, reward=0.0, done=False, finished=False))
    # step in lockstep across rollouts, batching the per-step generation
    for step in range(MAX_STEPS):
        act = [s for s in st if not s["finished"]]
        if not act:
            break
        prompts = []
        for s in act:
            for a in ASIN_SCAN.findall(s["obs"]):
                if ASIN_RE.match(a):
                    seen_asins.add(a.upper())
            clickables = s["env"].get_available_actions().get("clickables", [])
            prompts.append(build_agent_prompt(instruction, s["history"], s["obs"], clickables))
        raws = generate_batch(prompts)
        for s, raw in zip(act, raws):
            action, think = parse_gen(raw)
            rec = dict(step=step, obs=s["obs"], think=think, action=action)
            if action is None:
                action = f"search[{goal['query']}]" if step == 0 else None
                rec["fallback"] = True
            if action is None:
                s["steps"].append(rec); s["finished"] = True; continue
            m = re.match(r"click\[(.+)\]", action, re.I)
            if m and ASIN_RE.match(m.group(1).strip().upper()):
                s["cur_asin"] = m.group(1).strip().upper()
            state, reward, dn, _ = s["env"].step(action)
            rec["next_obs_has_gold"] = gold_asin.lower() in state.lower()
            s["steps"].append(rec)
            s["history"].append((action, s["obs"][:400]))
            if dn:
                s["reward"], s["done"], s["finished"] = float(reward), True, True
            else:
                s["obs"] = state

    trajs = []
    for s in st:
        buy_title = title_of(s["cur_asin"]) if s["cur_asin"] else None
        trajs.append(dict(rollout=s["k"], steps=s["steps"], done=s["done"],
                          reward=s["reward"], buy_asin=s["cur_asin"], buy_title=buy_title))
        print(f"  r{s['k']}: {len(s['steps'])} steps done={s['done']} "
              f"R={s['reward']:.2f} buy={s['cur_asin']}", flush=True)

    # ---- candidate set = gold + hard negatives ----
    cand = [gold_title]; seen = {gold_title}
    # hard negs from products seen in search results (same query neighborhood)
    neg_titles = []
    for a in seen_asins:
        if a == gold_asin:
            continue
        t = title_of(a)
        if t and t not in seen:
            neg_titles.append(t); seen.add(t)
    # also include the group's own wrong purchases (very hard negatives)
    for tr in trajs:
        t = tr["buy_title"]
        if t and t not in seen:
            neg_titles.insert(0, t); seen.add(t)
    cand += neg_titles[:MAX_NEG]
    gold_idx = 0
    print(f"  candidates ({len(cand)}): gold + {len(cand)-1} hard-neg", flush=True)

    # ---- belief per step over candidate set ----
    for tr in trajs:
        history = []
        for rec in tr["steps"]:
            ctx = belief_context(instruction, history, rec["obs"])
            rec["lp_pre"] = score_candidates(ctx, cand)
            if rec.get("action"):
                history.append((rec["action"], rec["obs"][:400]))

    all_tasks.append(dict(task=ti, gid=gid, instruction=instruction,
                          gold_asin=gold_asin, gold_title=gold_title,
                          gold_idx=gold_idx, goal_options=goal_options,
                          candidates=cand, trajs=trajs))
    with open(os.path.join(OUT, f"records_g{N_ROLLOUT}{TAG}.json"), "w") as f:
        json.dump(all_tasks, f)
    print(f"  [saved through task {ti}]", flush=True)

print(f"\n[g{N_ROLLOUT}{TAG}] ALL DONE -> records_g{N_ROLLOUT}{TAG}.json", flush=True)

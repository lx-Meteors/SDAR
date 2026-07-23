"""
Validation: think-as-belief-operator anchor (event anchor) vs GiGPO obs anchor.

Per turn t we compute, over a label-free candidate set C (final purchases of the
GRPO group), two belief distributions:
    p_pre  = p(c | instruction, history, obs_t)            (before this turn's think)
    p_post = p(c | instruction, history, obs_t, think_t)   (after  this turn's think)
Signature: dH = H(post)-H(pre), flip = argmax changed, KL(post||pre).
Events (sharpen / pivot / expand / inert) are classified in analyze.py.

Rollout data also stores everything needed for the GiGPO baseline (full obs text)
and for oracle step labels (gold asin clicks, option correctness, final reward).
"""
import os, sys, json, re
import numpy as np
import torch

WS = "/home/test/yyy/SDAR/agent_system/environments/env_package/webshop/webshop"
sys.path.insert(0, WS)
OUT = "/home/test/yyy/SDAR/tmp_event_anchor"
MODEL = "/home/test/models/Qwen2.5-3B-Instruct"

N_TASKS = 5
N_ROLLOUT = 4
MAX_STEPS = 10
TEMP = 0.7
MAX_NEW = 320
SEED = 0

torch.manual_seed(SEED); np.random.seed(SEED)

# ----------------------------------------------------------------- model
from transformers import AutoModelForCausalLM, AutoTokenizer
print("loading model ...", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cuda:0")
model.eval()
DEV = model.device
print("model ready", flush=True)

# ----------------------------------------------------------------- env
from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv
print("building webshop server (num_products=1000) ...", flush=True)
base_env = WebAgentTextEnv(observation_mode="text", num_products=1000, human_goals=False,
                           session_prefix="e0_")
server = base_env.server
envs = [base_env] + [
    WebAgentTextEnv(observation_mode="text", server=server, session_prefix=f"e{k}_")
    for k in range(1, N_ROLLOUT)
]
print("num goals", len(server.goals), flush=True)

ASIN_RE = re.compile(r"^[A-Z0-9]{10}$")
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
        # model reasons in free text without tags: take everything before <action>
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
    return p + "<think>\n"   # prefill to force an explicit reasoning trace

@torch.no_grad()
def generate(prompt):
    ids = tok(prompt, return_tensors="pt").to(DEV)
    out = model.generate(**ids, do_sample=True, temperature=TEMP, top_p=0.9,
                         max_new_tokens=MAX_NEW, pad_token_id=tok.eos_token_id)
    gen = out[0][ids.input_ids.shape[1]:]
    return "<think>\n" + tok.decode(gen, skip_special_tokens=True)

# ----------------------------------------------------------------- belief probe
PROBE = "\n\nBased on all the information above, the exact title of the single best product to buy is:"

def belief_context(instruction, history, obs, think=None):
    hist = ""
    for (a, o) in history:
        hist += f"Action: {a}\nObservation: {o}\n"
    txt = f"Instruction: {instruction}\n"
    if hist:
        txt += f"\n{hist}"
    txt += f"\nCurrent observation:\n{obs}\n"
    if think:
        txt += f"\nMy reasoning about what to do:\n{think}\n"
    return txt

@torch.no_grad()
def score_candidates(context, candidates):
    """length-normalized logprob of each candidate title under context+probe (batched)."""
    msgs = [{"role": "user", "content": context + PROBE}]
    pre = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False) + " "
    pre_ids = tok(pre, add_special_tokens=False).input_ids
    seqs, tlens = [], []
    for c in candidates:
        t = tok(c, add_special_tokens=False).input_ids
        seqs.append(pre_ids + t)
        tlens.append(len(t))
    maxlen = max(len(s) for s in seqs)
    pad = tok.eos_token_id
    input_ids = torch.full((len(seqs), maxlen), pad, dtype=torch.long)
    attn = torch.zeros((len(seqs), maxlen), dtype=torch.long)
    for i, s in enumerate(seqs):
        input_ids[i, :len(s)] = torch.tensor(s)
        attn[i, :len(s)] = 1
    input_ids, attn = input_ids.to(DEV), attn.to(DEV)
    logits = model(input_ids=input_ids, attention_mask=attn).logits.float()
    out = []
    L0 = len(pre_ids)
    for i, s in enumerate(seqs):
        tl = tlens[i]
        lp = torch.log_softmax(logits[i, L0 - 1:L0 - 1 + tl, :], dim=-1)
        tgt = torch.tensor(s[L0:L0 + tl], device=DEV)
        out.append(float(lp[torch.arange(tl, device=DEV), tgt].mean().item()))
    return out

# ----------------------------------------------------------------- rollout
rng = np.random.RandomState(SEED)
goal_ids = rng.choice(range(len(server.goals)), size=N_TASKS, replace=False).tolist()
print("task goal ids:", goal_ids, flush=True)

all_tasks = []
for ti, gid in enumerate(goal_ids):
    goal = server.goals[gid]
    instruction = goal["instruction_text"]
    gold_asin = goal["asin"].upper()
    gold_title = (goal["name"] or "")[:120]
    goal_options = [str(o).lower() for o in (goal.get("goal_options") or [])]
    print(f"\n=== task {ti} gid={gid} gold={gold_asin} '{gold_title[:60]}' ===", flush=True)
    print(f"    instruction: {instruction}", flush=True)

    trajs = []
    for k in range(N_ROLLOUT):
        env = envs[k]
        obs, _ = env.reset(session=gid)
        history = []
        steps = []
        cur_asin, final_reward, done = None, 0.0, False
        for step in range(MAX_STEPS):
            aa = env.get_available_actions()
            clickables = aa.get("clickables", [])
            prompt = build_agent_prompt(instruction, history, obs, clickables)
            raw = generate(prompt)
            action, think = parse_gen(raw)

            rec = dict(step=step, obs=obs, think=think, action=action, raw=raw[:600])
            if action is None:
                action = f"search[{goal['query']}]" if step == 0 else None
                rec["fallback"] = True
            if action is None:
                steps.append(rec); break

            m = re.match(r"click\[(.+)\]", action, re.I)
            if m and ASIN_RE.match(m.group(1).strip().upper()):
                cur_asin = m.group(1).strip().upper()

            state, reward, dn, _ = env.step(action)
            rec["next_obs_has_gold"] = gold_asin.lower() in state.lower()
            steps.append(rec)
            history.append((action, obs[:400]))
            if dn:
                final_reward, done = float(reward), True
                break
            obs = state

        buy_title = title_of(cur_asin) if cur_asin else None
        trajs.append(dict(rollout=k, steps=steps, done=done, reward=final_reward,
                          buy_asin=cur_asin, buy_title=buy_title))
        print(f"  rollout {k}: {len(steps)} steps done={done} R={final_reward:.2f} "
              f"buy={cur_asin} '{(buy_title or '')[:50]}'", flush=True)

    # ---------------- candidate set (label-free: group outcomes), gold flagged
    cand, seen = [], set()
    for tr in trajs:
        t = tr["buy_title"]
        if t and t not in seen:
            cand.append(t); seen.add(t)
    gold_in_free = gold_title in seen
    if not gold_in_free:
        cand.append(gold_title)
    gold_idx = cand.index(gold_title)
    print(f"  candidates ({len(cand)}, gold_label_free={gold_in_free}):", flush=True)
    for c in cand:
        print(f"    - {c[:70]}", flush=True)

    # ---------------- belief signatures per turn
    for tr in trajs:
        history = []
        for rec in tr["steps"]:
            ctx_pre = belief_context(instruction, history, rec["obs"])
            lp_pre = score_candidates(ctx_pre, cand)
            if rec["think"]:
                ctx_post = belief_context(instruction, history, rec["obs"], rec["think"])
                lp_post = score_candidates(ctx_post, cand)
            else:
                lp_post = lp_pre
            rec["lp_pre"], rec["lp_post"] = lp_pre, lp_post
            if rec.get("action"):
                history.append((rec["action"], rec["obs"][:400]))
        print(f"  rollout {tr['rollout']}: beliefs done", flush=True)

    all_tasks.append(dict(task=ti, gid=gid, instruction=instruction,
                          gold_asin=gold_asin, gold_title=gold_title,
                          gold_idx=gold_idx, gold_label_free=gold_in_free,
                          goal_options=goal_options, candidates=cand, trajs=trajs))
    with open(os.path.join(OUT, "event_records.json"), "w") as f:
        json.dump(all_tasks, f)
    print(f"  [saved through task {ti}]", flush=True)

print("\nALL DONE", flush=True)

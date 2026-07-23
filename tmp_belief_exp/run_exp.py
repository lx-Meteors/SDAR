"""
Small validation experiment.

Anchor idea under test:  belief-gain  b_t = logπ(a*|h_t) - logπ(a*|h_0)
  where a* = target product title of the webshop task, h_t = agent context after t turns.

We roll out a few webshop episodes with Qwen2.5-3B-Instruct, grouped as GRPO groups
(same task x N rollouts), and for every step record:
  - anchor_obs   : the raw page observation  (the anchor GiGPO uses, exact-match grouping)
  - belief       : logπ(a*|h_t)              (length-normalized target log-prob)
  - belief_gain  : belief - belief_0         (our proposed anchor coordinate)

Then we compare the two anchors:
  - GiGPO: how populated are the exact-obs step-groups within a task group (singleton => no signal)
  - Ours : is belief_gain a smooth/comparable coordinate; does it rise when the target appears
"""
import os, sys, json, time, re
import numpy as np
import torch

WS = "/home/test/yyy/SDAR/agent_system/environments/env_package/webshop/webshop"
sys.path.insert(0, WS)
OUT = "/home/test/yyy/SDAR/tmp_belief_exp"
MODEL = "/home/test/models/Qwen2.5-3B-Instruct"

N_TASKS = 5
N_ROLLOUT = 4          # GRPO group size
MAX_STEPS = 6
TEMP = 0.7
MAX_NEW = 256
SEED = 0

torch.manual_seed(SEED); np.random.seed(SEED)

# ----------------------------------------------------------------------------- model
from transformers import AutoModelForCausalLM, AutoTokenizer
print("loading model ...", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cuda:0")
model.eval()
DEV = model.device
print("model ready", flush=True)

# ----------------------------------------------------------------------------- env
from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv
print("building webshop server (num_products=1000) ...", flush=True)
base_env = WebAgentTextEnv(observation_mode="text", num_products=1000, human_goals=False,
                           session_prefix="r0_")
server = base_env.server
envs = [base_env] + [
    WebAgentTextEnv(observation_mode="text", server=server, session_prefix=f"r{k}_")
    for k in range(1, N_ROLLOUT)
]
print("num goals", len(server.goals), flush=True)

# ----------------------------------------------------------------------------- helpers
ACTION_RE_A = re.compile(r"<action>(.*?)</action>", re.S | re.I)

def parse_action(text):
    m = ACTION_RE_A.search(text)
    has_think = "<think>" in text.lower() and "</think>" in text.lower()
    if not m:
        return None, has_think
    return m.group(1).strip(), has_think

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
        "First think step-by-step inside <think> </think>. Then output exactly one "
        "action inside <action> </action>. An action is either search[keywords] "
        "(only on the search page) or click[value] where value is one of the "
        "available clickables (e.g. a product id, 'Buy Now', 'Next >', 'Back to Search')."
    )
    msgs = [{"role": "user", "content": user}]
    return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)

@torch.no_grad()
def generate(prompt):
    ids = tok(prompt, return_tensors="pt").to(DEV)
    out = model.generate(**ids, do_sample=True, temperature=TEMP, top_p=0.9,
                         max_new_tokens=MAX_NEW, pad_token_id=tok.eos_token_id)
    gen = out[0][ids.input_ids.shape[1]:]
    return tok.decode(gen, skip_special_tokens=True)

def build_belief_context(instruction, history, obs=None):
    """Context h_t for the belief probe: instruction + all past actions/observations
    (+ current obs if given). h_0 uses instruction only (history empty, obs None)."""
    hist = ""
    for (a, o) in history:
        hist += f"Action: {a}\nObservation: {o}\n"
    txt = f"Instruction: {instruction}\n"
    if hist:
        txt += f"\n{hist}"
    if obs is not None:
        txt += f"\nCurrent observation:\n{obs}\n"
    return txt

PROBE = "\n\nBased on all the information above, the exact title of the single best product to buy is:"

@torch.no_grad()
def belief_logprob(context, target):
    """length-normalized logπ(target | context+probe) under the instruct model."""
    msgs = [{"role": "user", "content": context + PROBE}]
    pre = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False) + " "
    pre_ids = tok(pre, return_tensors="pt").input_ids.to(DEV)
    tgt_ids = tok(target, return_tensors="pt", add_special_tokens=False).input_ids.to(DEV)
    full = torch.cat([pre_ids, tgt_ids], dim=1)
    logits = model(full).logits  # [1, L, V]
    # logprob of each target token predicted from previous position
    L_pre = pre_ids.shape[1]
    logp = torch.log_softmax(logits[0, L_pre-1:-1, :].float(), dim=-1)
    tgt = tgt_ids[0]
    tok_lp = logp[torch.arange(tgt.shape[0]), tgt]
    return float(tok_lp.mean().item()), int(tgt.shape[0])

# ----------------------------------------------------------------------------- rollout
records = []           # flat per-step records
rng = np.random.RandomState(SEED)
goal_ids = rng.choice(range(len(server.goals)), size=N_TASKS, replace=False).tolist()
print("task goal ids:", goal_ids, flush=True)

for ti, gid in enumerate(goal_ids):
    goal = server.goals[gid]
    instruction = goal["instruction_text"]
    astar = goal["name"]
    astar_short = astar[:120]
    # h_0 belief (instruction only), shared across the group
    b0, alen = belief_logprob(build_belief_context(instruction, []), astar_short)
    print(f"\n=== task {ti} gid={gid} | a*='{astar_short[:60]}...' | b0={b0:.3f} (alen={alen}) ===", flush=True)
    print(f"    instruction: {instruction}", flush=True)
    for k in range(N_ROLLOUT):
        env = envs[k]
        obs, _ = env.reset(session=gid)
        history = []
        for step in range(MAX_STEPS):
            aa = env.get_available_actions()
            clickables = aa.get("clickables", [])
            prompt = build_agent_prompt(instruction, history, obs, clickables)
            raw = generate(prompt)
            action, has_think = parse_action(raw)

            # belief at this state (before acting): context includes current obs
            ctx = build_belief_context(instruction, history, obs)
            b_t, _ = belief_logprob(ctx, astar_short)

            target_in_obs = (goal["asin"].lower() in obs.lower()) or (astar_short[:40].lower() in obs.lower())
            records.append(dict(
                task=ti, gid=gid, rollout=k, step=step,
                anchor_obs=obs.strip(),
                action=action, has_think=has_think,
                belief=b_t, belief0=b0, belief_gain=b_t - b0,
                target_in_obs=bool(target_in_obs),
                obs_len=len(obs),
            ))

            if action is None:
                # invalid -> fallback so the trajectory can still progress a bit
                if step == 0:
                    action = f"search[{goal['query']}]"
                else:
                    break
            state, reward, done, info = env.step(action)
            history.append((action, obs[:400]))
            if done:
                records[-1]["done_reward"] = reward
                break
            obs = state
        print(f"  rollout {k}: {len([r for r in records if r['task']==ti and r['rollout']==k])} steps"
              f" | belief_gain path: "
              + ", ".join(f"{r['belief_gain']:+.2f}" for r in records if r['task']==ti and r['rollout']==k),
              flush=True)

with open(os.path.join(OUT, "records.json"), "w") as f:
    json.dump(records, f, indent=2)
print(f"\nsaved {len(records)} step records -> records.json", flush=True)

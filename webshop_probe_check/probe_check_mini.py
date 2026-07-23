"""Minimal pre-training validation of the hindsight-gold probe on WebShop.

3 tasks x 4 rollouts x <=6 steps with base Qwen2.5-3B-Instruct (temp 1.0),
then probe every visited state h_t for:
    p_g = mean logprob of y* (hindsight gold = best rollout's action plan)
    p_o = mean logprob of the trajectory's OWN final action plan
Checks (directional, tiny-n by design):
  A. anchor collision structure across rollouts (GiGPO's precondition)
  B. probe vs group outcome at collision states
  C. within-trajectory probe progression: won vs lost
  D. GiGPO active coverage vs probe coverage
Run in conda env `verl-agent-webshop`, GPU5.
"""
import json, os, sys
from collections import defaultdict

WEBSHOP_ROOT = "/home/test/yyy/verl-agent/agent_system/environments/env_package/webshop/webshop"
sys.path.append(WEBSHOP_ROOT)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "5")

import numpy as np
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

MODEL = "/home/test/models/Qwen2.5-3B-Instruct"
N_TASKS, GROUP, MAX_STEPS = 3, 4, 8
FORCE_BUY_FROM = 5  # from this step on, buy now when possible (guarantees graded outcome)
TASK_IDS = [500, 507, 514]
HERE = os.path.dirname(os.path.abspath(__file__))

TEMPLATE = """
You are an expert autonomous agent operating in the WebShop e\u2011commerce environment. 
Your task is to: {task}.
{hist}Your current observation is: {obs}.
Your admissible actions of the current situation are: 
[
{avail}
].

Now it's your turn to take one action for the current step.
You should first reason step-by-step about the current situation, then think carefully which admissible action best advances the shopping goal. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""

def fmt_avail(av):
    acts = (["search[<your query>]"] if av.get("has_search_bar") else [])
    acts += [f"click[{c}]" for c in av.get("clickables", [])]
    return "\n".join(f"'{a}'," for a in acts)

def build_prompt(task, hist, obs, av):
    h = ""
    if hist:
        recent = hist[-2:]
        h = ("Your recent actions were: " +
             "; ".join(a for _, a in recent) + ".\n")
    return TEMPLATE.format(task=task, hist=h, obs=obs[:6000], avail=fmt_avail(av))

def project(resp):
    low = resp.lower()
    s, e = low.find("<action>"), low.find("</action>")
    if s == -1 or e == -1: return None
    return low[s + 8:e].strip()

print("init vllm...", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL)
llm = LLM(model=MODEL, gpu_memory_utilization=0.5, max_model_len=8192)
gen_sp = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=320,
                        stop=["</action>"], include_stop_str_in_output=True)
print("vllm ok; init envs...", flush=True)

# import after LLM() so vLLM's engine subprocess forks from a CUDA-clean parent
from web_agent_site.envs import WebAgentTextEnv  # noqa: E402
env_kwargs = dict(observation_mode="text", num_products=None, human_goals=False,
                  file_path=os.path.join(WEBSHOP_ROOT, "data/items_shuffle_1000.json"),
                  attr_path=os.path.join(WEBSHOP_ROOT, "data/items_ins_v2_1000.json"))
envs = [WebAgentTextEnv(**env_kwargs) for _ in range(GROUP)]
print("envs ok", flush=True)

trajs = []
for ti, gid in enumerate(TASK_IDS):
    obs, task = [], None
    for env in envs:
        r = env.reset(session=gid)
        obs.append(r[0] if isinstance(r, tuple) else r)
    task = envs[0].get_instruction_text()
    hist = [[] for _ in range(GROUP)]
    steps = [[] for _ in range(GROUP)]
    done, reward = [False] * GROUP, [0.0] * GROUP
    for t in range(MAX_STEPS):
        idxs = [k for k in range(GROUP) if not done[k]]
        if not idxs: break
        convs, forced = [], {}
        for k in idxs:
            av = envs[k].get_available_actions()
            p = build_prompt(task, hist[k], obs[k], av)
            steps[k].append(dict(t=t, anchor=obs[k], prompt=p))
            if t >= FORCE_BUY_FROM and any(c.lower() == "buy now" for c in av.get("clickables", [])):
                forced[k] = "click[buy now]"
            else:
                convs.append((k, [{"role": "user", "content": p}]))
        outs = llm.chat([c for _, c in convs], gen_sp, use_tqdm=False) if convs else []
        resp_of = {k: o.outputs[0].text for (k, _), o in zip(convs, outs)}
        for k in idxs:
            resp = resp_of.get(k, "(forced buy)")
            act = forced.get(k) or project(resp) or "invalid"
            steps[k][-1].update(response=resp[-400:], action=act)
            try:
                o2, r2, d2, _ = envs[k].step(act)
            except Exception:
                o2, r2, d2 = obs[k], 0.0, False
            hist[k].append((obs[k], act)); obs[k] = o2
            if d2: done[k], reward[k] = True, float(r2)
        print(f"task{ti} t{t} active={len(idxs)}", flush=True)
    for k in range(GROUP):
        trajs.append(dict(task=ti, goal=gid, k=k, instruction=task, score=reward[k],
                          won=reward[k] == 1.0, steps=steps[k]))
    print(f"== task{ti} scores={[round(reward[k],2) for k in range(GROUP)]}", flush=True)

json.dump([{**tr, "steps": [{kk: v for kk, v in s.items() if kk != "prompt"}
            for s in tr["steps"]]} for tr in trajs], open(os.path.join(HERE, "mini_rollouts.json"), "w"))

# ---------------- probe ----------------
def plan_of(tr):
    return "; ".join(s["action"] for s in tr["steps"] if s.get("action") and s["action"] != "invalid")[:300]

ystar = {}
for ti in range(N_TASKS):
    grp = [tr for tr in trajs if tr["task"] == ti]
    best = max(grp, key=lambda tr: tr["score"])
    ystar[ti] = plan_of(best) if best["score"] > 0 else None
    print(f"task{ti}: best score={best['score']:.2f} y*={ystar[ti]!r:.120}", flush=True)

HINT = "\n\nThe correct complete action plan for this task is:"
reqs, meta = [], []
for tr in trajs:
    tgts = {"g": ystar[tr["task"]], "o": plan_of(tr) or None}
    for s_i, s in enumerate(tr["steps"]):
        for typ, tgt in tgts.items():
            if not tgt: continue
            pre = tok.apply_chat_template([{"role": "user", "content": s["prompt"] + HINT}],
                                          add_generation_prompt=True, tokenize=False) + " "
            pre_ids = tok(pre, add_special_tokens=False).input_ids
            tgt_ids = tok(tgt, add_special_tokens=False).input_ids
            reqs.append({"prompt_token_ids": (pre_ids + tgt_ids)[-8000:]})
            meta.append((tr["task"], tr["k"], s_i, typ, len(tgt_ids)))
probe_sp = SamplingParams(temperature=0, max_tokens=1, prompt_logprobs=0)
print(f"probing {len(reqs)} states...", flush=True)
pouts = llm.generate(reqs, probe_sp, use_tqdm=False)

probe = {}
for (ti, k, s_i, typ, L), o in zip(meta, pouts):
    lps = [list(d.values())[0].logprob for d in o.prompt_logprobs[-L:] if d]
    probe[(ti, k, s_i, typ)] = float(np.mean(lps))

# ---------------- analysis ----------------
print("\n========== ANALYSIS ==========")
# A/D: anchor collisions & GiGPO coverage
tot_steps, multi_steps, pairs = 0, 0, []
for ti in range(N_TASKS):
    grp = [tr for tr in trajs if tr["task"] == ti]
    groups = defaultdict(list)
    for tr in grp:
        for s_i, s in enumerate(tr["steps"]):
            groups[" ".join(s["anchor"].split())].append((tr["k"], s_i, tr["won"]))
    tot = sum(len(v) for v in groups.values())
    multi = sum(len(v) for v in groups.values() if len({k for k, _, _ in v}) >= 2)
    tot_steps += tot; multi_steps += multi
    print(f"[A] task{ti}: states={tot} in-multi-anchor-group={multi} ({multi/max(1,tot):.0%})"
          f" groups={sum(1 for v in groups.values() if len({k for k,_,_ in v})>=2)}")
    score_of = {tr["k"]: tr["score"] for tr in grp}
    for key, v in groups.items():
        ks = {k for k, _, _ in v}
        if len(ks) >= 2:
            gbar = np.mean([score_of[k] for k, _, _ in v])
            for k, s_i, _ in v:
                pg = probe.get((ti, k, s_i, "g"))
                if pg is not None: pairs.append((pg, gbar, score_of[k]))
print(f"[D] GiGPO-active coverage={multi_steps}/{tot_steps} ({multi_steps/max(1,tot_steps):.0%}); probe coverage=100%")

# B: probe vs outcome at collision states
if pairs:
    pg = np.array([p[0] for p in pairs]); gb = np.array([p[1] for p in pairs]); w = np.array([p[2] for p in pairs])
    if len(set(gb)) > 1:
        from scipy.stats import spearmanr
        r1, _ = spearmanr(pg, gb)
        print(f"[B] collision states n={len(pairs)}: spearman(probe_g, group-mean-outcome)={r1:.2f}")
    med = float(np.median(w))
    hi = pg[w > med]; lo = pg[w < med]
    if len(hi) and len(lo):
        auc = float(np.mean([(a > b) + 0.5 * (a == b) for a in hi for b in lo]))
        print(f"[B] within-collision AUC(probe_g: high-score>low-score members)={auc:.2f}  (n_hi={len(hi)} n_lo={len(lo)})")

# C: within-trajectory progression (high = score>0.5, low = score==0)
for label, cond in (("high-score", lambda s: s > 0.5), ("zero-score", lambda s: s == 0)):
    ds = []
    for tr in trajs:
        if not cond(tr["score"]) or len(tr["steps"]) < 2: continue
        a = probe.get((tr["task"], tr["k"], 0, "g")); b = probe.get((tr["task"], tr["k"], len(tr["steps"]) - 1, "g"))
        if a is not None and b is not None: ds.append(b - a)
    if ds: print(f"[C] {label}: mean Δlogp_g(first→last)={np.mean(ds):+.3f} (n={len(ds)})")
mg = []
for tr in trajs:
    for s_i in range(len(tr["steps"])):
        g = probe.get((tr["task"], tr["k"], s_i, "g")); o = probe.get((tr["task"], tr["k"], s_i, "o"))
        if g is not None and o is not None: mg.append((tr["score"], s_i, g - o))
if mg:
    for lab, cond in (("high-score", lambda s: s > 0.5), ("zero-score", lambda s: s == 0)):
        v = [m for sc, _, m in mg if cond(sc)]
        if v: print(f"[C] margin logp_g-logp_o mean ({lab})={np.mean(v):+.3f} (n={len(v)})")

json.dump({f"{k[0]}_{k[1]}_{k[2]}_{k[3]}": v for k, v in probe.items()}, open(os.path.join(HERE, "mini_probe.json"), "w"))
print("MINI_CHECK_DONE")

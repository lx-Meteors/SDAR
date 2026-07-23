"""Behavioral eval for the routing mini-test: roll out a model on the 9
variance-group tasks (held-in) + 6 fresh tasks (held-out), 4 rollouts each,
temp 1.0, same harness as ws_collect_ckpt3.py.

Usage (verl-agent-webshop env):
    python ws_route_eval.py <model_path> <out_json> [gpu]
"""
import json, os, sys

WEBSHOP_ROOT = "/home/test/yyy/verl-agent/agent_system/environments/env_package/webshop/webshop"
sys.path.append(WEBSHOP_ROOT)
MODEL, OUT = sys.argv[1], sys.argv[2]
os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[3] if len(sys.argv) > 3 else "6"

from vllm import LLM, SamplingParams

HELD_IN = [542, 577, 605, 626, 640, 654, 661, 668, 675]
HELD_OUT = [717, 724, 731, 738, 745, 752]
TASK_IDS = HELD_IN + HELD_OUT
GROUP, MAX_STEPS, FORCE_BUY_FROM = 4, 8, 5
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
        h = "Your recent actions were: " + "; ".join(a for _, a in hist[-2:]) + ".\n"
    return TEMPLATE.format(task=task, hist=h, obs=obs[:6000], avail=fmt_avail(av))

def project(resp):
    low = resp.lower()
    s, e = low.find("<action>"), low.find("</action>")
    if s == -1 or e == -1:
        return None
    return low[s + 8:e].strip()

print("init vllm...", flush=True)
llm = LLM(model=MODEL, gpu_memory_utilization=0.5, max_model_len=8192)
print("vllm ok; init envs...", flush=True)

from web_agent_site.envs import WebAgentTextEnv  # noqa: E402
env_kwargs = dict(observation_mode="text", num_products=None, human_goals=False,
                  file_path=os.path.join(WEBSHOP_ROOT, "data/items_shuffle_1000.json"),
                  attr_path=os.path.join(WEBSHOP_ROOT, "data/items_ins_v2_1000.json"))
envs = [WebAgentTextEnv(**env_kwargs) for _ in range(GROUP)]
print("envs ok", flush=True)

trajs = []
for ti, gid in enumerate(TASK_IDS):
    obs = []
    for env in envs:
        r = env.reset(session=gid)
        obs.append(r[0] if isinstance(r, tuple) else r)
    task = envs[0].get_instruction_text()
    hist = [[] for _ in range(GROUP)]
    done, reward = [False] * GROUP, [0.0] * GROUP
    nsteps = [0] * GROUP
    for t in range(MAX_STEPS):
        idxs = [k for k in range(GROUP) if not done[k]]
        if not idxs:
            break
        convs, forced = [], {}
        for k in idxs:
            av = envs[k].get_available_actions()
            p = build_prompt(task, hist[k], obs[k], av)
            if t >= FORCE_BUY_FROM and any(c.lower() == "buy now" for c in av.get("clickables", [])):
                forced[k] = "click[buy now]"
            else:
                convs.append((k, [{"role": "user", "content": p}]))
        sp = [SamplingParams(temperature=1.0, top_p=1.0, max_tokens=320,
                             stop=["</action>"], include_stop_str_in_output=True,
                             seed=100000 + gid * 100 + k * 10 + t) for k, _ in convs]
        outs = llm.chat([c for _, c in convs], sp, use_tqdm=False) if convs else []
        resp_of = {k: o.outputs[0].text for (k, _), o in zip(convs, outs)}
        for k in idxs:
            resp = resp_of.get(k, "(forced)")
            act = forced.get(k) or project(resp) or "invalid"
            try:
                o2, r2, d2, _ = envs[k].step(act)
            except Exception:
                o2, r2, d2 = obs[k], 0.0, False
            hist[k].append((obs[k], act))
            obs[k] = o2
            nsteps[k] = t + 1
            if d2:
                done[k], reward[k] = True, float(r2)
    for k in range(GROUP):
        trajs.append(dict(gid=gid, k=k, held="in" if gid in HELD_IN else "out",
                          score=reward[k], done=done[k], nsteps=nsteps[k]))
    print(f"== gid{gid} scores={[round(reward[k], 2) for k in range(GROUP)]}", flush=True)

json.dump(trajs, open(OUT, "w"))
import numpy as np
sc = np.array([t["score"] for t in trajs])
hi = np.array([t["score"] for t in trajs if t["held"] == "in"])
ho = np.array([t["score"] for t in trajs if t["held"] == "out"])
print(f"MODEL={MODEL}")
print(f"ALL mean={sc.mean():.4f} succ={np.mean(sc == 1.0):.3f}   "
      f"IN mean={hi.mean():.4f}   OUT mean={ho.mean():.4f}")
print("WS_ROUTE_EVAL_DONE")

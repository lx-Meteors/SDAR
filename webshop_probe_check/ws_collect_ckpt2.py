"""Collect WebShop rollouts with the TRAINED policy (BEACON 1.5B step150) to
measure succ-succ sibling fratricide.  Same harness as collect_ws_full.py,
different model + output file.  3 tasks x 4 rollouts x <=8 steps, temp 1.0.
Run in conda env `verl-agent-webshop`, GPU5.
"""
import json, os, sys

WEBSHOP_ROOT = "/home/test/yyy/verl-agent/agent_system/environments/env_package/webshop/webshop"
sys.path.append(WEBSHOP_ROOT)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "5")

from vllm import LLM, SamplingParams

MODEL = "/data1/test/yyy/BEACON/checkpoints/verl_agent_webshop/beacon_qwen2.5_1.5b/step150_hf"
N_TASKS, GROUP, MAX_STEPS, FORCE_BUY_FROM = 8, 4, 8, 5
TASK_IDS = [521, 528, 535, 542, 549, 556, 563, 570]
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "ws_ckpt_rollouts2.json")

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
    if s == -1 or e == -1: return None
    return low[s + 8:e].strip()

print("init vllm...", flush=True)
llm = LLM(model=MODEL, gpu_memory_utilization=0.5, max_model_len=8192)
gen_sp = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=320,
                        stop=["</action>"], include_stop_str_in_output=True)
print("vllm ok; init envs...", flush=True)

from web_agent_site.envs import WebAgentTextEnv  # noqa: E402
env_kwargs = dict(observation_mode="text", num_products=None, human_goals=False,
                  file_path=os.path.join(WEBSHOP_ROOT, "data/items_shuffle_1000.json"),
                  attr_path=os.path.join(WEBSHOP_ROOT, "data/items_ins_v2_1000.json"))
envs = [WebAgentTextEnv(**env_kwargs) for _ in range(GROUP)]
print("envs ok", flush=True)

def goal_of(env, gid):
    try:
        g = env.server.user_sessions[str(gid)]["goal"]
    except Exception:
        g = env.server.goals[gid]
    return {k: g.get(k) for k in ("asin", "name", "category", "query",
                                  "instruction_text", "attributes", "price_upper", "goal_options")}

trajs = []
for ti, gid in enumerate(TASK_IDS):
    obs = []
    for env in envs:
        r = env.reset(session=gid)
        obs.append(r[0] if isinstance(r, tuple) else r)
    task = envs[0].get_instruction_text()
    goal = goal_of(envs[0], gid)
    print(f"task{ti} goal: asin={goal['asin']} name={str(goal['name'])[:60]}", flush=True)
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
            steps[k].append(dict(t=t, anchor=obs[k], prompt=p, avail=fmt_avail(av)))
            if t >= FORCE_BUY_FROM and any(c.lower() == "buy now" for c in av.get("clickables", [])):
                forced[k] = "click[buy now]"
            else:
                convs.append((k, [{"role": "user", "content": p}]))
        outs = llm.chat([c for _, c in convs], gen_sp, use_tqdm=False) if convs else []
        resp_of = {k: o.outputs[0].text for (k, _), o in zip(convs, outs)}
        for k in idxs:
            resp = resp_of.get(k, "(forced)")
            act = forced.get(k) or project(resp) or "invalid"
            steps[k][-1].update(response=resp, action=act)
            try:
                o2, r2, d2, _ = envs[k].step(act)
            except Exception:
                o2, r2, d2 = obs[k], 0.0, False
            hist[k].append((obs[k], act)); obs[k] = o2
            if d2: done[k], reward[k] = True, float(r2)
        print(f"task{ti} t{t} active={len(idxs)}", flush=True)
    for k in range(GROUP):
        trajs.append(dict(task=ti, goal_id=gid, k=k, instruction=task, goal=goal,
                          score=reward[k], steps=steps[k]))
    print(f"== task{ti} scores={[round(reward[k], 2) for k in range(GROUP)]}", flush=True)

json.dump(trajs, open(OUT, "w"))
print(f"saved {len(trajs)} trajs -> {OUT}")
print("WS_CKPT_COLLECT_DONE")

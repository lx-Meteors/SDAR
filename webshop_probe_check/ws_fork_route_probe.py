"""Direct test of counter-mass routing at REAL success/fail forks.

Find GiGPO-style anchor collisions (same task, same observation) between
success and fail trajectories that chose DIFFERENT actions.  At the first
divergent action token:

  fail side:  of the mass released by pushing down y_f, what share does the
              SUCCESS token y_s get under student menu p~ vs teacher menu q~?
              (routing better iff q~(y_s) > p~(y_s))
  succ side:  of the mass drawn by sharpening y_s, what share comes from the
              FAIL token y_f under p~ vs q~?
              (routing better iff q~(y_f) > p~(y_f))

Also counts same-action collisions = exact-token cancellation that routing
cannot fix (honesty statistic).

Memory-light: lm_head applied only to response positions (fits ~5GB GPU).
Run: qwen-infer env.  CUDA_VISIBLE_DEVICES set by caller.
"""
import json, os
import numpy as np
import torch
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "/data1/test/yyy/BEACON/checkpoints/verl_agent_webshop/beacon_qwen2.5_1.5b/step150_hf"
HERE = os.path.dirname(os.path.abspath(__file__))
SDAR = os.path.dirname(HERE)
SKILLS_DIR = os.path.join(SDAR, "skills", "webshop")
FILES = ["ws_ckpt_rollouts2.json", "ws_ckpt_rollouts3.json", "ws_ckpt_rollouts4.json"]
OUT = os.path.join(HERE, "ws_fork_route_rows.json")
MAX_FORKS = 120

_MAP = json.load(open(os.path.join(SKILLS_DIR, "skill_mapping.json")))
_CONTENT = {n: open(os.path.join(SKILLS_DIR, f)).read().strip()
            for n, f in _MAP["skill_files"].items()}

def skill_for_prompt(prompt_text):
    parts = [_CONTENT.get("general_skills", "")]
    tl = prompt_text.lower()
    for tt, kws in _MAP.get("task_keywords", {}).items():
        if kws and any(kw in tl for kw in kws):
            mapped = _MAP["task_to_skill"].get(tt)
            if mapped and mapped in _CONTENT:
                parts.append(_CONTENT[mapped])
    return "\n\n".join(parts)

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                             device_map="cuda:0")
model.eval()
DEV = model.device

@torch.no_grad()
def resp_probs(user_content, resp_ids):
    """p over vocab at each response position; lm_head on response slice only."""
    pre = tok.apply_chat_template([{"role": "user", "content": user_content}],
                                  add_generation_prompt=True, tokenize=False)
    pre_ids = tok(pre, add_special_tokens=False, truncation=True, max_length=6500).input_ids
    full = torch.tensor([pre_ids + resp_ids], device=DEV)
    hs = model.model(full).last_hidden_state[0]
    L0 = len(pre_ids)
    sl = hs[L0 - 1:L0 - 1 + len(resp_ids)]
    return torch.softmax(model.lm_head(sl).float(), dim=-1)      # (T, V)

def action_token_positions(resp, resp_ids, offs):
    o = resp.find("<action>")
    c = resp.find("</action>")
    if o == -1 or c == -1:
        return []
    a0 = o + len("<action>")
    return [i for i in range(len(resp_ids)) if a0 <= offs[i][0] < c]

# ---- variance groups & anchor collisions ----
groups = defaultdict(list)
for fn in FILES:
    for t in json.load(open(os.path.join(HERE, fn))):
        groups[(fn, t["task"])].append(t)

forks, same_action, diff_action = [], 0, 0
for gk, gtr in sorted(groups.items()):
    scores = np.array([t["score"] for t in gtr])
    if scores.std() < 1e-6:
        continue
    succs = [t for t in gtr if t["score"] > scores.mean()]
    fails = [t for t in gtr if t["score"] < scores.mean()]
    for ts in succs:
        for tf in fails:
            for ss in ts["steps"]:
                if "anchor" not in ss or not ss.get("response") or ss["response"] == "(forced)":
                    continue
                for sf in tf["steps"]:
                    if sf.get("anchor") != ss["anchor"] or not sf.get("response") \
                       or sf["response"] == "(forced)":
                        continue
                    if ss["action"] == sf["action"]:
                        same_action += 1
                        continue
                    diff_action += 1
                    forks.append((gk, ts, ss, tf, sf))
print(f"collisions: same-action={same_action}  diff-action={diff_action}  "
      f"fork pairs kept={min(len(forks), MAX_FORKS)}", flush=True)
forks = forks[:MAX_FORKS]

# ---- per-fork distribution probe ----
cache = {}
def get_p_q(step):
    key = id(step)
    if key in cache:
        return cache[key]
    enc = tok(step["response"], add_special_tokens=False, return_offsets_mapping=True)
    resp_ids, offs = enc.input_ids, enc.offset_mapping
    apos = action_token_positions(step["response"], resp_ids, offs)
    p = resp_probs(step["prompt"], resp_ids)
    teacher_user = ("[Privileged Skill Information]\n" + skill_for_prompt(step["prompt"])
                    + "\n\n" + step["prompt"])
    q = resp_probs(teacher_user, resp_ids)
    cache[key] = (resp_ids, apos, p.cpu(), q.cpu())
    return cache[key]

rows = []
for n, (gk, ts, ss, tf, sf) in enumerate(forks):
    try:
        s_ids, s_apos, s_p, s_q = get_p_q(ss)
        f_ids, f_apos, f_p, f_q = get_p_q(sf)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print(f"skip fork {n} OOM", flush=True)
        continue
    s_act = [s_ids[i] for i in s_apos]
    f_act = [f_ids[i] for i in f_apos]
    d = 0
    while d < min(len(s_act), len(f_act)) and s_act[d] == f_act[d]:
        d += 1
    if d >= min(len(s_act), len(f_act)):
        continue
    y_s, y_f = s_act[d], f_act[d]
    # fail side: share of released mass reaching y_s
    fp = f_p[f_apos[d]]; fq = f_q[f_apos[d]]
    pm_s = float(fp[y_s] / (1 - fp[y_f] + 1e-9))
    qm_s = float(fq[y_s] / (1 - fq[y_f] + 1e-9))
    # succ side: share of drawn mass coming from y_f
    sp = s_p[s_apos[d]]; sq = s_q[s_apos[d]]
    pm_f = float(sp[y_f] / (1 - sp[y_s] + 1e-9))
    qm_f = float(sq[y_f] / (1 - sq[y_s] + 1e-9))
    rows.append(dict(gk=list(gk), t_s=ss["t"], t_f=sf["t"], d=d,
                     ytok_s=tok.decode([y_s]), ytok_f=tok.decode([y_f]),
                     act_s=ss["action"][:60], act_f=sf["action"][:60],
                     fail_pm=pm_s, fail_qm=qm_s,
                     succ_pm=pm_f, succ_qm=qm_f,
                     py_f=float(fp[y_f]), qy_f=float(fq[y_f]),
                     py_s=float(sp[y_s]), qy_s=float(sq[y_s])))
    if n % 10 == 0:
        print(f"fork {n}/{len(forks)}", flush=True)

json.dump(rows, open(OUT, "w"), ensure_ascii=False)
print(f"probed forks: {len(rows)} -> {OUT}", flush=True)

# ---- analysis ----
if rows:
    fpm = np.array([r["fail_pm"] for r in rows])
    fqm = np.array([r["fail_qm"] for r in rows])
    spm = np.array([r["succ_pm"] for r in rows])
    sqm = np.array([r["succ_qm"] for r in rows])
    print("\n=== FAIL side: released-mass share reaching the SUCCESS token ===")
    print(f"  student p~(y_s): median={np.median(fpm):.4f}  mean={fpm.mean():.4f}")
    print(f"  teacher q~(y_s): median={np.median(fqm):.4f}  mean={fqm.mean():.4f}")
    print(f"  routing wins (q~>p~): {(fqm > fpm).mean():.1%}   "
          f"median ratio={np.median(fqm / (fpm + 1e-9)):.2f}x")
    print("\n=== SUCC side: drawn-mass share coming from the FAIL token ===")
    print(f"  student p~(y_f): median={np.median(spm):.4f}  mean={spm.mean():.4f}")
    print(f"  teacher q~(y_f): median={np.median(sqm):.4f}  mean={sqm.mean():.4f}")
    print(f"  routing wins (q~>p~): {(sqm > spm).mean():.1%}")
    net = (fqm - fpm) + (sqm - spm)
    print(f"\n=== NET pair contrast gain (fail-side + succ-side): "
          f"median={np.median(net):+.4f}  mean={net.mean():+.4f}  "
          f"positive share={(net > 0).mean():.1%}")
    print("\n--- examples (sorted by fail-side routing gain) ---")
    for r in sorted(rows, key=lambda r: -(r["fail_qm"] - r["fail_pm"]))[:10]:
        print(f"  y_f={r['ytok_f']!r} -> y_s={r['ytok_s']!r}  "
              f"fail p~={r['fail_pm']:.3f} q~={r['fail_qm']:.3f} | "
              f"succ p~={r['succ_pm']:.3f} q~={r['succ_qm']:.3f} | "
              f"{r['act_f'][:30]} vs {r['act_s'][:30]}")
print("\nWS_FORK_ROUTE_DONE")

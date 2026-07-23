"""
Level-1 offline diagnostic:
Do our intrinsic *reflection events* (think-as-belief-operator: pivot/sharpen)
recover BEACON's hand-crafted milestones -- WITHOUT any privileged goal info?

Ground truth: BEACON's MilestoneDetector run on the SAME trajectories
              (uses goal attributes/options/price + product catalog).
Candidate anchors compared against that ground truth:
  A. reflection events   = non-inert belief-operator steps (ours, intrinsic)
  B. obs-change (GiGPO-style) = steps where the observation differs from prev
     (a proxy for GiGPO's distinct-state anchor)

We report precision / recall / F1 of each anchor at predicting milestone steps.
"""
import os, sys, json, re, io, contextlib
import numpy as np
from collections import defaultdict

WS = "/home/test/yyy/SDAR/agent_system/environments/env_package/webshop/webshop"
sys.path.insert(0, WS)
sys.path.insert(0, "/home/test/yyy/BEACON")   # for `import migpo...` and agent_system
OUT = "/home/test/yyy/SDAR/tmp_event_anchor"

tasks = json.load(open(os.path.join(OUT, "event_records.json")))

# ---- build a webshop server to recover full goals + product catalog ----------
from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv
print("building webshop server ...", flush=True)
env = WebAgentTextEnv(observation_mode="text", num_products=1000, human_goals=False,
                      session_prefix="lvl1_")
server = env.server
print("num goals", len(server.goals), flush=True)

# ---- BEACON milestone detector (silence its logging) -------------------------
from migpo.webshop_milestone_detector import MilestoneDetector
MilestoneDetector._log_step = lambda *a, **k: None
MilestoneDetector._log_session_init = lambda *a, **k: None

def softmax(lp):
    lp = np.array(lp, float); p = np.exp(lp - lp.max()); return p / p.sum()
def entropy(p):
    p = np.clip(p, 1e-12, 1); return float(-(p * np.log(p)).sum())
def kl(p, q):
    p, q = np.clip(p, 1e-12, 1), np.clip(q, 1e-12, 1); return float((p * np.log(p / q)).sum())

# ---- per-task processing -----------------------------------------------------
def classify_events(steps):
    """adaptive: below-median-KL -> inert; else flip->pivot / dH<0->sharpen / else expand"""
    kls = np.array([s["_kl"] for s in steps])
    thr = max(np.median(kls), 1e-3) if len(kls) else 1e-3
    for s in steps:
        if s["_kl"] < thr:
            s["_etype"] = "inert"
        elif s["_flip"]:
            s["_etype"] = "pivot"
        elif s["_dH"] < 0:
            s["_etype"] = "sharpen"
        else:
            s["_etype"] = "expand"

def prf(pred_set, gold_set, n):
    tp = len(pred_set & gold_set)
    prec = tp / max(1, len(pred_set))
    rec = tp / max(1, len(gold_set))
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    return prec, rec, f1, tp

agg = defaultdict(lambda: [0, 0, 0])   # method -> [tp, pred, gold]
milestone_total = 0
step_total = 0
proximity = defaultdict(list)          # method -> distance of each gold milestone to nearest pred

for T in tasks:
    gid = T["gid"]
    goal = server.goals[gid]
    assert goal["asin"].upper() == T["gold_asin"].upper(), \
        f"goal misalignment task {T['task']}: {goal['asin']} vs {T['gold_asin']}"

    for tr in T["trajs"]:
        steps = tr["steps"]
        # belief signatures
        for s in steps:
            p_pre, p_post = softmax(s["lp_pre"]), softmax(s["lp_post"])
            s["_dH"] = entropy(p_post) - entropy(p_pre)
            s["_flip"] = int(np.argmax(p_post) != np.argmax(p_pre))
            s["_kl"] = kl(p_post, p_pre)
        classify_events(steps)

        # ---- ground-truth milestones via BEACON detector ----
        det = MilestoneDetector(goal, server.product_item_dict, server.product_prices)
        det.reset(goal)
        gold_ms = set()
        with contextlib.redirect_stdout(io.StringIO()):
            for i, s in enumerate(steps):
                action = s.get("action") or ""
                state = s["obs"]
                next_state = steps[i + 1]["obs"] if i + 1 < len(steps) else s["obs"]
                res = det.process(action=action, state=state, next_state=next_state, info=None)
                if res.achieved:
                    gold_ms.add(i)

        n = len(steps)
        step_total += n
        milestone_total += len(gold_ms)

        # ---- candidate anchors ----
        refl = set(i for i, s in enumerate(steps) if s["_etype"] != "inert")
        refl_ps = set(i for i, s in enumerate(steps) if s["_etype"] in ("pivot", "sharpen"))
        prev = None
        obschange = set()
        for i, s in enumerate(steps):
            o = " ".join(s["obs"].split())
            if o != prev:
                obschange.add(i)
            prev = o

        for name, pred in (("reflection(all-event)", refl),
                           ("reflection(pivot+sharpen)", refl_ps),
                           ("obs-change(GiGPO-style)", obschange)):
            _, _, _, tp = prf(pred, gold_ms, n)
            agg[name][0] += tp; agg[name][1] += len(pred); agg[name][2] += len(gold_ms)
            # proximity: for each gold milestone, nearest predicted step distance
            for g in gold_ms:
                if pred:
                    proximity[name].append(min(abs(g - p) for p in pred))

print("\n" + "=" * 78)
print(f"GROUND TRUTH: {milestone_total} milestone steps over {step_total} steps "
      f"({milestone_total/max(1,step_total):.0%} of steps are milestones)")
print("=" * 78)
print(f"{'anchor':<28} {'precision':>10} {'recall':>8} {'F1':>7} {'pred/step':>10} {'prox':>6}")
for name in ("reflection(all-event)", "reflection(pivot+sharpen)", "obs-change(GiGPO-style)"):
    tp, npred, ngold = agg[name]
    prec = tp / max(1, npred); rec = tp / max(1, ngold)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    prox = np.mean(proximity[name]) if proximity[name] else float("nan")
    print(f"{name:<28} {prec:>10.2f} {rec:>8.2f} {f1:>7.2f} {npred/max(1,step_total):>10.2f} {prox:>6.2f}")

print("\nNotes:")
print(" - precision = fraction of proposed anchor steps that ARE gold milestones")
print(" - pred/step = how many anchors proposed per step (density; lower=sparser=more selective)")
print(" - prox = mean distance (in steps) from each gold milestone to nearest proposed anchor")

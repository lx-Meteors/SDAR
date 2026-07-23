import json, os, re
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

OUT = "/home/test/yyy/SDAR/tmp_event_anchor"
tasks = json.load(open(os.path.join(OUT, "event_records.json")))
rows = json.load(open(os.path.join(OUT, "rows.json")))

def softmax(lp):
    lp = np.array(lp, float); p = np.exp(lp - lp.max()); return p / p.sum()

# ---- forward belief-gain toward gold (the OLD coordinate) per step, to show front-loading
fwd_by_step = defaultdict(list)   # |Δ p_gold| from obs across turns (forward info gain)
for T in tasks:
    gi = T["gold_idx"]
    for tr in T["trajs"]:
        prev = None
        for rec in tr["steps"]:
            pg = softmax(rec["lp_pre"])[gi]
            if prev is not None:
                fwd_by_step[rec["step"]].append(abs(pg - prev))
            prev = pg

# ---- event (think-operator) magnitude per step = KL(post||pre)
kl_by_step = defaultdict(list)
for r in rows:
    kl_by_step[r["step"]].append(r["kl"])

steps = sorted(set(list(fwd_by_step) + list(kl_by_step)))
fwd = [np.mean(fwd_by_step[s]) if fwd_by_step[s] else 0 for s in steps]
klm = [np.mean(kl_by_step[s]) if kl_by_step[s] else 0 for s in steps]

fig, axes = plt.subplots(1, 3, figsize=(16, 4.3))

ax = axes[0]
ax.bar(steps, fwd, color="#c0392b")
ax.set_title("Forward belief-gain toward gold\n(|Δp_gold| from new obs) — FRONT-LOADED")
ax.set_xlabel("turn"); ax.set_ylabel("mean |Δ p_gold|")

ax = axes[1]
ax.bar(steps, klm, color="#2980b9")
ax.set_title("Reflection-operator magnitude\nKL(post_think || pre_think) — SPREAD OUT")
ax.set_xlabel("turn"); ax.set_ylabel("mean KL")

ax = axes[2]
et_by_step = defaultdict(lambda: defaultdict(int))
for r in rows:
    et_by_step[r["step"]][r["etype"]] += 1
types = ["sharpen", "pivot", "expand"]
colors = {"sharpen": "#27ae60", "pivot": "#e67e22", "expand": "#8e44ad"}
bottom = np.zeros(len(steps))
for et in types:
    vals = [et_by_step[s][et] for s in steps]
    ax.bar(steps, vals, bottom=bottom, label=et, color=colors[et])
    bottom += np.array(vals)
ax.set_title("Cognitive events per turn (non-inert)\nevents occur throughout, not only at t=0")
ax.set_xlabel("turn"); ax.set_ylabel("# events"); ax.legend()

plt.tight_layout()
p = os.path.join(OUT, "event_compare.png")
plt.savefig(p, dpi=120)
print("saved", p)

# front-loading quantification
tot = sum(fwd); early = sum(fwd[:2])
print(f"forward belief-gain: {early/tot:.0%} of mass in turns 0-1  (front-loaded)")
tot2 = sum(klm); early2 = sum(klm[:2])
print(f"reflection KL:       {early2/tot2:.0%} of mass in turns 0-1  (spread)")

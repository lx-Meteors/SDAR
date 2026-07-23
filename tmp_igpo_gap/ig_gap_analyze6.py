"""Credit-assignment form shootout: additive vs multiplicative vs gated.

Labels (what a correct credit signal must separate):
  GOOD action: led_discovery | gold_click | click_opt_match
  BAD  action: wrong-asin click that hides a visible gold | non-matching
               option click
  (neutral actions excluded from AUC)

Candidates (all on z-scores; zS = z(dS_ans), zT = z(dT1_ans)):
  additive        zS + b*zT
  gap             zS - b*zT
  product         zS * zT               (AND-like)
  min             min(zS, zT)           (AND-like)
  sign-geo        sign-agree ? sign*sqrt(|zS*zT|) : 0
  mult-uplift     dS * exp(b*dT)        (raw scale)
  gated-additive  zS + b*zT*G, G=1 only on option/buy turns (execution gate)
  phase-additive  zS on first half, zT on second half of trajectory

Also report: per-phase AUC (early/late), trajectory-level corr(sum credit,
score), and credit assigned to BAD turns (should be negative).
"""
import collections
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
rows = json.load(open(os.path.join(HERE, "ig_gap_rows.json")))

traj = collections.defaultdict(list)
for r in rows:
    traj[(r["fn"], r["task"], r["k"])].append(r)

gains = []
for key, ss in traj.items():
    ss.sort(key=lambda r: r["t"])
    for i in range(1, len(ss)):
        a, b = ss[i - 1], ss[i]
        gains.append(dict(
            key=key, ta=a["t"], score=a["score"], n_steps=a["n_steps"],
            action=a["action"], act_type=a["act_type"],
            gold_click=a["gold_click"], led_discovery=(b["first_vis_t"] == b["t"]),
            vis_before=a["gold_visible"], vis_after=b["gold_visible"],
            dS=b["lp_S_ans"] - a["lp_S_ans"],
            dT=b["lp_T1_ans"] - a["lp_T1_ans"]))

for g in gains:
    g["good"] = bool(g["led_discovery"] or g["gold_click"]
                     or g["act_type"] == "click_opt_match")
    g["bad"] = bool((g["act_type"] == "click_asin" and not g["gold_click"]
                     and g["vis_before"] and not g["vis_after"])
                    or g["act_type"] == "click_other")

dS = np.array([g["dS"] for g in gains])
dT = np.array([g["dT"] for g in gains])
zS = (dS - dS.mean()) / dS.std()
zT = (dT - dT.mean()) / dT.std()
good = np.array([g["good"] for g in gains])
bad = np.array([g["bad"] for g in gains])
lab_mask = good | bad
pos_frac = np.array([g["ta"] / max(g["n_steps"] - 1, 1) for g in gains])
exec_gate = np.array([g["act_type"] in ("click_opt_match", "click_other", "buy")
                      for g in gains], dtype=float)
print(f"n={len(gains)} good={good.sum()} bad={bad.sum()} neutral={(~lab_mask).sum()}")


def auc(s, l):
    npos, nneg = l.sum(), (~l).sum()
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    return (ranks[l].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def ev(name, s):
    m = lab_mask
    a_all = auc(s[m], good[m])
    early = m & (pos_frac <= 0.34)
    late = m & (pos_frac > 0.34)
    a_e = auc(s[early], good[early]) if early.sum() > 5 else float("nan")
    a_l = auc(s[late], good[late]) if late.sum() > 5 else float("nan")
    mean_bad = s[bad].mean()
    # traj-level: corr(sum credit, score)
    agg = collections.defaultdict(float); sc = {}
    for si, g in zip(s, gains):
        agg[g["key"]] += si; sc[g["key"]] = g["score"]
    ks = list(agg)
    c = np.corrcoef([agg[k] for k in ks], [sc[k] for k in ks])[0, 1]
    print(f"  {name:24s} AUC={a_all:.3f} (early={a_e:.3f} late={a_l:.3f})  "
          f"mean@bad={mean_bad:+.3f}  corr(sum,score)={c:+.3f}")


print("\n--- single channels ---")
ev("zS (IGPO)", zS)
ev("zT", zT)

print("\n--- additive ---")
for b in (0.25, 0.5, 1.0, 2.0):
    ev(f"zS + {b}*zT", zS + b * zT)

print("\n--- gap (subtractive) ---")
for b in (0.5, 1.0):
    ev(f"zS - {b}*zT", zS - b * zT)

print("\n--- multiplicative ---")
ev("zS * zT", zS * zT)
ev("min(zS, zT)", np.minimum(zS, zT))
sgn = np.sign(zS)
agreem = (np.sign(zS) == np.sign(zT))
ev("sign-geo (agree only)", np.where(agreem, sgn * np.sqrt(np.abs(zS * zT)), 0.0))
for b in (5.0, 20.0):
    ev(f"dS * exp({b}*dT)", dS * np.exp(b * dT))

print("\n--- gated / phase additive ---")
for b in (0.5, 1.0, 2.0):
    ev(f"zS + {b}*zT*execgate", zS + b * zT * exec_gate)
ev("phase: zS early, zT late", np.where(pos_frac <= 0.34, zS, zT))
ev("phase-smooth (1-p)zS+p*zT", (1 - pos_frac) * zS + pos_frac * zT)

# oracle-ish reference: how separable is it at all with both channels?
print("\n--- reference ---")
from numpy.linalg import lstsq
X = np.stack([zS, zT, zS * exec_gate, zT * exec_gate], 1)[lab_mask]
y = good[lab_mask].astype(float)
w, *_ = lstsq(np.column_stack([X, np.ones(len(X))]), y, rcond=None)
s_fit = (np.stack([zS, zT, zS * exec_gate, zT * exec_gate], 1) @ w[:4]) + w[4]
ev("lstsq(4 feats) oracle", s_fit)

print("\nIG_GAP_ANALYZE6_DONE")

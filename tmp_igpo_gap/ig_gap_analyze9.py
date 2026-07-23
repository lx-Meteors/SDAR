"""Additive vs multiplicative combination with the REAL group advantage.

WebShop probe data has 4 sibling rollouts per (fn, task) -> reconstruct the
GRPO outcome advantage A = (score - group mean) / group std.  Shaping term
S_t = Delta_t (p_S + p_T)  (parameter-free, prob space).

Forms:
  outcome only   A
  additive       A + b*S_t
  multiplicative A * (1 + b*S_t)
  mult clamped   A * max(0, 1 + b*S_t)
  hybrid         A + b*|A|*S_t

Checks:
  1. turn-level AUC(good vs bad) of combined credit
  2. sign correctness: %good turns credit>0, %bad turns credit<0
  3. same restricted to FAILED trajectories (A<0)  [mult inversion regime]
  4. same restricted to zero-advantage groups (|A|<0.05) [shaping death]
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

# group advantage per (fn, task)
scores = {}
for (fn, task, k), ss in traj.items():
    scores.setdefault((fn, task), {})[k] = ss[0]["score"]
adv = {}
for gk, d in scores.items():
    v = np.array(list(d.values()))
    mu, sd = v.mean(), v.std()
    for k, s in d.items():
        adv[(gk[0], gk[1], k)] = (s - mu) / (sd + 1e-8) if sd > 1e-8 else 0.0

gains = []
for key, ss in traj.items():
    ss.sort(key=lambda r: r["t"])
    for i in range(1, len(ss)):
        a, b = ss[i - 1], ss[i]
        S = (np.exp(b["lp_S_ans"]) + np.exp(b["lp_T1_ans"])
             - np.exp(a["lp_S_ans"]) - np.exp(a["lp_T1_ans"]))
        good = bool((b["first_vis_t"] == b["t"]) or a["gold_click"]
                    or a["act_type"] == "click_opt_match")
        bad = bool((a["act_type"] == "click_asin" and not a["gold_click"]
                    and a["gold_visible"] and not b["gold_visible"])
                   or a["act_type"] == "click_other")
        gains.append(dict(key=key, S=S, A=adv[key], good=good, bad=bad,
                          score=a["score"]))

S = np.array([g["S"] for g in gains])
A = np.array([g["A"] for g in gains])
good = np.array([g["good"] for g in gains])
bad = np.array([g["bad"] for g in gains])
mask = good | bad
print(f"n={len(gains)}  A: mean={A.mean():+.2f} std={A.std():.2f} "
      f"zero-adv turns={np.mean(np.abs(A)<0.05):.0%} "
      f"S: std={S.std():.3f} p95|S|={np.percentile(np.abs(S),95):.3f}")


def auc(s, l):
    npos, nneg = l.sum(), (~l).sum()
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    return (ranks[l].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def report(name, C):
    m = mask
    a_gb = auc(C[m], good[m])
    sg = np.mean(C[good] > 0)          # good turns positive credit
    sb = np.mean(C[bad] < 0)           # bad turns negative credit
    fail = mask & (A < -0.05)
    zg = mask & (np.abs(A) < 0.05)
    a_fail = auc(C[fail], good[fail]) if fail.sum() > 10 else float("nan")
    sb_fail = np.mean(C[bad & (A < -0.05)] < 0) if (bad & (A < -0.05)).sum() else float("nan")
    a_zero = auc(C[zg], good[zg]) if zg.sum() > 10 else float("nan")
    print(f"  {name:22s} AUC={a_gb:.3f}  good>0:{sg:.0%} bad<0:{sb:.0%}  "
          f"failed-traj AUC={a_fail:.3f} bad<0:{sb_fail if isinstance(sb_fail,float) else 0:.0%}  "
          f"zeroAdv AUC={a_zero:.3f}")


print("\nb chosen so b*S typical magnitude ~ A magnitude (A std=1, S std={:.3f})".format(S.std()))
report("A (outcome only)", A.copy())
for b in (1.0, 2.0, 5.0):
    report(f"add  A+{b}*S", A + b * S)
for b in (1.0, 2.0, 5.0):
    report(f"mult A*(1+{b}*S)", A * (1 + b * S))
for b in (2.0,):
    report(f"multc A*max(0,1+{b}S)", A * np.maximum(0, 1 + b * S))
for b in (2.0, 5.0):
    report(f"hyb  A+{b}*|A|*S", A + b * np.abs(A) * S)

# the decisive micro-example: bad turns inside failed trajectories
print("\n--- bad turns in failed trajectories: mean credit (should be strongly negative) ---")
sel = bad & (A < -0.05)
print(f"  n={sel.sum()}")
for name, C in (("A", A), ("add A+2S", A + 2 * S), ("mult A*(1+2S)", A * (1 + 2 * S)),
                ("hyb A+2|A|S", A + 2 * np.abs(A) * S)):
    print(f"  {name:14s} mean={C[sel].mean():+.3f}")

print("\n--- good turns in zero-advantage groups: mean credit (should be >0, not 0) ---")
sel = good & (np.abs(A) < 0.05)
print(f"  n={sel.sum()}")
for name, C in (("A", A), ("add A+2S", A + 2 * S), ("mult A*(1+2S)", A * (1 + 2 * S))):
    print(f"  {name:14s} mean={C[sel].mean():+.3f}  nonzero={np.mean(np.abs(C[sel])>1e-6):.0%}")

print("\nIG_GAP_ANALYZE9_DONE")

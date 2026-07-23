"""Analyze deep-search teacher/student per-turn gold-answer logp gains.

Labels: arrival (gold numeric core first appears in this turn's observations),
        err (all tool calls of the turn errored).
Evaluate the same gate-free formula family as on WebShop."""
import collections
import json
import os

import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
rows = json.load(open(os.path.join(HERE, "ds_gap_rows.json")))

eps = collections.defaultdict(list)
for r in rows:
    eps[(r["fn"], r["ep"])].append(r)

gains = []
for key, ss in eps.items():
    ss.sort(key=lambda r: r["t"])
    for i in range(1, len(ss)):
        a, b = ss[i - 1], ss[i]
        gains.append(dict(key=key, t=b["t"], n=b["n_acts"],
                          arrival=b["arrival"], err=b["err"],
                          dS=b["lpS"] - a["lpS"], dT=b["lpT"] - a["lpT"],
                          lpS=b["lpS"], lpT=b["lpT"]))

dS = np.array([g["dS"] for g in gains])
dT = np.array([g["dT"] for g in gains])
arr_ = np.array([g["arrival"] for g in gains])
err = np.array([g["err"] for g in gains])
print(f"episodes={len(eps)} turn-gains={len(gains)} arrival={arr_.sum()} err={err.sum()}")

lvS = np.array([g["lpS"] for g in gains]); lvT = np.array([g["lpT"] for g in gains])
print(f"levels: lpS mean={lvS.mean():+.3f}  lpT mean={lvT.mean():+.3f}")
print(f"gains:  dS mean={dS.mean():+.4f} std={dS.std():.4f} | "
      f"dT mean={dT.mean():+.4f} std={dT.std():.4f}  ratio std {dS.std()/dT.std():.1f}x")
print(f"corr(dS,dT): pearson={np.corrcoef(dS,dT)[0,1]:+.3f} "
      f"spearman={spearmanr(dS,dT).correlation:+.3f}")

zS = (dS - dS.mean()) / dS.std()
zT = (dT - dT.mean()) / dT.std()


def auc(s, l):
    s = np.asarray(s, float); l = np.asarray(l, bool)
    npos, nneg = l.sum(), (~l).sum()
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    return (ranks[l].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def ev(name, s):
    a_arr = auc(s, arr_)
    a_err = auc(-s, err)      # err turns should get LOW credit
    print(f"  {name:22s} AUC(arrival)={a_arr:.3f}  AUC(err gets low)={a_err:.3f}  "
          f"mean@arrival={s[arr_].mean():+.3f} mean@err={s[err].mean():+.3f} "
          f"mean@rest={s[~arr_ & ~err].mean():+.3f}")


print("\n--- per-turn credit candidates ---")
ev("zS (IGPO)", zS)
ev("zT", zT)
for b in (0.25, 0.5, 1.0):
    ev(f"zS + {b}*zT", zS + b * zT)
ev("z(dS - dT) PMI-close", ((dS - dT) - (dS - dT).mean()) / (dS - dT).std())
# residual per episode
res = np.zeros(len(gains))
idx = collections.defaultdict(list)
for i, g in enumerate(gains):
    idx[g["key"]].append(i)
for k, ii in idx.items():
    x, y = dS[ii], dT[ii]
    x0 = x - x.mean()
    den = (x0 ** 2).sum()
    rho = (x0 * (y - y.mean())).sum() / den if den > 1e-12 else 0.0
    res[ii] = y - y.mean() - rho * x0
zr = (res - res.mean()) / (res.std() + 1e-12)
for b in (0.25, 0.5):
    ev(f"zS + {b}*z(residT)", zS + b * zr)

# where do big teacher/student gains sit?
print("\n--- top-decile turns ---")
for name, v in (("dS", dS), ("dT", dT)):
    k = max(1, len(v) // 10)
    top = np.argsort(-v)[:k]
    print(f"  top10% {name}: arrival={np.mean(arr_[top]):.0%} err={np.mean(err[top]):.0%} "
          f"(base arrival={arr_.mean():.0%} err={err.mean():.0%})")
    bot = np.argsort(v)[:k]
    print(f"  bot10% {name}: arrival={np.mean(arr_[bot]):.0%} err={np.mean(err[bot]):.0%}")

# per-episode: is arrival the argmax turn?
print("\n--- within-episode argmax hit rate (episodes with exactly 1 arrival) ---")
hitS = hitT = hitC = n1 = 0
for k, ii in idx.items():
    la = arr_[ii]
    if la.sum() != 1:
        continue
    n1 += 1
    tgt = int(np.argmax(la))
    hitS += int(np.argmax(dS[ii]) == tgt)
    hitT += int(np.argmax(dT[ii]) == tgt)
    hitC += int(np.argmax(zS[ii] + 0.5 * zT[ii]) == tgt)
print(f"  n={n1}  argmax(dS)={hitS/n1:.0%}  argmax(dT)={hitT/n1:.0%}  "
      f"argmax(zS+0.5zT)={hitC/n1:.0%}  chance≈{np.mean([1/len(v) for v in idx.values()]):.0%}")

print("\nDS_GAP_ANALYZE_DONE")

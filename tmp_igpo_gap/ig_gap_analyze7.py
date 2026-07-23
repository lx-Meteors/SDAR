"""Gate-free formula family shootout on WebShop rows.

Constraint: credit A_t must be a function of the two series {dS_t}, {dT_t}
only (per trajectory / batch), no action-type or page features, no gates.

Families (all potential-based up to monotone per-channel maps):
  raw-z additive      A = zS + b*zT                     (global z)
  traj-z additive     A = zS' + b*zT'                   (per-trajectory z)
  prob additive       A = dpS + b*dpT   (IGPO prob_diff space, bounded)
  rank additive       A = nq(dS) + b*nq(dT)             (per-traj normal scores)
  residual additive   A = zS + b*z(dT - rho*dS)         (common-mode rejection,
                                                         rho per trajectory)
  PMI closure         A = z(dS - dT)
Metrics: good-vs-bad AUC (all/early/late), mean credit on bad turns,
corr(sum credit, final score).
"""
import collections
import json
import os

import numpy as np
from scipy.stats import norm, rankdata

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
            act_type=a["act_type"], gold_click=a["gold_click"],
            led_discovery=(b["first_vis_t"] == b["t"]),
            vis_before=a["gold_visible"], vis_after=b["gold_visible"],
            lpS_a=a["lp_S_ans"], lpS_b=b["lp_S_ans"],
            lpT_a=a["lp_T1_ans"], lpT_b=b["lp_T1_ans"]))

for g in gains:
    g["dS"] = g["lpS_b"] - g["lpS_a"]
    g["dT"] = g["lpT_b"] - g["lpT_a"]
    g["dpS"] = np.exp(g["lpS_b"]) - np.exp(g["lpS_a"])
    g["dpT"] = np.exp(g["lpT_b"]) - np.exp(g["lpT_a"])
    g["good"] = bool(g["led_discovery"] or g["gold_click"]
                     or g["act_type"] == "click_opt_match")
    g["bad"] = bool((g["act_type"] == "click_asin" and not g["gold_click"]
                     and g["vis_before"] and not g["vis_after"])
                    or g["act_type"] == "click_other")

dS = np.array([g["dS"] for g in gains])
dT = np.array([g["dT"] for g in gains])
dpS = np.array([g["dpS"] for g in gains])
dpT = np.array([g["dpT"] for g in gains])
good = np.array([g["good"] for g in gains])
bad = np.array([g["bad"] for g in gains])
mask = good | bad
pos_frac = np.array([g["ta"] / max(g["n_steps"] - 1, 1) for g in gains])
keys = [g["key"] for g in gains]
scores = np.array([g["score"] for g in gains])

zS = (dS - dS.mean()) / dS.std()
zT = (dT - dT.mean()) / dT.std()

# per-trajectory transforms
def per_traj(fn):
    out = np.zeros(len(gains))
    idx = collections.defaultdict(list)
    for i, k in enumerate(keys):
        idx[k].append(i)
    for k, ii in idx.items():
        out[ii] = fn(np.array(ii))
    return out

def traj_z(x):
    def f(ii):
        v = x[ii]
        s = v.std()
        return (v - v.mean()) / (s if s > 1e-8 else 1.0)
    return per_traj(f)

def traj_nq(x):
    def f(ii):
        v = x[ii]
        r = rankdata(v) / (len(v) + 1)
        return norm.ppf(r)
    return per_traj(f)

def traj_resid(y, x):
    """residual of y on x, slope fit per trajectory"""
    def f(ii):
        vx, vy = x[ii], y[ii]
        vx0 = vx - vx.mean()
        den = (vx0 ** 2).sum()
        rho = (vx0 * (vy - vy.mean())).sum() / den if den > 1e-12 else 0.0
        return vy - vy.mean() - rho * vx0
    return per_traj(f)

tzS, tzT = traj_z(dS), traj_z(dT)
nqS, nqT = traj_nq(dS), traj_nq(dT)
rT = traj_resid(dT, dS)
zrT = (rT - rT.mean()) / rT.std()
# prob-space z for comparability of beta
zpS = (dpS - dpS.mean()) / dpS.std()
zpT = (dpT - dpT.mean()) / dpT.std()


def auc(s, l):
    npos, nneg = l.sum(), (~l).sum()
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    return (ranks[l].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def ev(name, s):
    m = mask
    a_all = auc(s[m], good[m])
    early = m & (pos_frac <= 0.34); late = m & (pos_frac > 0.34)
    a_e = auc(s[early], good[early]); a_l = auc(s[late], good[late])
    agg = collections.defaultdict(float); sc = {}
    for si, k, scv in zip(s, keys, scores):
        agg[k] += si; sc[k] = scv
    ks = list(agg)
    c = np.corrcoef([agg[k] for k in ks], [sc[k] for k in ks])[0, 1]
    print(f"  {name:26s} AUC={a_all:.3f} (e={a_e:.3f} l={a_l:.3f}) "
          f"mean@bad={s[bad].mean():+.3f} corr(sum,score)={c:+.3f}")


print(f"n={len(gains)} good={good.sum()} bad={bad.sum()}")
print("\n--- single ---")
ev("zS", zS); ev("zT", zT); ev("nq(dS)", nqS); ev("nq(dT)", nqT)
ev("dpS (prob)", dpS); ev("dpT (prob)", dpT); ev("z(residual dT|dS)", zrT)

for fam, (A, B) in {
    "global-z": (zS, zT),
    "traj-z": (tzS, tzT),
    "rank-nq": (nqS, nqT),
    "prob-z": (zpS, zpT),
    "resid": (zS, zrT),
}.items():
    print(f"\n--- {fam} additive ---")
    for b in (0.25, 0.5, 1.0, 2.0):
        ev(f"{fam}: A + {b}*B", A + b * B)

print("\n--- prob additive raw (IGPO natural units) ---")
for b in (0.5, 1.0, 2.0, 4.0):
    ev(f"dpS + {b}*dpT", dpS + b * dpT)

print("\n--- PMI closure ---")
ev("z(dS - dT)", ((dS - dT) - (dS - dT).mean()) / (dS - dT).std())

print("\n--- rank + resid combo ---")
nqrT = traj_nq(rT)
for b in (0.5, 1.0):
    ev(f"nqS + {b}*nq(residT)", nqS + b * nqrT)

print("\nIG_GAP_ANALYZE7_DONE")

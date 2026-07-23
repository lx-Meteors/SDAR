"""Outcome-anchored belief residual: shaping WITHOUT group centering.

User direction: use log-gap but drop the group mean.  The bounded form of
log-gap is the absorption ratio
    rho_t = exp(-(log pT - log pS)) = pS / pT  in [0,1]
where the TEACHER is the per-task difficulty normalizer (replaces the group
mean's calibration role).  Anchor with the normalized outcome R~ in [0,1]
(replaces the group mean's centering role):

    v3:   A_t = A_GRPO + beta * (R~ - rho_t)        "belief Bellman residual"

Semantics (four quadrants):
  success & low rho  -> strong +  (succeeded beyond own knowledge)
  success & high rho -> ~0        (already knew; no extra push)
  fail    & high rho -> strong -  (knew everything, still failed = the
                                   browse-don't-buy collapse pattern)
  fail    & low rho  -> ~0        (blind failure, forgivable)

Unbiasedness: rho(s_t) is a state baseline (zero expected gradient);
beta*R~ is plain REINFORCE reweighting. No farmable channel:
on failures the term is -beta*rho_t <= 0, so re-search farming earns nothing.

Compared against v0 (A), v2 group-centered variants, and a fully
parameter-free fusion  v3z: z_group(R~ - rho_t).
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

scores = {}
for (fn, task, k), ss in traj.items():
    scores.setdefault((fn, task), {})[k] = ss[0]["score"]
adv = {}
for gk, d in scores.items():
    v = np.array(list(d.values()))
    mu, sd = v.mean(), v.std()
    for k, s in d.items():
        adv[(gk[0], gk[1], k)] = (s - mu) / (sd + 1e-8) if sd > 1e-8 else 0.0

EPS = 1e-6
recs = []
for key, ss in traj.items():
    ss.sort(key=lambda r: r["t"])
    phis = []
    for r in ss:
        pS, pT = np.exp(r["lp_S_ans"]), np.exp(r["lp_T1_ans"])
        phis.append(dict(sum=pS + pT, rho=pS / max(pT, EPS)))
    for i in range(1, len(ss)):
        a, b = ss[i - 1], ss[i]
        good = bool((b["first_vis_t"] == b["t"]) or a["gold_click"]
                    or a["act_type"] == "click_opt_match")
        bad = bool((a["act_type"] == "click_asin" and not a["gold_click"]
                    and a["gold_visible"] and not b["gold_visible"])
                   or a["act_type"] == "click_other")
        recs.append(dict(g=(key[0], key[1]), A=adv[key], score=a["score"],
                         sum=phis[i - 1]["sum"], rho=phis[i - 1]["rho"],
                         act=a["act_type"], t=a["t"], good=good, bad=bad,
                         is_last_pre_end=(i == len(ss) - 1)))

A = np.array([x["A"] for x in recs])
score = np.array([x["score"] for x in recs])           # already in [0,1]
sm = np.array([x["sum"] for x in recs])
rho = np.array([x["rho"] for x in recs])
good = np.array([x["good"] for x in recs])
bad = np.array([x["bad"] for x in recs])
act = np.array([x["act"] for x in recs])
t = np.array([x["t"] for x in recs])
last_pre = np.array([x["is_last_pre_end"] for x in recs])
groups = np.array(["%s|%s" % (x["g"][0], x["g"][1]) for x in recs])
mask = good | bad
buy_sel = last_pre & (score > 0.9)
farm1 = (act == "search") & (t >= 2) & (score < 0.1)


def center_by_group(x):
    out = np.zeros_like(x)
    for g in np.unique(groups):
        s = groups == g
        out[s] = x[s] - x[s].mean()
    return out


def z_by_group(x):
    out = np.zeros_like(x)
    for g in np.unique(groups):
        s = groups == g
        sd = x[s].std()
        out[s] = (x[s] - x[s].mean()) / (sd + 1e-8) if sd > 1e-8 else 0.0
    return out


def auc(s, l):
    npos, nneg = l.sum(), (~l).sum()
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    return (ranks[l].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def report(name, C):
    fail = mask & (A < -0.05)
    zg = mask & (np.abs(A) < 0.05)
    print(f"  {name:22s} AUC={auc(C[mask], good[mask]):.3f}  "
          f"good>0:{np.mean(C[good]>0):.0%} bad<0:{np.mean(C[bad]<0):.0%}  "
          f"failAUC={auc(C[fail], good[fail]):.3f} zeroAUC={auc(C[zg], good[zg]):.3f}  "
          f"farm1={C[farm1].mean():+.3f}  buy={C[buy_sel].mean():+.3f}")


print(f"n={len(recs)} labeled={mask.sum()} farm1_n={farm1.sum()} buy_n={buy_sel.sum()}\n")

report("v0   A", A.copy())
csum = center_by_group(sm); csum /= (csum.std() + 1e-8)
report("v2   A-2*cen(sum)", A - 2 * csum)
crho = center_by_group(rho); crho /= (crho.std() + 1e-8)
report("v2r  A-2*cen(rho)", A - 2 * crho)
for b in (1.0, 2.0, 4.0):
    report(f"v3   A+{b}*(R-rho)", A + b * (score - rho))
report("v3b  A+2*(1[R=1]-rho)", A + 2 * ((score > 0.99).astype(float) - rho))
report("v3s  A+2*(R-sum/2)", A + 2 * (score - sm / 2))
zres = z_by_group(score - rho)
report("v3z  z(R-rho) alone", zres)
report("v3zA A+z(R-rho)", A + zres)

print("\n--- quadrant sanity of raw residual (R - rho) ---")
for nm, sel in (("succ&low-rho ", (score > 0.9) & (rho < 0.3)),
                ("succ&high-rho", (score > 0.9) & (rho > 0.7)),
                ("fail&high-rho", (score < 0.1) & (rho > 0.7)),
                ("fail&low-rho ", (score < 0.1) & (rho < 0.3))):
    v = (score - rho)[sel]
    print(f"  {nm} n={sel.sum():4d}  mean={v.mean() if len(v) else float('nan'):+.3f}")

print("\n--- all-fail-group behavior (the collapse regime): mean credit by action ---")
allfail = np.abs(A) < 0.05
for nm, C in (("v1  A+2*dS(old)", None), ("v2  A-2*cen(sum)", A - 2 * csum),
              ("v3  A+2*(R-rho)", A + 2 * (score - rho))):
    if C is None:
        continue
    line = "  " + nm + "  "
    for a_ in ("search", "click_asin", "click_opt_match", "click_other", "buy"):
        s = allfail & (act == a_) & (score < 0.1)
        line += f"{a_}={C[s].mean() if s.sum() else float('nan'):+.2f}({s.sum()}) "
    print(line)

print("\nIG_GAP_ANALYZE14_DONE")

"""The safe design family and the best per-state predictor g.

Safety boundary (from the online collapse + PBRS analysis):
  * per-turn DELTA terms  -> farmable (v1 collapse)
  * terminal-anchored     -> farmable (idle on a high-phi page pumps rho_T)
  * outcome-anchored      A_t = A_GRPO + beta*(R~ - g(s_t))  -> safe:
    R~ is env-verified (policy cannot fake it), g(s_t) is a state baseline
    (zero expected gradient).  Within this family the ONLY design choice is g.

g should be the best calibrated per-state predictor of R~ built from the two
belief channels.  Candidates ("logp difference" vs "logp ratio" and friends):

  rho    exp(lpS - lpT) = pS/pT      log-DIFFERENCE via exp link  (current v3)
  kappa  lpT / lpS                   log-RATIO (surprisal ratio, in (0,1])
  pS     student prob alone
  avg    (pS + pT)/2
  geo    sqrt(pS*pT) = exp((lpS+lpT)/2)   joint-belief geometric mean
  fit    linear regression score ~ (lpS, lpT), clipped to [0,1]
         -> HEADROOM upper bound (not a proposal: fitted, not parameter-free)

Tests (WebShop probe rows):
  [1] predictor calibration: corr(g, score) and MSE(R~ - g)
  [2] credit battery for A + 2*(R~ - g): AUC / signs / failAUC / zeroAUC /
      farming fuel / buy credit
  [3] QUADRANT check (per analyze4): median-split |dS|,|dT| per-turn deltas;
      the level-based residual should reproduce sensible quadrant credit
      without ever seeing deltas (C=both-move discovery turns should get the
      most positive credit in successful trajs; stagnant D turns in failed
      trajs the most negative).
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

recs = []
for key, ss in traj.items():
    ss.sort(key=lambda r: r["t"])
    for i in range(1, len(ss)):
        a, b = ss[i - 1], ss[i]
        good = bool((b["first_vis_t"] == b["t"]) or a["gold_click"]
                    or a["act_type"] == "click_opt_match")
        bad = bool((a["act_type"] == "click_asin" and not a["gold_click"]
                    and a["gold_visible"] and not b["gold_visible"])
                   or a["act_type"] == "click_other")
        recs.append(dict(A=adv[key], score=a["score"],
                         lpS=a["lp_S_ans"], lpT=a["lp_T1_ans"],
                         dS=b["lp_S_ans"] - a["lp_S_ans"],
                         dT=b["lp_T1_ans"] - a["lp_T1_ans"],
                         act=a["act_type"], t=a["t"], good=good, bad=bad,
                         is_last_pre_end=(i == len(ss) - 1)))

A = np.array([x["A"] for x in recs])
score = np.array([x["score"] for x in recs])
lpS = np.array([x["lpS"] for x in recs])
lpT = np.array([x["lpT"] for x in recs])
dS = np.array([x["dS"] for x in recs])
dT = np.array([x["dT"] for x in recs])
good = np.array([x["good"] for x in recs])
bad = np.array([x["bad"] for x in recs])
act = np.array([x["act"] for x in recs])
t = np.array([x["t"] for x in recs])
last_pre = np.array([x["is_last_pre_end"] for x in recs])
mask = good | bad
buy_sel = last_pre & (score > 0.9)
farm1 = (act == "search") & (t >= 2) & (score < 0.1)

pS, pT = np.exp(lpS), np.exp(lpT)
X = np.stack([lpS, lpT, np.ones_like(lpS)], 1)
w, *_ = np.linalg.lstsq(X, score, rcond=None)
fit = np.clip(X @ w, 0, 1)

G = {
    "rho (logdiff)": np.exp(lpS - lpT),
    "kappa (logratio)": np.clip(lpT / np.minimum(lpS, -1e-3), 0, 1),
    "pS": pS,
    "avg": (pS + pT) / 2,
    "geo": np.sqrt(pS * pT),
    "fit*(bound)": fit,
}


def auc(s, l):
    npos, nneg = l.sum(), (~l).sum()
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    return (ranks[l].sum() - npos * (npos + 1) / 2) / (npos * nneg)


print("[1] predictor quality of g   +   [2] credit battery for A + 2*(R~ - g)")
for name, g in G.items():
    C = A + 2.0 * (score - g)
    fail = mask & (A < -0.05)
    zg = mask & (np.abs(A) < 0.05)
    print(f"  {name:16s} corr(g,R)={np.corrcoef(g, score)[0,1]:+.3f} "
          f"MSE={np.mean((score-g)**2):.3f} | AUC={auc(C[mask], good[mask]):.3f} "
          f"good>0:{np.mean(C[good]>0):.0%} bad<0:{np.mean(C[bad]<0):.0%} "
          f"failAUC={auc(C[fail], good[fail]):.3f} zeroAUC={auc(C[zg], good[zg]):.3f} "
          f"farm1={C[farm1].mean():+.3f} buy={C[buy_sel].mean():+.3f}")

print("\n[3] quadrant analysis (median split of |dS|,|dT|):"
      " mean credit, v3(rho) vs v2-style centered-sum")
mS, mT = np.median(np.abs(dS)), np.median(np.abs(dT))
quad = np.where((np.abs(dS) >= mS) & (np.abs(dT) >= mT), "C both-large",
        np.where((np.abs(dS) >= mS), "B S-only",
        np.where((np.abs(dT) >= mT), "A T-only", "D both-small")))
C3 = A + 2.0 * (score - G["rho (logdiff)"])
# v2 comparator: centered sum within (fn,task) group
gid = collections.defaultdict(list)
for i, x in enumerate(recs):
    gid[(x["t"] * 0, )].append(i)  # dummy; use score-based groups instead
sumphi = pS + pT
# group centering by task group
gkeys = {}
idx = 0
for key, ss in traj.items():
    for i in range(1, len(ss)):
        gkeys.setdefault((key[0], key[1]), []).append(idx)
        idx += 1
cen = np.zeros_like(sumphi)
for g_, ii in gkeys.items():
    ii = np.array(ii)
    cen[ii] = sumphi[ii] - sumphi[ii].mean()
cen /= (cen.std() + 1e-8)
C2 = A - 2.0 * cen

for q in ("A T-only", "B S-only", "C both-large", "D both-small"):
    s_succ = (quad == q) & (score > 0.9)
    s_fail = (quad == q) & (score < 0.1)
    print(f"  {q:13s} succ(n={s_succ.sum():3d}): v3={C3[s_succ].mean():+.2f} "
          f"v2={C2[s_succ].mean():+.2f}   fail(n={s_fail.sum():3d}): "
          f"v3={C3[s_fail].mean() if s_fail.sum() else float('nan'):+.2f} "
          f"v2={C2[s_fail].mean() if s_fail.sum() else float('nan'):+.2f}")

print("\nIG_GAP_ANALYZE15_DONE")

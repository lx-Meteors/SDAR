"""Parameter-free credit formulas, evaluated uniformly on BOTH domains.

All candidates are per-turn differences of a potential built from the two
belief levels p_S = exp(lpS), p_T = exp(lpT).  No z-scores, no fitted
lambda/beta/rho, no gates.

  dpS            = Delta p_S                    (IGPO prob_diff baseline)
  dS             = Delta log p_S                (IGPO log_prob_diff baseline)
  SUM            = Delta (p_S + p_T)
  PROD           = Delta (p_S * p_T)
  GEO            = Delta sqrt(p_S * p_T)
  LOGSUM         = Delta (log p_S + log p_T)
  PMIC           = Delta (log p_S - log p_T)    (PMI closure)
"""
import collections
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def auc(s, l):
    s = np.asarray(s, float); l = np.asarray(l, bool)
    npos, nneg = l.sum(), (~l).sum()
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    return (ranks[l].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def formulas(lpS_a, lpS_b, lpT_a, lpT_b):
    pS_a, pS_b = np.exp(lpS_a), np.exp(lpS_b)
    pT_a, pT_b = np.exp(lpT_a), np.exp(lpT_b)
    return {
        "dS  (log, IGPO)": lpS_b - lpS_a,
        "dpS (prob, IGPO)": pS_b - pS_a,
        "SUM  D(pS+pT)": (pS_b + pT_b) - (pS_a + pT_a),
        "PROD D(pS*pT)": pS_b * pT_b - pS_a * pT_a,
        "GEO  D(sqrt(pS*pT))": np.sqrt(pS_b * pT_b) - np.sqrt(pS_a * pT_a),
        "LOGSUM D(lpS+lpT)": (lpS_b + lpT_b) - (lpS_a + lpT_a),
        "PMIC D(lpS-lpT)": (lpS_b - lpT_b) - (lpS_a - lpT_a),
    }


# ================= WebShop =================
rows = json.load(open(os.path.join(HERE, "ig_gap_rows.json")))
traj = collections.defaultdict(list)
for r in rows:
    traj[(r["fn"], r["task"], r["k"])].append(r)

W = dict(lpS_a=[], lpS_b=[], lpT_a=[], lpT_b=[], good=[], bad=[], pos=[],
         key=[], score=[])
for key, ss in traj.items():
    ss.sort(key=lambda r: r["t"])
    for i in range(1, len(ss)):
        a, b = ss[i - 1], ss[i]
        good = bool((b["first_vis_t"] == b["t"]) or a["gold_click"]
                    or a["act_type"] == "click_opt_match")
        bad = bool((a["act_type"] == "click_asin" and not a["gold_click"]
                    and a["gold_visible"] and not b["gold_visible"])
                   or a["act_type"] == "click_other")
        W["lpS_a"].append(a["lp_S_ans"]); W["lpS_b"].append(b["lp_S_ans"])
        W["lpT_a"].append(a["lp_T1_ans"]); W["lpT_b"].append(b["lp_T1_ans"])
        W["good"].append(good); W["bad"].append(bad)
        W["pos"].append(a["t"] / max(a["n_steps"] - 1, 1))
        W["key"].append(key); W["score"].append(a["score"])
W = {k: np.array(v) if k != "key" else v for k, v in W.items()}
good, bad = W["good"], W["bad"]
mask = good | bad
early = mask & (W["pos"] <= 0.34); late = mask & (W["pos"] > 0.34)

print("================ WebShop ================")
print(f"n={len(good)} good={good.sum()} bad={bad.sum()}")
F = formulas(W["lpS_a"], W["lpS_b"], W["lpT_a"], W["lpT_b"])
for name, s in F.items():
    agg = collections.defaultdict(float); sc = {}
    for si, k, scv in zip(s, W["key"], W["score"]):
        agg[k] += si; sc[k] = scv
    ks = list(agg)
    c = np.corrcoef([agg[k] for k in ks], [sc[k] for k in ks])[0, 1]
    print(f"  {name:20s} AUC={auc(s[mask], good[mask]):.3f} "
          f"(e={auc(s[early], good[early]):.3f} l={auc(s[late], good[late]):.3f}) "
          f"mean@bad={s[bad].mean():+.4f} corr(sum,score)={c:+.3f}")

# ================= Deep search =================
rows = json.load(open(os.path.join(HERE, "ds_gap_rows.json")))
eps = collections.defaultdict(list)
for r in rows:
    eps[(r["fn"], r["ep"])].append(r)

D = dict(lpS_a=[], lpS_b=[], lpT_a=[], lpT_b=[], arr=[], err=[], key=[])
for key, ss in eps.items():
    ss.sort(key=lambda r: r["t"])
    for i in range(1, len(ss)):
        a, b = ss[i - 1], ss[i]
        D["lpS_a"].append(a["lpS"]); D["lpS_b"].append(b["lpS"])
        D["lpT_a"].append(a["lpT"]); D["lpT_b"].append(b["lpT"])
        D["arr"].append(b["arrival"]); D["err"].append(b["err"])
        D["key"].append(key)
D = {k: np.array(v) if k != "key" else v for k, v in D.items()}
arr_, err = D["arr"], D["err"]

print("\n================ Deep search ================")
print(f"n={len(arr_)} arrival={arr_.sum()} err={err.sum()}")
F = formulas(D["lpS_a"], D["lpS_b"], D["lpT_a"], D["lpT_b"])
idx = collections.defaultdict(list)
for i, k in enumerate(D["key"]):
    idx[k].append(i)
for name, s in F.items():
    hit = n1 = 0
    for k, ii in idx.items():
        la = arr_[ii]
        if la.sum() != 1:
            continue
        n1 += 1
        hit += int(np.argmax(s[ii]) == int(np.argmax(la)))
    print(f"  {name:20s} AUC(arr)={auc(s, arr_):.3f} AUC(err low)={auc(-s, err):.3f} "
          f"mean@arr={s[arr_].mean():+.4f} mean@err={s[err].mean():+.4f} "
          f"argmax-hit={hit}/{n1}={hit/n1:.0%}")

print("\nIG_GAP_ANALYZE8_DONE")

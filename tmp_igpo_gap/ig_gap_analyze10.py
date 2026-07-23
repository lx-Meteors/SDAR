"""Canonical-scale shootout: prob vs log vs LOG-ODDS potentials, both domains.

All parameter-free, single-potential candidates:
  dS        Delta log pS                      (IGPO)
  SUM       Delta (pS + pT)                   (current pick)
  LGT-S     Delta logit(pS)                   (control)
  LGT-T     Delta logit(pT)                   (control)
  LOGIT     Delta [logit(pS) + logit(pT)]
  CROSS     Delta [log pS - log(1 - pT)]

Extra WebShop diagnostics: OPT+ sign %, gold-page OPT+ sign %, WRONGCLK %,
DISC %.  Deep search: AUC(arrival), argmax top1, err ~ 0.
"""
import collections
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EPS = 1e-6


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def cands(lpS0, lpS1, lpT0, lpT1):
    pS0, pS1 = np.exp(lpS0), np.exp(lpS1)
    pT0, pT1 = np.exp(lpT0), np.exp(lpT1)
    return {
        "dS (IGPO log)": lpS1 - lpS0,
        "SUM D(pS+pT)": (pS1 + pT1) - (pS0 + pT0),
        "LGT-S": logit(pS1) - logit(pS0),
        "LGT-T": logit(pT1) - logit(pT0),
        "LOGIT sum": (logit(pS1) + logit(pT1)) - (logit(pS0) + logit(pT0)),
        "CROSS lpS-l(1-pT)": (lpS1 - np.log(1 - np.clip(pT1, EPS, 1 - EPS))) -
                             (lpS0 - np.log(1 - np.clip(pT0, EPS, 1 - EPS))),
    }


def auc(s, l):
    npos, nneg = l.sum(), (~l).sum()
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    return (ranks[l].sum() - npos * (npos + 1) / 2) / (npos * nneg)


# ---------------- WebShop ----------------
rows = json.load(open(os.path.join(HERE, "ig_gap_rows.json")))
traj = collections.defaultdict(list)
for r in rows:
    traj[(r["fn"], r["task"], r["k"])].append(r)

W = collections.defaultdict(list)
on_gold_page = []
for key, ss in traj.items():
    ss.sort(key=lambda r: r["t"])
    on_gold = None
    for i in range(1, len(ss)):
        a, b = ss[i - 1], ss[i]
        if a["act_type"] == "click_asin":
            on_gold = a["gold_click"]
        elif a["act_type"] == "search":
            on_gold = None
        W["lpS0"].append(a["lp_S_ans"]); W["lpS1"].append(b["lp_S_ans"])
        W["lpT0"].append(a["lp_T1_ans"]); W["lpT1"].append(b["lp_T1_ans"])
        W["good"].append(bool((b["first_vis_t"] == b["t"]) or a["gold_click"]
                              or a["act_type"] == "click_opt_match"))
        W["bad"].append(bool((a["act_type"] == "click_asin" and not a["gold_click"]
                              and a["gold_visible"] and not b["gold_visible"])
                             or a["act_type"] == "click_other"))
        W["disc"].append(b["first_vis_t"] == b["t"])
        W["wrong"].append(a["act_type"] == "click_asin" and not a["gold_click"]
                          and a["gold_visible"] and not b["gold_visible"])
        W["opt"].append(a["act_type"] == "click_opt_match")
        on_gold_page.append(bool(on_gold) and a["act_type"] == "click_opt_match")
        W["pos"].append(a["t"] / max(a["n_steps"] - 1, 1))
        W["key"].append(key); W["score"].append(a["score"])
W = {k: (np.array(v) if k != "key" else v) for k, v in W.items()}
ogp = np.array(on_gold_page)
good, bad = W["good"], W["bad"]
mask = good | bad
late = mask & (W["pos"] > 0.34)

print("================ WebShop ================")
F = cands(W["lpS0"], W["lpS1"], W["lpT0"], W["lpT1"])
for name, s in F.items():
    agg = collections.defaultdict(float); sc = {}
    for si, k, scv in zip(s, W["key"], W["score"]):
        agg[k] += si; sc[k] = scv
    ks = list(agg)
    c = np.corrcoef([agg[k] for k in ks], [sc[k] for k in ks])[0, 1]
    print(f"  {name:18s} AUC={auc(s[mask], good[mask]):.3f} late={auc(s[late], good[late]):.3f} | "
          f"DISC>0:{np.mean(s[W['disc']]>0):.0%} WRONG<0:{np.mean(s[W['wrong']]<0):.0%} "
          f"OPT+>0:{np.mean(s[W['opt']]>0):.0%} goldpageOPT>0:{np.mean(s[ogp]>0):.0%} | "
          f"corr={c:+.3f}")

# ---------------- Deep search ----------------
rows = json.load(open(os.path.join(HERE, "ds_gap_rows.json")))
eps = collections.defaultdict(list)
for r in rows:
    eps[(r["fn"], r["ep"])].append(r)

D = collections.defaultdict(list)
for key, ss in eps.items():
    ss.sort(key=lambda r: r["t"])
    for i in range(1, len(ss)):
        a, b = ss[i - 1], ss[i]
        D["lpS0"].append(a["lpS"]); D["lpS1"].append(b["lpS"])
        D["lpT0"].append(a["lpT"]); D["lpT1"].append(b["lpT"])
        D["arr"].append(b["arrival"]); D["err"].append(b["err"])
        D["key"].append(key)
D = {k: (np.array(v) if k != "key" else v) for k, v in D.items()}
arr_, err = D["arr"], D["err"]
idx = collections.defaultdict(list)
for i, k in enumerate(D["key"]):
    idx[k].append(i)

print("\n================ Deep search ================")
F = cands(D["lpS0"], D["lpS1"], D["lpT0"], D["lpT1"])
for name, s in F.items():
    hit = n1 = 0
    for k, ii in idx.items():
        la = arr_[ii]
        if la.sum() != 1:
            continue
        n1 += 1
        hit += int(np.argmax(s[ii]) == int(np.argmax(la)))
    # err turns should sit near zero relative to the candidate's own scale
    scale = np.percentile(np.abs(s), 90)
    err_small = np.mean(np.abs(s[err]) < 0.1 * scale)
    print(f"  {name:18s} AUC(arr)={auc(s, arr_):.3f} argmax-top1={hit}/{n1}={hit/n1:.0%} "
          f"err-near0={err_small:.0%} mean@arr(in own p90 units)={s[arr_].mean()/scale:+.2f}")

print("\nIG_GAP_ANALYZE10_DONE")

"""If exactly ONE knob is allowed: SUM + gamma * Delta logit(pT).
Check what it buys on WebShop (option credit) and costs on deep search."""
import collections
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EPS = 1e-6


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def auc(s, l):
    npos, nneg = l.sum(), (~l).sum()
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    return (ranks[l].sum() - npos * (npos + 1) / 2) / (npos * nneg)


# WebShop
rows = json.load(open(os.path.join(HERE, "ig_gap_rows.json")))
traj = collections.defaultdict(list)
for r in rows:
    traj[(r["fn"], r["task"], r["k"])].append(r)
W = collections.defaultdict(list)
for key, ss in traj.items():
    ss.sort(key=lambda r: r["t"])
    on_gold = None
    for i in range(1, len(ss)):
        a, b = ss[i - 1], ss[i]
        if a["act_type"] == "click_asin":
            on_gold = a["gold_click"]
        elif a["act_type"] == "search":
            on_gold = None
        pS0, pS1 = np.exp(a["lp_S_ans"]), np.exp(b["lp_S_ans"])
        pT0, pT1 = np.exp(a["lp_T1_ans"]), np.exp(b["lp_T1_ans"])
        W["SUM"].append((pS1 + pT1) - (pS0 + pT0))
        W["LT"].append(logit(pT1) - logit(pT0))
        W["good"].append(bool((b["first_vis_t"] == b["t"]) or a["gold_click"]
                              or a["act_type"] == "click_opt_match"))
        W["bad"].append(bool((a["act_type"] == "click_asin" and not a["gold_click"]
                              and a["gold_visible"] and not b["gold_visible"])
                             or a["act_type"] == "click_other"))
        W["wrong"].append(a["act_type"] == "click_asin" and not a["gold_click"]
                          and a["gold_visible"] and not b["gold_visible"])
        W["ogp"].append(bool(on_gold) and a["act_type"] == "click_opt_match")
        W["pos"].append(a["t"] / max(a["n_steps"] - 1, 1))
W = {k: np.array(v) for k, v in W.items()}
mask = W["good"] | W["bad"]
late = mask & (W["pos"] > 0.34)

# deep search
rows = json.load(open(os.path.join(HERE, "ds_gap_rows.json")))
eps = collections.defaultdict(list)
for r in rows:
    eps[(r["fn"], r["ep"])].append(r)
D = collections.defaultdict(list)
for key, ss in eps.items():
    ss.sort(key=lambda r: r["t"])
    for i in range(1, len(ss)):
        a, b = ss[i - 1], ss[i]
        pS0, pS1 = np.exp(a["lpS"]), np.exp(b["lpS"])
        pT0, pT1 = np.exp(a["lpT"]), np.exp(b["lpT"])
        D["SUM"].append((pS1 + pT1) - (pS0 + pT0))
        D["LT"].append(logit(pT1) - logit(pT0))
        D["arr"].append(b["arrival"]); D["key"].append(key)
Dk = D["key"]
D = {k: np.array(v) for k, v in D.items() if k != "key"}
idx = collections.defaultdict(list)
for i, k in enumerate(Dk):
    idx[k].append(i)

print("gamma  | WS AUC  late   WRONG<0 goldpgOPT>0 | DS AUC(arr) argmax-top1")
for g in (0.0, 0.02, 0.05, 0.1, 0.2, 0.5):
    sw = W["SUM"] + g * W["LT"]
    sd = D["SUM"] + g * D["LT"]
    hit = n1 = 0
    for k, ii in idx.items():
        la = D["arr"][ii]
        if la.sum() != 1:
            continue
        n1 += 1
        hit += int(np.argmax(sd[ii]) == int(np.argmax(la)))
    print(f"  {g:4.2f} |  {auc(sw[mask], W['good'][mask]):.3f}  {auc(sw[late], W['good'][late]):.3f}  "
          f"{np.mean(sw[W['wrong']]<0):.0%}     {np.mean(sw[W['ogp']]>0):.0%}      |   "
          f"{auc(sd, D['arr']):.3f}      {hit}/{n1}={hit/n1:.0%}")

print("\nIG_GAP_ANALYZE11_DONE")

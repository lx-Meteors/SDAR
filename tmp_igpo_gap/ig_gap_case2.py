"""Fairness check on the OPT+ 'miss' cases: split option clicks by whether
the agent is currently on the GOLD item's page (last asin click before this
turn was the gold item) vs on a wrong item's page."""
import collections
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
rows = json.load(open(os.path.join(HERE, "ig_gap_rows.json")))

traj = collections.defaultdict(list)
for r in rows:
    traj[(r["fn"], r["task"], r["k"])].append(r)

groups = collections.defaultdict(list)
for key, ss in traj.items():
    ss.sort(key=lambda r: r["t"])
    on_gold = None   # None = not on an item page yet
    for i in range(1, len(ss)):
        a, b = ss[i - 1], ss[i]
        if a["act_type"] == "click_asin":
            on_gold = a["gold_click"]
        elif a["act_type"] == "search":
            on_gold = None
        if a["act_type"] != "click_opt_match":
            continue
        S = (np.exp(b["lp_S_ans"]) + np.exp(b["lp_T1_ans"])
             - np.exp(a["lp_S_ans"]) - np.exp(a["lp_T1_ans"]))
        dT_log = b["lp_T1_ans"] - a["lp_T1_ans"]
        ctx = ("gold-page" if on_gold else
               "wrong-page" if on_gold is not None else "unknown")
        groups[ctx].append((S, dT_log, a["score"]))

print("option clicks matching goal options, split by current item page:")
for ctx in ("gold-page", "wrong-page", "unknown"):
    v = groups[ctx]
    if not v:
        continue
    S = np.array([x[0] for x in v]); dT = np.array([x[1] for x in v])
    sc = np.array([x[2] for x in v])
    print(f"  {ctx:10s} n={len(v):3d}  S>0: {np.mean(S>0):.0%}  meanS={S.mean():+.4f}  "
          f"dT_log>0: {np.mean(dT>0):.0%}  traj succ={np.mean(sc>=0.9):.0%}")

print("\nIG_GAP_CASE2_DONE")

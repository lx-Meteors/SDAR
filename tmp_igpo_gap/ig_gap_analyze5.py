"""Deep dive on the two agreement regimes:
  C: |dS| large AND |dT1| large   (both move)
  D: |dS| small AND |dT1| small   (neither moves)
Large = top quartile within own condition's |gain| scale, small = below median
(same thresholds as ig_gap_analyze4).

For C: split by sign pattern (dS+/-, dT+/-), action type, gold-visibility
transition; check the two mechanistic hypotheses:
  - search rows: dT negative because results page floods context with
    distractor product text (interference) -> test: dT drop vs number of
    asin candidates that appear after the search
  - click_asin rows: dS negative = observation-copy channel severed
    (list page with gold name -> item page without it)
For D: composition, position in trajectory, does anything there predict
outcome at all."""
import collections
import json
import os
import re

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
            fn=a["fn"], task=a["task"], k=a["k"], ta=a["t"], score=a["score"],
            n_steps=a["n_steps"], action=a["action"], act_type=a["act_type"],
            gold_click=a["gold_click"], led_discovery=(b["first_vis_t"] == b["t"]),
            gold_vis_before=a["gold_visible"], gold_vis_after=b["gold_visible"],
            # number of asin candidates visible after the action (proxy for
            # distractor volume on results pages) -- count from next avail
            n_asin_after=len(re.findall(r"click\[b0[0-9a-z]+\]", "")),
            dS=b["lp_S_ans"] - a["lp_S_ans"],
            dT=b["lp_T1_ans"] - a["lp_T1_ans"]))

# recompute n_asin_after properly from the NEXT step's avail
for key, ss in traj.items():
    ss.sort(key=lambda r: r["t"])
idx = 0
for key, ss in traj.items():
    for i in range(1, len(ss)):
        pass
# simpler: rebuild with avail
gains = []
for key, ss in traj.items():
    ss.sort(key=lambda r: r["t"])
    for i in range(1, len(ss)):
        a, b = ss[i - 1], ss[i]
        gains.append(dict(
            fn=a["fn"], task=a["task"], k=a["k"], ta=a["t"], score=a["score"],
            n_steps=a["n_steps"], action=a["action"], act_type=a["act_type"],
            gold_click=a["gold_click"], led_discovery=(b["first_vis_t"] == b["t"]),
            gold_vis_before=a["gold_visible"], gold_vis_after=b["gold_visible"],
            dS=b["lp_S_ans"] - a["lp_S_ans"],
            dT=b["lp_T1_ans"] - a["lp_T1_ans"]))

dS = np.array([g["dS"] for g in gains])
dT = np.array([g["dT"] for g in gains])

def prank(x):
    o = np.argsort(np.argsort(np.abs(x)))
    return o / (len(x) - 1)
pS, pT = prank(dS), prank(dT)
HI, LO = 0.75, 0.50

C = [g for i, g in enumerate(gains) if pS[i] >= HI and pT[i] >= HI]
D = [g for i, g in enumerate(gains) if pS[i] < LO and pT[i] < LO]

# ---------------- C ----------------
print(f"===== C: both large  n={len(C)} =====")
pat = collections.defaultdict(list)
for g in C:
    pat[("S+" if g["dS"] > 0 else "S-") + ("T+" if g["dT"] > 0 else "T-")].append(g)
for name in ("S+T+", "S+T-", "S-T+", "S-T-"):
    b = pat[name]
    if not b:
        print(f"  {name}: none"); continue
    at = collections.Counter(g["act_type"] for g in b)
    print(f"  {name}: n={len(b):3d} act={dict(at.most_common())} "
          f"disc={np.mean([g['led_discovery'] for g in b]):.0%} "
          f"gold_click={np.mean([g['gold_click'] for g in b]):.0%} "
          f"succ={np.mean([g['score']>=0.9 for g in b]):.0%} "
          f"vis {np.mean([g['gold_vis_before'] for g in b]):.0%}->"
          f"{np.mean([g['gold_vis_after'] for g in b]):.0%}")

# hypothesis 1: search rows in C -- dT sign vs whether the search FOUND gold
print("\n  -- C search rows: does finding gold determine dT sign?")
cs = [g for g in C if g["act_type"] == "search"]
for lab, sel in (("found gold (disc)", [g for g in cs if g["led_discovery"]]),
                 ("no gold found", [g for g in cs if not g["led_discovery"]])):
    if sel:
        print(f"     {lab:18s} n={len(sel):3d} dT mean={np.mean([g['dT'] for g in sel]):+.4f} "
              f"dT>0 {np.mean([g['dT']>0 for g in sel]):.0%}  "
              f"dS mean={np.mean([g['dS'] for g in sel]):+.3f}")

# hypothesis 2: click_asin rows in C -- copy channel severed?
print("\n  -- C click_asin rows: gold visible before click -> after?")
cc = [g for g in C if g["act_type"] == "click_asin"]
sev = [g for g in cc if g["gold_vis_before"] and not g["gold_vis_after"]]
oth = [g for g in cc if not (g["gold_vis_before"] and not g["gold_vis_after"])]
for lab, sel in (("vis->hidden (severed)", sev), ("other transition", oth)):
    if sel:
        print(f"     {lab:22s} n={len(sel):3d} dS mean={np.mean([g['dS'] for g in sel]):+.3f} "
              f"dT mean={np.mean([g['dT'] for g in sel]):+.4f} "
              f"gold_click={np.mean([g['gold_click'] for g in sel]):.0%} "
              f"succ={np.mean([g['score']>=0.9 for g in sel]):.0%}")

# does C's dT sign on click_asin carry ANY outcome info?
print("\n  -- C click_asin: dT sign vs trajectory outcome")
for lab, sel in (("dT>0", [g for g in cc if g["dT"] > 0]),
                 ("dT<0", [g for g in cc if g["dT"] < 0])):
    if sel:
        print(f"     {lab}: n={len(sel)} succ={np.mean([g['score']>=0.9 for g in sel]):.0%} "
              f"gold_click={np.mean([g['gold_click'] for g in sel]):.0%}")

# ---------------- D ----------------
print(f"\n===== D: both small  n={len(D)} =====")
at = collections.Counter(g["act_type"] for g in D)
print(f"  act={dict(at.most_common())}")
print(f"  disc={np.mean([g['led_discovery'] for g in D]):.0%} "
      f"gold_click={np.mean([g['gold_click'] for g in D]):.0%} "
      f"succ={np.mean([g['score']>=0.9 for g in D]):.0%} "
      f"vis {np.mean([g['gold_vis_before'] for g in D]):.0%}->"
      f"{np.mean([g['gold_vis_after'] for g in D]):.0%}")
# position in trajectory
pos = np.array([g["ta"] / max(g["n_steps"] - 1, 1) for g in D])
posC = np.array([g["ta"] / max(g["n_steps"] - 1, 1) for g in C])
print(f"  relative position in traj: D mean={pos.mean():.2f}  C mean={posC.mean():.2f}")
# D option clicks: does the tiny dT still separate match vs other?
dop = [g for g in D if g["act_type"] in ("click_opt_match", "click_other")]
lab = np.array([g["act_type"] == "click_opt_match" for g in dop])
dt = np.array([g["dT"] for g in dop]); ds = np.array([g["dS"] for g in dop])
def auc(s, l):
    npos, nneg = l.sum(), (~l).sum()
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(s); ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    return (ranks[l].sum() - npos * (npos + 1) / 2) / (npos * nneg)
print(f"  D option clicks n={len(dop)}: AUC(match) via dT={auc(dt, lab):.3f} via dS={auc(ds, lab):.3f}")
# gold_clicks that landed in D
dg = [g for g in D if g["gold_click"]]
print(f"  gold clicks inside D: n={len(dg)} "
      f"succ={np.mean([g['score']>=0.9 for g in dg]):.0%}" if dg else "  no gold clicks in D")
# do D turns matter at all? within-traj: share of D turns vs score
shares, scores = [], []
keyf = lambda g: (g["fn"], g["task"], g["k"])
Dset = {(keyf(g), g["ta"]) for g in D}
for key, ss in traj.items():
    n = len(ss) - 1
    if n < 1:
        continue
    nd = sum(1 for t in range(n) if (key, ss[t]["t"]) in Dset)
    shares.append(nd / n); scores.append(ss[0]["score"])
print(f"  corr(share of D-turns in traj, score) = {np.corrcoef(shares, scores)[0,1]:+.3f}")

# sign agreement inside D (is the tiny movement still directional?)
agree = np.mean([(g["dS"] > 0) == (g["dT"] > 0) for g in D])
print(f"  sign agreement dS vs dT inside D: {agree:.0%}")

print("\nIG_GAP_ANALYZE5_DONE")

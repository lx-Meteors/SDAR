"""Characterize the two disagreement regimes (magnitude, not sign):
  A: |dT1| large, |dS| small   -- teacher moves, student doesn't
  B: |dS| large, |dT1| small   -- student moves, teacher doesn't
Magnitudes compared in within-condition percentile space (scales differ ~50x).
Split each regime by gain sign, action type; print concrete examples."""
import collections
import json
import os

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
            action=a["action"], act_type=a["act_type"],
            gold_click=a["gold_click"], led_discovery=(b["first_vis_t"] == b["t"]),
            gold_vis_before=a["gold_visible"], gold_vis_after=b["gold_visible"],
            dS=b["lp_S_ans"] - a["lp_S_ans"],
            dT=b["lp_T1_ans"] - a["lp_T1_ans"]))

dS = np.array([g["dS"] for g in gains])
dT = np.array([g["dT"] for g in gains])
# percentile rank of |gain| within each condition
def prank(x):
    o = np.argsort(np.argsort(np.abs(x)))
    return o / (len(x) - 1)
pS, pT = prank(dS), prank(dT)
HI, LO = 0.75, 0.50   # large = top quartile of own scale, small = below median

def describe(name, idx):
    sub = [gains[i] for i in idx]
    if not sub:
        print(f"\n### {name}: EMPTY"); return
    at = collections.Counter(g["act_type"] for g in sub)
    pos_T = np.mean([gains[i]["dT"] > 0 for i in idx])
    pos_S = np.mean([gains[i]["dS"] > 0 for i in idx])
    print(f"\n### {name}  n={len(sub)}")
    print(f"    act={dict(at.most_common())}")
    print(f"    disc={np.mean([g['led_discovery'] for g in sub]):.0%} "
          f"goldclick={np.mean([g['gold_click'] for g in sub]):.0%} "
          f"succ={np.mean([g['score']>=0.9 for g in sub]):.0%} "
          f"sign: dT>0 {pos_T:.0%}, dS>0 {pos_S:.0%}")
    print(f"    gold visible before->after: "
          f"{np.mean([g['gold_vis_before'] for g in sub]):.0%} -> "
          f"{np.mean([g['gold_vis_after'] for g in sub]):.0%}")
    # split by sign of the DOMINANT side
    dom = "dT" if "T-large" in name else "dS"
    for sgn, lab in ((1, "positive"), (-1, "negative")):
        ss2 = [g for g in sub if np.sign(g[dom]) == sgn]
        if not ss2:
            continue
        at2 = collections.Counter(g["act_type"] for g in ss2)
        print(f"    {dom} {lab}: n={len(ss2)} act={dict(at2.most_common())} "
              f"succ={np.mean([g['score']>=0.9 for g in ss2]):.0%}")
    # examples: most extreme dominant gain, dedup by action string
    seen = set(); ex = []
    for g in sorted(sub, key=lambda g: -abs(g[dom])):
        if g["action"] in seen:
            continue
        seen.add(g["action"]); ex.append(g)
        if len(ex) == 10:
            break
    for g in ex:
        print(f"      dS={g['dS']:+.3f} dT={g['dT']:+.4f} score={g['score']:.2f} "
              f"disc={int(g['led_discovery'])} gold={int(g['gold_click'])} "
              f"{g['act_type']:14s} {g['action'][:58]!r}")

idx_A = [i for i in range(len(gains)) if pT[i] >= HI and pS[i] < LO]
idx_B = [i for i in range(len(gains)) if pS[i] >= HI and pT[i] < LO]
idx_both = [i for i in range(len(gains)) if pS[i] >= HI and pT[i] >= HI]
idx_none = [i for i in range(len(gains)) if pS[i] < LO and pT[i] < LO]

print(f"total gain-steps={len(gains)}  |dS| p75={np.percentile(np.abs(dS),75):.3f} "
      f"|dT| p75={np.percentile(np.abs(dT),75):.4f}")
describe("A: T-large & S-small (teacher moves, student flat)", idx_A)
describe("B: S-large & T-small (student moves, teacher flat)", idx_B)
describe("C: both large (S-large, T-large)", idx_both)
describe("D: both small (S-small, T-small)", idx_none)

print("\nIG_GAP_ANALYZE4_DONE")

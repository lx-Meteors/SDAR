"""Case-level audit of the final credit  C_t = A_grpo + beta * S_t,
S_t = Delta_t(p_S + p_T), beta = 2.

1. print representative trajectories turn by turn (success / failed / zero-adv
   group) with per-turn credit and a verdict vs. the derivable label
2. aggregate sign-correctness by action category
3. list the worst mis-credited turns (good&most-negative, bad&most-positive)
4. loop-farming check: net S over list->item->list round trips (should ~0)
"""
import collections
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BETA = 2.0
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
    sd = v.std()
    for k, s in d.items():
        adv[(gk[0], gk[1], k)] = (s - v.mean()) / (sd + 1e-8) if sd > 1e-8 else 0.0

recs = collections.defaultdict(list)   # key -> list of per-turn records
for key, ss in traj.items():
    ss.sort(key=lambda r: r["t"])
    for i in range(1, len(ss)):
        a, b = ss[i - 1], ss[i]
        pS0, pS1 = np.exp(a["lp_S_ans"]), np.exp(b["lp_S_ans"])
        pT0, pT1 = np.exp(a["lp_T1_ans"]), np.exp(b["lp_T1_ans"])
        S = (pS1 + pT1) - (pS0 + pT0)
        good = bool((b["first_vis_t"] == b["t"]) or a["gold_click"]
                    or a["act_type"] == "click_opt_match")
        bad = bool((a["act_type"] == "click_asin" and not a["gold_click"]
                    and a["gold_visible"] and not b["gold_visible"])
                   or a["act_type"] == "click_other")
        lab = ("DISC" if b["first_vis_t"] == b["t"] else
               "GOLDCLK" if a["gold_click"] else
               "OPT+" if a["act_type"] == "click_opt_match" else
               "WRONGCLK" if bad and a["act_type"] == "click_asin" else
               "OPT-/NAV" if bad else "neutral")
        recs[key].append(dict(
            t=a["t"], action=a["action"][:56], lab=lab, good=good, bad=bad,
            act_type=a["act_type"], pS0=pS0, pS1=pS1, pT0=pT0, pT1=pT1,
            S=S, A=adv[key], C=adv[key] + BETA * S, score=a["score"]))


def verdict(r):
    if r["good"]:
        return "OK " if r["C"] > 0 else "MISS"
    if r["bad"]:
        return "OK " if r["C"] < 0 else "MISS"
    return "  -"


def show(key, title):
    rs = recs[key]
    print(f"\n### {title}  {key[0][-6:]}:task{key[1]} k{key[2]} "
          f"score={rs[0]['score']:.2f} A={rs[0]['A']:+.2f}")
    print("   t  label     credit    S      pS->      pT->      v  action")
    for r in rs:
        print(f"  {r['t']:2d}  {r['lab']:8s} {r['C']:+.3f} {r['S']:+.3f} "
              f"{r['pS0']:.2f}>{r['pS1']:.2f} {r['pT0']:.2f}>{r['pT1']:.2f} "
              f"{verdict(r)} {r['action']!r}")


# pick representative trajectories
all_keys = list(recs)
succ = [k for k in all_keys if recs[k][0]["score"] >= 0.99 and abs(recs[k][0]["A"]) > 0.1
        and any(r["lab"] == "DISC" for r in recs[k]) and any(r["lab"] == "OPT+" for r in recs[k])]
fail = [k for k in all_keys if recs[k][0]["A"] < -0.3
        and any(r["lab"] == "WRONGCLK" for r in recs[k])]
zero = [k for k in all_keys if abs(recs[k][0]["A"]) < 1e-6
        and any(r["lab"] == "DISC" for r in recs[k]) and any(r["bad"] for r in recs[k])]
if succ:
    show(succ[0], "SUCCESS traj (A>0)")
if fail:
    show(fail[0], "FAILED traj (A<0) with wrong click")
if zero:
    show(zero[0], "ZERO-ADVANTAGE group traj (A=0, shaping only)")

# 2. aggregate sign correctness
print("\n===== sign correctness by category (C = A + 2S) =====")
flat = [r for rs in recs.values() for r in rs]
for lab in ("DISC", "GOLDCLK", "OPT+", "WRONGCLK", "OPT-/NAV", "neutral"):
    sel = [r for r in flat if r["lab"] == lab]
    if not sel:
        continue
    want_pos = sel[0]["good"]
    C = np.array([r["C"] for r in sel]); Sv = np.array([r["S"] for r in sel])
    ok = np.mean(C > 0) if want_pos else np.mean(C < 0)
    okS = np.mean(Sv > 0) if want_pos else np.mean(Sv < 0)
    tgt = ">0" if want_pos else ("<0" if sel[0]["bad"] else "~0")
    print(f"  {lab:9s} n={len(sel):3d} want{tgt}: C-correct={ok:.0%} "
          f"(S alone {okS:.0%})  meanC={C.mean():+.3f} meanS={Sv.mean():+.3f}")

# 3. worst mis-credits
print("\n===== worst mis-credited turns =====")
gm = sorted([r for r in flat if r["good"]], key=lambda r: r["C"])[:6]
print("  good turns with most NEGATIVE credit:")
for r in gm:
    print(f"    C={r['C']:+.3f} S={r['S']:+.3f} A={r['A']:+.2f} {r['lab']:8s} "
          f"score={r['score']:.2f} {r['action']!r}")
bm = sorted([r for r in flat if r["bad"]], key=lambda r: -r["C"])[:6]
print("  bad turns with most POSITIVE credit:")
for r in bm:
    print(f"    C={r['C']:+.3f} S={r['S']:+.3f} A={r['A']:+.2f} {r['lab']:8s} "
          f"score={r['score']:.2f} {r['action']!r}")

# 4. loop-farming: trajectories that leave a list page and come back
print("\n===== loop-farming check (net S over leave-and-return cycles) =====")
cyc_sums = []
for key, rs in recs.items():
    # find i<j where vis drops at i (click away) and comes back by j
    for i, r in enumerate(rs):
        if r["act_type"] == "click_asin" and r["pS1"] < r["pS0"]:
            # look for a later return where pS recovers to within 10%
            for j in range(i + 1, len(rs)):
                if rs[j]["pS1"] >= r["pS0"] * 0.9:
                    cyc_sums.append(sum(x["S"] for x in rs[i:j + 1]))
                    break
            break
print(f"  cycles found={len(cyc_sums)}  net S: mean={np.mean(cyc_sums):+.4f} "
      f"median={np.median(cyc_sums):+.4f} p90|.|={np.percentile(np.abs(cyc_sums),90):.4f}"
      if cyc_sums else "  no cycles found")

print("\nIG_GAP_CASE_DONE")

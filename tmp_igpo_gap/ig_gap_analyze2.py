"""Follow-up: (a) strict critical label AUC, (b) can teacher gain discriminate
gold vs wrong asin-clicks (something the student fundamentally cannot know),
(c) sign-quadrant characterization of dS vs dT1."""
import collections
import json
import os

import numpy as np
from scipy.stats import mannwhitneyu

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
        g = dict(fn=a["fn"], task=a["task"], k=a["k"], ta=a["t"], score=a["score"],
                 action=a["action"], act_type=a["act_type"], forced=a["forced"],
                 gold_click=a["gold_click"], led_discovery=(b["first_vis_t"] == b["t"]),
                 gold_visible_now=a["gold_visible"])
        for tgt in ("ans", "act"):
            for c in ("S", "T1", "T2"):
                la, lb = a[f"lp_{c}_{tgt}"], b[f"lp_{c}_{tgt}"]
                g[f"d{c}_{tgt}"] = (lb - la) if (la is not None and lb is not None) else None
        gains.append(g)


def auc(scores, labels):
    s = np.asarray(scores, float); l = np.asarray(labels, bool)
    npos = l.sum(); nneg = (~l).sum()
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    return (ranks[l].sum() - npos * (npos + 1) / 2) / (npos * nneg)


# ---- (a) strict critical label: discovery search or gold click ----
print("===== (a) STRICT critical = led_discovery | gold_click =====")
for tgt in ("ans", "act"):
    ok = [g for g in gains if all(g[f"d{c}_{tgt}"] is not None for c in ("S", "T1", "T2"))]
    lab = [bool(g["led_discovery"] or g["gold_click"]) for g in ok]
    dS = np.array([g[f"dS_{tgt}"] for g in ok])
    dT1 = np.array([g[f"dT1_{tgt}"] for g in ok])
    dT2 = np.array([g[f"dT2_{tgt}"] for g in ok])
    print(f"  target={tgt} n={len(ok)} crit_rate={np.mean(lab):.0%}")
    for name, s in [("dS (IGPO)", dS), ("dT1", dT1), ("dT2", dT2),
                    ("gap dS-dT1", dS - dT1), ("gap dT1-dS", dT1 - dS),
                    ("dS+1*dT1", dS + dT1), ("dS+2*dT1", dS + 2 * dT1),
                    ("dS+5*dT1", dS + 5 * dT1), ("dS+20*dT1", dS + 20 * dT1),
                    ("dS+1*dT2", dS + dT2)]:
        print(f"    {name:12s} AUC={auc(s, lab):.3f}")

# ---- (b) click-decision discrimination: gold vs wrong asin click ----
print("\n===== (b) among click_asin actions: gold vs wrong click =====")
clicks = [g for g in gains if g["act_type"] == "click_asin"]
print(f"  n={len(clicks)}  gold={sum(g['gold_click'] for g in clicks)}")
for tgt in ("ans", "act"):
    ok = [g for g in clicks if all(g[f"d{c}_{tgt}"] is not None for c in ("S", "T1", "T2"))]
    lab = [g["gold_click"] for g in ok]
    for c in ("S", "T1", "T2"):
        d = np.array([g[f"d{c}_{tgt}"] for g in ok])
        dg = d[np.array(lab)]; dw = d[~np.array(lab)]
        u = mannwhitneyu(dg, dw, alternative="two-sided")
        print(f"  {tgt}/d{c}: AUC(gold vs wrong)={auc(d, lab):.3f}  "
              f"gold mean={dg.mean():+.4f} wrong mean={dw.mean():+.4f}  p={u.pvalue:.2g}")

# same but for option clicks: matching vs non-matching option
print("\n===== (b2) among option clicks: goal-matching vs other =====")
opts = [g for g in gains if g["act_type"] in ("click_opt_match", "click_other")]
for tgt in ("ans", "act"):
    ok = [g for g in opts if all(g[f"d{c}_{tgt}"] is not None for c in ("S", "T1", "T2"))]
    lab = [g["act_type"] == "click_opt_match" for g in ok]
    for c in ("S", "T1", "T2"):
        d = np.array([g[f"d{c}_{tgt}"] for g in ok])
        print(f"  {tgt}/d{c}: AUC(match vs other)={auc(d, lab):.3f} "
              f"match mean={d[np.array(lab)].mean():+.4f} other mean={d[~np.array(lab)].mean():+.4f}")

# ---- (c) sign quadrants dS x dT1 (ans) ----
print("\n===== (c) sign quadrants of (dS_ans, dT1_ans) =====")
ok = [g for g in gains if g["dS_ans"] is not None and g["dT1_ans"] is not None]
qs = collections.defaultdict(list)
for g in ok:
    qs[("S+" if g["dS_ans"] > 0 else "S-") + ("T+" if g["dT1_ans"] > 0 else "T-")].append(g)
for name in ("S+T+", "S+T-", "S-T+", "S-T-"):
    b = qs[name]
    if not b:
        continue
    at = collections.Counter(g["act_type"] for g in b)
    print(f"  {name}: n={len(b):3d} disc={np.mean([g['led_discovery'] for g in b]):.0%} "
          f"goldclick={np.mean([g['gold_click'] for g in b]):.0%} "
          f"succ={np.mean([g['score']>=0.9 for g in b]):.0%} act={dict(at)}")

# ---- (d) does dT1 on the click step predict trajectory outcome? ----
print("\n===== (d) first asin-click's dT1_ans vs final score =====")
first_click = {}
for g in gains:
    if g["act_type"] == "click_asin" and (g["fn"], g["task"], g["k"]) not in first_click:
        first_click[(g["fn"], g["task"], g["k"])] = g
xs = [g["dT1_ans"] for g in first_click.values() if g["dT1_ans"] is not None]
ys = [g["score"] for g in first_click.values() if g["dT1_ans"] is not None]
print(f"  n={len(xs)} corr(dT1_ans@first_click, score)={np.corrcoef(xs, ys)[0,1]:+.3f}")
xs2 = [g["dS_ans"] for g in first_click.values() if g["dS_ans"] is not None]
print(f"  corr(dS_ans@first_click, score)={np.corrcoef(xs2, ys)[0,1]:+.3f}")

print("\nIG_GAP_ANALYZE2_DONE")

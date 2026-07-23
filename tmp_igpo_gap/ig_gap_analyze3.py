"""Robustness: (a) dedupe near-identical sibling steps, (b) z-normalized
A = z(dS) + beta*z(dT1) over beta grid, (c) per-task AUC (no cross-task
pooling leakage), (d) explicit ratio dT1/dS test."""
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
            dS=(b["lp_S_ans"] - a["lp_S_ans"]),
            dT1=(b["lp_T1_ans"] - a["lp_T1_ans"]),
            dT2=(b["lp_T2_ans"] - a["lp_T2_ans"]) if (a["lp_T2_ans"] is not None and
                                                      b["lp_T2_ans"] is not None) else None))


def auc(s, l):
    s = np.asarray(s, float); l = np.asarray(l, bool)
    npos, nneg = l.sum(), (~l).sum()
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    return (ranks[l].sum() - npos * (npos + 1) / 2) / (npos * nneg)


# (a) dedupe: identical (fn, task, ta, action) across siblings count once
seen, dedup = set(), []
for g in gains:
    key = (g["fn"], g["task"], g["ta"], g["action"])
    if key in seen:
        continue
    seen.add(key)
    dedup.append(g)
print(f"steps: raw={len(gains)} dedup={len(dedup)}")

for name, G in (("RAW", gains), ("DEDUP", dedup)):
    lab = [bool(g["led_discovery"] or g["gold_click"]) for g in G]
    dS = np.array([g["dS"] for g in G])
    dT1 = np.array([g["dT1"] for g in G])
    zS = (dS - dS.mean()) / dS.std()
    zT = (dT1 - dT1.mean()) / dT1.std()
    print(f"\n===== {name} strict-critical AUC (n={len(G)}, rate={np.mean(lab):.0%}) =====")
    print(f"  z(dS) alone      AUC={auc(zS, lab):.3f}")
    print(f"  z(dT1) alone     AUC={auc(zT, lab):.3f}")
    for b in (0.1, 0.25, 0.5, 1.0, 2.0):
        print(f"  z(dS)+{b:4.2f}*z(dT1) AUC={auc(zS + b * zT, lab):.3f}")
    for b in (0.25, 0.5, 1.0):
        print(f"  z(dS)-{b:4.2f}*z(dT1) AUC={auc(zS - b * zT, lab):.3f}")
    # ratio (guard tiny denominators)
    ratio = dT1 / np.where(np.abs(dS) < 1e-3, np.nan, dS)
    m = ~np.isnan(ratio)
    print(f"  ratio dT1/dS     AUC={auc(ratio[m], np.array(lab)[m]):.3f} (n={m.sum()})")

# (c) per-task AUC of best candidates (avoid cross-task pooling artifacts)
print("\n===== per-task AUC (median over tasks with both classes) =====")
bytask = collections.defaultdict(list)
for g in dedup:
    bytask[(g["fn"], g["task"])].append(g)
res = collections.defaultdict(list)
for key, G in bytask.items():
    lab = [bool(g["led_discovery"] or g["gold_click"]) for g in G]
    if not (0 < sum(lab) < len(lab)):
        continue
    dS = np.array([g["dS"] for g in G]); dT1 = np.array([g["dT1"] for g in G])
    sd_s = dS.std() or 1.0; sd_t = dT1.std() or 1.0
    res["dS"].append(auc(dS, lab))
    res["dT1"].append(auc(dT1, lab))
    res["z+0.5z"].append(auc(dS / sd_s + 0.5 * dT1 / sd_t, lab))
    res["z+1.0z"].append(auc(dS / sd_s + 1.0 * dT1 / sd_t, lab))
for k, v in res.items():
    print(f"  {k:8s} median={np.median(v):.3f} mean={np.mean(v):.3f} n_tasks={len(v)}")

# (d) option-click discrimination after dedup
opts = [g for g in dedup if g["act_type"] in ("click_opt_match", "click_other")]
lab = [g["act_type"] == "click_opt_match" for g in opts]
for c in ("dS", "dT1"):
    d = np.array([g[c] for g in opts])
    print(f"\noption clicks dedup n={len(opts)}: {c} AUC(match)={auc(d, lab):.3f}")

print("\nIG_GAP_ANALYZE3_DONE")

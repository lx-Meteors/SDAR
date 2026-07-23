"""Analysis for ig_gap_collect.py output.

Per-turn gain dX(t) = lp_X(prompt_t) - lp_X(prompt_{t-1}) is credited to the
ACTION taken at step t-1 (it produced observation_t).  Questions:
  1. teacher (T1 outcome-gold / T2 procedural) gain distribution vs student;
     does T1's verbatim-answer prefix flatten gains (ceiling)?
  2. do large-|gain| turns coincide across teacher and student?
  3. do gains / gaps localize on "critical" actions
     (discovery search, gold-asin click, matching-option click)?
  4. candidate credit A = dS + beta*dT (or gap / ratio): AUC for critical turns.
"""
import collections
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
rows = json.load(open(os.path.join(HERE, "ig_gap_rows.json")))

# ---- rebuild per-trajectory sequences, compute gains ----
traj = collections.defaultdict(list)
for r in rows:
    traj[(r["fn"], r["task"], r["k"])].append(r)

gains = []   # one record per action step ta = t-1
for key, ss in traj.items():
    ss.sort(key=lambda r: r["t"])
    for i in range(1, len(ss)):
        a, b = ss[i - 1], ss[i]     # action at a["t"] produced state b
        g = dict(fn=a["fn"], task=a["task"], k=a["k"], ta=a["t"],
                 score=a["score"], action=a["action"], act_type=a["act_type"],
                 forced=a["forced"], gold_click=a["gold_click"],
                 led_discovery=(b["first_vis_t"] == b["t"]),
                 gold_visible_after=b["gold_visible"])
        for tgt in ("ans", "act"):
            for c in ("S", "T1", "T2"):
                la, lb = a[f"lp_{c}_{tgt}"], b[f"lp_{c}_{tgt}"]
                g[f"d{c}_{tgt}"] = (lb - la) if (la is not None and lb is not None) else None
                g[f"lp{c}_{tgt}"] = lb if lb is not None else None
        gains.append(g)

print(f"trajs={len(traj)}  action-steps with gain={len(gains)}")

# critical action label
for g in gains:
    g["crit"] = bool(g["led_discovery"] or g["gold_click"]
                     or g["act_type"] == "click_opt_match")
ncrit = sum(g["crit"] for g in gains)
print(f"critical actions: {ncrit} ({ncrit/len(gains):.0%})  "
      f"[led_discovery={sum(g['led_discovery'] for g in gains)}, "
      f"gold_click={sum(g['gold_click'] for g in gains)}, "
      f"opt_match={sum(g['act_type']=='click_opt_match' for g in gains)}]")

def arr(key, sel=None):
    xs = [(g[key], g) for g in gains if g[key] is not None and (sel is None or sel(g))]
    return np.array([x for x, _ in xs]), [g for _, g in xs]

# ---- 1. levels and gain spread ----
print("\n===== 1. logp LEVELS (per-state, mean) and GAIN spread =====")
for tgt in ("ans", "act"):
    for c in ("S", "T1", "T2"):
        lv = np.array([r[f"lp_{c}_{tgt}"] for r in rows if r[f"lp_{c}_{tgt}"] is not None])
        d, _ = arr(f"d{c}_{tgt}")
        print(f"  {tgt}/{c}: level mean={lv.mean():+.3f}  "
              f"gain mean={d.mean():+.4f} std={d.std():.4f} "
              f"|gain| p50={np.median(np.abs(d)):.4f} p90={np.percentile(np.abs(d),90):.4f}")

# ---- 2. consistency student vs teacher ----
print("\n===== 2. consistency of per-turn gains (pooled corr / per-traj Spearman) =====")
from scipy.stats import spearmanr
for tgt in ("ans", "act"):
    for c in ("T1", "T2"):
        pairs = [(g[f"dS_{tgt}"], g[f"d{c}_{tgt}"], g) for g in gains
                 if g[f"dS_{tgt}"] is not None and g[f"d{c}_{tgt}"] is not None]
        x = np.array([p[0] for p in pairs]); y = np.array([p[1] for p in pairs])
        pear = np.corrcoef(x, y)[0, 1]
        sp_all = spearmanr(x, y).correlation
        # per-traj
        bytr = collections.defaultdict(list)
        for a, b, g in pairs:
            bytr[(g["fn"], g["task"], g["k"])].append((a, b))
        sps = [spearmanr([p[0] for p in v], [p[1] for p in v]).correlation
               for v in bytr.values() if len(v) >= 4]
        sps = [s for s in sps if not np.isnan(s)]
        # top-quintile overlap: is a top-|dS| turn also top-|dC|?
        n = len(x); k = max(1, n // 5)
        topS = set(np.argsort(-np.abs(x))[:k]); topC = set(np.argsort(-np.abs(y))[:k])
        jac = len(topS & topC) / len(topS | topC)
        print(f"  {tgt}: dS vs d{c}  pearson={pear:+.3f} spearman={sp_all:+.3f} "
              f"per-traj spearman med={np.median(sps):+.3f}  top20%|.|-overlap jaccard={jac:.2f}")

# ---- 3. what are the large-gain turns ----
def describe(name, key, top_frac=0.15):
    d, gs = arr(key)
    k = max(1, int(len(d) * top_frac))
    idx = np.argsort(-d)[:k]           # most POSITIVE gains
    sub = [gs[i] for i in idx]
    at = collections.Counter(g["act_type"] for g in sub)
    print(f"  {name} top{top_frac:.0%} positive (n={k}): crit={np.mean([g['crit'] for g in sub]):.0%} "
          f"disc={np.mean([g['led_discovery'] for g in sub]):.0%} "
          f"goldclick={np.mean([g['gold_click'] for g in sub]):.0%} "
          f"succ={np.mean([g['score']>=0.9 for g in sub]):.0%}  act={dict(at)}")
    idxn = np.argsort(d)[:k]           # most NEGATIVE
    subn = [gs[i] for i in idxn]
    atn = collections.Counter(g["act_type"] for g in subn)
    print(f"  {name} top{top_frac:.0%} negative (n={k}): crit={np.mean([g['crit'] for g in subn]):.0%} "
          f"succ={np.mean([g['score']>=0.9 for g in subn]):.0%}  act={dict(atn)}")

print("\n===== 3. composition of large-gain turns =====")
base_crit = np.mean([g["crit"] for g in gains])
base_succ = np.mean([g["score"] >= 0.9 for g in gains])
print(f"  base rates: crit={base_crit:.0%} succ={base_succ:.0%}")
for tgt in ("ans", "act"):
    for c in ("S", "T1", "T2"):
        describe(f"d{c}_{tgt}", f"d{c}_{tgt}")

# ---- 4. per-act-type mean gains ----
print("\n===== 4. mean gain by action type =====")
for tgt in ("ans",):
    print(f"  target={tgt}")
    for at in ("search", "click_asin", "click_opt_match", "click_other", "buy", "invalid"):
        sel = [g for g in gains if g["act_type"] == at and g[f"dS_{tgt}"] is not None]
        if not sel:
            continue
        m = lambda k: np.mean([g[k] for g in sel if g[k] is not None])
        print(f"    {at:16s} n={len(sel):4d}  dS={m(f'dS_{tgt}'):+.4f} "
              f"dT1={m(f'dT1_{tgt}'):+.4f} dT2={m(f'dT2_{tgt}'):+.4f} "
              f"gap1={m(f'dT1_{tgt}')-m(f'dS_{tgt}'):+.4f}")
    for lab, sel_f in (("led_discovery", lambda g: g["led_discovery"]),
                       ("gold_click", lambda g: g["gold_click"]),
                       ("non-critical", lambda g: not g["crit"])):
        sel = [g for g in gains if sel_f(g) and g[f"dS_{tgt}"] is not None]
        m = lambda k: np.mean([g[k] for g in sel if g[k] is not None])
        print(f"    {lab:16s} n={len(sel):4d}  dS={m(f'dS_{tgt}'):+.4f} "
              f"dT1={m(f'dT1_{tgt}'):+.4f} dT2={m(f'dT2_{tgt}'):+.4f}")

# ---- 5. AUC for identifying critical turns ----
def auc(scores, labels):
    s = np.asarray(scores, float); l = np.asarray(labels, bool)
    if l.all() or (~l).any() == 0:
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    npos = l.sum(); nneg = (~l).sum()
    return (ranks[l].sum() - npos * (npos + 1) / 2) / (npos * nneg)

print("\n===== 5. AUC(critical-turn detection) =====")
for tgt in ("ans", "act"):
    ok = [g for g in gains if all(g[f"d{c}_{tgt}"] is not None for c in ("S", "T1", "T2"))]
    lab = [g["crit"] for g in ok]
    dS = np.array([g[f"dS_{tgt}"] for g in ok])
    dT1 = np.array([g[f"dT1_{tgt}"] for g in ok])
    dT2 = np.array([g[f"dT2_{tgt}"] for g in ok])
    cands = {
        "dS (IGPO)": dS, "dT1": dT1, "dT2": dT2,
        "|dS|": np.abs(dS), "|dT1|": np.abs(dT1),
        "gap dT1-dS": dT1 - dS, "gap dS-dT1": dS - dT1,
        "gap dT2-dS": dT2 - dS,
        "min(dS,dT1)": np.minimum(dS, dT1), "dS*dT1>0 both+": ((dS > 0) & (dT1 > 0)).astype(float),
    }
    for b in (0.5, 1.0, 2.0):
        cands[f"dS+{b}*dT1"] = dS + b * dT1
        cands[f"dS+{b}*dT2"] = dS + b * dT2
    print(f"  target={tgt}  (n={len(ok)}, crit rate={np.mean(lab):.0%})")
    for name, s in cands.items():
        print(f"    {name:16s} AUC={auc(s, lab):.3f}")

# ---- 6. outcome correlation: sum of gains vs final score ----
print("\n===== 6. trajectory-level: cumulative gain vs final score =====")
for tgt in ("ans", "act"):
    for c in ("S", "T1", "T2"):
        xs, ys = [], []
        for key, ss in traj.items():
            gs = [g for g in gains if (g["fn"], g["task"], g["k"]) == key
                  and g[f"d{c}_{tgt}"] is not None]
            if not gs:
                continue
            xs.append(sum(g[f"d{c}_{tgt}"] for g in gs)); ys.append(gs[0]["score"])
        if len(xs) > 3:
            print(f"  {tgt}/{c}: corr(sum gain, score)={np.corrcoef(xs, ys)[0,1]:+.3f} n={len(xs)}")

# ---- 7. concrete examples ----
print("\n===== 7. examples: largest teacher-student disagreement (ans) =====")
ok = [g for g in gains if g["dS_ans"] is not None and g["dT1_ans"] is not None]
for title, keyf in (("dT1 >> dS (teacher sees gain, student doesn't)",
                     lambda g: g["dT1_ans"] - g["dS_ans"]),
                    ("dS >> dT1 (student gain, teacher already knew)",
                     lambda g: g["dS_ans"] - g["dT1_ans"])):
    print(f"  -- {title}")
    for g in sorted(ok, key=keyf, reverse=True)[:8]:
        print(f"     {g['fn'][-6:-5]}:t{g['task']}k{g['k']}ta{g['ta']} score={g['score']:.2f} "
              f"dS={g['dS_ans']:+.3f} dT1={g['dT1_ans']:+.3f} dT2="
              f"{'--' if g['dT2_ans'] is None else format(g['dT2_ans'], '+.3f')} "
              f"crit={int(g['crit'])} disc={int(g['led_discovery'])} {g['act_type']:14s} "
              f"{g['action'][:52]!r}")

print("\nIG_GAP_ANALYZE_DONE")

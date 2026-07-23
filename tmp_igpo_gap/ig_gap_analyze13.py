"""Which potential channel for the v2 baseline: p_S+p_T vs p_T-p_S vs log-gap?

Candidates (per state c_t):
  sum      pS + pT                    total belief (current choice)
  gap_p    pT - pS                    privileged-information deficit
  gap_log  log pT - log pS            pointwise info-gain of privilege
  pS       student belief alone       (IGPO channel)
  pT       teacher belief alone
  ratio    pS / pT                    absorption fraction (0..1)

Theory to test:
  * A baseline should be a good VALUE proxy: monotone predictor of outcome R.
    gap conflates "both know" (near success) with "both blind" (hopeless):
    both give small gap -> non-monotone in true value -> weaker baseline.
  * log-gap explodes when pS -> 0 (noise amplification, cf. analyze10).

Tests on WebShop rows (4 siblings per task -> real group structure):
  1. within-group corr(phi_t, R): control-variate quality (rho, rho^2 = the
     fraction of variance a linear baseline can remove).
  2. v2 credit  A - beta * phi_centered  with per-candidate global std
     normalization; metrics: good/bad AUC, sign rates, failed-traj AUC,
     buy-turn credit (pre-terminal turn of successful trajs).
Tests on deep-search rows (all-success trajs, no outcome variance):
  3. progress tracking: corr(phi, t/n) and step-diff SNR
     = |mean(phi_end - phi_0)| / std(per-step diffs)  (monotone progress vs noise).
"""
import collections
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
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
    mu, sd = v.mean(), v.std()
    for k, s in d.items():
        adv[(gk[0], gk[1], k)] = (s - mu) / (sd + 1e-8) if sd > 1e-8 else 0.0

EPS = 1e-6
CANDS = {
    "sum":     lambda pS, pT: pS + pT,
    "gap_p":   lambda pS, pT: pT - pS,
    "gap_log": lambda pS, pT: np.log(pT + EPS) - np.log(pS + EPS),
    "pS":      lambda pS, pT: pS,
    "pT":      lambda pS, pT: pT,
    "ratio":   lambda pS, pT: pS / np.maximum(pT, EPS),
}

# flatten turns with group ids
recs = []
for key, ss in traj.items():
    ss.sort(key=lambda r: r["t"])
    for i, r in enumerate(ss):
        recs.append(dict(g=(key[0], key[1]), key=key, i=i, n=len(ss),
                         pS=np.exp(r["lp_S_ans"]), pT=np.exp(r["lp_T1_ans"]),
                         score=r["score"], A=adv[key], row=r))

print("=" * 30, "WebShop", "=" * 30)
print("\n[1] within-group corr(phi_t, R)  (control-variate quality)")
for name, f in CANDS.items():
    xs, ys = [], []
    bygroup = collections.defaultdict(list)
    for rc in recs:
        bygroup[rc["g"]].append(rc)
    for g, rs in bygroup.items():
        ph = np.array([f(rc["pS"], rc["pT"]) for rc in rs])
        sc = np.array([rc["score"] for rc in rs])
        if sc.std() < 1e-8 or ph.std() < 1e-8:
            continue
        xs.append((ph - ph.mean()) / ph.std())
        ys.append((sc - sc.mean()) / sc.std())
    x, y = np.concatenate(xs), np.concatenate(ys)
    rho = float(np.mean(x * y))
    print(f"  {name:8s} rho={rho:+.3f}   rho^2={rho*rho:.3f}")

# [2] v2 credit metrics -- reuse analyze12 label construction
gphi = {}
for name, f in CANDS.items():
    d = collections.defaultdict(list)
    for rc in recs:
        d[rc["g"]].append(f(rc["pS"], rc["pT"]))
    gphi[name] = {g: float(np.mean(v)) for g, v in d.items()}

gains = []
for key, ss in traj.items():
    ss.sort(key=lambda r: r["t"])
    for i in range(1, len(ss)):
        a, b = ss[i - 1], ss[i]
        good = bool((b["first_vis_t"] == b["t"]) or a["gold_click"]
                    or a["act_type"] == "click_opt_match")
        bad = bool((a["act_type"] == "click_asin" and not a["gold_click"]
                    and a["gold_visible"] and not b["gold_visible"])
                   or a["act_type"] == "click_other")
        rec = dict(A=adv[key], good=good, bad=bad, score=a["score"],
                   is_last_pre_end=(i == len(ss) - 1), g=(key[0], key[1]))
        for name, f in CANDS.items():
            rec[name] = f(np.exp(a["lp_S_ans"]), np.exp(a["lp_T1_ans"]))
        gains.append(rec)

A = np.array([x["A"] for x in gains])
good = np.array([x["good"] for x in gains])
bad = np.array([x["bad"] for x in gains])
score = np.array([x["score"] for x in gains])
last_pre = np.array([x["is_last_pre_end"] for x in gains])
mask = good | bad
buy_sel = last_pre & (score > 0.9)


def auc(s, l):
    npos, nneg = l.sum(), (~l).sum()
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    return (ranks[l].sum() - npos * (npos + 1) / 2) / (npos * nneg)


print("\n[2] v2 credit  A - 2*phi_centered/std(phi_centered)  (sign auto-oriented)")
for name in CANDS:
    ph = np.array([x[name] for x in gains], dtype=float)
    cen = ph - np.array([gphi[name][x["g"]] for x in gains])
    cen = cen / (cen.std() + 1e-8)
    # orient so that potential is POSITIVELY correlated with outcome (value proxy)
    sgn = 1.0 if np.corrcoef(ph, score)[0, 1] >= 0 else -1.0
    C = A - 2.0 * sgn * cen
    fail = mask & (A < -0.05)
    print(f"  {name:8s} AUC={auc(C[mask], good[mask]):.3f}  "
          f"good>0:{np.mean(C[good]>0):.0%} bad<0:{np.mean(C[bad]<0):.0%}  "
          f"failAUC={auc(C[fail], good[fail]):.3f}  buy={C[buy_sel].mean():+.3f}")

# [3] deep search progress tracking
DS = os.path.join(HERE, "ds_gap_rows.json")
if os.path.exists(DS):
    print("\n" + "=" * 30, "Deep Search", "=" * 30)
    ds = json.load(open(DS))
    eps_map = collections.defaultdict(list)
    for r in ds:
        eps_map[(r["fn"], r["ep"])].append(r)
    print("[3] corr(phi, progress t/n) and SNR = |phi_end-phi_0| / std(step diffs)")
    for name, f in CANDS.items():
        cors, snrs = [], []
        for ep, rs in eps_map.items():
            rs.sort(key=lambda r: r["t"])
            ph = np.array([f(np.exp(r["lpS"]), np.exp(r["lpT"])) for r in rs])
            tt = np.arange(len(ph), dtype=float)
            if len(ph) < 3 or ph.std() < 1e-9:
                continue
            cors.append(np.corrcoef(ph, tt)[0, 1])
            dd = np.diff(ph)
            if dd.std() > 1e-9:
                snrs.append(abs(ph[-1] - ph[0]) / dd.std())
        print(f"  {name:8s} corr={np.mean(cors):+.3f}   SNR={np.mean(snrs):.2f}")

print("\nIG_GAP_ANALYZE13_DONE")

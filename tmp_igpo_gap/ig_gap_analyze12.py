"""v1 (delta-phi bonus) vs centering variants vs v2 (phi as state baseline).

Motivated by the ONLINE collapse of A + beta*Delta(phi): the policy learned
risk-averse belief farming (search = safe +S, click = risky -S, buy = 0).
Centering removes constant offsets but not the ACTION-TYPE conditional means
E[S|search] > 0 > E[S|click] that fuel the hack.

v2 derivation (strict PBRS, gamma=1, terminal potential = 0):
    r'_t = r_t + phi(s_{t+1}) - phi(s_t)   =>   G'_t = R - phi(s_t)
i.e. telescoped shaped return = outcome minus belief BEFORE the turn.
Equivalent to REINFORCE with a state baseline (Wiewiora 2003) -> unbiased,
and unfarmable: a turn's credit does not depend on the Delta(phi) it causes.

Compared on WebShop probe rows (ig_gap_rows.json):
  v0        A                                (GRPO outcome only)
  v1        A + b*S_t                        (online-collapsed form)
  v1cT      A + b*(S_t - mean_traj S)
  v1cG      A + b*(S_t - mean_group S)
  v2        A - b*(phi_t - mean_group phi)   (belief as baseline)

Metrics: good/bad AUC, sign correctness, failed-traj AUC, zero-adv AUC,
plus FARMING PROXIES:
  farm1 = mean credit of re-search turns (search at t>=2) in FAILED trajs
          (v1's positive value here is exactly the hack fuel)
  farm2 = mean credit of 'stagnate at high belief' turns in FAILED trajs
          (phi_t in top quartile, action is not buy/click_asin)
  buy   = mean credit of the turn RIGHT BEFORE success-side termination
          (v1 structurally gives ~0 to final buy; v2 should reward it)
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

phi_of = lambda r: np.exp(r["lp_S_ans"]) + np.exp(r["lp_T1_ans"])

# per-group phi mean (over all turns of all siblings, mirrors online batch stats)
gphi = collections.defaultdict(list)
gS = collections.defaultdict(list)
for key, ss in traj.items():
    ss.sort(key=lambda r: r["t"])
    for r in ss:
        gphi[(key[0], key[1])].append(phi_of(r))
    for i in range(1, len(ss)):
        gS[(key[0], key[1])].append(phi_of(ss[i]) - phi_of(ss[i - 1]))
gphi_mean = {g: float(np.mean(v)) for g, v in gphi.items()}
gS_mean = {g: float(np.mean(v)) for g, v in gS.items()}

gains = []
for key, ss in traj.items():
    ss.sort(key=lambda r: r["t"])
    phis = [phi_of(r) for r in ss]
    Ss = [phis[i] - phis[i - 1] for i in range(1, len(ss))]
    S_traj_mean = float(np.mean(Ss)) if Ss else 0.0
    g = (key[0], key[1])
    for i in range(1, len(ss)):
        a, b = ss[i - 1], ss[i]
        S = Ss[i - 1]
        good = bool((b["first_vis_t"] == b["t"]) or a["gold_click"]
                    or a["act_type"] == "click_opt_match")
        bad = bool((a["act_type"] == "click_asin" and not a["gold_click"]
                    and a["gold_visible"] and not b["gold_visible"])
                   or a["act_type"] == "click_other")
        gains.append(dict(key=key, A=adv[key], S=S,
                          ScT=S - S_traj_mean, ScG=S - gS_mean[g],
                          phi=phis[i - 1], phic=phis[i - 1] - gphi_mean[g],
                          act=a["act_type"], t=a["t"], score=a["score"],
                          good=good, bad=bad,
                          is_last_pre_end=(i == len(ss) - 1)))

A = np.array([x["A"] for x in gains])
S = np.array([x["S"] for x in gains])
ScT = np.array([x["ScT"] for x in gains])
ScG = np.array([x["ScG"] for x in gains])
phic = np.array([x["phic"] for x in gains])
good = np.array([x["good"] for x in gains])
bad = np.array([x["bad"] for x in gains])
act = np.array([x["act"] for x in gains])
t = np.array([x["t"] for x in gains])
score = np.array([x["score"] for x in gains])
last_pre = np.array([x["is_last_pre_end"] for x in gains])
mask = good | bad


def auc(s, l):
    npos, nneg = l.sum(), (~l).sum()
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    return (ranks[l].sum() - npos * (npos + 1) / 2) / (npos * nneg)


farm1_sel = (act == "search") & (t >= 2) & (score < 0.1)          # re-search in failed
farm2_sel = (phic > np.percentile(phic, 75)) & (score < 0.1) & \
            ~np.isin(act, ["buy", "click_asin"])                   # stagnate at high belief
buy_sel = last_pre & (score > 0.9)                                 # pre-terminal turn of successes


def report(name, C):
    m = mask
    fail = mask & (A < -0.05)
    zg = mask & (np.abs(A) < 0.05)
    print(f"  {name:8s} AUC={auc(C[m], good[m]):.3f}  good>0:{np.mean(C[good]>0):.0%} "
          f"bad<0:{np.mean(C[bad]<0):.0%}  failAUC={auc(C[fail], good[fail]):.3f} "
          f"zeroAUC={auc(C[zg], good[zg]):.3f}  | farm1={C[farm1_sel].mean():+.3f} "
          f"farm2={C[farm2_sel].mean():+.3f} buy={C[buy_sel].mean():+.3f}")


print(f"n={len(gains)} labeled={mask.sum()} farm1_n={farm1_sel.sum()} "
      f"farm2_n={farm2_sel.sum()} buy_n={buy_sel.sum()}")
print("farming proxies: want farm1/farm2 <= 0 (no fuel), buy > 0 (terminal credited)\n")

report("v0  A", A.copy())
for b in (2.0, 5.0):
    report(f"v1 b={b}", A + b * S)
for b in (2.0, 5.0):
    report(f"v1cT b={b}", A + b * ScT)
for b in (2.0, 5.0):
    report(f"v1cG b={b}", A + b * ScG)
for b in (2.0, 5.0):
    report(f"v2 b={b}", A - b * phic)

# shaping-term-only view (what drives all-fail groups where A ~ 0)
print("\n--- shaping term alone (all-fail regime driver) ---")
report("S", S.copy())
report("ScT", ScT.copy())
report("ScG", ScG.copy())
report("-phic", -phic)

print("\nIG_GAP_ANALYZE12_DONE")

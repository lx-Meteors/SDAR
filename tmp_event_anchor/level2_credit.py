"""
Level-2 (correct objective): CREDIT QUALITY, not milestone recovery.

Claim to test on webshop: in the sparse-reward regime, GiGPO gives ZERO step
advantage on all-same-outcome obs-groups (esp. all-fail groups, the majority),
whereas a belief-gain anchor still produces oracle-consistent, dense credit.

Coordinate (already in data): b_t = lp_pre[gold_idx] = logπ(gold_title | h_t).
  belief-gain  g_t = b_t - b_{t-1}   (dense, defined every step t>=1)

Anchors compared, each producing a per-step advantage:
  A. GiGPO           : group by exact obs; adv = R_i - mean_group(R)     (RTG = terminal R)
  B. belief-gain     : adv = z(g_t)                                       (dense shaping)
  C. belief-bin GiGPO: group by round(b_t) bin; adv = R_i - mean_group(R) (obs-free grouping)

Evaluated by sign-agreement with webshop ORACLE step labels, overall AND split by
whether the step lives in a GiGPO-dead group (all rollouts same outcome).
"""
import json, os, re
import numpy as np
from collections import defaultdict

OUT = "/home/test/yyy/SDAR/tmp_event_anchor"
tasks = json.load(open(os.path.join(OUT, "event_records.json")))
ASIN_RE = re.compile(r"^[A-Z0-9]{10}$")

# ---------- build per-step rows with belief toward gold ----------
rows = []
for T in tasks:
    gi = T["gold_idx"]
    goal_opts = T["goal_options"]
    gasin = T["gold_asin"].lower()
    for tr in T["trajs"]:
        steps = tr["steps"]
        prev_b = None
        for i, rec in enumerate(steps):
            b = rec["lp_pre"][gi]                 # logπ(gold | h_t)
            g = 0.0 if prev_b is None else b - prev_b
            prev_b = b
            rows.append(dict(task=T["task"], rollout=tr["rollout"], step=rec["step"],
                             obs=" ".join(rec["obs"].split()), action=rec.get("action"),
                             R=tr["reward"], b=b, g=g,
                             next_obs_has_gold=rec.get("next_obs_has_gold", False),
                             gasin=gasin, gopts=goal_opts))

# ---------- oracle step label ----------
def oracle(r):
    a = (r["action"] or "").lower()
    m = re.match(r"click\[(.+)\]", a)
    if m:
        v = m.group(1).strip()
        if ASIN_RE.match(v.upper()):
            return +1 if v == r["gasin"] else -1
        if any(v == o for o in r["gopts"]):
            return +1
    if a.startswith("search["):
        return +1 if r["next_obs_has_gold"] else -1
    return 0

# ---------- advantages ----------
# A. GiGPO: group by (task, obs); RTG = terminal R (undiscounted, single terminal reward)
for r in rows:
    r["A_gig"] = 0.0
grp = defaultdict(list)
for r in rows:
    grp[(r["task"], r["obs"])].append(r)
for v in grp.values():
    if len(set(x["rollout"] for x in v)) >= 2:
        m = np.mean([x["R"] for x in v])
        for x in v:
            x["A_gig"] = x["R"] - m
        x_all_same = len(set(round(x["R"], 3) for x in v)) == 1
    else:
        x_all_same = True
    for x in v:
        x["gig_dead"] = (len(set(x["rollout"] for x in v)) < 2) or (len(set(round(y["R"],3) for y in v)) == 1)

# B. belief-gain: z-normalized per task
for t in set(r["task"] for r in rows):
    sub = [r for r in rows if r["task"] == t]
    gs = np.array([r["g"] for r in sub])
    mu, sd = gs.mean(), gs.std() + 1e-6
    for r in sub:
        r["A_bg"] = (r["g"] - mu) / sd

# C. belief-bin GiGPO: group by (task, round(b)) ; adv = R - group mean
for r in rows:
    r["bbin"] = int(round(r["b"] * 2))     # 0.5-nat bins
grp2 = defaultdict(list)
for r in rows:
    grp2[(r["task"], r["bbin"])].append(r)
for v in grp2.values():
    if len(set(x["rollout"] for x in v)) >= 2:
        m = np.mean([x["R"] for x in v])
        for x in v:
            x["A_bb"] = x["R"] - m
    else:
        for x in v:
            x["A_bb"] = 0.0

# ---------- evaluation ----------
def evalset(rs, key):
    lab = [r for r in rs if oracle(r) != 0]
    cov = [r for r in lab if abs(r[key]) > 1e-9]
    good = sum(1 for r in cov if np.sign(r[key]) == oracle(r))
    return len(lab), len(cov), good

labeled = [r for r in rows if oracle(r) != 0]
dead = [r for r in labeled if r.get("gig_dead", True)]
live = [r for r in labeled if not r.get("gig_dead", True)]

print("=" * 82)
print(f"data: {len(rows)} steps, {len(labeled)} oracle-labeled steps")
print(f"      GiGPO-dead labeled steps (all-same-outcome obs-group): {len(dead)}  "
      f"| live: {len(live)}")
print("=" * 82)
hdr = f"{'anchor':<20}{'coverage':>18}{'sign-acc(cov)':>15}{'correct/labeled':>18}"
for scope, rs in (("ALL labeled", labeled), ("GiGPO-DEAD only", dead), ("LIVE only", live)):
    print(f"\n-- {scope} ({len(rs)} steps) " + "-" * 40)
    print(hdr)
    for key, name in (("A_gig", "GiGPO(obs)"), ("A_bg", "belief-gain"),
                      ("A_bb", "belief-bin GiGPO")):
        nlab, ncov, good = evalset(rs, key)
        sacc = good / max(1, ncov)
        print(f"{name:<20}{ncov:>6}/{nlab:<11}{sacc:>14.0%}{good:>10}/{nlab:<7} ({good/max(1,nlab):.0%})")

print("\nlegend: coverage = labeled steps given non-zero credit; "
      "sign-acc = of those, fraction matching oracle;")
print("        correct/labeled = correctly-signed as fraction of ALL labeled steps (coverage×acc)")

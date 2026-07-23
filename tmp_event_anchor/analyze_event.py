"""
Analysis of the think-as-belief-operator (event) anchor vs GiGPO obs anchor.

For each turn we already have lp_pre / lp_post over the candidate set.
1. signature: dH, flip, KL  ->  event type (sharpen / pivot / expand / inert)
2. event-aligned grouping within task group -> step advantage A_evt
   (sign from group-relative outcome R_i - mean(R_j) among same-type events,
    matched by event type; inert steps fall back to episode advantage)
3. GiGPO baseline: group steps by exact obs match within task group,
   A_gigpo = R_i - mean(R over the obs-group)  (singleton -> 0)
4. Oracle step labels (webshop-specific, evaluation only):
   good step: click[gold_asin], click[goal option], search whose next obs has gold
   bad  step: click[wrong asin], buy w/o required options
   -> sign-agreement of each method's step advantage with oracle
"""
import json, os, re
import numpy as np
from collections import defaultdict

OUT = "/home/test/yyy/SDAR/tmp_event_anchor"
tasks = json.load(open(os.path.join(OUT, "event_records.json")))
ASIN_RE = re.compile(r"^[A-Z0-9]{10}$")

def softmax(lp):
    lp = np.array(lp, dtype=np.float64)
    p = np.exp(lp - lp.max()); return p / p.sum()

def entropy(p):
    p = np.clip(p, 1e-12, 1); return float(-(p * np.log(p)).sum())

def kl(p, q):
    p, q = np.clip(p, 1e-12, 1), np.clip(q, 1e-12, 1)
    return float((p * np.log(p / q)).sum())

# ---------------------------------------------------------------- signatures
rows = []
for T in tasks:
    G = T["trajs"]
    Rs = [tr["reward"] for tr in G]
    for tr in G:
        for rec in tr["steps"]:
            p_pre = softmax(rec["lp_pre"]); p_post = softmax(rec["lp_post"])
            dH = entropy(p_post) - entropy(p_pre)
            flip = int(np.argmax(p_post) != np.argmax(p_pre))
            k = kl(p_post, p_pre)
            rows.append(dict(task=T["task"], rollout=tr["rollout"], step=rec["step"],
                             obs=" ".join(rec["obs"].split()), action=rec.get("action"),
                             think_len=len(rec.get("think") or ""),
                             dH=dH, flip=flip, kl=k,
                             R=tr["reward"], done=tr["done"],
                             next_obs_has_gold=rec.get("next_obs_has_gold", False),
                             argmax_post=int(np.argmax(p_post)),
                             p_gold_pre=float(p_pre[T["gold_idx"]]),
                             p_gold_post=float(p_post[T["gold_idx"]])))

# adaptive thresholds per task-group (quantile based, no hand tuning)
for t in set(r["task"] for r in rows):
    sub = [r for r in rows if r["task"] == t]
    kls = np.array([r["kl"] for r in sub])
    thr = max(np.quantile(kls, 0.5), 1e-3)   # below median KL -> inert
    for r in sub:
        if r["kl"] < thr:
            r["etype"] = "inert"
        elif r["flip"]:
            r["etype"] = "pivot"
        elif r["dH"] < 0:
            r["etype"] = "sharpen"
        else:
            r["etype"] = "expand"

print("=" * 78)
print("1) EVENT DISTRIBUTION: where do cognitive events happen along the trajectory?")
print("=" * 78)
bystep = defaultdict(lambda: defaultdict(int))
for r in rows:
    bystep[r["step"]][r["etype"]] += 1
print(f"{'step':>4} | {'inert':>6} {'sharpen':>8} {'pivot':>6} {'expand':>7}")
for s in sorted(bystep):
    d = bystep[s]
    print(f"{s:>4} | {d['inert']:>6} {d['sharpen']:>8} {d['pivot']:>6} {d['expand']:>7}")
n_evt = sum(1 for r in rows if r["etype"] != "inert")
late_evt = sum(1 for r in rows if r["etype"] != "inert" and r["step"] >= 2)
print(f"\n non-inert events: {n_evt}/{len(rows)} steps; "
      f"{late_evt}/{n_evt} = {late_evt/max(1,n_evt):.0%} of events at step>=2 "
      f"(front-loading check: >50% means credit NOT concentrated at step 0/1)")

# ---------------------------------------------------------------- advantages
print("\n" + "=" * 78)
print("2) STEP ADVANTAGES: event-aligned (ours) vs GiGPO obs-anchor")
print("=" * 78)
for r in rows:
    r["A_evt"] = 0.0; r["A_gig"] = 0.0

for t in set(r["task"] for r in rows):
    sub = [r for r in rows if r["task"] == t]
    Rmean_ep = np.mean([r["R"] for r in {x["rollout"]: x for x in sub}.values()])
    # ours: group same-type events across rollouts (event type as the key)
    for et in ("pivot", "sharpen", "expand"):
        grp = [r for r in sub if r["etype"] == et]
        if len(set(x["rollout"] for x in grp)) >= 2:
            m = np.mean([x["R"] for x in grp])
            for x in grp:
                x["A_evt"] = x["R"] - m
    # GiGPO: exact obs matching
    og = defaultdict(list)
    for r in sub:
        og[r["obs"]].append(r)
    for v in og.values():
        if len(set(x["rollout"] for x in v)) >= 2:
            m = np.mean([x["R"] for x in v])
            for x in v:
                x["A_gig"] = x["R"] - m

def nonzero_frac(key):
    return sum(1 for r in rows if abs(r[key]) > 1e-9) / len(rows)

print(f" steps with non-zero step-advantage:  ours={nonzero_frac('A_evt'):.0%}  "
      f"GiGPO={nonzero_frac('A_gig'):.0%}   (higher = denser credit signal)")

# ---------------------------------------------------------------- oracle labels
print("\n" + "=" * 78)
print("3) ORACLE SIGN AGREEMENT (webshop ground truth, evaluation only)")
print("=" * 78)
gold_asin = {T["task"]: T["gold_asin"].lower() for T in tasks}
goal_opts = {T["task"]: T["goal_options"] for T in tasks}

def oracle_label(r):
    a = (r["action"] or "").lower()
    m = re.match(r"click\[(.+)\]", a)
    if m:
        v = m.group(1).strip()
        if ASIN_RE.match(v.upper()):
            return +1 if v == gold_asin[r["task"]] else -1
        if any(v == o for o in goal_opts[r["task"]]):
            return +1
    if a.startswith("search["):
        return +1 if r["next_obs_has_gold"] else -1
    return 0

agree = defaultdict(lambda: [0, 0])
for r in rows:
    lab = oracle_label(r)
    if lab == 0:
        continue
    for key, name in (("A_evt", "ours"), ("A_gig", "gigpo")):
        if abs(r[key]) > 1e-9:
            agree[name][1] += 1
            if np.sign(r[key]) == lab:
                agree[name][0] += 1
for name in ("ours", "gigpo"):
    c, n = agree[name]
    print(f" {name:>6}: sign agreement with oracle = {c}/{n} = {c/max(1,n):.0%} "
          f"(on labeled steps where the method gives non-zero credit)")

# coverage x correctness combined
lab_steps = [r for r in rows if oracle_label(r) != 0]
print(f"\n labeled steps total: {len(lab_steps)}")
for key, name in (("A_evt", "ours"), ("A_gig", "gigpo")):
    covered = [r for r in lab_steps if abs(r[key]) > 1e-9]
    good = sum(1 for r in covered if np.sign(r[key]) == oracle_label(r))
    print(f" {name:>6}: covers {len(covered)}/{len(lab_steps)} labeled steps, "
          f"correct-sign on {good} ({good/max(1,len(lab_steps)):.0%} of all labeled steps)")

# ---------------------------------------------------------------- faithfulness probe
print("\n" + "=" * 78)
print("4) THINK OPERATOR SANITY: does think actually move the belief distribution?")
print("=" * 78)
kls = np.array([r["kl"] for r in rows if r["think_len"] > 0])
print(f" KL(post||pre) over steps with think: mean={kls.mean():.4f} median={np.median(kls):.4f} "
      f"p90={np.quantile(kls,0.9):.4f} max={kls.max():.4f}")
flips = sum(r["flip"] for r in rows)
print(f" argmax flips: {flips}/{len(rows)} steps")
dgold = np.array([r["p_gold_post"] - r["p_gold_pre"] for r in rows])
print(f" think effect on p(gold): mean={dgold.mean():+.4f}  "
      f"(pos = thinks tend to push toward gold)")
# faithfulness vs outcome
for tag, cond in (("success(R>0.5)", lambda r: r["R"] > 0.5),
                  ("fail(R<=0.5)", lambda r: r["R"] <= 0.5)):
    d = np.array([r["p_gold_post"] - r["p_gold_pre"] for r in rows if cond(r)])
    if len(d):
        print(f"   {tag:>14}: mean d p(gold) per think = {d.mean():+.4f} (n={len(d)})")

json.dump(rows, open(os.path.join(OUT, "rows.json"), "w"))
print("\nsaved per-step rows -> rows.json")

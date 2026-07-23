"""
Analyze credit-assignment coverage & sign-agreement vs GRPO group size on webshop.

For each records_g{N}.json we compute, per oracle-labeled step, the advantage from
three anchors and report coverage (fraction given non-zero credit) and sign-acc
(of covered, fraction matching the webshop oracle):

  A. GiGPO(obs)      : group by (task, exact obs); adv = R_i - mean_group(R)
  C. belief-bin GiGPO: group by (task, round(logπ(gold|h_t))); adv = R_i - mean_group(R)
  B. belief-gain     : z(Δ logπ(gold|h_t))   (dense shaping, reference)

Prints one row per (group_size, anchor) plus the ALL / GiGPO-DEAD split.
"""
import json, os, re, sys
import numpy as np
from collections import defaultdict

OUT = "/home/test/yyy/SDAR/tmp_scale"
ASIN_RE = re.compile(r"^[A-Z0-9]{10}$")

def load_rows(path):
    tasks = json.load(open(path))
    rows = []
    for T in tasks:
        gi = T["gold_idx"]; gopts = T["goal_options"]; gasin = T["gold_asin"].lower()
        for tr in T["trajs"]:
            prev_b = None
            for rec in tr["steps"]:
                if "lp_pre" not in rec:
                    continue
                b = rec["lp_pre"][gi]
                g = 0.0 if prev_b is None else b - prev_b
                prev_b = b
                rows.append(dict(task=T["task"], rollout=tr["rollout"], step=rec["step"],
                                 obs=" ".join(rec["obs"].split()), action=rec.get("action"),
                                 R=tr["reward"], b=b, g=g,
                                 next_obs_has_gold=rec.get("next_obs_has_gold", False),
                                 gasin=gasin, gopts=gopts))
    return rows

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

def compute_adv(rows):
    for r in rows:
        r["A_gig"] = 0.0
    grp = defaultdict(list)
    for r in rows:
        grp[(r["task"], r["obs"])].append(r)
    for v in grp.values():
        rollouts = set(x["rollout"] for x in v)
        outcomes = set(round(y["R"], 3) for y in v)
        if len(rollouts) >= 2:
            m = np.mean([x["R"] for x in v])
            for x in v:
                x["A_gig"] = x["R"] - m
        dead = (len(rollouts) < 2) or (len(outcomes) == 1)
        for x in v:
            x["gig_dead"] = dead
    for t in set(r["task"] for r in rows):
        sub = [r for r in rows if r["task"] == t]
        gs = np.array([r["g"] for r in sub]); mu, sd = gs.mean(), gs.std() + 1e-6
        for r in sub:
            r["A_bg"] = (r["g"] - mu) / sd
    for r in rows:
        r["bbin"] = int(round(r["b"] * 2))     # 0.5-nat bins
    grp2 = defaultdict(list)
    for r in rows:
        grp2[(r["task"], r["bbin"])].append(r)
    for v in grp2.values():
        if len(set(x["rollout"] for x in v)) >= 2 and len(set(round(y["R"],3) for y in v)) >= 2:
            m = np.mean([x["R"] for x in v])
            for x in v:
                x["A_bb"] = x["R"] - m
        else:
            for x in v:
                x["A_bb"] = 0.0
    return rows

def evalset(rs, key):
    lab = [r for r in rs if oracle(r) != 0]
    cov = [r for r in lab if abs(r[key]) > 1e-9]
    good = sum(1 for r in cov if np.sign(r[key]) == oracle(r))
    return len(lab), len(cov), good

def report(path):
    N = re.search(r"_g(\d+)", path).group(1)
    rows = compute_adv(load_rows(path))
    labeled = [r for r in rows if oracle(r) != 0]
    dead = [r for r in labeled if r.get("gig_dead", True)]
    live = [r for r in labeled if not r.get("gig_dead", True)]
    print("\n" + "=" * 90)
    print(f"GROUP SIZE g={N}   | steps={len(rows)}  oracle-labeled={len(labeled)}  "
          f"GiGPO-dead={len(dead)} ({len(dead)/max(1,len(labeled)):.0%})  live={len(live)}")
    print("=" * 90)
    hdr = f"{'scope':<16}{'anchor':<20}{'coverage':>16}{'sign-acc':>10}{'correct/lab':>14}"
    print(hdr)
    for scope, rs in (("ALL", labeled), ("GiGPO-DEAD", dead)):
        for key, name in (("A_gig", "GiGPO(obs)"), ("A_bb", "belief-bin GiGPO"),
                          ("A_bg", "belief-gain")):
            nlab, ncov, good = evalset(rs, key)
            cov = ncov / max(1, nlab); sacc = good / max(1, ncov)
            print(f"{scope:<16}{name:<20}{ncov:>5}/{nlab:<9}({cov:>3.0%}){sacc:>9.0%}"
                  f"{good:>8}/{nlab:<5}")
        print("-" * 90)
    return N, labeled

if __name__ == "__main__":
    files = sorted([os.path.join(OUT, f) for f in os.listdir(OUT)
                    if re.match(r"records_g\d+\.json", f)],
                   key=lambda p: int(re.search(r"_g(\d+)", p).group(1)))
    summ = []
    for p in files:
        N, labeled = report(p)
        rows = compute_adv(load_rows(p))
        lab = [r for r in rows if oracle(r) != 0]
        def cov_acc(key):
            n, c, g = evalset(lab, key)
            return c / max(1, n), (g / max(1, c))
        gc, ga = cov_acc("A_gig"); bc, ba = cov_acc("A_bb")
        summ.append((N, gc, ga, bc, ba))
    print("\n\n### SCALING SUMMARY (coverage rises with group size, sign-acc stable) ###")
    print(f"{'group':<8}{'GiGPO cov':>12}{'GiGPO acc':>12}{'bbin cov':>12}{'bbin acc':>12}")
    for N, gc, ga, bc, ba in summ:
        print(f"g={N:<6}{gc:>11.0%}{ga:>12.0%}{bc:>11.0%}{ba:>12.0%}")

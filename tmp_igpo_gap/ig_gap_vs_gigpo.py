"""Head-to-head: GiGPO anchor-state turn advantage vs ours, same data/labels.

GiGPO (step level): group all steps across sibling rollouts that share the
same anchor observation (task, obs-text); step advantage = z-norm of
discounted return-to-go within the anchor group.  Terminal-reward WebShop:
R = gamma^(T-1-t) * score.

Compare on:
  coverage (% steps with a usable anchor group)
  good/bad turn AUC + sign correctness (same labels as before)
  fork test (outcome pairs)
  combination GiGPO + beta*S_ours
Then: anchor-group feasibility statistics on the deep-search episodes.
"""
import collections
import hashlib
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SDAR = os.path.dirname(HERE)
PROBE = os.path.join(SDAR, "webshop_probe_check")
FILES = ["ws_ckpt_rollouts2.json", "ws_ckpt_rollouts3.json", "ws_ckpt_rollouts4.json"]
GAMMA = 0.95

# ---- load raw rollouts for anchors ----
steps_by_key = {}
score_by_key = {}
for fn in FILES:
    for tr in json.load(open(os.path.join(PROBE, fn))):
        key = (fn, tr["task"], tr["k"])
        steps_by_key[key] = tr["steps"]
        score_by_key[key] = tr["score"]

# ---- GiGPO anchor groups ----
groups = collections.defaultdict(list)   # (fn, task, anchor_hash) -> [(key, t, T)]
for key, steps in steps_by_key.items():
    T = len(steps)
    for s in steps:
        h = hashlib.md5(s["anchor"].encode()).hexdigest()[:16]
        groups[(key[0], key[1], h)].append((key, s["t"], T))

gig_adv = {}   # (key, t) -> advantage
grp_sizes = []
for gk, members in groups.items():
    Rs = np.array([GAMMA ** (T - 1 - t) * score_by_key[k] for k, t, T in members])
    grp_sizes.append(len(members))
    if len(members) < 2 or Rs.std() < 1e-8:
        for k, t, T in members:
            gig_adv[(k, t)] = 0.0
    else:
        z = (Rs - Rs.mean()) / Rs.std()
        for (k, t, T), zi in zip(members, z):
            gig_adv[(k, t)] = float(zi)

sizes = np.array(grp_sizes)
print(f"anchor groups={len(groups)}  size>=2: {np.mean(sizes>=2):.0%}  "
      f"mean size={sizes.mean():.2f}")

# ---- our rows + labels ----
rows = json.load(open(os.path.join(HERE, "ig_gap_rows.json")))
traj = collections.defaultdict(list)
for r in rows:
    traj[(r["fn"], r["task"], r["k"])].append(r)
scores_g = {}
for (fn, task, k), ss in traj.items():
    scores_g.setdefault((fn, task), {})[k] = ss[0]["score"]
adv_ep = {}
for gk, d in scores_g.items():
    v = np.array(list(d.values()))
    sd = v.std()
    for k, s in d.items():
        adv_ep[(gk[0], gk[1], k)] = (s - v.mean()) / (sd + 1e-8) if sd > 1e-8 else 0.0

recs = []
for key, ss in traj.items():
    ss.sort(key=lambda r: r["t"])
    for i in range(1, len(ss)):
        a, b = ss[i - 1], ss[i]
        pS0, pS1 = np.exp(a["lp_S_ans"]), np.exp(b["lp_S_ans"])
        pT0, pT1 = np.exp(a["lp_T1_ans"]), np.exp(b["lp_T1_ans"])
        good = bool((b["first_vis_t"] == b["t"]) or a["gold_click"]
                    or a["act_type"] == "click_opt_match")
        bad = bool((a["act_type"] == "click_asin" and not a["gold_click"]
                    and a["gold_visible"] and not b["gold_visible"])
                   or a["act_type"] == "click_other")
        recs.append(dict(
            key=key, t=a["t"], good=good, bad=bad, score=a["score"],
            act=a["action"], S=(pS1 + pT1) - (pS0 + pT0),
            A_ep=adv_ep[key], A_gig=gig_adv.get((key, a["t"]), 0.0)))

good = np.array([r["good"] for r in recs])
bad = np.array([r["bad"] for r in recs])
mask = good | bad
S = np.array([r["S"] for r in recs])
A_ep = np.array([r["A_ep"] for r in recs])
A_gig = np.array([r["A_gig"] for r in recs])
print(f"turn steps={len(recs)}  GiGPO nonzero step-adv coverage={np.mean(A_gig!=0):.0%}")


def auc(s, l):
    npos, nneg = l.sum(), (~l).sum()
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    return (ranks[l].sum() - npos * (npos + 1) / 2) / (npos * nneg)


print("\n===== good/bad turn discrimination (same labels as before) =====")
for name, C in (("GiGPO step-adv", A_gig),
                ("GiGPO full (ep + step)", A_ep + A_gig),
                ("ours S only", S),
                ("ours full (ep + 2S)", A_ep + 2 * S),
                ("GiGPO + 2S (combined)", A_ep + A_gig + 2 * S)):
    print(f"  {name:24s} AUC={auc(C[mask], good[mask]):.3f}  "
          f"good>0:{np.mean(C[good]>0):.0%} bad<0:{np.mean(C[bad]<0):.0%}")

# where does GiGPO have signal vs not
has = A_gig != 0
print(f"\n  GiGPO-covered turns: {has.sum()} ({has.mean():.0%});  "
      f"AUC on covered={auc((A_ep+A_gig)[mask&has], good[mask&has]):.3f}  "
      f"ours on covered={auc((A_ep+2*S)[mask&has], good[mask&has]):.3f}")
print(f"  uncovered turns: AUC GiGPO-full={auc((A_ep+A_gig)[mask&~has], good[mask&~has]):.3f} "
      f"(=episode only)  ours={auc((A_ep+2*S)[mask&~has], good[mask&~has]):.3f}")

# ---- fork test with GiGPO ----
byk = collections.defaultdict(dict)
for r in recs:
    byk[r["key"]][r["t"]] = r
groups2 = collections.defaultdict(list)
for key in byk:
    groups2[(key[0], key[1])].append(key)
pairs = []
for gk, keys in groups2.items():
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            A = [byk[keys[i]][t] for t in sorted(byk[keys[i]])]
            B = [byk[keys[j]][t] for t in sorted(byk[keys[j]])]
            fork = None
            for t in range(min(len(A), len(B))):
                if A[t]["act"] != B[t]["act"]:
                    fork = t
                    break
            if fork is None:
                continue
            a, b = A[fork], B[fork]
            if abs(a["score"] - b["score"]) >= 0.2:
                w, l = (a, b) if a["score"] > b["score"] else (b, a)
                pairs.append((w, l))
print(f"\n===== fork test (outcome pairs n={len(pairs)}) =====")
for name, f in (("GiGPO step-adv", lambda r: r["A_gig"]),
                ("ours S", lambda r: r["S"])):
    acc = np.mean([f(w) > f(l) for w, l in pairs])
    nz = np.mean([f(w) != 0 or f(l) != 0 for w, l in pairs])
    print(f"  {name:16s} pairwise acc={acc:.0%}  (pairs with signal: {nz:.0%})")

# ---- deep search: does GiGPO's anchor grouping even exist there? ----
print("\n===== deep-search anchor feasibility =====")
ds = json.load(open(os.path.join(HERE, "ds_gap_rows.json")))
eps = collections.defaultdict(list)
for r in ds:
    eps[(r["fn"], r["ep"])].append(r)
print(f"  episodes={len(eps)}; rollouts per question=1 in this corpus -> "
      f"no sibling rollouts, GiGPO step groups impossible by construction")
print("  (even with sibling sampling: anchor = full evidence context; two deep-search")
print("   rollouts virtually never reproduce an identical observation state after t=1)")

print("\nIG_GAP_VS_GIGPO_DONE")

"""Fork test: sibling rollouts that share an identical action prefix and
diverge at turn t.  At the fork, context is (near) identical, so per-turn
credit MUST rank the better fork action above the worse one.

Pair labels (who should win):
  gold-vs-wrong : one sibling clicks the gold asin, the other a non-gold asin
  opt-vs-nonopt : one clicks a goal-matching option, the other a non-matching one
  outcome       : final scores differ by >= 0.2 (weaker label)

Candidates compared on pairwise accuracy at the fork turn:
  dS_log  = Delta log pS            (IGPO log_prob_diff)
  dpS     = Delta pS                (IGPO prob_diff)
  dpT     = Delta pT                (teacher channel alone)
  OURS    = Delta (pS + pT)
Also prints concrete fork cases for manual inspection.
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
for key in traj:
    traj[key].sort(key=lambda r: r["t"])

# per-trajectory: actions and per-turn credits (gain of action at index t)
def build(key):
    ss = traj[key]
    out = []
    for i in range(1, len(ss)):
        a, b = ss[i - 1], ss[i]
        pS0, pS1 = np.exp(a["lp_S_ans"]), np.exp(b["lp_S_ans"])
        pT0, pT1 = np.exp(a["lp_T1_ans"]), np.exp(b["lp_T1_ans"])
        out.append(dict(
            t=a["t"], action=a["action"], act_type=a["act_type"],
            gold_click=a["gold_click"], score=a["score"],
            dS_log=b["lp_S_ans"] - a["lp_S_ans"],
            dpS=pS1 - pS0, dpT=pT1 - pT0, OURS=(pS1 + pT1) - (pS0 + pT0)))
    return out

trajs = {key: build(key) for key in traj}

groups = collections.defaultdict(list)
for key in trajs:
    groups[(key[0], key[1])].append(key)

CANDS = ("dS_log", "dpS", "dpT", "OURS")
pairs = collections.defaultdict(list)   # label -> list of (winner_rec, loser_rec)
cases = []

for gk, keys in groups.items():
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            A, B = trajs[keys[i]], trajs[keys[j]]
            # find fork: first index where both defined and actions differ,
            # with identical action prefix
            fork = None
            for t in range(min(len(A), len(B))):
                if A[t]["action"] != B[t]["action"]:
                    fork = t
                    break
            if fork is None:
                continue
            a, b = A[fork], B[fork]
            # local labels
            lab = None
            if (a["act_type"] == "click_asin" and b["act_type"] == "click_asin"
                    and a["gold_click"] != b["gold_click"]):
                lab = "gold-vs-wrong"
                win, lose = (a, b) if a["gold_click"] else (b, a)
            elif ({a["act_type"], b["act_type"]} == {"click_opt_match", "click_other"}):
                lab = "opt-vs-nonopt"
                win, lose = (a, b) if a["act_type"] == "click_opt_match" else (b, a)
            if lab:
                pairs[lab].append((win, lose))
                cases.append((lab, gk, win, lose))
            # outcome label (may coexist)
            if abs(a["score"] - b["score"]) >= 0.2:
                win, lose = (a, b) if a["score"] > b["score"] else (b, a)
                pairs["outcome"].append((win, lose))

print("===== pairwise accuracy at fork turn (winner's credit > loser's) =====")
for lab, pl in pairs.items():
    print(f"\n  label={lab}  n_pairs={len(pl)}")
    for c in CANDS:
        acc = np.mean([w[c] > l[c] for w, l in pl])
        margin = np.mean([w[c] - l[c] for w, l in pl])
        print(f"    {c:7s} acc={acc:.0%}  mean margin={margin:+.4f}")

print("\n===== concrete fork cases =====")
seen = set()
for lab, gk, win, lose in cases:
    sig = (gk, win["action"][:30], lose["action"][:30])
    if sig in seen:
        continue
    seen.add(sig)
    print(f"\n  [{lab}] {gk[0][-6:]}:task{gk[1]} t={win['t']}")
    for tag, r in (("WIN ", win), ("LOSE", lose)):
        print(f"    {tag} score={r['score']:.2f} OURS={r['OURS']:+.4f} "
              f"(dpS={r['dpS']:+.4f} dpT={r['dpT']:+.4f} dS_log={r['dS_log']:+.3f}) "
              f"{r['action'][:52]!r}")
    if len(seen) >= 10:
        break

# additional deep check: same-context wrong clicks that later recovered vs not
print("\n===== does OURS punish the wrong click MORE in trajs that never recover? =====")
wr = []
for key, T in trajs.items():
    for r in T:
        if (r["act_type"] == "click_asin" and not r["gold_click"]):
            wr.append((r, r["score"]))
lo = [r["OURS"] for r, s in wr if s < 0.5]
hi = [r["OURS"] for r, s in wr if s >= 0.9]
print(f"  wrong-clicks in low-score trajs (<0.5): n={len(lo)} mean OURS={np.mean(lo):+.4f}")
print(f"  wrong-clicks in high-score trajs (>=0.9): n={len(hi)} mean OURS={np.mean(hi):+.4f}")

print("\nIG_GAP_FORK_DONE")

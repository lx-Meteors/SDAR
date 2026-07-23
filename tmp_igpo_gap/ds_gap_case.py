"""Case-level audit of S_t = Delta_t(p_S + p_T) on deep-search episodes.
All episodes are answer-correct (A would be ~equal in-group), so we audit the
shaping term itself: does the arrival turn get the top credit, do err turns
get ~0, what does the credit profile look like turn by turn."""
import collections
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
rows = json.load(open(os.path.join(HERE, "ds_gap_rows.json")))

eps = collections.defaultdict(list)
for r in rows:
    eps[(r["fn"], r["ep"])].append(r)

recs = {}
for key, ss in eps.items():
    ss.sort(key=lambda r: r["t"])
    out = []
    for i in range(1, len(ss)):
        a, b = ss[i - 1], ss[i]
        S = (np.exp(b["lpS"]) + np.exp(b["lpT"])) - (np.exp(a["lpS"]) + np.exp(a["lpT"]))
        out.append(dict(t=b["t"], S=S, arr=b["arrival"], err=b["err"],
                        pS0=np.exp(a["lpS"]), pS1=np.exp(b["lpS"]),
                        pT0=np.exp(a["lpT"]), pT1=np.exp(b["lpT"]),
                        gold=b["gold"]))
    recs[key] = out

# rank of arrival turn within episode
ranks, errC, restC, arrC = [], [], [], []
for key, rs in recs.items():
    Svals = np.array([r["S"] for r in rs])
    for i, r in enumerate(rs):
        if r["arr"]:
            ranks.append(int((Svals > r["S"]).sum()) + 1)
            arrC.append(r["S"])
        elif r["err"]:
            errC.append(r["S"])
        else:
            restC.append(r["S"])
n_arr = len(ranks)
print(f"episodes={len(recs)}  arrival-turns={n_arr}")
print(f"arrival turn rank within episode: top1={np.mean([r==1 for r in ranks]):.0%} "
      f"top2={np.mean([r<=2 for r in ranks]):.0%} top3={np.mean([r<=3 for r in ranks]):.0%} "
      f"(episode len median={int(np.median([len(v) for v in recs.values()]))})")
print(f"credit: arrival mean={np.mean(arrC):+.3f}  err mean={np.mean(errC):+.4f}  "
      f"other mean={np.mean(restC):+.4f}")
print(f"err turns with |S|<0.02 (correctly ~zero): {np.mean(np.abs(errC)<0.02):.0%}")

# show 3 episodes: one where arrival is top-1, one where it isn't, one with errors
show_top, show_miss, show_err = None, None, None
for key, rs in recs.items():
    Svals = np.array([r["S"] for r in rs])
    arr_i = [i for i, r in enumerate(rs) if r["arr"]]
    if len(arr_i) == 1:
        is_top = Svals.argmax() == arr_i[0]
        if is_top and show_top is None and len(rs) >= 5:
            show_top = key
        if not is_top and show_miss is None and len(rs) >= 5:
            show_miss = key
    if show_err is None and sum(r["err"] for r in rs) >= 2:
        show_err = key

def show(key, title):
    rs = recs[key]
    print(f"\n### {title}  {key[1]}  gold={rs[0]['gold'][:40]!r}  turns={len(rs)}")
    print("   t  flag  S        pS->        pT->")
    for r in rs:
        flag = "ARR" if r["arr"] else ("ERR" if r["err"] else "   ")
        star = " <== max" if r["S"] == max(x["S"] for x in rs) else ""
        print(f"  {r['t']:2d}  {flag}  {r['S']:+.3f}  "
              f"{r['pS0']:.3f}>{r['pS1']:.3f}  {r['pT0']:.3f}>{r['pT1']:.3f}{star}")

for key, title in ((show_top, "arrival correctly gets max credit"),
                   (show_miss, "arrival NOT max (miss case)"),
                   (show_err, "episode with error turns")):
    if key:
        show(key, title)

print("\nDS_GAP_CASE_DONE")

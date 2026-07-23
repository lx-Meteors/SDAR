import json, os, numpy as np
from collections import defaultdict
OUT="/home/test/yyy/SDAR/tmp_belief_exp"
R=json.load(open(os.path.join(OUT,"records.json")))
tasks=sorted(set(r["task"] for r in R))

def norm_obs(o):
    return " ".join(o.split())

print("="*70)
print("A) GiGPO anchor = exact-match observation, grouped WITHIN each task")
print("="*70)
gigpo_singleton_all=0; gigpo_n_all=0
gigpo_singleton_info=0; gigpo_info=0
belief_std_within_obsgroup=[]
for t in tasks:
    steps=[r for r in R if r["task"]==t]
    groups=defaultdict(list)
    for r in steps:
        groups[norm_obs(r["anchor_obs"])].append(r)
    sizes=sorted((len(v) for v in groups.values()),reverse=True)
    singﾟ=sum(1 for v in groups.values() if len(v)==1)
    n_steps=len(steps)
    gigpo_singleton_all+=sum(len(v) for v in groups.values() if len(v)==1)
    gigpo_n_all+=n_steps
    # informative steps = step>=1 (after first search, where real evidence differs)
    info=[r for r in steps if r["step"]>=1]
    info_groups=defaultdict(list)
    for r in info: info_groups[norm_obs(r["anchor_obs"])].append(r)
    info_sing=sum(len(v) for v in info_groups.values() if len(v)==1)
    gigpo_singleton_info+=info_sing; gigpo_info+=len(info)
    # belief reproducibility within identical-obs groups
    for v in groups.values():
        if len(v)>=2:
            belief_std_within_obsgroup.append(np.std([x["belief"] for x in v]))
    print(f" task{t}: {n_steps} steps -> {len(groups)} distinct obs | group sizes {sizes} | "
          f"singleton steps={sum(len(v) for v in groups.values() if len(v)==1)}")
print(f"\n GiGPO: {gigpo_singleton_all}/{gigpo_n_all} = {gigpo_singleton_all/gigpo_n_all:.0%} of ALL steps are in singleton obs-groups (=> step-advantage=0)")
print(f" GiGPO: {gigpo_singleton_info}/{gigpo_info} = {gigpo_singleton_info/max(1,gigpo_info):.0%} of INFORMATIVE steps (step>=1) are singletons")
if belief_std_within_obsgroup:
    print(f" belief std within identical-obs groups: mean={np.mean(belief_std_within_obsgroup):.3f} "
          f"(near 0 => length-averaged belief is reproducible, local-position noise cancels)")

print("\n"+"="*70)
print("B) Belief-gain anchor = band on b_t=logπ(a*|h_t)-logπ(a*|h_0), WITHIN each task")
print("="*70)
# adaptive equal-frequency bands per task (K=3), only if range significant
K=3
bg_singleton_info=0; bg_info=0
for t in tasks:
    steps=[r for r in R if r["task"]==t]
    info=[r for r in steps if r["step"]>=1]
    vals=np.array([r["belief_gain"] for r in info])
    rng=vals.max()-vals.min()
    if rng<0.5:
        band=np.zeros(len(info),dtype=int)  # flat -> single band (fall back to episode adv)
        note="(flat range -> 1 band, fallback)"
    else:
        qs=np.quantile(vals,np.linspace(0,1,K+1))
        qs[0]-=1e-6
        band=np.digitize(vals,qs[1:-1])
        note=""
    bsz=defaultdict(int)
    for b in band: bsz[int(b)]+=1
    sizes=sorted(bsz.values(),reverse=True)
    sing=sum(1 for v in bsz.values() if v==1)
    bg_singleton_info+=sum(v for v in bsz.values() if v==1); bg_info+=len(info)
    print(f" task{t}: {len(info)} info steps -> band sizes {sizes} range={rng:.2f} {note}")
print(f"\n Belief-band: {bg_singleton_info}/{bg_info} = {bg_singleton_info/max(1,bg_info):.0%} of INFORMATIVE steps are singletons")

print("\n"+"="*70)
print("C) head-to-head on INFORMATIVE steps (where credit assignment matters)")
print("="*70)
print(f"   GiGPO obs-anchor singleton rate : {gigpo_singleton_info/max(1,gigpo_info):.0%}")
print(f"   belief-gain band singleton rate : {bg_singleton_info/max(1,bg_info):.0%}")

# ----------------------------------------------------------------- plot
try:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(13,4.5))
    ax=axes[0]
    for t in tasks:
        for k in sorted(set(r["rollout"] for r in R if r["task"]==t)):
            seq=[r for r in R if r["task"]==t and r["rollout"]==k]
            seq=sorted(seq,key=lambda r:r["step"])
            ax.plot([r["step"] for r in seq],[r["belief_gain"] for r in seq],
                    marker="o",ms=3,alpha=0.7,label=f"t{t}" if k==0 else None)
    ax.set_title("belief-gain  b_t = logπ(a*|h_t)-logπ(a*|h_0)\n(jump at first search, then plateau)")
    ax.set_xlabel("turn"); ax.set_ylabel("belief gain (nats/token)"); ax.axhline(0,color="k",lw=.5)
    ax.legend(fontsize=7,ncol=2)

    ax=axes[1]
    # anchor group-size distribution on informative steps
    gig_sizes=[]; 
    for t in tasks:
        info=[r for r in R if r["task"]==t and r["step"]>=1]
        g=defaultdict(int)
        for r in info: g[norm_obs(r["anchor_obs"])]+=1
        gig_sizes+=list(g.values())
    ax.hist(gig_sizes,bins=range(1,N_MAX:=max(gig_sizes+[2])+2),align="left",rwidth=.8)
    ax.set_title(f"GiGPO obs-anchor group sizes (informative steps)\n"
                 f"{gigpo_singleton_info/max(1,gigpo_info):.0%} are singletons -> step-adv=0")
    ax.set_xlabel("group size (# steps sharing identical obs)"); ax.set_ylabel("# groups")
    plt.tight_layout(); p=os.path.join(OUT,"compare.png"); plt.savefig(p,dpi=120)
    print("\nsaved plot ->",p)
except Exception as e:
    print("plot skipped:",e)

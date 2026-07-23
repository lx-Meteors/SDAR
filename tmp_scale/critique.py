"""
Stress-test belief-bin GiGPO honestly.

Adds the baselines/metrics that decide whether the idea is REALLY effective:
  - depth-bin GiGPO      : group by (task, step_index)  -> is belief doing real work, or
                           does ANY non-obs grouping break singletons?
  - belief-value baseline: A = R_i - V_hat(b_t), V_hat = isotonic reg of R on belief
                           (the elegant, hyperparameter-free form of belief-bin)
Metrics (imbalance-robust):
  - oracle +/- balance
  - coverage, raw sign-acc
  - BALANCED-acc = mean(+recall, -recall)   (chance = 50% regardless of imbalance)
  - point-biserial corr(A, oracle01)         (does larger A mean a better step?)
"""
import json, os, re, sys
import numpy as np
from collections import defaultdict
try:
    from sklearn.isotonic import IsotonicRegression
    HAVE_ISO = True
except Exception:
    HAVE_ISO = False

ASIN_RE = re.compile(r"^[A-Z0-9]{10}$")

# ---------------- loaders ----------------
def load_webshop(path):
    tasks = json.load(open(path)); rows = []
    for T in tasks:
        gi = T["gold_idx"]; gopts = T["goal_options"]; gasin = T["gold_asin"].lower()
        for tr in T["trajs"]:
            prev = None
            for rec in tr["steps"]:
                if "lp_pre" not in rec: continue
                b = rec["lp_pre"][gi]; g = 0.0 if prev is None else b-prev; prev = b
                rows.append(dict(task=T["task"], rollout=tr["rollout"], step=rec["step"],
                                 obs=" ".join(rec["obs"].split()), action=rec.get("action"),
                                 R=tr["reward"], b=b, g=g,
                                 next_obs_has_gold=rec.get("next_obs_has_gold", False),
                                 gasin=gasin, gopts=gopts))
    def orc(r):
        a=(r["action"] or "").lower(); m=re.match(r"click\[(.+)\]",a)
        if m:
            v=m.group(1).strip()
            if ASIN_RE.match(v.upper()): return +1 if v==r["gasin"] else -1
            if any(v==o for o in r["gopts"]): return +1
        if a.startswith("search["): return +1 if r["next_obs_has_gold"] else -1
        return 0
    for r in rows: r["oracle"]=orc(r)
    return rows

def load_deepsearch(path):
    rows = json.load(open(path))
    for r in rows: r["task"]=r["q"]; r["oracle"]= +1 if r["res_has_gold"] else -1
    return rows

# ---------------- anchors ----------------
def add_group_adv(rows, keyfn, name):
    grp=defaultdict(list)
    for r in rows: grp[keyfn(r)].append(r)
    for r in rows: r[name]=0.0
    for v in grp.values():
        if len(set(x["rollout"] for x in v))>=2 and len(set(round(x["R"],3) for x in v))>=2:
            m=np.mean([x["R"] for x in v])
            for x in v: x[name]=x["R"]-m

def add_belief_value(rows, name):
    # V_hat(b) = E[R|b] via isotonic regression per task; A = R - V_hat(b)
    for r in rows: r[name]=0.0
    for t in set(r["task"] for r in rows):
        sub=[r for r in rows if r["task"]==t]
        b=np.array([r["b"] for r in sub]); R=np.array([r["R"] for r in sub])
        if len(sub)<3 or len(set(R))<2:
            continue
        if HAVE_ISO:
            try:
                iso=IsotonicRegression(out_of_bounds="clip").fit(b,R)
                V=iso.predict(b)
            except Exception:
                V=np.full_like(R, R.mean())
        else:
            V=np.full_like(R, R.mean())
        for r,vi in zip(sub,V): r[name]=r["R"]-vi

def add_belief_gain(rows, name):
    for t in set(r["task"] for r in rows):
        sub=[r for r in rows if r["task"]==t]
        gs=np.array([r["g"] for r in sub]); mu,sd=gs.mean(),gs.std()+1e-6
        for r in sub: r[name]=(r["g"]-mu)/sd

# ---------------- metrics ----------------
def metrics(rows, key):
    lab=[r for r in rows if r["oracle"]!=0]
    cov=[r for r in lab if abs(r[key])>1e-9]
    if not cov: return len(lab),0,0,0,0,0.0
    pos=[r for r in cov if r["oracle"]>0]; neg=[r for r in cov if r["oracle"]<0]
    pr=np.mean([np.sign(r[key])>0 for r in pos]) if pos else 0.0   # +recall
    nr=np.mean([np.sign(r[key])<0 for r in neg]) if neg else 0.0   # -recall
    bal=0.5*(pr+nr)
    sacc=np.mean([np.sign(r[key])==r["oracle"] for r in cov])
    A=np.array([r[key] for r in cov]); O=np.array([1.0 if r["oracle"]>0 else 0.0 for r in cov])
    pb=float(np.corrcoef(A,O)[0,1]) if A.std()>0 and O.std()>0 else 0.0
    return len(lab), len(cov), sacc, bal, pb, (len(pos),len(neg))

def run(name, rows):
    add_group_adv(rows, lambda r:(r["task"], r["obs"]), "A_gig")
    add_group_adv(rows, lambda r:(r["task"], r["step"]), "A_depth")
    # belief-bin: per-task quantile terciles
    for t in set(r["task"] for r in rows):
        sub=[r for r in rows if r["task"]==t]; bs=np.array([r["b"] for r in sub])
        edges=np.quantile(bs,[1/3,2/3]) if len(bs)>=3 else []
        for r in sub: r["bbin"]=int(np.searchsorted(edges,r["b"]))
    add_group_adv(rows, lambda r:(r["task"], r["bbin"]), "A_bb")
    add_belief_value(rows, "A_bv")
    add_belief_gain(rows, "A_bg")
    lab=[r for r in rows if r["oracle"]!=0]
    npos=sum(1 for r in lab if r["oracle"]>0); nneg=sum(1 for r in lab if r["oracle"]<0)
    print("\n"+"="*100)
    print(f"{name}: {len(rows)} steps | labeled={len(lab)}  (+{npos} / -{nneg}, "
          f"pos-rate={npos/max(1,len(lab)):.0%})")
    print("="*100)
    print(f"{'anchor':<22}{'coverage':>14}{'sign-acc':>10}{'BAL-acc':>10}{'corr(A,oracle)':>16}")
    for key,nm in (("A_gig","GiGPO(obs)"),("A_depth","depth-bin GiGPO"),
                   ("A_bb","belief-bin GiGPO"),("A_bv","belief-value baseline"),
                   ("A_bg","belief-gain")):
        nl,nc,sacc,bal,pb,br=metrics(rows,key)
        print(f"{nm:<22}{nc:>6}/{nl:<7}({nc/max(1,nl):>3.0%}){sacc:>9.0%}{bal:>10.0%}{pb:>15.2f}")
    print("BAL-acc>50% 且 corr>0 才算真正有效(优于按基率乱猜)。")

if __name__=="__main__":
    ws="/home/test/yyy/SDAR/tmp_scale/records_g16.json"
    if os.path.exists(ws): run("WEBSHOP g16", load_webshop(ws))
    ds="/data1/test/yyy/ARPO/evaluation/ds_rows.json"
    if os.path.exists(ds): run("DEEP-SEARCH (bamboogle 16x7)", load_deepsearch(ds))

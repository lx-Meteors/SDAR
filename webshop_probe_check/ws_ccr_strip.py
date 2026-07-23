"""WebShop replication of the sibling-strip statistic (CCR v2 step 1).

Token-level menu collisions (same task, same student top-5 candidate id set,
different rollouts, different sampled tokens = real decision divergence, no
teacher needed).  Two directions:
  STRIP: rollout with positive GRPO advantage fires a positive update; what
         share of its counter-mass lands on the token a sibling chose there?
         (deep-search numbers: 46% from successful siblings, 20% from failed)
  FEED:  rollout with negative advantage fires a negative update; what share
         of the freed mass does GRPO already hand to the token the task's
         BEST sibling chose there?  (large -> failure side self-corrects at
         collisions; small -> S+ targeted feeding is a second intervention)

Data: 12 existing trajs (3 tasks x 4 rollouts), one student forward per step.
Run: qwen-infer env, CUDA_VISIBLE_DEVICES=5, ~2 min.
"""
import json, os
import numpy as np
import torch
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
MODEL = sys.argv[2] if len(sys.argv) > 2 else "/home/test/models/Qwen2.5-3B-Instruct"
HERE = os.path.dirname(os.path.abspath(__file__))
IN = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "ws_full_rollouts.json")
TAG = os.path.basename(IN).replace(".json", "")
trajs = json.load(open(IN))

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cuda:0")
model.eval(); DEV = model.device

@torch.no_grad()
def dists(user_content, response):
    msgs = [{"role": "user", "content": user_content}]
    pre = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    pre_ids = tok(pre, add_special_tokens=False, truncation=True, max_length=7000).input_ids
    out_ids = tok(response, add_special_tokens=False).input_ids[:400]
    full = torch.tensor([pre_ids + out_ids], device=DEV)
    lg = model(full).logits[0].float()
    L0 = len(pre_ids)
    return torch.softmax(lg[L0 - 1:L0 - 1 + len(out_ids), :], dim=-1), out_ids

gmean = defaultdict(list)
for tr in trajs: gmean[tr["task"]].append(tr["score"])
gmean = {t: float(np.mean(v)) for t, v in gmean.items()}
best_k = {}
for tr in trajs:
    if tr["task"] not in best_k or tr["score"] > best_k[tr["task"]][1]:
        best_k[tr["task"]] = (tr["k"], tr["score"])

rows = []
for tr in trajs:
    adv = tr["score"] - gmean[tr["task"]]
    for s in tr["steps"]:
        resp = s.get("response", "")
        if not resp or resp == "(forced)": continue
        p, out_ids = dists(s["prompt"], resp)
        t5v, t5i = p.topk(5, dim=-1)
        t5v, t5i = t5v.cpu().numpy(), t5i.cpu().numpy()
        idx = torch.arange(len(out_ids), device=DEV)
        py = p[idx, torch.tensor(out_ids, device=DEV)].cpu().numpy()
        enc = tok(resp, add_special_tokens=False, return_offsets_mapping=True)
        offs = enc.offset_mapping[:len(out_ids)]
        tspan = (resp.find("<think>"), resp.find("</think>"))
        aspan = (resp.find("<action>"), resp.find("</action>"))
        def region(c):
            if aspan[0] != -1 and aspan[0] <= c < (aspan[1] if aspan[1] != -1 else 1e9): return "action"
            if tspan[0] != -1 and tspan[0] <= c < (tspan[1] if tspan[1] != -1 else 1e9): return "think"
            return "other"
        for i in range(len(out_ids)):
            rows.append(dict(task=tr["task"], k=tr["k"], t=s["t"], i=i, adv=adv,
                             score=tr["score"], y=int(out_ids[i]), p_y=float(py[i]),
                             reg=region(offs[i][0]),
                             menu=tuple(sorted(int(x) for x in t5i[i])),
                             pm={int(v): float(x) for v, x in zip(t5i[i], t5v[i])}))
        del p; torch.cuda.empty_cache()
    print(f"task{tr['task']} k{tr['k']} done ({len(rows)} pos)", flush=True)

groups = defaultdict(list)
for r in rows: groups[(r["task"], r["menu"])].append(r)
multi = {g: m for g, m in groups.items() if len({x["k"] for x in m}) >= 2}
cov = sum(len(m) for m in multi.values()) / max(1, len(rows))
div = {g: m for g, m in multi.items() if len({x["y"] for x in m}) >= 2}
print(f"\npositions={len(rows)}  collision groups={len(multi)} (coverage {cov:.0%} of all positions)  "
      f"choice-diverse={len(div)}")

# group degeneracy (GRPO advantage all-zero tasks)
tscores = defaultdict(set)
for tr in trajs: tscores[tr["task"]].add(tr["score"])
dead = [t for t, v in tscores.items() if len(v) == 1]
print(f"degenerate GRPO groups (all-equal scores, zero advantage everywhere): "
      f"{len(dead)}/{len(tscores)} tasks {dead}")

strip, feed = defaultdict(list), defaultdict(list)
seen = set()
for g, mem in div.items():
    task = g[0]
    bk = best_k[task][0]
    adv_of = {m["k"]: m["adv"] for m in mem}
    for a in mem:
        if a["p_y"] > 0.95: continue          # no meaningful counter-mass; also kills fp-noise ratios
        others = [b for b in mem if b["k"] != a["k"] and b["y"] != a["y"] and b["y"] in a["pm"]]
        if not others: continue
        if a["adv"] > 0:
            for b in others:
                key = ("s", a["k"], a["t"], a["i"], b["k"], b["y"])
                if key in seen: continue
                seen.add(key)
                share = min(1.0, a["pm"][b["y"]] / max(1e-9, 1 - a["p_y"]))
                strip[(b["adv"] > 0, a["reg"])].append(share)
        elif a["adv"] < 0:
            bs = [b for b in others if b["k"] == bk]
            if bs:
                key = ("f", task, a["k"], a["t"], a["i"])
                if key in seen: continue
                seen.add(key)
                share = min(1.0, max(a["pm"][b["y"]] for b in bs) / max(1e-9, 1 - a["p_y"]))
                feed[a["reg"]].append(share)

def summ(v): return f"mean={np.mean(v):.2f} med={np.median(v):.2f} p90={np.quantile(v, .9):.2f} n={len(v)}"

print("\nSTRIP (positive update's counter-mass on sibling's chosen token):")
allpos = [x for (succ, reg), v in strip.items() for x in v]
if allpos: print(f"  all siblings          : {summ(allpos)}")
for succ in (True, False):
    v = [x for (s2, reg), vv in strip.items() if s2 == succ for x in vv]
    if v: print(f"  sibling adv {'>0 (FRATRICIDE)' if succ else '<=0            '}: {summ(v)}")
for reg in ("think", "action"):
    v = [x for (s2, r2), vv in strip.items() if r2 == reg for x in vv]
    if v: print(f"  region {reg:6s}         : {summ(v)}")

print("\nFEED (negative update's freed mass GRPO hands to BEST sibling's token):")
allf = [x for v in feed.values() for x in v]
if allf: print(f"  all                   : {summ(allf)}")
for reg in ("think", "action"):
    if feed.get(reg): print(f"  region {reg:6s}         : {summ(feed[reg])}")

json.dump(dict(strip={f"{k[0]}_{k[1]}": v for k, v in strip.items()},
               feed=dict(feed), coverage=cov, groups=len(multi), diverse=len(div)),
          open(os.path.join(HERE, "ws_ccr_strip.json"), "w"))
print("\nWS_CCR_STRIP_DONE")

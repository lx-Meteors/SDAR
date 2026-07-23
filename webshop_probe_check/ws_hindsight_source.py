"""Hindsight-source comparison at fratricide pair sites: does the privilege
need to be the BEST sibling's terminal solution, or is each positive sample's
OWN outcome enough?  Also test gold (goal asin+options) as the oracle bound.

Teachers (success side only):
  own  = this trajectory's own final purchase (HER-style, zero group compute)
  best = best OTHER successful sibling's final purchase (current v5 scheme)
  gold = hidden goal asin + goal_options (oracle; instruction text excluded
         since the agent already sees it)

Metrics per teacher, gate lam = TV/(TV + mean_group(TV)):
  strip   = share of counter-mass landing on sibling token y_b after gated
            flattening (lower = better protection; GRPO = proportional baseline)
  siteLam = lambda at the pair sites (should be high: forks must open the gate)
  openBg  = fraction of all positions with lam > 0.5 (background distortion,
            lower = better localization)
  contrast= mean site TV / mean overall TV (localization sharpness)

Run: qwen-infer env, GPU5, ~5 min.
"""
import json, os, re, sys
import numpy as np
import torch
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "/data1/test/yyy/BEACON/checkpoints/verl_agent_webshop/beacon_qwen2.5_1.5b/step150_hf"
HERE = os.path.dirname(os.path.abspath(__file__))
FILES = sys.argv[1:] or ["ws_ckpt_rollouts2.json", "ws_ckpt_rollouts3.json"]

trajs = []
for fn in FILES:
    for t in json.load(open(os.path.join(HERE, fn))):
        t["gk"] = (fn, t["task"])
        trajs.append(t)

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

ASIN_RE = re.compile(r"click\[(b0\w+)\]")
NAV = ("back to search", "< prev", "next >", "description", "features", "reviews",
       "buy now", "search")

def terminal_solution(tr):
    asin, opts = None, []
    for s in tr["steps"]:
        a = s["action"].lower()
        m = ASIN_RE.match(a)
        if m: asin = m.group(1); opts = []
        elif a.startswith("click[") and a[6:-1] not in NAV and not a[6:-1].startswith("b0"):
            opts.append(a[6:-1])
    return asin, opts

def priv_outcome(asin, opts, score):
    return ("Privileged information (invisible to the agent): this task was solved "
            f"(final reward {score:.2f}); the successful final purchase was item ID "
            f"{asin} with selected options {opts}.\n\n")

def priv_gold(tr):
    g = tr["goal"]
    opts = list(g.get("goal_options", {}).values())
    return ("Privileged information (invisible to the agent): the correct final "
            f"purchase for this task is item ID {g['asin'].lower()} with selected "
            f"options {opts}.\n\n")

groups_by_task = defaultdict(list)
for t in trajs: groups_by_task[t["gk"]].append(t)

TEACHERS = ("own", "best", "gold")
# step_recs: dict(gk, tv={name: vec}, sites=[(i, y_a, y_b, lp_row)])
step_recs = []
for gk, gtr in sorted(groups_by_task.items()):
    scores = [t["score"] for t in gtr]
    gmean = float(np.mean(scores))
    for t in gtr: t["adv"] = t["score"] - gmean
    succ = [t for t in gtr if t["adv"] > 0]
    if len(set(scores)) < 2 or len(succ) < 2: continue
    menu_rows = []
    for tr in gtr:
        if tr["adv"] <= 0: continue
        for s in tr["steps"]:
            resp = s.get("response", "")
            if not resp or resp == "(forced)": continue
            p, out_ids = dists(s["prompt"], resp)
            t5v, t5i = p.topk(5, dim=-1)
            t5v, t5i = t5v.cpu().numpy(), t5i.cpu().numpy()
            idx = torch.arange(len(out_ids), device=DEV)
            py = p[idx, torch.tensor(out_ids, device=DEV)].cpu().numpy()
            for i in range(len(out_ids)):
                menu_rows.append(dict(k=tr["k"], t=s["t"], i=i, y=int(out_ids[i]),
                                      p_y=float(py[i]),
                                      menu=tuple(sorted(int(x) for x in t5i[i])),
                                      pm={int(v): float(x) for v, x in zip(t5i[i], t5v[i])}))
            del p; torch.cuda.empty_cache()
    mg = defaultdict(list)
    for r in menu_rows: mg[r["menu"]].append(r)
    pairs, seen = defaultdict(list), set()
    for mem in mg.values():
        if len({m["k"] for m in mem}) < 2 or len({m["y"] for m in mem}) < 2: continue
        for a in mem:
            if a["p_y"] > 0.95: continue
            for b in mem:
                if b["k"] == a["k"] or b["y"] == a["y"] or b["y"] not in a["pm"]: continue
                key = (a["k"], a["t"], a["i"], b["y"])
                if key in seen: continue
                seen.add(key)
                pairs[(a["k"], a["t"])].append((a["i"], a["y"], b["y"]))
    if not pairs: continue
    by_k = {t["k"]: t for t in gtr}
    best_of = {t["k"]: max([o for o in gtr if o["k"] != t["k"]], key=lambda o: o["score"])
               for t in gtr}
    for (k, t_), plist in sorted(pairs.items()):
        tr = by_k[k]
        s = next(st for st in tr["steps"] if st["t"] == t_)
        p, out_ids = dists(s["prompt"], s["response"])
        a_own, o_own = terminal_solution(tr)
        b = best_of[k]
        a_best, o_best = terminal_solution(b)
        privs = dict(own=priv_outcome(a_own, o_own, tr["score"]),
                     best=priv_outcome(a_best, o_best, b["score"]),
                     gold=priv_gold(tr))
        tvs = {}
        for name, pv in privs.items():
            q, _ = dists(pv + s["prompt"], s["response"])
            tvs[name] = (0.5 * (q - p).abs().sum(-1)).cpu().numpy()
            del q
        lp = torch.log(p + 1e-12)
        sites = [(i, y_a, y_b, lp[i].cpu().numpy().astype(np.float32))
                 for (i, y_a, y_b) in plist]
        step_recs.append(dict(gk=gk, tv=tvs, sites=sites))
        del p, lp; torch.cuda.empty_cache()
    print(f"group {gk}: cached", flush=True)

# ---- evaluate ----
grp_mean = {name: {} for name in TEACHERS}
for name in TEACHERS:
    for gk in {r["gk"] for r in step_recs}:
        grp_mean[name][gk] = float(np.mean(np.concatenate(
            [r["tv"][name] for r in step_recs if r["gk"] == gk])))

base = []
for r in step_recs:
    for (i, y_a, y_b, lp_row) in r["sites"]:
        pr = np.exp(lp_row); pr[y_a] = 0; pr = pr / pr.sum()
        base.append(float(pr[y_b]))
print(f"\npairs={len(base)}  GRPO strip: mean={np.mean(base):.3f} med={np.median(base):.3f}")

for name in TEACHERS:
    shares, site_lams, all_lams, site_tv, all_tv = [], [], [], [], []
    for r in step_recs:
        c = max(grp_mean[name][r["gk"]], 1e-4)
        tvv = r["tv"][name]
        lams = tvv / (tvv + c)
        all_lams.extend(lams.tolist()); all_tv.extend(tvv.tolist())
        for (i, y_a, y_b, lp_row) in r["sites"]:
            lam = float(lams[i])
            site_lams.append(lam); site_tv.append(float(tvv[i]))
            w = np.exp((1 - lam) * lp_row - ((1 - lam) * lp_row).max())
            w[y_a] = 0; w = w / w.sum()
            shares.append(float(w[y_b]))
    all_lams = np.array(all_lams)
    print(f"  {name:5s}: strip mean={np.mean(shares):.3f} med={np.median(shares):.3f} "
          f"| siteLam={np.mean(site_lams):.3f} | openBg={np.mean(all_lams > 0.5):.3f} "
          f"| meanTV={np.mean(all_tv):.4f} siteTV={np.mean(site_tv):.4f} "
          f"contrast={np.mean(site_tv)/max(np.mean(all_tv),1e-9):.2f}")

# how often own == best terminal solution (same asin+options)?
same = 0; tot = 0
for gk, gtr in sorted(groups_by_task.items()):
    scores = [t["score"] for t in gtr]
    if len(set(scores)) < 2: continue
    gmean = float(np.mean(scores))
    succ = [t for t in gtr if t["score"] - gmean > 0]
    if len(succ) < 2: continue
    for tr in succ:
        b = max([o for o in gtr if o["k"] != tr["k"]], key=lambda o: o["score"])
        tot += 1
        if terminal_solution(tr) == terminal_solution(b): same += 1
print(f"\nown==best terminal solution: {same}/{tot}")
print("WS_HINDSIGHT_SOURCE_DONE")

"""v7 probe: conserved credit redistribution ON ADVANTAGES (time axis), driven
by the candidate-distribution triple (p, q_own, q_sib).

  lam_rel  <- TV(p, q_own)      outcome relevance      (odds gate, group mean)
  lam_frat <- TV(q_own, q_sib)  certificate idiosyncrasy (odds gate, group mean)
  w~_i = (1 - lam_frat_i) * (1 + lam_rel_i)
  w_i  = w~_i * sum(m) / sum(m * w~)   per training row (= per step response)

Reports mean credit weight w at four position classes plus the effective
sibling-strip change (strip scales with the advantage weight at the site):
  solution-fork sites   (succ siblings bought different things)  -> expect w << 1
  wording-fork sites    (same purchase, different phrasing)      -> expect w >= 1
  consensus positions   (lam_rel > .5 & lam_frat < .5, non-site) -> expect w > 1
  quiet background      (lam_rel < .1)                           -> expect w ~ 1

Run: slime-qwen35 env, one free GPU, ~1 min.  Usage: python ws_v7_credit.py [files...]
"""
import json, os, re, sys
import numpy as np
import torch
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "/data1/test/yyy/BEACON/checkpoints/verl_agent_webshop/beacon_qwen2.5_1.5b/step150_hf"
HERE = os.path.dirname(os.path.abspath(__file__))
FILES = sys.argv[1:] or ["ws_ckpt_rollouts4.json"]

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
    pre_ids = tok(pre, add_special_tokens=False, truncation=True, max_length=6000).input_ids
    out_ids = tok(response, add_special_tokens=False).input_ids[:400]
    full = torch.tensor([pre_ids + out_ids], device=DEV)
    lg = model(full, use_cache=False, logits_to_keep=len(out_ids) + 1).logits[0].float()
    return torch.softmax(lg[:-1], dim=-1), out_ids

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

groups_by_task = defaultdict(list)
for t in trajs: groups_by_task[t["gk"]].append(t)

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
        sib = best_of[k]
        sol_own, sol_sib = terminal_solution(tr), terminal_solution(sib)
        q_own, _ = dists(priv_outcome(*sol_own, tr["score"]) + s["prompt"], s["response"])
        q_sib, _ = dists(priv_outcome(*sol_sib, sib["score"]) + s["prompt"], s["response"])
        tv_rel = (0.5 * (q_own - p).abs().sum(-1)).cpu().numpy()
        tv_frat = (0.5 * (q_own - q_sib).abs().sum(-1)).cpu().numpy()
        d1, d2 = q_own - p, q_sib - p
        agree = (torch.sign(d1) == torch.sign(d2))
        common = torch.where(agree, torch.minimum(d1.abs(), d2.abs()), torch.zeros_like(d1))
        tv_cons = (0.5 * common.sum(-1)).cpu().numpy()
        lp = torch.log(p + 1e-12)
        sites = [(i, y_a, y_b, lp[i].cpu().numpy().astype(np.float32))
                 for (i, y_a, y_b) in plist]
        step_recs.append(dict(gk=gk, tv_rel=tv_rel, tv_frat=tv_frat, tv_cons=tv_cons,
                              sites=sites, same_sol=(sol_own == sol_sib)))
        del p, q_own, q_sib, lp; torch.cuda.empty_cache()
    print(f"group {gk}: cached", flush=True)

# ---- gates (group-mean self-calibration) ----
def gate(recs, key):
    gm = {}
    for gk in {r["gk"] for r in recs}:
        gm[gk] = max(float(np.mean(np.concatenate(
            [r[key] for r in recs if r["gk"] == gk]))), 1e-4)
    return {id(r): r[key] / (r[key] + gm[r["gk"]]) for r in recs}

lam_rel = gate(step_recs, "tv_rel")
lam_frat = gate(step_recs, "tv_frat")
lam_cons = gate(step_recs, "tv_cons")

# ---- w per row (conserved within each step response) ----
cls = dict(sol_site=[], word_site=[], consensus=[], quiet=[], all=[])
strip_grpo, strip_v7 = [], []
for r in step_recs:
    lr, lf, lc = lam_rel[id(r)], lam_frat[id(r)], lam_cons[id(r)]
    wt = (1 - lf) * (1 + lc)
    w = wt * len(wt) / max(wt.sum(), 1e-6)
    site_idx = {i for (i, _, _, _) in r["sites"]}
    cls["all"].extend(w.tolist())
    for j in range(len(w)):
        if j in site_idx:
            cls["sol_site" if not r["same_sol"] else "word_site"].append(float(w[j]))
        elif lr[j] > 0.5 and lf[j] < 0.5:
            cls["consensus"].append(float(w[j]))
        elif lr[j] < 0.1:
            cls["quiet"].append(float(w[j]))
    for (i, y_a, y_b, lp_row) in r["sites"]:
        pr = np.exp(lp_row); pr[y_a] = 0; pr = pr / pr.sum()
        kind = "word" if r["same_sol"] else "sol"
        strip_grpo.append((kind, float(pr[y_b])))
        strip_v7.append((kind, float(w[i] * pr[y_b])))

print(f"\ncredit weight w by position class (GRPO == 1.0 everywhere):")
for name, lab in (("sol_site", "solution-fork sites"), ("word_site", "wording-fork sites"),
                  ("consensus", "consensus positions"), ("quiet", "quiet background"),
                  ("all", "all positions")):
    v = np.array(cls[name])
    if len(v):
        print(f"  {lab:22s} n={len(v):5d}  mean={v.mean():.3f}  med={np.median(v):.3f}  "
              f"p90={np.percentile(v,90):.3f}  max={v.max():.3f}")

print(f"\nsibling strip (scales with w):")
for kind in ("sol", "word"):
    g = [v for k, v in strip_grpo if k == kind]; s = [v for k, v in strip_v7 if k == kind]
    print(f"  {kind:4s} forks: GRPO {np.mean(g):.3f}/{np.median(g):.3f} -> v7c {np.mean(s):.3f}/{np.median(s):.3f}  (n={len(g)})")
print("WS_V7_CREDIT_DONE")

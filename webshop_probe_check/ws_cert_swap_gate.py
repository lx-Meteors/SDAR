"""Certificate-swap gate probe (v6 design): can TV(q_own, q_sib) -- the
divergence between two sibling-certificate teachers -- separate "arbitrary
commitment" fork positions (attenuate credit) from "consensus correction"
positions (keep credit), which the old student-teacher gate TV(p, q_own)
cannot distinguish?

Gates compared at succ-succ fratricide pair sites (out-of-sample rollouts4):
  old  = TV(p, q_own)        opens at ANY outcome-relevant position
  new  = TV(q_own, q_sib)    opens only where the two certificates disagree,
                             i.e. where the choice is arbitrary among sibling
                             successes (zero counterfactual advantage)

Sites are split by whether own/sib terminal solutions differ:
  solution forks (different purchase) -> new gate SHOULD open
  wording forks (same purchase)       -> new gate SHOULD stay closed
Also reports consensus positions: lam_old high but lam_new low = credit the
old gate would wrongly attenuate but the new gate preserves.

Run: qwen-infer env, GPU5, ~5 min.
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
    # memory-lean: only materialize logits for the response span (+1 for the
    # position that predicts the first response token)
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

# recs: per success-side step, tv_old / tv_new vectors, pair sites, fork type
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
                pairs[(a["k"], a["t"])].append((a["i"], a["y"], b["y"], b["k"]))
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
        pv_own = priv_outcome(*sol_own, tr["score"])
        pv_sib = priv_outcome(*sol_sib, sib["score"])
        q_own, _ = dists(pv_own + s["prompt"], s["response"])
        q_sib, _ = dists(pv_sib + s["prompt"], s["response"])
        tv_old = (0.5 * (q_own - p).abs().sum(-1)).cpu().numpy()
        tv_new = (0.5 * (q_own - q_sib).abs().sum(-1)).cpu().numpy()
        lp = torch.log(p + 1e-12)
        sites = [(i, y_a, y_b, lp[i].cpu().numpy().astype(np.float32))
                 for (i, y_a, y_b, _bk) in plist]
        step_recs.append(dict(gk=gk, tv_old=tv_old, tv_new=tv_new, sites=sites,
                              same_sol=(sol_own == sol_sib)))
        del p, q_own, q_sib, lp; torch.cuda.empty_cache()
    print(f"group {gk}: cached", flush=True)

# ---- gate stats ----
def lam_of(recs, key):
    grp_mean = {}
    for gk in {r["gk"] for r in recs}:
        grp_mean[gk] = max(float(np.mean(np.concatenate(
            [r[key] for r in recs if r["gk"] == gk]))), 1e-4)
    return {id(r): r[key] / (r[key] + grp_mean[r["gk"]]) for r in recs}

lam_old = lam_of(step_recs, "tv_old")
lam_new = lam_of(step_recs, "tv_new")

rows = dict(sol=dict(o=[], n=[]), word=dict(o=[], n=[]))
bg_o, bg_n = [], []
atten_strip_old, atten_strip_new, base_strip = [], [], []
for r in step_recs:
    lo, ln = lam_old[id(r)], lam_new[id(r)]
    bg_o.extend(lo.tolist()); bg_n.extend(ln.tolist())
    kind = "word" if r["same_sol"] else "sol"
    for (i, y_a, y_b, lp_row) in r["sites"]:
        rows[kind]["o"].append(float(lo[i])); rows[kind]["n"].append(float(ln[i]))
        pr = np.exp(lp_row); pr[y_a] = 0; pr = pr / pr.sum()
        base_strip.append(float(pr[y_b]))
        # attenuation lever: strip after A*(1-lam) is simply (1-lam)*GRPO strip
        atten_strip_old.append(float((1 - lo[i]) * pr[y_b]))
        atten_strip_new.append(float((1 - ln[i]) * pr[y_b]))

bg_o, bg_n = np.array(bg_o), np.array(bg_n)
n_sol, n_word = len(rows["sol"]["o"]), len(rows["word"]["o"])
print(f"\npair sites: {n_sol} solution-fork, {n_word} wording-fork; "
      f"positions total {len(bg_o)}")
print(f"{'':16s}{'lam_old':>9s}{'lam_new':>9s}")
for kind, lab in (("sol", "solution forks"), ("word", "wording forks")):
    if rows[kind]["o"]:
        print(f"{lab:16s}{np.mean(rows[kind]['o']):9.3f}{np.mean(rows[kind]['n']):9.3f}")
print(f"{'background':16s}{np.mean(bg_o):9.3f}{np.mean(bg_n):9.3f}")
print(f"{'openBg(>0.5)':16s}{np.mean(bg_o > 0.5):9.3f}{np.mean(bg_n > 0.5):9.3f}")

print(f"\nfratricide strip via advantage attenuation (GRPO base "
      f"{np.mean(base_strip):.3f}/{np.median(base_strip):.3f} mean/med):")
print(f"  old gate A*(1-lam_old): {np.mean(atten_strip_old):.3f}/{np.median(atten_strip_old):.3f}")
print(f"  new gate A*(1-lam_new): {np.mean(atten_strip_new):.3f}/{np.median(atten_strip_new):.3f}")

# consensus positions: old gate open but new gate closed = credit preserved
hi_old = bg_o > 0.5
if hi_old.any():
    saved = float(np.mean(bg_n[hi_old] < 0.5))
    print(f"\nconsensus check: of positions with lam_old>0.5 ({hi_old.sum()}), "
          f"{saved:.1%} have lam_new<0.5 (credit the new gate preserves)")
print("WS_CERT_SWAP_GATE_DONE")

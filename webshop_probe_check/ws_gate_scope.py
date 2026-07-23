"""Calibration-scope test for the self-calibrated gate lam = TV/(TV+mean(TV)):
does the averaging scope of mean(TV) matter?  per-STEP vs per-GROUP vs GLOBAL.
Two-pass over the fresh fratricide pairs (221): pass 1 caches TV vectors and
the student log-prob rows at pair sites; pass 2 evaluates gates under each
scope.  Metrics: strip share on sibling token + quiet-position lambda.
Run: qwen-infer env, GPU5, ~2 min.
"""
import json, os, re
import numpy as np
import torch
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "/data1/test/yyy/BEACON/checkpoints/verl_agent_webshop/beacon_qwen2.5_1.5b/step150_hf"
HERE = os.path.dirname(os.path.abspath(__file__))
FILES = ["ws_ckpt_rollouts2.json", "ws_ckpt_rollouts3.json"]
FLOOR = 0.005

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

def terminal_solution(btr):
    asin, opts = None, []
    for s in btr["steps"]:
        a = s["action"].lower()
        m = ASIN_RE.match(a)
        if m: asin = m.group(1); opts = []
        elif a.startswith("click[") and a[6:-1] not in NAV and not a[6:-1].startswith("b0"):
            opts.append(a[6:-1])
    return asin, opts

def term_priv(btr):
    asin, opts = terminal_solution(btr)
    return ("Privileged information (invisible to the agent): this task was solved "
            f"(final reward {btr['score']:.2f}); the successful final purchase was item ID "
            f"{asin} with selected options {opts}.\n\n")

groups_by_task = defaultdict(list)
for t in trajs: groups_by_task[t["gk"]].append(t)

# ---- pass 1: build pairs and cache tvv + lp rows ----
step_recs = []   # dict(gk, tvv, sites=[(i, y_a, y_b, lp_row)])
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
        s = next(s for s in tr["steps"] if s["t"] == t_)
        p, out_ids = dists(s["prompt"], s["response"])
        q, _ = dists(term_priv(best_of[k]) + s["prompt"], s["response"])
        tvv = (0.5 * (q - p).abs().sum(-1)).cpu().numpy()
        lp = torch.log(p + 1e-12)
        sites = [(i, y_a, y_b, lp[i].cpu().numpy().astype(np.float32))
                 for (i, y_a, y_b) in plist]
        step_recs.append(dict(gk=gk, tvv=tvv, sites=sites))
        del p, q, lp; torch.cuda.empty_cache()
    print(f"group {gk}: cached", flush=True)

# ---- pass 2: scopes ----
glob_mean = float(np.mean(np.concatenate([r["tvv"] for r in step_recs])))
grp_mean = {}
for gk in {r["gk"] for r in step_recs}:
    grp_mean[gk] = float(np.mean(np.concatenate([r["tvv"] for r in step_recs if r["gk"] == gk])))
print(f"\nglobal meanTV={glob_mean:.4f}")
for gk, v in sorted(grp_mean.items()): print(f"  group {gk}: meanTV={v:.4f}")

def gate_eval(scope):
    shares, lam_quiet = [], []
    for r in step_recs:
        c = dict(step=float(np.mean(r["tvv"])), group=grp_mean[r["gk"]],
                 global_=glob_mean)[scope]
        c = max(c, FLOOR)
        lams = r["tvv"] / (r["tvv"] + c)
        lam_quiet.extend(lams[r["tvv"] < 0.02].tolist())
        for (i, y_a, y_b, lp_row) in r["sites"]:
            lam = float(lams[i])
            w = np.exp((1 - lam) * lp_row - ((1 - lam) * lp_row).max())
            w[y_a] = 0; w = w / w.sum()
            shares.append(float(w[y_b]))
    return shares, lam_quiet

base = []
for r in step_recs:
    for (i, y_a, y_b, lp_row) in r["sites"]:
        pr = np.exp(lp_row); pr[y_a] = 0; pr = pr / pr.sum()
        base.append(float(pr[y_b]))
print(f"\npairs={len(base)}  GRPO strip: mean={np.mean(base):.2f} med={np.median(base):.2f}")
for scope in ("step", "group", "global_"):
    sh, lq = gate_eval(scope)
    print(f"  scope={scope:8s}: strip mean={np.mean(sh):.2f} med={np.median(sh):.2f} "
          f"p90={np.quantile(sh, .9):.2f} | quiet-lam={np.mean(lq):.3f}")
print("WS_GATE_SCOPE_DONE")

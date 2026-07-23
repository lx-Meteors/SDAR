"""CCR v5 validation on FRESH rollouts (new tasks, never used to design the
method): OPSD-gated flattening with short terminal-purchase privilege.

For every variance group among the new tasks: find succ-succ fratricide pairs
(menu collision, both adv>0, p_y<=0.95, dedup) and compare strip share on the
sibling's token under GRPO vs v5 (omega ~ p^(1-lam), lam=min(1,TV/c)).
Also reports gate background stats (TV over all positions of touched steps).
Run: qwen-infer env, GPU5.
"""
import json, os, re, sys
import numpy as np
import torch
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "/data1/test/yyy/BEACON/checkpoints/verl_agent_webshop/beacon_qwen2.5_1.5b/step150_hf"
HERE = os.path.dirname(os.path.abspath(__file__))
FILES = sys.argv[1:] or ["ws_ckpt_rollouts2.json", "ws_ckpt_rollouts3.json"]
C_GATE = (0.1, 0.2)
GAMMA = 0.5

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

# gate fns take (tv_i, bc_i, st): bc_i = Bhattacharyya coeff at position i,
# st = per-step stats (mean/q90 of TV over that step's positions)
GATES = {
    "smooth c=.01  ": lambda tv, bc, st: tv / (tv + 0.01),
    "smooth c=.03  ": lambda tv, bc, st: tv / (tv + 0.03),
    "smooth c=.05  ": lambda tv, bc, st: tv / (tv + 0.05),
    "smooth c=.1   ": lambda tv, bc, st: tv / (tv + 0.1),
    "smooth c=.2   ": lambda tv, bc, st: tv / (tv + 0.2),
    "self c=meanTV ": lambda tv, bc, st: tv / (tv + max(st["mean"], 1e-3)),
    "self c=q90    ": lambda tv, bc, st: min(1.0, tv / max(st["q90"], 1e-3)),
    "1-BC (0par)   ": lambda tv, bc, st: 1.0 - bc,
}
grand_base, grand_v5 = [], {c: [] for c in C_GATE}
gate_share = {g: [] for g in GATES}
gate_lam_all = {g: [] for g in GATES}    # lambda over ALL positions (background)
gate_lam_quiet = {g: [] for g in GATES}  # lambda where tv < 0.02
grand_tv_pairs, grand_tv_all = [], []
n_var_groups = 0
for gk, gtr in sorted(groups_by_task.items()):
    scores = [t["score"] for t in gtr]
    gmean = float(np.mean(scores))
    for t in gtr: t["adv"] = t["score"] - gmean
    succ = [t for t in gtr if t["adv"] > 0]
    if len(set(scores)) < 2 or len(succ) < 2:
        continue
    n_var_groups += 1
    print(f"\n== group {gk} scores={scores}", flush=True)
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
    np_ = sum(len(v) for v in pairs.values())
    print(f"   succ-succ pairs: {np_}")
    if not np_: continue
    by_k = {t["k"]: t for t in gtr}
    best_of = {t["k"]: max([o for o in gtr if o["k"] != t["k"]], key=lambda o: o["score"])
               for t in gtr}
    for (k, t_), plist in sorted(pairs.items()):
        tr = by_k[k]
        s = next(s for s in tr["steps"] if s["t"] == t_)
        pv = term_priv(best_of[k])
        p, out_ids = dists(s["prompt"], s["response"])
        q, _ = dists(pv + s["prompt"], s["response"])
        lp = torch.log(p + 1e-12)
        tvv = (0.5 * (q - p).abs().sum(-1)).cpu().numpy()
        bcv = (p.sqrt() * q.sqrt()).sum(-1).clamp(max=1.0).cpu().numpy()
        st = dict(mean=float(np.mean(tvv)), q90=float(np.quantile(tvv, 0.9)))
        grand_tv_all.extend(tvv.tolist())
        for g, fn in GATES.items():
            lams = np.array([fn(t2, b2, st) for t2, b2 in zip(tvv, bcv)])
            gate_lam_all[g].extend(lams.tolist())
            gate_lam_quiet[g].extend(lams[tvv < 0.02].tolist())
        for (i, y_a, y_b) in plist:
            pr = p[i].clone(); pr[y_a] = 0; pr = pr / pr.sum()
            grand_base.append(float(pr[y_b]))
            tv_i = float(tvv[i])
            grand_tv_pairs.append(tv_i)
            for c in C_GATE:
                lam = min(1.0, tv_i / c)
                w = torch.softmax((1 - lam) * lp[i], dim=-1)
                w[y_a] = 0; w = w / w.sum()
                grand_v5[c].append(float(w[y_b]))
            for g, fn in GATES.items():
                lam = fn(tv_i, float(bcv[i]), st)
                w = torch.softmax((1 - lam) * lp[i], dim=-1)
                w[y_a] = 0; w = w / w.sum()
                gate_share[g].append(float(w[y_b]))
        del p, q; torch.cuda.empty_cache()

def summ(v): return f"mean={np.mean(v):.2f} med={np.median(v):.2f} p90={np.quantile(v, .9):.2f}"
print(f"\n===== FRESH-DATA RESULT: {n_var_groups} variance groups, {len(grand_base)} pairs =====")
if grand_base:
    print(f"  gate background: TV med={np.median(grand_tv_all):.3f} "
          f"frac>0.1={np.mean(np.array(grand_tv_all) > 0.1):.0%} (n={len(grand_tv_all)})")
    print(f"  TV at pair sites: med={np.median(grand_tv_pairs):.3f} "
          f"p10={np.quantile(grand_tv_pairs, .1):.3f}")
    print(f"  GRPO strip on sibling token : {summ(grand_base)}")
    for c in C_GATE:
        print(f"  v5 flatten c={c:<4}          : {summ(grand_v5[c])}")
    print(f"\n  ---- gate-shape comparison (protection | background distortion) ----")
    print(f"  {'gate':16s} {'strip mean/med':>16s} {'bg mean-lam':>12s} {'quiet mean-lam':>15s} {'frac lam>.5':>12s}")
    for g in GATES:
        sh = gate_share[g]; la = np.array(gate_lam_all[g]); lq = np.array(gate_lam_quiet[g])
        print(f"  {g:16s} {np.mean(sh):8.2f}/{np.median(sh):.2f}   {np.mean(la):12.3f} {np.mean(lq):15.3f} {np.mean(la > 0.5):12.1%}")
print("WS_CCR_V5_FRESH_DONE")

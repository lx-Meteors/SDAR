"""Field-form CCR: does the full-vocab OPSD tilt subsume the S+ set version?

omega_v ~ p_v * exp(-gamma * clip(log(q_v/p_v), +-5)),  v != y  (success side)
q = student conditioned on best sibling's action sequence (hindsight, T2).

Test on the exact pairs where GRPO fratricide was measured (BEACON step150
rollouts, task1 = the only variance group): strip share on the succ sibling's
token under GRPO (baseline ~0.98) vs field-omega, NO menu matching in the
method (menus only used to define eval pairs).  Also: where does the counter-
mass go instead (top stripped tokens, decoded).
Run: qwen-infer env, GPU5, ~3 min.
"""
import json, os
import numpy as np
import torch
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "/data1/test/yyy/BEACON/checkpoints/verl_agent_webshop/beacon_qwen2.5_1.5b/step150_hf"
HERE = os.path.dirname(os.path.abspath(__file__))
trajs = [t for t in json.load(open(os.path.join(HERE, "ws_ckpt_rollouts.json"))) if t["task"] == 1]
GAMMAS = (0.5, 1.0, 2.0)

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cuda:0")
model.eval(); DEV = model.device

@torch.no_grad()
def dists(user_content, response):
    msgs = [{"role": "user", "content": user_content}]
    pre = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    pre_ids = tok(pre, add_special_tokens=False, truncation=True, max_length=7000).input_ids
    enc = tok(response, add_special_tokens=False, return_offsets_mapping=True)
    out_ids = enc.input_ids[:400]
    offs = enc.offset_mapping[:400]
    full = torch.tensor([pre_ids + out_ids], device=DEV)
    lg = model(full).logits[0].float()
    L0 = len(pre_ids)
    return torch.softmax(lg[L0 - 1:L0 - 1 + len(out_ids), :], dim=-1), out_ids, offs

import sys, re
MODE = sys.argv[1] if len(sys.argv) > 1 else "acts"   # acts | full | term
LOCAL = len(sys.argv) > 2 and sys.argv[2] == "local"   # inject right before decision instead of front

def teacher_ctx(pv, prompt):
    if LOCAL and "Now it's your turn" in prompt:
        return prompt.replace("Now it's your turn", pv + "Now it's your turn")
    return pv + prompt

ASIN_RE = re.compile(r"click\[(b0\w+)\]")
NAV = ("back to search", "< prev", "next >", "description", "features", "reviews",
       "buy now", "search")

def terminal_solution(btr):
    """Short xi: the successful final purchase, reconstructed from the sibling's
    own actions (zero external labels): last clicked item + option clicks."""
    asin, opts = None, []
    for s in btr["steps"]:
        a = s["action"].lower()
        m = ASIN_RE.match(a)
        if m: asin = m.group(1); opts = []
        elif a.startswith("click[") and a[6:-1] not in NAV and not a[6:-1].startswith("b0"):
            opts.append(a[6:-1])
    return asin, opts

def proc_priv(btr, wtr=None, mode=None):
    MODE = mode or globals()["MODE"]
    if MODE == "contrast":
        asin, opts = terminal_solution(btr)
        s = ("Privileged information (invisible to the agent): this task was attempted "
             f"multiple times. A successful attempt (final reward {btr['score']:.2f}) "
             f"purchased item ID {asin} with selected options {opts}.")
        if wtr is not None:
            wasin, wopts = terminal_solution(wtr)
            s += (f" A failed attempt (final reward {wtr['score']:.2f}) purchased "
                  f"item ID {wasin} with selected options {wopts}.")
        return s + "\n\n"
    if MODE == "full":
        body = "\n".join(s.get("response", "")[:800] for s in btr["steps"]
                         if s.get("response") and s["response"] != "(forced)")
        return ("Privileged information (invisible to the agent): an expert solved this exact task "
                f"(final reward {btr['score']:.2f}). The expert's full reasoning and actions were:\n"
                f"{body[:3500]}\n\n")
    if MODE == "term":
        asin, opts = terminal_solution(btr)
        return ("Privileged information (invisible to the agent): this task was solved "
                f"(final reward {btr['score']:.2f}); the successful final purchase was item ID "
                f"{asin} with selected options {opts}.\n\n")
    acts = " -> ".join(s["action"] for s in btr["steps"])
    return ("Privileged information (invisible to the agent): an expert solved this exact task "
            f"(final reward {btr['score']:.2f}) with the action sequence: {acts}.\n\n")

gmean = float(np.mean([t["score"] for t in trajs]))
for t in trajs: t["adv"] = t["score"] - gmean
print("scores:", [t["score"] for t in trajs], "advs:", [round(t["adv"], 2) for t in trajs])

# pass 1: menus for eval-pair definition (p only, cheap fields)
menu_rows = []
for tr in trajs:
    for s in tr["steps"]:
        resp = s.get("response", "")
        if not resp or resp == "(forced)": continue
        act_start = resp.lower().find(s["action"].lower())
        if act_start < 0: act_start = len(resp)
        p, out_ids, offs = dists(s["prompt"], resp)
        t5v, t5i = p.topk(5, dim=-1)
        t5v, t5i = t5v.cpu().numpy(), t5i.cpu().numpy()
        idx = torch.arange(len(out_ids), device=DEV)
        py = p[idx, torch.tensor(out_ids, device=DEV)].cpu().numpy()
        for i in range(len(out_ids)):
            menu_rows.append(dict(k=tr["k"], adv=tr["adv"], t=s["t"], i=i,
                                  y=int(out_ids[i]), p_y=float(py[i]),
                                  reg="act" if offs[i][0] >= act_start else "think",
                                  menu=tuple(sorted(int(x) for x in t5i[i])),
                                  pm={int(v): float(x) for v, x in zip(t5i[i], t5v[i])}))
        del p; torch.cuda.empty_cache()
    print(f"pass1 k{tr['k']} done", flush=True)

groups = defaultdict(list)
for r in menu_rows: groups[r["menu"]].append(r)
pairs = defaultdict(list)   # (a_k, a_t) -> [(i, y_a, y_b), ...]
seen = set()
for mem in groups.values():
    if len({m["k"] for m in mem}) < 2 or len({m["y"] for m in mem}) < 2: continue
    for a in mem:
        if a["adv"] <= 0 or a["p_y"] > 0.95: continue
        for b in mem:
            if b["k"] == a["k"] or b["adv"] <= 0 or b["y"] == a["y"] or b["y"] not in a["pm"]: continue
            key = (a["k"], a["t"], a["i"], b["y"])
            if key in seen: continue
            seen.add(key)
            pairs[(a["k"], a["t"])].append((a["i"], a["y"], b["y"], a["reg"]))
npairs = sum(len(v) for v in pairs.values())
print(f"succ-succ eval pairs: {npairs} at {len(pairs)} (traj,step) sites")

# pass 2: p and q at needed steps, field-omega shares
by_k = {t["k"]: t for t in trajs}
best_of, worst_of = {}, {}
for t in trajs:
    sibs = [o for o in trajs if o["k"] != t["k"]]
    best_of[t["k"]] = max(sibs, key=lambda o: o["score"])
    fails = [o for o in sibs if o["adv"] <= 0]
    worst_of[t["k"]] = min(fails, key=lambda o: o["score"]) if fails else None

res = {g: [] for g in GAMMAS}; base = []; examples = []; regs = []
flat = defaultdict(list)   # OPSD-gated flattening variants
tvs, all_tv, ens_rows = [], [], []
for (k, t_), plist in sorted(pairs.items()):
    tr = by_k[k]
    s = next(s for s in tr["steps"] if s["t"] == t_)
    pv = proc_priv(best_of[k], worst_of[k], mode="term" if MODE == "ens" else None)
    p, out_ids, _ = dists(s["prompt"], s["response"])
    q, _, _ = dists(teacher_ctx(pv, s["prompt"]), s["response"])
    if MODE == "ens":
        # teacher ensemble across the three SHORT privilege forms
        qs = []
        for m2 in ("term", "contrast", "acts"):
            pv2 = proc_priv(best_of[k], worst_of[k], mode=m2)
            q2, _, _ = dists(teacher_ctx(pv2, s["prompt"]), s["response"])
            qs.append(q2)
        lps = torch.log(p + 1e-12)
        dirs = torch.stack([ (torch.log(q2 + 1e-12) - lps).clamp(-5, 5) for q2 in qs ])
        dmean, dstd = dirs.mean(0), dirs.std(0)
        for (i, y_a, y_b, reg) in plist:
            ens_rows.append(dict(
                d=[float(dirs[j, i, y_b]) for j in range(3)],
                m=float(dmean[i, y_b]), s=float(dstd[i, y_b])))
        del qs, dirs
    lp, lq = torch.log(p + 1e-12), torch.log(q + 1e-12)
    ratio = (lq - lp).clamp(-5, 5)
    all_tv.extend((0.5 * (q - p).abs().sum(-1)).cpu().numpy().tolist())
    for (i, y_a, y_b, reg) in plist:
        pr = p[i].clone(); pr[y_a] = 0; pr = pr / pr.sum()
        base.append(float(pr[y_b])); regs.append(reg)
        tv_i = float(0.5 * (q[i] - p[i]).abs().sum())
        tvs.append(tv_i)
        for c in (0.1, 0.2, 0.4):
            lam = min(1.0, tv_i / c)
            for g2 in (0.0, 0.5):
                w = ((1 - lam) * lp[i] - g2 * ratio[i])
                w = torch.softmax(w, dim=-1)
                w[y_a] = 0; w = w / w.sum()
                flat[(c, g2)].append(float(w[y_b]))
        for g in GAMMAS:
            w = (lp[i] - g * ratio[i]); w = torch.softmax(w, dim=-1)
            w[y_a] = 0; w = w / w.sum()
            res[g].append(float(w[y_b]))
            if g == 1.0 and len(examples) < 6:
                top = torch.topk(w, 3)
                examples.append(dict(
                    k=k, t=t_, i=i, y_a=tok.decode([y_a]), y_b=tok.decode([y_b]),
                    grpo=float(pr[y_b]), ccr=float(w[y_b]),
                    q_yb=float(q[i, y_b]), p_yb=float(p[i, y_b]),
                    goes_to=[(tok.decode([int(j)]), round(float(v), 2))
                             for v, j in zip(top.values, top.indices)]))
    del p, q; torch.cuda.empty_cache()

def summ(v): return f"mean={np.mean(v):.2f} med={np.median(v):.2f} p90={np.quantile(v, .9):.2f}"
print(f"\nstrip share on succ sibling token (n={len(base)}):")
print(f"  GRPO (omega~p)        : {summ(base)}")
for g in GAMMAS:
    print(f"  CCR field gamma={g:<4}   : {summ(res[g])}")
print(f"\nOPSD divergence at pair sites: TV mean={np.mean(tvs):.3f} med={np.median(tvs):.3f} "
      f"p10={np.quantile(tvs, .1):.3f}")
print(f"OPSD divergence at ALL positions of touched steps: mean={np.mean(all_tv):.3f} "
      f"med={np.median(all_tv):.3f} p90={np.quantile(all_tv, .9):.3f}  "
      f"frac(tv>0.1)={np.mean(np.array(all_tv) > 0.1):.0%}  n={len(all_tv)}")
print("flattening variants  omega ~ p^(1-lam)*(q/p)^(-g),  lam=min(1, tv/c):")
for (c, g2), v in sorted(flat.items()):
    print(f"  c={c:<4} gamma={g2:<4} : {summ(v)}")

if ens_rows:
    print(f"\nensemble direction on KNOWN-GOOD y_b (n={len(ens_rows)}), truth = direction should be >0:")
    m = np.array([e["m"] for e in ens_rows]); sd = np.array([e["s"] for e in ens_rows])
    signs = np.array([[np.sign(x) for x in e["d"]] for e in ens_rows])
    agree = (signs == signs[:, [0]]).all(1) & (np.abs(m) > 0.1)
    print(f"  frac(mean dir > 0)             : {np.mean(m > 0):.0%}   (50% = coin flip)")
    print(f"  sign-agreement across 3 forms  : {np.mean(agree):.0%} of pairs")
    if agree.sum() and (~agree).sum():
        print(f"  frac(dir>0) | forms agree      : {np.mean(m[agree] > 0):.0%}  (n={agree.sum()})")
        print(f"  frac(dir>0) | forms disagree   : {np.mean(m[~agree] > 0):.0%}  (n={(~agree).sum()})")
    snr = m / (sd + 0.05)
    print(f"  |SNR| distribution: med={np.median(np.abs(snr)):.2f} p90={np.quantile(np.abs(snr), .9):.2f}")
print("\nexamples (gamma=1):")
for e in examples:
    print(f"  k{e['k']}t{e['t']}i{e['i']} y_a={e['y_a']!r} y_b={e['y_b']!r} "
          f"share {e['grpo']:.2f}->{e['ccr']:.2f}  (p_yb={e['p_yb']:.2f} q_yb={e['q_yb']:.2f})  "
          f"mass goes to {e['goes_to']}")
print("WS_CCR_FIELD_DONE")

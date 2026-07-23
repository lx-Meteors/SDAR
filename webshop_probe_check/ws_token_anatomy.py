"""WebShop token-position anatomy under env-gold privileged conditioning.

For every generated token position: student p (no gold) vs teacher q (gold =
true target item name+options+price+asin in context). Reports:
 1. metric census (TV/JS/KL/dH/surprisal) + what tokens each metric selects
 2. taxonomy fractions & where forks live (<think> vs <action>)
 3. action-menu probability transfer: p vs q mass on the GOLD item's action
    token at click decisions (the WebShop version of candidate-token transfer)
 4. hindsight potential Phi_t = logp(y*|prompt_t): spearman vs final score;
    B-criterion redo at anchor collisions (vs group mean score)
Run with qwen-infer python on GPU5 after collect_ws_full.py.
"""
import json, os, re, sys
import numpy as np
import torch
from collections import Counter, defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "/home/test/models/Qwen2.5-3B-Instruct"
HERE = os.path.dirname(os.path.abspath(__file__))
trajs = json.load(open(os.path.join(HERE, "ws_full_rollouts.json")))

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cuda:0")
model.eval(); DEV = model.device

def priv_text(goal):
    return ("Privileged information (invisible to the agent): the correct final purchase is "
            f"\"{goal['name']}\" (item ID {goal['asin']}), options {goal['goal_options']}, "
            f"price up to {goal['price_upper']}.\n\n")

@torch.no_grad()
def dists(user_content, response):
    msgs = [{"role": "user", "content": user_content}]
    pre = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    pre_ids = tok(pre, add_special_tokens=False, truncation=True, max_length=7000).input_ids
    out_ids = tok(response, add_special_tokens=False).input_ids[:400]
    full = torch.tensor([pre_ids + out_ids], device=DEV)
    lg = model(full).logits[0].float()
    L0 = len(pre_ids)
    lp = torch.log_softmax(lg[L0 - 1:L0 - 1 + len(out_ids), :], dim=-1)
    return lp.exp(), out_ids

@torch.no_grad()
def lp_target(context, target):
    msgs = [{"role": "user", "content": context}]
    pre = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False) + " "
    pre_ids = tok(pre, add_special_tokens=False, truncation=True, max_length=7000).input_ids
    tgt_ids = tok(target, add_special_tokens=False).input_ids
    full = torch.tensor([pre_ids + tgt_ids], device=DEV)
    lg = model(full).logits[0].float()
    L0 = len(pre_ids)
    lp = torch.log_softmax(lg[L0 - 1:L0 - 1 + len(tgt_ids), :], dim=-1)
    return float(lp[torch.arange(len(tgt_ids), device=DEV), torch.tensor(tgt_ids, device=DEV)].mean().item())

@torch.no_grad()
def lp_cont(user_content, assistant_prefix, target):
    msgs = [{"role": "user", "content": user_content}]
    pre = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False) + assistant_prefix
    pre_ids = tok(pre, add_special_tokens=False, truncation=True, max_length=7000).input_ids
    tgt_ids = tok(target, add_special_tokens=False).input_ids
    full = torch.tensor([pre_ids + tgt_ids], device=DEV)
    lg = model(full).logits[0].float()
    L0 = len(pre_ids)
    lp = torch.log_softmax(lg[L0 - 1:L0 - 1 + len(tgt_ids), :], dim=-1)
    return float(lp[torch.arange(len(tgt_ids), device=DEV), torch.tensor(tgt_ids, device=DEV)].mean().item())

def tok_set(text, cap=8):
    ids = set()
    for pre in ("", " "):
        ids.update(tok(str(text) + "", add_special_tokens=False).input_ids[:cap])
        ids.update(tok(pre + str(text), add_special_tokens=False).input_ids[:cap])
    return ids

pos_rows, transfer, phis = [], [], []
for tr in trajs:
    goal = tr["goal"]
    gold_ids = tok_set(goal["asin"].lower()) | tok_set(goal["asin"]) | tok_set("buy now")
    for w in str(goal["name"]).split()[:8]: gold_ids |= tok_set(w, cap=3)
    for v in (goal.get("goal_options") or {}).values(): gold_ids |= tok_set(v, cap=4)
    gid_t = torch.tensor(sorted(gold_ids), device=DEV)
    pv = priv_text(goal)
    for s in tr["steps"]:
        resp = s.get("response", "")
        if not resp or resp == "(forced)": continue
        p, out_ids = dists(s["prompt"], resp)
        q, _ = dists(pv + s["prompt"], resp)
        lp_, lq_ = torch.log(p + 1e-12), torch.log(q + 1e-12)
        m = 0.5 * (p + q); lm = torch.log(m + 1e-12)
        js = (0.5 * (p * (lp_ - lm)).sum(-1) + 0.5 * (q * (lq_ - lm)).sum(-1)).cpu().numpy()
        kl = (q * (lq_ - lp_)).sum(-1).cpu().numpy()
        tv = (0.5 * (q - p).abs().sum(-1)).cpu().numpy()
        Hp = -(p * lp_).sum(-1).cpu().numpy(); Hq = -(q * lq_).sum(-1).cpu().numpy()
        leak = (q - p)[:, gid_t].clamp(min=0).sum(-1).cpu().numpy()
        idx = torch.arange(len(out_ids), device=DEV)
        oi = torch.tensor(out_ids, device=DEV)
        sur = (-lp_[idx, oi]).cpu().numpy()
        pt_p, pt_i = p.topk(3, dim=-1); qt_p, qt_i = q.topk(3, dim=-1)
        pt_p, pt_i, qt_p, qt_i = [x.cpu().numpy() for x in (pt_p, pt_i, qt_p, qt_i)]
        # char offsets for region tags
        enc = tok(resp, add_special_tokens=False, return_offsets_mapping=True)
        offs = enc.offset_mapping[:len(out_ids)]
        tspan = (resp.find("<think>"), resp.find("</think>"))
        aspan = (resp.find("<action>"), resp.find("</action>"))
        def region(c):
            if aspan[0] != -1 and aspan[0] <= c < (aspan[1] if aspan[1] != -1 else 1e9): return "action"
            if tspan[0] != -1 and tspan[0] <= c < (tspan[1] if tspan[1] != -1 else 1e9): return "think"
            return "other"
        for i in range(len(out_ids)):
            pos_rows.append(dict(
                task=tr["task"], k=tr["k"], t=s["t"], i=i, reg=region(offs[i][0]),
                ctx=resp[max(0, offs[i][0] - 45):offs[i][0]].replace("\n", " "),
                y=tok.decode([out_ids[i]]), js=float(js[i]), kl=float(kl[i]),
                tv=float(tv[i]), dh=float(Hp[i] - Hq[i]), sur=float(sur[i]),
                leakfrac=float(leak[i] / max(tv[i], 1e-9)),
                ptop=[(tok.decode([pt_i[i][j]]).strip(), round(float(pt_p[i][j]), 2)) for j in range(3)],
                qtop=[(tok.decode([qt_i[i][j]]).strip(), round(float(qt_p[i][j]), 2)) for j in range(3)]))
        # action-menu transfer: logprob of the GOLD action as continuation at
        # the <action> emission point, student ctx vs privileged ctx
        gasin = goal["asin"].lower()
        if f"click[{gasin}]" in s["avail"].lower() and aspan[0] != -1:
            pre_act = resp[:aspan[0] + len("<action>")]
            tgt = f"click[{gasin}]"
            lp_g_s = lp_cont(s["prompt"], pre_act, tgt)
            lp_g_t = lp_cont(pv + s["prompt"], pre_act, tgt)
            ch = s["action"][:60]
            lp_c_s = lp_cont(s["prompt"], pre_act, ch)
            lp_c_t = lp_cont(pv + s["prompt"], pre_act, ch)
            transfer.append(dict(task=tr["task"], k=tr["k"], t=s["t"],
                                 d_gold=lp_g_t - lp_g_s, d_chosen=lp_c_t - lp_c_s,
                                 margin_s=lp_g_s - lp_c_s, margin_t=lp_g_t - lp_c_t,
                                 chosen=ch[:40], is_gold=gasin in s["action"]))
        del p, q; torch.cuda.empty_cache()
    # potential probe per step: semantic target and action-grounded target
    ystar = f"{goal['name']}, options {goal['goal_options']}"
    yact = f"click[{goal['asin'].lower()}]"
    for s in tr["steps"]:
        ph = lp_target(s["prompt"] + "\n\nThe correct final purchase for this task is:", ystar)
        pa = lp_target(s["prompt"] + "\n\nThe single best next action towards the correct purchase is:", yact)
        phis.append(dict(task=tr["task"], k=tr["k"], t=s["t"], phi=ph, phi_act=pa,
                         anchor=" ".join(s["anchor"].split())[:400], score=tr["score"]))
    print(f"traj task{tr['task']} k{tr['k']} done", flush=True)

print(f"\n===== positions={len(pos_rows)} =====")
tvv = np.array([r["tv"] for r in pos_rows])
agree = tvv < 0.1
leakm = np.array([r["leakfrac"] > 0.5 for r in pos_rows]) & ~agree
fork = ~agree & ~leakm
print(f"agree={agree.mean():.0%} leak={leakm.mean():.0%} fork={fork.mean():.0%}")
for reg in ("think", "action", "other"):
    sel = [r for r, f in zip(pos_rows, fork) if f and r["reg"] == reg]
    tot = [r for r in pos_rows if r["reg"] == reg]
    print(f"  fork in {reg}: {len(sel)} ({len(sel)/max(1,len(tot)):.0%} of region)")

print("\n----- metric top-decile overlap (Jaccard) -----")
names = ("js", "kl", "dh", "sur", "tv")
tops = {}
for n in names:
    v = np.array([r[n] for r in pos_rows])
    tops[n] = set(np.argsort(v)[-len(v) // 10:])
for a in range(len(names)):
    row = " ".join(f"{names[b]}:{len(tops[names[a]] & tops[names[b]])/len(tops[names[a]] | tops[names[b]]):.2f}"
                   for b in range(len(names)) if b > a)
    if row: print(f"  {names[a]} vs {row}")

print("\n----- what tokens does each metric select (top-decile sampled-token counts) -----")
for n in ("js", "dh", "sur"):
    c = Counter(pos_rows[i]["y"].strip() for i in tops[n]).most_common(12)
    print(f"  {n}: {c}")

print("\n----- top-10 JS fork positions (concrete) -----")
order = np.argsort([-r["js"] for r in pos_rows])
shown = 0
for i in order:
    r = pos_rows[i]
    if r["leakfrac"] > 0.5: continue
    print(f"  [{r['reg']}] t{r['t']} task{r['task']}k{r['k']} JS={r['js']:.2f} dH={r['dh']:+.2f} "
          f"...{r['ctx'][-38:]}| y={r['y']!r} p={r['ptop']} q={r['qtop']}")
    shown += 1
    if shown >= 10: break

print("\n----- leak positions: what q pushes -----")
lk = [r for r, f in zip(pos_rows, leakm) if f]
print(f"  n={len(lk)}; q-argmax counts: {Counter(r['qtop'][0][0] for r in lk).most_common(10)}")

print("\n----- action-menu probability transfer (gold item clickable) -----")
if transfer:
    dg = np.mean([x["d_gold"] for x in transfer]); dc = np.mean([x["d_chosen"] for x in transfer])
    ng = [x for x in transfer if not x["is_gold"]]
    print(f"  n={len(transfer)} decisions w/ gold on page: mean dlogp(gold action)={dg:+.2f}, "
          f"mean dlogp(chosen)={dc:+.2f}")
    if ng:
        ms = np.mean([x["margin_s"] for x in ng]); mt = np.mean([x["margin_t"] for x in ng])
        flip = np.mean([x["margin_t"] > x["margin_s"] for x in ng])
        print(f"  non-gold choices n={len(ng)}: margin(gold-chosen) student={ms:+.2f} teacher={mt:+.2f}; "
          f"teacher raises margin in {flip:.0%}")
    for x in transfer[:10]:
        print(f"    task{x['task']}k{x['k']}t{x['t']} dgold={x['d_gold']:+.2f} dchosen={x['d_chosen']:+.2f} "
              f"m_s={x['margin_s']:+.2f} m_t={x['margin_t']:+.2f} gold={x['is_gold']} act={x['chosen']!r}")
else:
    print("  no click-decisions with gold item on page")

print("\n----- hindsight potential vs outcome -----")
from scipy.stats import spearmanr
ph = np.array([r["phi"] for r in phis]); sc = np.array([r["score"] for r in phis])
pa = np.array([r["phi_act"] for r in phis])
rho, _ = spearmanr(ph, sc)
rhoa, _ = spearmanr(pa, sc)
print(f"  all steps n={len(phis)}: spearman(Phi_sem, score)={rho:.2f}  spearman(Phi_act, score)={rhoa:.2f}")
last = defaultdict(lambda: None)
for r in phis: last[(r["task"], r["k"])] = r
lp_ = [v["phi"] for v in last.values()]; ls = [v["score"] for v in last.values()]
rho2, _ = spearmanr(lp_, ls)
print(f"  last-step only n={len(lp_)}: spearman={rho2:.2f}")
groups = defaultdict(list)
for r in phis: groups[(r["task"], r["anchor"])].append(r)
colp, cola, cols, colown = [], [], [], []
ncol = 0
for g in groups.values():
    ks = {r["k"] for r in g}
    if len(ks) < 2: continue
    ncol += len(g)
    gbar = np.mean([r["score"] for r in g])
    for r in g:
        colp.append(r["phi"]); cola.append(r["phi_act"]); cols.append(gbar); colown.append(r["score"])
if colp and len(set(cols)) > 1:
    rho3, _ = spearmanr(colp, cols); rho4, _ = spearmanr(cola, cols)
    rho5, _ = spearmanr(colp, colown); rho6, _ = spearmanr(cola, colown)
    print(f"  B-criterion at collisions: n={len(colp)} states ({ncol/len(phis):.0%} cov)")
    print(f"    vs group-mean-score: Phi_sem={rho3:.2f} Phi_act={rho4:.2f}")
    print(f"    vs OWN final score:  Phi_sem={rho5:.2f} Phi_act={rho6:.2f}")
json.dump(phis, open(os.path.join(HERE, "ws_phi.json"), "w"))
print("WS_ANATOMY_DONE")

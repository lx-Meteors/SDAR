"""Hindsight-sibling (T2) teacher deep-dive on WebShop, three questions at once:

1. TOKEN ANATOMY under the EFFECTIVE teacher (q = student conditioned on best
   sibling's action sequence -- zero external labels): agree/replay-leak/fork
   census, metric comparison (JS/KL/TV/dH/surprisal) with concrete token dumps.
   The old anatomy used the inert outcome-gold teacher; this is the missing one.
2. SUPPORT-RESTRICTED PROBABILITY TRANSFER: at wrong click-decisions with the
   gold item on page, is the gold action's first-divergence token inside the
   student's top-p support?  Does restricting q to supp(p) keep the T2 margin
   lift (mechanised anti-replay instead of leak rules)?
3. LONG-RANGE CREDIT vs GiGPO: env-gold rule labels per step (EVAL ONLY);
   compare AUC/coverage of  (a) GiGPO anchor-collision group advantage,
   (b) potential telescoping delta_t = Phi_{t+1}-Phi_t with y* = sibling's
   hindsight purchase (student-only probe), (c) token-level T2 endorsement
   e_t = sum_fork JS_i * (q_i(y_i)-p_i(y_i)) raw & support-restricted.

Run: conda env qwen-infer, CUDA_VISIBLE_DEVICES=5.  Reads ws_full_rollouts.json.
"""
import json, os, re
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

best = {}
for tr in trajs:
    if tr["task"] not in best or tr["score"] > best[tr["task"]]["score"]:
        best[tr["task"]] = tr

def teacher_of(tr):
    b = best[tr["task"]]
    return b if b["score"] > tr["score"] else None

def proc_priv(btr):
    # sibling-only privilege: no env gold anywhere (asin appears only if the
    # sibling itself clicked it)
    acts = " -> ".join(s["action"] for s in btr["steps"])
    return ("Privileged information (invisible to the agent): an expert solved this exact task "
            f"(final reward {btr['score']:.2f}) with the action sequence: {acts}.\n\n")

ASIN_RE = re.compile(r"click\[(b0\w+)\]")

def purchased(tr):
    asin = None
    for s in tr["steps"]:
        m = ASIN_RE.match(s["action"].lower())
        if m: asin = m.group(1)
    return asin

# ---------- env-gold step labels (OFFLINE EVAL ORACLE ONLY) ----------
NAV = ("back to search", "< prev", "next >", "description", "features", "reviews")

def label_steps(tr):
    gold = tr["goal"]["asin"].lower()
    opts = [str(v).lower() for v in (tr["goal"].get("goal_options") or {}).values()]
    labels, cur = [], None
    steps = tr["steps"]
    for j, s in enumerate(steps):
        a = s["action"].lower(); lab = None
        nxt = steps[j + 1]["anchor"].lower() if j + 1 < len(steps) else None
        m = ASIN_RE.match(a)
        if a.startswith("search["):
            lab = (gold in nxt) if nxt is not None else None
        elif m:
            cur = m.group(1); lab = (cur == gold)
        elif a == "click[buy now]":
            lab = (cur == gold)
        elif a.startswith("click[") and a[6:-1] in NAV:
            lab = None
        elif a.startswith("click[") and any(o and o in a[6:-1] for o in opts):
            lab = (cur == gold)
        elif a == "invalid":
            lab = False
        labels.append(lab)
    return labels

# ---------- model probes ----------
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
def next_dist(user_content, assistant_prefix):
    msgs = [{"role": "user", "content": user_content}]
    pre = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False) + assistant_prefix
    pre_ids = tok(pre, add_special_tokens=False, truncation=True, max_length=7200).input_ids
    lg = model(torch.tensor([pre_ids], device=DEV)).logits[0, -1].float()
    return torch.softmax(lg, dim=-1)

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

def auc(pos, neg):
    if not pos or not neg: return float("nan")
    import itertools
    wins = sum((a > b) + 0.5 * (a == b) for a, b in itertools.product(pos, neg))
    return wins / (len(pos) * len(neg))

# =====================================================================
# PART 1: token anatomy under T2 + per-step endorsement
# =====================================================================
pos_rows, step_rows = [], {}
for tr in trajs:
    btr = teacher_of(tr)
    if btr is None: continue
    pv = proc_priv(btr)
    replay_ids = set()
    for s in btr["steps"]:
        for prefix in ("", " "):
            replay_ids.update(tok(prefix + s["action"], add_special_tokens=False).input_ids[:16])
    rid_t = torch.tensor(sorted(replay_ids), device=DEV)
    for s in tr["steps"]:
        resp = s.get("response", "")
        if not resp or resp == "(forced)": continue
        p, out_ids = dists(s["prompt"], resp)
        q, _ = dists(pv + s["prompt"], resp)
        lp_, lq_ = torch.log(p + 1e-12), torch.log(q + 1e-12)
        m = 0.5 * (p + q); lm = torch.log(m + 1e-12)
        js = (0.5 * (p * (lp_ - lm)).sum(-1) + 0.5 * (q * (lq_ - lm)).sum(-1))
        kl = (q * (lq_ - lp_)).sum(-1)
        tv = (0.5 * (q - p).abs().sum(-1))
        Hp = -(p * lp_).sum(-1); Hq = -(q * lq_).sum(-1)
        leak = (q - p)[:, rid_t].clamp(min=0).sum(-1)
        idx = torch.arange(len(out_ids), device=DEV)
        oi = torch.tensor(out_ids, device=DEV)
        sur = -lp_[idx, oi]
        # student top-p 0.9 support mask (sampled token always kept)
        sp, si = p.sort(-1, descending=True)
        keep = (sp.cumsum(-1) - sp) < 0.9
        mask = torch.zeros_like(p, dtype=torch.bool).scatter_(1, si, keep)
        mask[idx, oi] = True
        qr = q * mask
        qr = qr / qr.sum(-1, keepdim=True).clamp(min=1e-9)
        leakfrac = leak / tv.clamp(min=1e-9)
        agree = tv < 0.1
        leak_m = (~agree) & (leakfrac > 0.5)
        fork_m = (~agree) & (~leak_m)
        d_raw = q[idx, oi] - p[idx, oi]
        d_res = qr[idx, oi] - p[idx, oi]
        key = (tr["task"], tr["k"], s["t"])
        step_rows[key] = dict(
            e_raw=float((js * d_raw)[fork_m].sum().item()),
            e_res=float((js * d_res)[fork_m].sum().item()),
            n_fork=int(fork_m.sum().item()), n_tok=len(out_ids),
            js_mass=float(js[fork_m].sum().item()))
        pt_p, pt_i = p.topk(3, dim=-1); qt_p, qt_i = q.topk(3, dim=-1)
        enc = tok(resp, add_special_tokens=False, return_offsets_mapping=True)
        offs = enc.offset_mapping[:len(out_ids)]
        tspan = (resp.find("<think>"), resp.find("</think>"))
        aspan = (resp.find("<action>"), resp.find("</action>"))
        def region(c):
            if aspan[0] != -1 and aspan[0] <= c < (aspan[1] if aspan[1] != -1 else 1e9): return "action"
            if tspan[0] != -1 and tspan[0] <= c < (tspan[1] if tspan[1] != -1 else 1e9): return "think"
            return "other"
        jsn, kln, tvn, dhn = js.cpu().numpy(), kl.cpu().numpy(), tv.cpu().numpy(), (Hp - Hq).cpu().numpy()
        surn, lfn = sur.cpu().numpy(), leakfrac.cpu().numpy()
        for i in range(len(out_ids)):
            pos_rows.append(dict(
                task=tr["task"], k=tr["k"], t=s["t"], i=i, reg=region(offs[i][0]),
                ctx=resp[max(0, offs[i][0] - 45):offs[i][0]].replace("\n", " "),
                y=tok.decode([out_ids[i]]), js=float(jsn[i]), kl=float(kln[i]),
                tv=float(tvn[i]), dh=float(dhn[i]), sur=float(surn[i]),
                leakfrac=float(lfn[i]),
                ptop=[(tok.decode([pt_i[i][j]]).strip(), round(float(pt_p[i][j]), 2)) for j in range(3)],
                qtop=[(tok.decode([qt_i[i][j]]).strip(), round(float(qt_p[i][j]), 2)) for j in range(3)]))
        del p, q, qr, mask; torch.cuda.empty_cache()
    print(f"anatomy task{tr['task']} k{tr['k']} done", flush=True)

print(f"\n===== PART 1: T2 anatomy, positions={len(pos_rows)} =====")
tvv = np.array([r["tv"] for r in pos_rows])
agree = tvv < 0.1
leakm = np.array([r["leakfrac"] > 0.5 for r in pos_rows]) & ~agree
fork = ~agree & ~leakm
print(f"agree={agree.mean():.0%} replay-leak={leakm.mean():.0%} fork={fork.mean():.0%}")
for reg in ("think", "action", "other"):
    sel = [1 for r, f in zip(pos_rows, fork) if f and r["reg"] == reg]
    tot = [1 for r in pos_rows if r["reg"] == reg]
    print(f"  fork in {reg}: {len(sel)} ({len(sel)/max(1,len(tot)):.0%} of region)")

print("\n----- metric top-decile overlap (Jaccard) under T2 -----")
names = ("js", "kl", "dh", "sur", "tv")
tops = {}
for n in names:
    v = np.array([r[n] for r in pos_rows])
    tops[n] = set(np.argsort(v)[-len(v) // 10:])
for a in range(len(names)):
    row = " ".join(f"{names[b]}:{len(tops[names[a]] & tops[names[b]])/len(tops[names[a]] | tops[names[b]]):.2f}"
                   for b in range(len(names)) if b > a)
    if row: print(f"  {names[a]} vs {row}")

print("\n----- what tokens each metric selects (top-decile sampled-token counts) -----")
for n in ("js", "kl", "dh", "sur"):
    c = Counter(pos_rows[i]["y"].strip() for i in tops[n]).most_common(12)
    print(f"  {n}: {c}")

print("\n----- top-12 JS fork positions under T2 (concrete) -----")
order = np.argsort([-r["js"] for r in pos_rows])
shown = 0
for i in order:
    r = pos_rows[i]
    if r["leakfrac"] > 0.5: continue
    print(f"  [{r['reg']}] task{r['task']}k{r['k']}t{r['t']} JS={r['js']:.2f} dH={r['dh']:+.2f} "
          f"...{r['ctx'][-38:]}| y={r['y']!r} p={r['ptop']} q={r['qtop']}")
    shown += 1
    if shown >= 12: break

print("\n----- replay-leak positions: what q pushes -----")
lk = [r for r, f in zip(pos_rows, leakm) if f]
print(f"  n={len(lk)}; q-argmax: {Counter(r['qtop'][0][0] for r in lk).most_common(12)}")
ex = [r for r in lk if r["reg"] == "think"][:5]
for r in ex:
    print(f"    [think] ...{r['ctx'][-38:]}| y={r['y']!r} q={r['qtop']}")

# =====================================================================
# PART 2: support-restricted transfer at wrong click-decisions
# =====================================================================
print("\n===== PART 2: support-restricted transfer at wrong decisions =====")
dec_rows = []
for tr in trajs:
    btr = teacher_of(tr)
    if btr is None: continue
    pv = proc_priv(btr)
    gasin = tr["goal"]["asin"].lower()
    for s in tr["steps"]:
        resp = s.get("response", "")
        if not resp or resp == "(forced)": continue
        if f"click[{gasin}]" not in s["avail"].lower(): continue
        if gasin in s["action"]: continue
        a0 = resp.find("<action>")
        if a0 == -1: continue
        pre_act = resp[:a0 + len("<action>")]
        gids = tok(f"click[{gasin}]", add_special_tokens=False).input_ids
        cids = tok(s["action"][:60], add_special_tokens=False).input_ids
        j = 0
        while j < min(len(gids), len(cids)) and gids[j] == cids[j]: j += 1
        if j >= len(gids): continue
        common = tok.decode(gids[:j])
        gtok = gids[j]
        pd = next_dist(s["prompt"], pre_act + common)
        qd = next_dist(pv + s["prompt"], pre_act + common)
        spd, sid = pd.sort(-1, descending=True)
        keep = (spd.cumsum(-1) - spd) < 0.9
        supp = torch.zeros_like(pd, dtype=torch.bool).scatter_(0, sid, keep)
        in_supp = bool(supp[gtok].item())
        qres = qd * supp
        qres = qres / qres.sum().clamp(min=1e-9)
        pres = pd * supp
        pres = pres / pres.sum().clamp(min=1e-9)
        rank = int((pd > pd[gtok]).sum().item()) + 1
        dec_rows.append(dict(
            task=tr["task"], k=tr["k"], t=s["t"], gtok=tok.decode([gtok]),
            p_g=float(pd[gtok]), q_g=float(qd[gtok]), rank_p=rank, in_supp=in_supp,
            pres_g=float(pres[gtok]) if in_supp else 0.0,
            qres_g=float(qres[gtok]) if in_supp else 0.0))
        print(f"  task{tr['task']}k{tr['k']}t{s['t']} div-tok={tok.decode([gtok])!r} "
              f"rank_p={rank} in_topp0.9={in_supp} p={pd[gtok]:.3f} q={qd[gtok]:.3f} "
              f"restricted: p̃={dec_rows[-1]['pres_g']:.3f} q̃={dec_rows[-1]['qres_g']:.3f}", flush=True)
        del pd, qd; torch.cuda.empty_cache()
if dec_rows:
    ins = np.mean([r["in_supp"] for r in dec_rows])
    lift_raw = np.mean([r["q_g"] - r["p_g"] for r in dec_rows])
    raise_raw = np.mean([r["q_g"] > r["p_g"] for r in dec_rows])
    sup = [r for r in dec_rows if r["in_supp"]]
    lift_res = np.mean([r["qres_g"] - r["pres_g"] for r in sup]) if sup else float("nan")
    raise_res = np.mean([r["qres_g"] > r["pres_g"] for r in sup]) if sup else float("nan")
    print(f"\n  n={len(dec_rows)} wrong decisions (gold clickable): gold div-token in student "
          f"top-p0.9 support: {ins:.0%}")
    print(f"  raw transfer:        mean dp(gold-tok)={lift_raw:+.3f}, raised in {raise_raw:.0%}")
    print(f"  supp-restricted:     mean dp̃(gold-tok)={lift_res:+.3f}, raised in {raise_res:.0%} (n={len(sup)})")

# =====================================================================
# PART 3: long-range credit --- delta / endorsement / GiGPO on labeled steps
# =====================================================================
print("\n===== PART 3: credit signals vs env-gold step labels =====")
ystar = {ti: purchased(b) for ti, b in best.items()}
print(f"  hindsight y* per task: {ystar}  (env gold: "
      f"{ {tr['task']: tr['goal']['asin'].lower() for tr in trajs} })")

phi = {}
for tr in trajs:
    ys = ystar[tr["task"]]
    if ys is None: continue
    for s in tr["steps"]:
        phi[(tr["task"], tr["k"], s["t"])] = lp_target(
            s["prompt"] + "\n\nThe single best next action towards completing this purchase is:",
            f"click[{ys}]")
    print(f"phi task{tr['task']} k{tr['k']} done", flush=True)

# GiGPO anchor groups (exact obs match across sibling rollouts)
groups = defaultdict(list)
for tr in trajs:
    for s in tr["steps"]:
        key = (tr["task"], " ".join(s["anchor"].split()))
        groups[key].append((tr["k"], tr["score"], s["t"]))
gigpo_adv = {}
for key, mem in groups.items():
    if len({k for k, _, _ in mem}) < 2: continue
    gbar = np.mean([sc for _, sc, _ in mem])
    for k, sc, t in mem:
        gigpo_adv[(key[0], k, t)] = sc - gbar

rows = []
for tr in trajs:
    labels = label_steps(tr)
    for j, s in enumerate(tr["steps"]):
        lab = labels[j]
        if lab is None: continue
        key = (tr["task"], tr["k"], s["t"])
        nkey = (tr["task"], tr["k"], tr["steps"][j + 1]["t"]) if j + 1 < len(tr["steps"]) else None
        delta = (phi[nkey] - phi[key]) if (nkey in phi and key in phi) else None
        er = step_rows.get(key)
        rows.append(dict(task=tr["task"], k=tr["k"], t=s["t"], lab=bool(lab),
                         score=tr["score"], delta=delta,
                         e_raw=er["e_raw"] if er else None,
                         e_res=er["e_res"] if er else None,
                         gig=gigpo_adv.get(key)))

lab_g = [r for r in rows if r["lab"]]
lab_b = [r for r in rows if not r["lab"]]
print(f"\n  labeled steps: good={len(lab_g)} bad={len(lab_b)} "
      f"(of {sum(len(tr['steps']) for tr in trajs)} total)")

def report(name, key, subset=lambda r: True, note=""):
    g = [r[key] for r in lab_g if r[key] is not None and subset(r)]
    b = [r[key] for r in lab_b if r[key] is not None and subset(r)]
    cov = (len(g) + len(b)) / max(1, len([r for r in rows if subset(r)]))
    print(f"  {name:34s} AUC={auc(g, b):.2f}  n=({len(g)}g,{len(b)}b)  cov={cov:.0%} {note}")

report("delta (hindsight potential)", "delta")
report("delta, t>=1 only", "delta", lambda r: r["t"] >= 1)
report("e_raw (T2 endorsement, fork JS)", "e_raw")
report("e_res (supp-restricted endorse)", "e_res")
report("GiGPO group adv (covered only)", "gig")
report("GiGPO adv, t>=1 only", "gig", lambda r: r["t"] >= 1)
gg = [r for r in rows if r["gig"] is not None]
sa = np.mean([(r["gig"] > 0) == r["lab"] for r in gg if r["gig"] != 0]) if gg else float("nan")
nz = [r for r in gg if r["gig"] == 0]
print(f"  GiGPO: coverage {len(gg)}/{len(rows)} labeled steps, sign-acc(nonzero adv)={sa:.0%}, "
      f"zero-adv (abstain)={len(nz)}")

print("\n  per-task delta AUC (teacher quality: t0 best=1.0, t1 best=0.02, t2 best=0.33):")
for ti in sorted({r["task"] for r in rows}):
    g = [r["delta"] for r in lab_g if r["task"] == ti and r["delta"] is not None]
    b = [r["delta"] for r in lab_b if r["task"] == ti and r["delta"] is not None]
    print(f"    task{ti}: AUC={auc(g, b):.2f} ({len(g)}g,{len(b)}b)")

print("\n  per-task e_res AUC:")
for ti in sorted({r["task"] for r in rows}):
    g = [r["e_res"] for r in lab_g if r["task"] == ti and r["e_res"] is not None]
    b = [r["e_res"] for r in lab_b if r["task"] == ti and r["e_res"] is not None]
    print(f"    task{ti}: AUC={auc(g, b):.2f} ({len(g)}g,{len(b)}b)")

comb = [r for r in rows if r["delta"] is not None and r["e_res"] is not None]
if comb:
    from scipy.stats import spearmanr
    rho, _ = spearmanr([r["delta"] for r in comb], [r["e_res"] for r in comb])
    print(f"\n  spearman(delta, e_res) on {len(comb)} steps = {rho:.2f} (complementarity check)")
    zg = [r["delta"] + r["e_res"] for r in comb if r["lab"]]
    zb = [r["delta"] + r["e_res"] for r in comb if not r["lab"]]
    # naive z-sum after per-signal standardization
    dv = np.array([r["delta"] for r in comb]); ev = np.array([r["e_res"] for r in comb])
    zs = (dv - dv.mean()) / (dv.std() + 1e-9) + (ev - ev.mean()) / (ev.std() + 1e-9)
    zg = [z for z, r in zip(zs, comb) if r["lab"]]; zb = [z for z, r in zip(zs, comb) if not r["lab"]]
    print(f"  z(delta)+z(e_res) combined AUC = {auc(zg, zb):.2f}")

json.dump(dict(pos_n=len(pos_rows), rows=rows, dec=dec_rows),
          open(os.path.join(HERE, "ws_hindsight_anatomy.json"), "w"), default=float)
print("\nWS_HINDSIGHT_ANATOMY_DONE")

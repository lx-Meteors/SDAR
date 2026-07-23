"""Skill-privilege probe (fresh-start plan, 2026-07-15): can a SDAR-style
strategy-skills prefix serve as a SECOND privileged teacher axis (q_skill)
next to the outcome certificate (q_cert)?

Checks (numbering follows research_decisions.md 重启 section):
  C1 localization quality: does TV(p, q_skill) concentrate on decision sites
     (fratricide pair sites) vs quiet background?  Self-gating ratio vs the
     cert-teacher benchmark on the SAME positions.
  C2 arbitrariness detection: at solution forks (own/sib terminal purchases
     differ, both succeed) q_cert should be biased to own token (share>0.5)
     while q_skill should be balanced (share~0.5).  Wording forks as control.
  C3 (WebShop step-ranking) do lam*decisiveness step signals rank bad steps
     above good ones on the labeled set (ws_step_rows.json)?  Reference: the
     stored hindsight-delta signal (AUC ~0.80).
  C4 menu-restricted vs full-vocab TV: how much q mass falls outside the
     student top-20 menu (leak proxy), and do full-TV gate openings survive
     the menu-restricted gate?

Run: qwen-infer env, CUDA_VISIBLE_DEVICES=5, ~15 min.
"""
import json, os, re, sys
import numpy as np
import torch
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "/data1/test/yyy/BEACON/checkpoints/verl_agent_webshop/beacon_qwen2.5_1.5b/step150_hf"
HERE = os.path.dirname(os.path.abspath(__file__))
SKILLS_MD = "/data1/test/yyy/SDAR/skills/webshop/general_skills.md"

skills_text = open(SKILLS_MD).read().strip()
PRIV_SKILL = ("Privileged strategy guide (invisible to the agent):\n"
              + skills_text + "\n\n")

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cuda:0")
model.eval(); DEV = model.device
print(f"skill prefix tokens: {len(tok(PRIV_SKILL, add_special_tokens=False).input_ids)}", flush=True)

@torch.no_grad()
def dists(user_content, response):
    msgs = [{"role": "user", "content": user_content}]
    pre = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    pre_ids = tok(pre, add_special_tokens=False, truncation=True, max_length=7000).input_ids
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
        a = (s.get("action") or "").lower()
        m = ASIN_RE.match(a)
        if m: asin = m.group(1); opts = []
        elif a.startswith("click[") and a[6:-1] not in NAV and not a[6:-1].startswith("b0"):
            opts.append(a[6:-1])
    return asin, opts

def priv_outcome(asin, opts, score):
    return ("Privileged information (invisible to the agent): this task was solved "
            f"(final reward {score:.2f}); the successful final purchase was item ID "
            f"{asin} with selected options {opts}.\n\n")

def tv_full(a, b):
    return (0.5 * (a - b).abs().sum(-1)).cpu().numpy()

def menu_metrics(p, q, k=5, kleak=20):
    """per-position: menu TV (student top-k restricted+renorm), leak mass of q
    outside student top-kleak."""
    tv_m, leak = [], []
    tkv, tki = p.topk(kleak, dim=-1)
    for i in range(p.shape[0]):
        ids = tki[i, :k]
        pm = p[i, ids]; qm = q[i, ids]
        pm = pm / pm.sum().clamp_min(1e-9); qm = qm / qm.sum().clamp_min(1e-9)
        tv_m.append(float(0.5 * (pm - qm).abs().sum()))
        leak.append(float(1.0 - q[i, tki[i]].sum()))
    return np.array(tv_m), np.array(leak)

# ---------------- phase 1: rollouts4 pair sites (C1/C2/C4) ----------------
trajs = []
for t in json.load(open(os.path.join(HERE, "ws_ckpt_rollouts4.json"))):
    t["gk"] = t["task"]; trajs.append(t)
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
        q_cert, _ = dists(priv_outcome(*sol_own, tr["score"]) + s["prompt"], s["response"])
        q_skill, _ = dists(PRIV_SKILL + s["prompt"], s["response"])
        tvc, tvs = tv_full(q_cert, p), tv_full(q_skill, p)
        tvc_m, leak_c = menu_metrics(p, q_cert)
        tvs_m, leak_s = menu_metrics(p, q_skill)
        sites = []
        for (i, y_a, y_b) in plist:
            qc_a, qc_b = float(q_cert[i, y_a]), float(q_cert[i, y_b])
            qs_a, qs_b = float(q_skill[i, y_a]), float(q_skill[i, y_b])
            p_a, p_b = float(p[i, y_a]), float(p[i, y_b])
            sites.append(dict(i=i, y_a=y_a, y_b=y_b, p_a=p_a, p_b=p_b,
                              cert_share=qc_a / max(qc_a + qc_b, 1e-9),
                              skill_share=qs_a / max(qs_a + qs_b, 1e-9),
                              p_share=p_a / max(p_a + p_b, 1e-9)))
        step_recs.append(dict(gk=gk, k=k, t=t_, tv_cert=tvc, tv_skill=tvs,
                              tvm_cert=tvc_m, tvm_skill=tvs_m,
                              leak_cert=leak_c, leak_skill=leak_s,
                              out_ids=out_ids, sites=sites,
                              same_sol=(sol_own == sol_sib)))
        del p, q_cert, q_skill; torch.cuda.empty_cache()
    print(f"group {gk}: cached", flush=True)

def lam_of(recs, key):
    gm = {}
    for gk in {r["gk"] for r in recs}:
        gm[gk] = max(float(np.mean(np.concatenate([r[key] for r in recs if r["gk"] == gk]))), 1e-4)
    return {id(r): r[key] / (r[key] + gm[r["gk"]]) for r in recs}, gm

lam_c, gm_c = lam_of(step_recs, "tv_cert")
lam_s, gm_s = lam_of(step_recs, "tv_skill")
lam_cm, _ = lam_of(step_recs, "tvm_cert")
lam_sm, _ = lam_of(step_recs, "tvm_skill")

print(f"\n=== C1 localization (mean TV, {len(step_recs)} steps) ===")
print(f"group mean TV  cert={np.mean(list(gm_c.values())):.4f}  skill={np.mean(list(gm_s.values())):.4f}")
site_tv_c, site_tv_s, bg_tv_c, bg_tv_s = [], [], [], []
site_lam_c, site_lam_s, bg_lam_c, bg_lam_s = [], [], [], []
site_lam_cm, site_lam_sm = [], []
shares = dict(sol=dict(c=[], s=[], p=[]), word=dict(c=[], s=[], p=[]))
site_leak_c, site_leak_s = [], []
strip_base, strip_cert, strip_skill = [], [], []
for r in step_recs:
    idx_sites = [st["i"] for st in r["sites"]]
    mask = np.zeros(len(r["tv_cert"]), dtype=bool); mask[idx_sites] = True
    site_tv_c.extend(r["tv_cert"][mask]); site_tv_s.extend(r["tv_skill"][mask])
    bg_tv_c.extend(r["tv_cert"][~mask]); bg_tv_s.extend(r["tv_skill"][~mask])
    lc, ls = lam_c[id(r)], lam_s[id(r)]
    lcm, lsm = lam_cm[id(r)], lam_sm[id(r)]
    site_lam_c.extend(lc[mask]); site_lam_s.extend(ls[mask])
    bg_lam_c.extend(lc[~mask]); bg_lam_s.extend(ls[~mask])
    site_lam_cm.extend(lcm[mask]); site_lam_sm.extend(lsm[mask])
    site_leak_c.extend(r["leak_cert"][mask]); site_leak_s.extend(r["leak_skill"][mask])
    kind = "word" if r["same_sol"] else "sol"
    for st in r["sites"]:
        shares[kind]["c"].append(st["cert_share"])
        shares[kind]["s"].append(st["skill_share"])
        shares[kind]["p"].append(st["p_share"])
        base = st["p_b"] / max(1 - st["p_a"], 1e-9)
        strip_base.append(base)
        strip_cert.append((1 - lc[st["i"]]) * base)
        strip_skill.append((1 - ls[st["i"]]) * base)
sg_c = np.mean(site_tv_c) / max(np.mean(bg_tv_c), 1e-9)
sg_s = np.mean(site_tv_s) / max(np.mean(bg_tv_s), 1e-9)
print(f"site TV      cert={np.mean(site_tv_c):.4f}  skill={np.mean(site_tv_s):.4f}")
print(f"bg TV        cert={np.mean(bg_tv_c):.4f}  skill={np.mean(bg_tv_s):.4f}")
print(f"self-gating  cert={sg_c:.1f}x  skill={sg_s:.1f}x")
print(f"site lam     cert={np.mean(site_lam_c):.3f}  skill={np.mean(site_lam_s):.3f}"
      f"   (menu-gate: cert={np.mean(site_lam_cm):.3f} skill={np.mean(site_lam_sm):.3f})")
print(f"bg openFrac(lam>0.5)  cert={np.mean(np.array(bg_lam_c) > 0.5):.3f}"
      f"  skill={np.mean(np.array(bg_lam_s) > 0.5):.3f}")

print(f"\n=== C2 arbitrariness: share of q-mass on OWN token within (y_a,y_b) ===")
for kind, lab in (("sol", "solution forks"), ("word", "wording forks")):
    if shares[kind]["c"]:
        n = len(shares[kind]["c"])
        print(f"{lab} (n={n}): p={np.mean(shares[kind]['p']):.3f}  "
              f"cert={np.mean(shares[kind]['c']):.3f}  skill={np.mean(shares[kind]['s']):.3f}"
              f"  | balance(|share-0.5|): cert={np.mean(np.abs(np.array(shares[kind]['c'])-0.5)):.3f}"
              f" skill={np.mean(np.abs(np.array(shares[kind]['s'])-0.5)):.3f}")
print(f"\nfratricide strip via A*(1-lam) (GRPO base {np.mean(strip_base):.3f}/{np.median(strip_base):.3f}):")
print(f"  cert-gate:  {np.mean(strip_cert):.3f}/{np.median(strip_cert):.3f}")
print(f"  skill-gate: {np.mean(strip_skill):.3f}/{np.median(strip_skill):.3f}")

print(f"\n=== C4 menu vs full-vocab gate ===")
print(f"q mass outside student top-20 @sites: cert={np.mean(site_leak_c):.3f}  skill={np.mean(site_leak_s):.3f}")
all_lam_s = np.concatenate([lam_s[id(r)] for r in step_recs])
all_lam_sm = np.concatenate([lam_sm[id(r)] for r in step_recs])
hi = all_lam_s > 0.5
if hi.any():
    print(f"skill full-TV openings (lam>0.5, n={hi.sum()}): {np.mean(all_lam_sm[hi] < 0.5):.1%} "
          f"closed under menu gate (style/leak suspects)")
# qualitative: top skill-TV tokens
print("\ntop-5 skill-TV positions (token | tv_skill | tv_cert):")
flat = []
for r in step_recs:
    for i in range(len(r["tv_skill"])):
        flat.append((float(r["tv_skill"][i]), float(r["tv_cert"][i]), r["out_ids"][i]))
flat.sort(reverse=True)
for tv_s, tv_c, tid in flat[:5]:
    print(f"  {tok.decode([tid])!r} | {tv_s:.3f} | {tv_c:.3f}")

# ---------------- phase 2: labeled steps (C3) ----------------
print("\n=== C3 step ranking on labeled set ===", flush=True)
full = json.load(open(os.path.join(HERE, "full_rollouts.json")))
rows = json.load(open(os.path.join(HERE, "ws_step_rows.json")))
by_task = defaultdict(list)
for t in full: by_task[t["task"]].append(t)
cert_of_task = {tk: priv_outcome(*terminal_solution(max(g, key=lambda x: x["score"])),
                                 max(x["score"] for x in g))
                for tk, g in by_task.items()}
by_key = {}
for t in full:
    for s in t["steps"]:
        by_key[(t["task"], t["k"], s["t"])] = s

sigs = defaultdict(list); labels = []; deltas = []
task_tv = defaultdict(lambda: defaultdict(list))
cache = {}
for r in rows:
    key = (r["task"], r["k"], r["t"])
    s = by_key.get(key)
    if s is None or not s.get("response"): continue
    p, out_ids = dists(s["prompt"], s["response"])
    q_c, _ = dists(cert_of_task[r["task"]] + s["prompt"], s["response"])
    q_s, _ = dists(PRIV_SKILL + s["prompt"], s["response"])
    tvc, tvs = tv_full(q_c, p), tv_full(q_s, p)
    dc, _ = menu_metrics(p, q_c); ds, _ = menu_metrics(p, q_s)
    cache[key] = (tvc, tvs, dc, ds)
    task_tv[r["task"]]["c"].append(tvc); task_tv[r["task"]]["s"].append(tvs)
    del p, q_c, q_s; torch.cuda.empty_cache()
gm2 = {tk: dict(c=max(float(np.mean(np.concatenate(v["c"]))), 1e-4),
                s=max(float(np.mean(np.concatenate(v["s"]))), 1e-4))
       for tk, v in task_tv.items()}
for r in rows:
    key = (r["task"], r["k"], r["t"])
    if key not in cache: continue
    tvc, tvs, dc, ds = cache[key]
    lamc = tvc / (tvc + gm2[r["task"]]["c"]); lams = tvs / (tvs + gm2[r["task"]]["s"])
    sigs["cert lam*d mean"].append(float(np.mean(lamc * dc)))
    sigs["cert lam*d max"].append(float(np.max(lamc * dc)))
    sigs["skill lam*d mean"].append(float(np.mean(lams * ds)))
    sigs["skill lam*d max"].append(float(np.max(lams * ds)))
    sigs["cert TV mean"].append(float(np.mean(tvc)))
    sigs["skill TV mean"].append(float(np.mean(tvs)))
    labels.append(0 if r["hit"] else 1)  # 1 = bad step
    deltas.append(-r["delta"])           # low delta = blame

def auc(sig, lab):
    sig, lab = np.array(sig), np.array(lab)
    pos, neg = sig[lab == 1], sig[lab == 0]
    if len(pos) == 0 or len(neg) == 0: return float("nan")
    return float(np.mean([np.mean((pos_i > neg).astype(float) + 0.5 * (pos_i == neg))
                          for pos_i in pos]))

lab = np.array(labels)
print(f"labeled steps used: {len(lab)} (bad={lab.sum()}, good={(1-lab).sum()})")
print(f"reference -delta (hindsight δ): AUC(bad>good)={auc(deltas, labels):.3f}")
for name, v in sigs.items():
    print(f"{name:18s}: AUC(bad>good)={auc(v, labels):.3f}")
print("WS_SKILL_PROBE_DONE")

"""Real-gradient A/B: apply K optimizer steps with GRPO vs v5 counter-mass on
the SAME success-side rollouts, then compare (1) p(y_a) growth (own learning),
(2) p(y_b) at fratricide sites (sibling protection), (3) entropy at gated
positions.  Saves both updated models for behavioral rollout comparison.

Surrogate loss with explicit counter-mass allocation omega:
  L = -A * sum_i (1-pi_y).detach() * ( z_y - sum_{v!=y} omega_v.detach() * z_v )
  d L/d z_y = -A(1-pi_y)          (same as standard PG)
  d L/d z_v = +A(1-pi_y)*omega_v  (GRPO: omega=pi_v/(1-pi_y); v5: flattened)
Run: qwen-infer env, GPU5.  Usage: python ws_v5_trainstep.py [grpo|v5]
"""
import json, os, re, sys
import numpy as np
import torch
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer

VARIANT = sys.argv[1] if len(sys.argv) > 1 else "v5"
MODEL = "/data1/test/yyy/BEACON/checkpoints/verl_agent_webshop/beacon_qwen2.5_1.5b/step150_hf"
HERE = os.path.dirname(os.path.abspath(__file__))
FILES = ["ws_ckpt_rollouts2.json", "ws_ckpt_rollouts3.json"]
K_STEPS, LR, C_GATE = 3, 2e-5, 0.03
KEEP = {("ws_ckpt_rollouts3.json", 7), ("ws_ckpt_rollouts3.json", 11)}
SAVE = os.environ.get("SAVE", "0") == "1"
OUT_DIR = os.path.join("/data1/test/yyy/BEACON/checkpoints/verl_agent_webshop",
                       f"step150_{VARIANT}{K_STEPS}")

trajs = []
for fn in FILES:
    for t in json.load(open(os.path.join(HERE, fn))):
        t["gk"] = (fn, t["task"])
        trajs.append(t)

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cuda:0")
DEV = model.device

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

def encode(user_content, response):
    msgs = [{"role": "user", "content": user_content}]
    pre = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    pre_ids = tok(pre, add_special_tokens=False, truncation=True, max_length=7000).input_ids
    out_ids = tok(response, add_special_tokens=False).input_ids[:400]
    return pre_ids, out_ids

# ---- build the training set: all success-side steps of variance groups ----
groups_by_task = defaultdict(list)
for t in trajs:
    if t["gk"] in KEEP:
        groups_by_task[t["gk"]].append(t)
train_items = []   # (traj, step, adv, best_sibling)
group_of = {}
for gk, gtr in sorted(groups_by_task.items()):
    scores = [t["score"] for t in gtr]
    gmean = float(np.mean(scores))
    for t in gtr: t["adv"] = t["score"] - gmean
    if len(set(scores)) < 2: continue
    for t in gtr:
        if t["adv"] <= 0: continue
        best = max([o for o in gtr if o["k"] != t["k"]], key=lambda o: o["score"])
        for s in t["steps"]:
            resp = s.get("response", "")
            if not resp or resp == "(forced)": continue
            train_items.append((t, s, t["adv"], best))
        group_of[(gk, t["k"])] = gtr
print(f"variant={VARIANT}  success-side sequences: {len(train_items)}", flush=True)

# ---- fratricide pair sites for metrics (same defn as ws_ccr_v5_fresh) ----
@torch.no_grad()
def p_of(step_prompt, resp):
    pre_ids, out_ids = encode(step_prompt, resp)
    full = torch.tensor([pre_ids + out_ids], device=DEV)
    lg = model(full).logits[0].float()
    L0 = len(pre_ids)
    return torch.softmax(lg[L0 - 1:L0 - 1 + len(out_ids), :], dim=-1), out_ids

pair_sites = []   # (traj, step, i, y_a, y_b)
model.eval()
for gk, gtr in sorted(groups_by_task.items()):
    scores = [t["score"] for t in gtr]
    if len(set(scores)) < 2 or len([t for t in gtr if t["adv"] > 0]) < 2: continue
    menu_rows = []
    for tr in gtr:
        if tr["adv"] <= 0: continue
        for s in tr["steps"]:
            resp = s.get("response", "")
            if not resp or resp == "(forced)": continue
            p, out_ids = p_of(s["prompt"], resp)
            t5v, t5i = p.topk(5, dim=-1)
            t5i = t5i.cpu().numpy()
            idx = torch.arange(len(out_ids), device=DEV)
            py = p[idx, torch.tensor(out_ids, device=DEV)].cpu().numpy()
            for i in range(len(out_ids)):
                menu_rows.append(dict(tr=tr, s=s, i=i, y=int(out_ids[i]), p_y=float(py[i]),
                                      menu=tuple(sorted(int(x) for x in t5i[i]))))
            del p; torch.cuda.empty_cache()
    mg = defaultdict(list)
    for r in menu_rows: mg[r["menu"]].append(r)
    seen = set()
    for mem in mg.values():
        if len({m["tr"]["k"] for m in mem}) < 2 or len({m["y"] for m in mem}) < 2: continue
        for a in mem:
            if a["p_y"] > 0.95: continue
            for b in mem:
                if b["tr"]["k"] == a["tr"]["k"] or b["y"] == a["y"]: continue
                key = (gk, a["tr"]["k"], a["s"]["t"], a["i"], b["y"])
                if key in seen: continue
                seen.add(key)
                pair_sites.append((a["tr"], a["s"], a["i"], a["y"], b["y"]))
print(f"pair sites: {len(pair_sites)}", flush=True)

@torch.no_grad()
def pair_metrics(tag):
    model.eval()
    pa, pb, ent = [], [], []
    cache = {}
    for (tr, s, i, y_a, y_b) in pair_sites:
        ck = (id(s))
        if ck not in cache:
            cache[ck] = p_of(s["prompt"], s["response"])[0]
        p = cache[ck]
        pa.append(float(p[i, y_a])); pb.append(float(p[i, y_b]))
        ent.append(float(-(p[i] * torch.log(p[i] + 1e-12)).sum()))
    print(f"  [{tag}] p(y_a)={np.mean(pa):.4f}  p(y_b)={np.mean(pb):.4f}  "
          f"H@sites={np.mean(ent):.3f}", flush=True)
    return np.mean(pa), np.mean(pb), np.mean(ent)

m0 = pair_metrics("before")

# ---- K optimizer steps ----
opt = torch.optim.Adam(model.parameters(), lr=LR)
for ep in range(K_STEPS):
    model.train()
    opt.zero_grad()
    tot = 0.0
    for (tr, s, adv, best) in train_items:
        pre_ids, out_ids = encode(s["prompt"], s["response"])
        full = torch.tensor([pre_ids + out_ids], device=DEV)
        lg = model(full).logits[0].float()
        L0 = len(pre_ids)
        z = lg[L0 - 1:L0 - 1 + len(out_ids), :]          # logits at response pos
        y = torch.tensor(out_ids, device=DEV)
        logp = torch.log_softmax(z, dim=-1)
        p = logp.exp()
        py = p.gather(1, y[:, None]).squeeze(1)
        if VARIANT == "grpo":
            loss = -(adv * logp.gather(1, y[:, None]).squeeze(1)).mean()
        else:
            with torch.no_grad():
                pv = term_priv(best)
                pre2, _ = encode(pv + s["prompt"], s["response"])
                full2 = torch.tensor([pre2 + out_ids], device=DEV)
                lg2 = model(full2).logits[0].float()
                q = torch.softmax(lg2[len(pre2) - 1:len(pre2) - 1 + len(out_ids), :], dim=-1)
                tv = 0.5 * (q - p.detach()).abs().sum(-1)
                lam = tv / (tv + C_GATE)                  # single-parameter gate
                lw = (1 - lam)[:, None] * torch.log(p.detach() + 1e-12)
                lw.scatter_(1, y[:, None], -1e30)         # exclude sampled token
                omega = torch.softmax(lw, dim=-1)
            s_det = (1 - py).detach()
            zy = z.gather(1, y[:, None]).squeeze(1)
            zc = (omega * z).sum(-1)
            loss = -(adv * s_det * (zy - zc)).mean()
        (loss / len(train_items)).backward()
        tot += float(loss)
        del lg, z, logp, p
        torch.cuda.empty_cache()
    opt.step()
    print(f"epoch {ep}: mean surrogate loss {tot / len(train_items):.4f}", flush=True)
    pair_metrics(f"after ep{ep}")

m1 = pair_metrics("final")
print(f"\nRESULT {VARIANT}: p(y_a) {m0[0]:.4f}->{m1[0]:.4f}  "
      f"p(y_b) {m0[1]:.4f}->{m1[1]:.4f}  H {m0[2]:.3f}->{m1[2]:.3f}", flush=True)

if SAVE:
    os.makedirs(OUT_DIR, exist_ok=True)
    model.save_pretrained(OUT_DIR)
    tok.save_pretrained(OUT_DIR)
    print(f"saved -> {OUT_DIR}")
print("WS_V5_TRAINSTEP_DONE")

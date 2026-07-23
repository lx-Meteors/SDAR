"""Causal mini-test of counter-mass ROUTING:  d = A(1-p_y)(e_y - omega),
omega = student menu p~ (== exact GRPO gradient)  vs  teacher menu q~ (route).

Surrogate (identical structure for both variants; only omega differs):
    L = -A * (1-p_y).detach() * ( z_y - sum_{v!=y} omega_v.detach() * z_v )
    dL/dz_y = -A(1-p_y)     dL/dz_v = +A(1-p_y)*omega_v
With omega = p_v/(1-p_y) this reproduces -A log pi(y) gradients exactly.

Teacher = same checkpoint + skill prefix (matches training pipeline / probes).
Data = variance groups from cached ckpt rollouts (A = z-scored group score).
K=3 Adam steps, save model for behavioral rollout comparison.

Run: qwen-infer env, GPU4.   python ws_route_trainstep.py [grpo|route]
"""
import json, os, sys
import numpy as np
import torch
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer

VARIANT = sys.argv[1] if len(sys.argv) > 1 else "route"
assert VARIANT in ("grpo", "route")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "4")

MODEL = "/data1/test/yyy/BEACON/checkpoints/verl_agent_webshop/beacon_qwen2.5_1.5b/step150_hf"
HERE = os.path.dirname(os.path.abspath(__file__))
SDAR = os.path.dirname(HERE)
SKILLS_DIR = os.path.join(SDAR, "skills", "webshop")
FILES = ["ws_ckpt_rollouts2.json", "ws_ckpt_rollouts3.json", "ws_ckpt_rollouts4.json"]
K_STEPS = int(os.environ.get("K_STEPS", "3"))
LR = float(os.environ.get("LR", "2e-5"))
TOPK = 32
TAG = os.environ.get("TAG", "")
TEACH_CACHE = os.path.join(HERE, "ws_route_teacher_cache.pt")
OUT_DIR = (f"/data1/test/yyy/BEACON/checkpoints/verl_agent_webshop/"
           f"step150_route_{VARIANT}{K_STEPS}{TAG}")

# ---- inline skill loader (same as ds_ratio_kl.py) ----
_MAP = json.load(open(os.path.join(SKILLS_DIR, "skill_mapping.json")))
_CONTENT = {n: open(os.path.join(SKILLS_DIR, f)).read().strip()
            for n, f in _MAP["skill_files"].items()}

def skill_for_prompt(prompt_text):
    parts = [_CONTENT.get("general_skills", "")]
    tl = prompt_text.lower()
    for tt, kws in _MAP.get("task_keywords", {}).items():
        if kws and any(kw in tl for kw in kws):
            mapped = _MAP["task_to_skill"].get(tt)
            if mapped and mapped in _CONTENT:
                parts.append(_CONTENT[mapped])
    return "\n\n".join(parts)

# ---- data: variance groups, z-scored advantages ----
groups = defaultdict(list)
for fn in FILES:
    for t in json.load(open(os.path.join(HERE, fn))):
        groups[(fn, t["task"])].append(t)

train_items = []          # (gk, k, step_dict, adv)
for gk, gtr in sorted(groups.items()):
    scores = np.array([t["score"] for t in gtr], dtype=np.float64)
    if scores.std() < 1e-6:
        continue
    for t in gtr:
        adv = float((t["score"] - scores.mean()) / (scores.std() + 1e-6))
        for s in t["steps"]:
            resp = s.get("response", "")
            if not resp or resp == "(forced)":
                continue
            train_items.append((gk, t["k"], s, adv))
print(f"variant={VARIANT}  sequences={len(train_items)} "
      f"from {len(set(i[0] for i in train_items))} variance groups", flush=True)

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                             device_map="cuda:0")
DEV = model.device

def encode(user_content, response):
    pre = tok.apply_chat_template([{"role": "user", "content": user_content}],
                                  add_generation_prompt=True, tokenize=False)
    pre_ids = tok(pre, add_special_tokens=False, truncation=True, max_length=7000).input_ids
    out_ids = tok(response, add_special_tokens=False).input_ids[:400]
    return pre_ids, out_ids

# ---- teacher menu cache: top-K of q with y masked, renormalized ----
def item_key(gk, k, s):
    return f"{gk[0]}|{gk[1]}|{k}|{s['t']}"

teach = {}
if VARIANT == "route":
    if os.path.exists(TEACH_CACHE):
        teach = torch.load(TEACH_CACHE)
        print(f"teacher cache loaded: {len(teach)}", flush=True)
    else:
        model.eval()
        with torch.no_grad():
            for n, (gk, k, s, adv) in enumerate(train_items):
                teacher_user = ("[Privileged Skill Information]\n"
                                + skill_for_prompt(s["prompt"]) + "\n\n" + s["prompt"])
                pre_ids, out_ids = encode(teacher_user, s["response"])
                full = torch.tensor([pre_ids + out_ids], device=DEV)
                lg = model(full).logits[0].float()
                L0 = len(pre_ids)
                q = torch.softmax(lg[L0 - 1:L0 - 1 + len(out_ids), :], dim=-1)
                y = torch.tensor(out_ids, device=DEV)
                q.scatter_(1, y[:, None], 0.0)                       # drop target
                vals, ids = q.topk(TOPK, dim=-1)
                vals = vals / vals.sum(-1, keepdim=True)             # renormalize menu
                teach[item_key(gk, k, s)] = (ids.cpu(), vals.cpu())
                if n % 20 == 0:
                    print(f"  teacher {n}/{len(train_items)}", flush=True)
                del lg, q
        torch.save(teach, TEACH_CACHE)
        print(f"teacher cache saved: {len(teach)}", flush=True)

# ---- K optimizer steps ----
opt = torch.optim.Adam(model.parameters(), lr=LR)
for ep in range(K_STEPS):
    model.train()
    opt.zero_grad()
    tot = 0.0
    for (gk, k, s, adv) in train_items:
        pre_ids, out_ids = encode(s["prompt"], s["response"])
        full = torch.tensor([pre_ids + out_ids], device=DEV)
        lg = model(full).logits[0].float()
        L0 = len(pre_ids)
        z = lg[L0 - 1:L0 - 1 + len(out_ids), :]
        y = torch.tensor(out_ids, device=DEV)
        p = torch.softmax(z, dim=-1)
        py = p.gather(1, y[:, None]).squeeze(1)
        zy = z.gather(1, y[:, None]).squeeze(1)
        if VARIANT == "grpo":
            with torch.no_grad():
                pd = p.detach()
                pyd = py.detach().clamp(max=1 - 1e-4)
            zc = ((pd * z).sum(-1) - pyd * zy) / (1 - pyd)           # E_{p~}[z]
        else:
            ids, vals = teach[item_key(gk, k, s)]
            ids = ids.to(DEV)
            vals = vals.to(DEV).float()
            zc = (vals * z.gather(1, ids)).sum(-1)                   # E_{q~}[z]
        loss = -(adv * (1 - py).detach() * (zy - zc)).mean()
        (loss / len(train_items)).backward()
        tot += float(loss)
        del lg, z, p
    opt.step()
    print(f"epoch {ep}: mean surrogate {tot / len(train_items):.4f}", flush=True)

os.makedirs(OUT_DIR, exist_ok=True)
model.save_pretrained(OUT_DIR)
tok.save_pretrained(OUT_DIR)
print(f"saved -> {OUT_DIR}")
print("WS_ROUTE_TRAINSTEP_DONE")

"""Degenerate-teacher test for the gate calibration.

Give one group a NULL privilege (no task information) -> all TV should be
noise.  Correct gate behavior: stay closed (lambda ~ 0, fall back to GRPO).
Compare false-open rates under:
  (a) pure group scale + floor:  lam = TV/(TV + max(m_g, 0.005))
  (b) equal-weight blend:        lam = TV/(TV + (m_g + m_bar)/2)
where m_bar = global running mean TV from informative groups (0.0386 measured).
Also verify (b) leaves the informative group's protection unchanged.
Run: qwen-infer env, GPU5, ~1 min.
"""
import json, os
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "/data1/test/yyy/BEACON/checkpoints/verl_agent_webshop/beacon_qwen2.5_1.5b/step150_hf"
HERE = os.path.dirname(os.path.abspath(__file__))
M_BAR = 0.0386
NULL_PRIV = ("Privileged information (invisible to the agent): no additional "
             "information is available for this task.\n\n")

trajs = [t for t in json.load(open(os.path.join(HERE, "ws_ckpt_rollouts3.json")))
         if t["task"] == 7]
gmean = float(np.mean([t["score"] for t in trajs]))
for t in trajs: t["adv"] = t["score"] - gmean

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

tvs = []
for tr in trajs:
    if tr["adv"] <= 0: continue
    for s in tr["steps"]:
        resp = s.get("response", "")
        if not resp or resp == "(forced)": continue
        p, out_ids = dists(s["prompt"], resp)
        q, _ = dists(NULL_PRIV + s["prompt"], resp)
        tvs.extend((0.5 * (q - p).abs().sum(-1)).cpu().numpy().tolist())
        del p, q; torch.cuda.empty_cache()

tvs = np.array(tvs)
m_g = float(np.mean(tvs))
print(f"NULL-teacher group: positions={len(tvs)}  meanTV={m_g:.4f}  "
      f"medTV={np.median(tvs):.4f}  p90={np.quantile(tvs, .9):.4f}")

for name, c in (("pure group + floor.005", max(m_g, 0.005)),
                ("blend (m_g+m_bar)/2   ", (m_g + M_BAR) / 2)):
    lam = tvs / (tvs + c)
    print(f"  {name}: c={c:.4f}  mean-lam={np.mean(lam):.3f}  "
          f"frac lam>0.5={np.mean(lam > 0.5):.1%}  frac lam>0.3={np.mean(lam > 0.3):.1%}")

# sanity: informative-group behavior under blend vs pure (m_g ~ m_bar -> same)
m_real = 0.0377   # measured group-7 informative meanTV
print(f"\ninformative group check: c_pure={max(m_real, 0.005):.4f}  "
      f"c_blend={(m_real + M_BAR) / 2:.4f}  (ratio {((m_real + M_BAR) / 2) / m_real:.2f} -> protection unchanged)")
print("WS_GATE_NULL_DONE")

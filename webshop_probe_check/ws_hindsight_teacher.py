"""Procedural-hindsight teacher test on WebShop.

Hypothesis: WebShop is an EXECUTION task; outcome-gold privileged info (item
name/options) barely re-ranks actions. Conditioning the teacher on a successful
sibling rollout's ACTION SEQUENCE (procedural hindsight) should re-rank much
more sharply. Compare three teachers at every click-decision where the gold
item is on the page:
  T0: no privilege (student)          -- baseline margins
  T1: outcome gold (name+options+asin)
  T2: procedural hindsight (expert action walkthrough from best sibling)
Metric: logp margin of gold action vs chosen non-gold action at the <action>
emission point; fraction of decisions where teacher raises the margin.
"""
import json, os
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "/home/test/models/Qwen2.5-3B-Instruct"
HERE = os.path.dirname(os.path.abspath(__file__))
trajs = json.load(open(os.path.join(HERE, "ws_full_rollouts.json")))

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cuda:0")
model.eval(); DEV = model.device

@torch.no_grad()
def lp_cont(user_content, assistant_prefix, target):
    msgs = [{"role": "user", "content": user_content}]
    pre = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False) + assistant_prefix
    pre_ids = tok(pre, add_special_tokens=False, truncation=True, max_length=7200).input_ids
    tgt_ids = tok(target, add_special_tokens=False).input_ids
    full = torch.tensor([pre_ids + tgt_ids], device=DEV)
    lg = model(full).logits[0].float()
    L0 = len(pre_ids)
    lp = torch.log_softmax(lg[L0 - 1:L0 - 1 + len(tgt_ids), :], dim=-1)
    return float(lp[torch.arange(len(tgt_ids), device=DEV), torch.tensor(tgt_ids, device=DEV)].mean().item())

# best sibling per task (highest score)
best = {}
for tr in trajs:
    if tr["task"] not in best or tr["score"] > best[tr["task"]]["score"]:
        best[tr["task"]] = tr

def outcome_priv(goal):
    return ("Privileged information (invisible to the agent): the correct final purchase is "
            f"\"{goal['name']}\" (item ID {goal['asin']}), options {goal['goal_options']}, "
            f"price up to {goal['price_upper']}.\n\n")

def proc_priv(btr):
    acts = " -> ".join(s["action"] for s in btr["steps"])
    return ("Privileged information (invisible to the agent): an expert solved this exact task "
            f"(final reward {btr['score']:.2f}) with the action sequence: {acts}. "
            f"The correct final purchase is item ID {btr['goal']['asin']}.\n\n")

rows = []
for tr in trajs:
    goal = tr["goal"]; gasin = goal["asin"].lower()
    btr = best[tr["task"]]
    if btr["k"] == tr["k"] and btr["score"] > 0:  # don't teach a traj with itself
        pass
    for s in tr["steps"]:
        resp = s.get("response", "")
        if not resp or resp == "(forced)": continue
        if f"click[{gasin}]" not in s["avail"].lower(): continue
        a0 = resp.find("<action>")
        if a0 == -1: continue
        if gasin in s["action"]: continue  # only study wrong choices
        pre_act = resp[:a0 + len("<action>")]
        tgt_g, tgt_c = f"click[{gasin}]", s["action"][:60]
        m = {}
        for name, pv in (("T0", ""), ("T1", outcome_priv(goal)),
                         ("T2", proc_priv(btr) if btr["score"] > 0 else None)):
            if pv is None: m[name] = None; continue
            m[name] = lp_cont(pv + s["prompt"], pre_act, tgt_g) - lp_cont(pv + s["prompt"], pre_act, tgt_c)
        rows.append(dict(task=tr["task"], k=tr["k"], t=s["t"], chosen=tgt_c[:40], **m))
        print(f"task{tr['task']}k{tr['k']}t{s['t']} m0={m['T0']:+.2f} m1={m['T1']:+.2f} "
              f"m2={m['T2'] if m['T2'] is None else format(m['T2'], '+.2f')} chose={tgt_c[:32]!r}", flush=True)

print("\n===== summary (margin = logp(gold act) - logp(chosen wrong act)) =====")
r2 = [r for r in rows if r["T2"] is not None]
for nm in ("T1", "T2"):
    rr = r2 if nm == "T2" else rows
    d = np.array([r[nm] - r["T0"] for r in rr])
    print(f"  {nm}: n={len(rr)} mean margin lift={d.mean():+.3f} raise%={np.mean(d > 0):.0%} "
          f"big-raise(>0.5)%={np.mean(d > 0.5):.0%}")
if r2:
    d12 = np.array([r["T2"] - r["T1"] for r in r2])
    print(f"  T2 vs T1 direct: mean={d12.mean():+.3f} T2 better in {np.mean(d12 > 0):.0%}")
print("WS_HINDSIGHT_DONE")

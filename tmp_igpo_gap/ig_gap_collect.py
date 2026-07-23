"""IGPO-extension probe: teacher-vs-student per-turn gold-answer logp.

IGPO defines turn reward as the delta of the STUDENT's logp of the gold
answer across turn boundaries.  New idea: also compute the same per-turn
logp under a TEACHER prompt (turn prompt prefixed with privileged info),
and study
  (1) which turns have large teacher gain vs large student gain,
  (2) whether they coincide,
  (3) whether gap = dT - dS (or ratio) can serve as turn credit (A + beta*A').

Conditions per (traj, step):
  S  : plain step prompt                      (student, IGPO baseline)
  T1 : outcome-gold privilege (answer verbatim: name+asin+options)
  T2 : procedural privilege (best sibling's expert action sequence)
Targets:
  ans : declarative gold answer  "\"{name}\" (item ID {asin}) with options ..."
  act : gold click action        "click[{asin}]"

Per-turn logp levels are stored; gains/gaps computed in the analysis script.
Copy effect of T1 (answer verbatim in prefix) shifts LEVELS but cancels in
per-turn deltas -- whether it flattens the gains (ceiling) is an empirical
question this probe answers.

Model = BEACON 1.5B step150 (the policy that generated the rollouts).
Run in conda env `sdar`, GPU 0.
"""
import json
import os
import re

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
SDAR = os.path.dirname(HERE)
PROBE_DIR = os.path.join(SDAR, "webshop_probe_check")
FILES = ["ws_ckpt_rollouts2.json", "ws_ckpt_rollouts3.json", "ws_ckpt_rollouts4.json"]
MODEL = "/home/test/yyy/BEACON/checkpoints/verl_agent_webshop/beacon_qwen2.5_1.5b/step150_hf"
OUT = os.path.join(HERE, "ig_gap_rows.json")

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                             device_map="cuda:0")
model.eval()
DEV = model.device


@torch.no_grad()
def lp_cont(user_content, assistant_prefix, target):
    """Mean logp of `target` tokens continuing `assistant_prefix` after chat prompt."""
    msgs = [{"role": "user", "content": user_content}]
    pre = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                  tokenize=False) + assistant_prefix
    pre_ids = tok(pre, add_special_tokens=False, truncation=True,
                  max_length=7200).input_ids
    tgt_ids = tok(target, add_special_tokens=False).input_ids
    full = torch.tensor([pre_ids + tgt_ids], device=DEV)
    L0 = len(pre_ids)
    # slice logits BEFORE float() to keep memory low (vocab 151k)
    lg = model(full).logits[0, L0 - 1:L0 - 1 + len(tgt_ids), :].float()
    lp = torch.log_softmax(lg, dim=-1)
    tgt = torch.tensor(tgt_ids, device=DEV)
    return float(lp[torch.arange(len(tgt_ids), device=DEV), tgt].mean().item())


def opts_str(goal):
    go = goal.get("goal_options") or {}
    if isinstance(go, dict):
        return ", ".join(f"{k}: {v}" for k, v in go.items()) or "none"
    return ", ".join(map(str, go)) or "none"


def outcome_priv(goal):
    return ("Privileged information (invisible to the agent): the correct final "
            f"purchase is \"{goal['name']}\" (item ID {goal['asin']}), "
            f"options {opts_str(goal)}, price up to {goal['price_upper']}.\n\n")


def proc_priv(btr):
    acts = " -> ".join(s["action"] for s in btr["steps"])
    return ("Privileged information (invisible to the agent): an expert solved this "
            f"exact task (final reward {btr['score']:.2f}) with the action sequence: "
            f"{acts}.\n\n")


ANS_PREFIX = ("<think>\nBased on all the information gathered so far, "
              "the correct final purchase is ")
ACT_PREFIX = "<action>"


def act_type(action, goal_options):
    a = (action or "").lower()
    if a.startswith("search["):
        return "search"
    if a == "click[buy now]":
        return "buy"
    if re.match(r"click\[b0[0-9a-z]+\]", a):
        return "click_asin"
    if a.startswith("click["):
        inner = a[6:-1]
        opts = [str(v).lower() for v in (goal_options or {}).values()] \
            if isinstance(goal_options, dict) else []
        return "click_opt_match" if inner in opts else "click_other"
    return "invalid"


rows = []
for fn in FILES:
    trajs = json.load(open(os.path.join(PROBE_DIR, fn)))
    # best sibling per task, chosen per-trajectory excluding itself
    by_task = {}
    for tr in trajs:
        by_task.setdefault(tr["task"], []).append(tr)
    for tr in trajs:
        goal = tr["goal"]
        gasin = goal["asin"].lower()
        sibs = [b for b in by_task[tr["task"]] if b["k"] != tr["k"] and b["score"] > 0]
        btr = max(sibs, key=lambda b: b["score"]) if sibs else None
        p1 = outcome_priv(goal)
        p2 = proc_priv(btr) if btr else None
        tgt_ans = (f"\"{goal['name']}\" (item ID {goal['asin']}) "
                   f"with options {opts_str(goal)}.")
        tgt_act = f"click[{gasin}]"

        first_vis = None
        for s in tr["steps"]:
            if gasin in s.get("avail", "").lower():
                first_vis = s["t"]
                break

        for s in tr["steps"]:
            r = dict(fn=fn, task=tr["task"], k=tr["k"], t=s["t"],
                     score=tr["score"], n_steps=len(tr["steps"]),
                     action=s.get("action", "")[:80],
                     act_type=act_type(s.get("action", ""), goal.get("goal_options")),
                     forced=s.get("response", "") == "(forced)",
                     gold_visible=gasin in s.get("avail", "").lower(),
                     gold_click=gasin in (s.get("action", "") or "").lower(),
                     first_vis_t=first_vis,
                     has_sib=btr is not None)
            for cond, priv in (("S", ""), ("T1", p1), ("T2", p2)):
                if priv is None:
                    r[f"lp_{cond}_ans"] = None
                    r[f"lp_{cond}_act"] = None
                    continue
                user = priv + s["prompt"]
                r[f"lp_{cond}_ans"] = lp_cont(user, ANS_PREFIX, tgt_ans)
                r[f"lp_{cond}_act"] = lp_cont(user, ACT_PREFIX, tgt_act)
            rows.append(r)
        print(f"{fn} task{tr['task']}k{tr['k']} done  rows={len(rows)}", flush=True)

json.dump(rows, open(OUT, "w"), ensure_ascii=False)
print(f"TOTAL rows {len(rows)} -> {OUT}")
print("IG_GAP_COLLECT_DONE")

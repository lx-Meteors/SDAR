"""Deep-search validation of the teacher/student per-turn logp-diff credit.

Data: mint-agent correct-answer deep search trajectories (finsearchcomp_t2 +
fingaia).  For each episode with 4..14 action turns, build per-turn student
context = question + accumulated tool evidence; teacher = same + privileged
line stating the golden answer.  Compute mean logp of the golden answer under
both at every turn boundary.

Labels (derivable, no manual annotation):
  arrival: the turn whose new observation first contains the gold numeric core
  err    : all of the turn's tool observations have status=error

Model = Qwen2.5-3B-Instruct (neutral measurement model), GPU 0.
"""
import json
import os
import re

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = "/home/test/gyz/mint-agent/output/correct_answer_trajectories/"
FILES = ["finsearchcomp_t2.answer_correct.jsonl", "fingaia.answer_correct.jsonl"]
MODEL = "/home/test/models/Qwen2.5-3B-Instruct"
OUT = os.path.join(HERE, "ds_gap_rows.json")
MAX_EP_PER_FILE = 30
OBS_CHAR_CAP = 600
CTX_CHAR_CAP = 11000

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                             device_map="cuda:0")
model.eval()
DEV = model.device


@torch.no_grad()
def lp_cont(user_content, assistant_prefix, target):
    msgs = [{"role": "user", "content": user_content}]
    pre = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                  tokenize=False) + assistant_prefix
    pre_ids = tok(pre, add_special_tokens=False, truncation=True,
                  max_length=6800).input_ids
    tgt_ids = tok(target, add_special_tokens=False).input_ids
    full = torch.tensor([pre_ids + tgt_ids], device=DEV)
    L0 = len(pre_ids)
    lg = model(full).logits[0, L0 - 1:L0 - 1 + len(tgt_ids), :].float()
    lp = torch.log_softmax(lg, dim=-1)
    tgt = torch.tensor(tgt_ids, device=DEV)
    return float(lp[torch.arange(len(tgt_ids), device=DEV), tgt].mean().item())


def clean_gold(g):
    g = str(g).strip()
    for sep in ("，答案允许", ",答案允许", "。答案允许", "（答案允许", "，答案", "。 答案允许"):
        i = g.find(sep)
        if i > 0:
            g = g[:i]
    return g.strip().rstrip("。，,")


def numeric_core(g):
    m = re.search(r"[-+]?\d[\d,]*\.?\d*", g)
    return m.group(0).replace(",", "") if m else None


def obs_text(turn):
    parts = []
    for o in (turn.get("observations") or []):
        c = str(o.get("content", ""))[:OBS_CHAR_CAP]
        parts.append(c)
    return "\n".join(parts)


def turn_is_err(turn):
    obs = turn.get("observations") or []
    if not obs:
        return False
    return all(o.get("status") == "error" for o in obs)


rows = []
for fn in FILES:
    n_ep = 0
    for line in open(BASE + fn):
        if n_ep >= MAX_EP_PER_FILE:
            break
        e = json.loads(line)
        turns = [t for t in e["turns"] if t.get("type") in ("plan", "action")]
        acts = [t for t in turns if t["type"] == "action"]
        if not (4 <= len(acts) <= 14):
            continue
        gold = clean_gold(e["golden_answer"])
        core = numeric_core(gold)
        if not gold or len(gold) > 120:
            continue
        q = e["question"].strip()[:1500]
        priv = ("Privileged information (invisible to the agent): the correct "
                f"final answer to this task is: {gold}.\n\n")
        n_ep += 1
        acc = ""
        found = False
        # state 0: no evidence yet
        states = [dict(t=0, err=False, arrival=False)]
        for i, t in enumerate(acts):
            ot = obs_text(t)
            arrival = False
            if core and not found:
                blob = ot.replace(",", "")
                if core in blob:
                    arrival, found = True, True
            acc += f"\n--- evidence from step {i+1} ---\n{ot}"
            if len(acc) > CTX_CHAR_CAP:
                acc = acc[-CTX_CHAR_CAP:]
            states.append(dict(t=i + 1, err=turn_is_err(t), arrival=arrival,
                               acc=acc))
        # compute logp per state
        prev = None
        ep_rows = []
        for st in states:
            ctx = st.get("acc", "")
            user = (f"Task: {q}\n\nEvidence collected so far:{ctx}"
                    if ctx else f"Task: {q}\n\n(No evidence collected yet.)")
            pref = ("Based on the evidence collected so far, "
                    "the final answer to the task is: ")
            lpS = lp_cont(user, pref, gold)
            lpT = lp_cont(priv + user, pref, gold)
            ep_rows.append(dict(fn=fn, ep=e["episode_id"], t=st["t"],
                                n_acts=len(acts), err=st["err"],
                                arrival=st["arrival"], lpS=lpS, lpT=lpT,
                                gold=gold))
        rows.extend(ep_rows)
        print(f"{fn} {e['episode_id']} acts={len(acts)} gold={gold[:30]!r} "
              f"rows={len(rows)}", flush=True)

json.dump(rows, open(OUT, "w"), ensure_ascii=False)
print(f"TOTAL rows {len(rows)} -> {OUT}")
print("DS_GAP_COLLECT_DONE")

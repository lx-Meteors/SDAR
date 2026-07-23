"""Do think-wording forks change the ACTION?  (value-relevance of fratricide)

For each measured succ-succ fratricide pair (all in think region on ckpt):
continue from prefix+y_a and prefix+y_b (greedy + 2 samples each), extract the
action produced at this step, compare.  If actions rarely differ, wording
fratricide is value-neutral -> protection there is unnecessary -> short
contrastive privilege suffices for everything that matters.
Run: qwen-infer env, GPU5, ~3 min.
"""
import json, os, re
import numpy as np
import torch
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "/data1/test/yyy/BEACON/checkpoints/verl_agent_webshop/beacon_qwen2.5_1.5b/step150_hf"
HERE = os.path.dirname(os.path.abspath(__file__))
trajs = [t for t in json.load(open(os.path.join(HERE, "ws_ckpt_rollouts.json"))) if t["task"] == 1]

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
    return torch.softmax(lg[L0 - 1:L0 - 1 + len(out_ids), :], dim=-1), out_ids, pre_ids

ACT_RE = re.compile(r"(search\[[^\]]*\]|click\[[^\]]*\])", re.I)

@torch.no_grad()
def actions_after(pre_ids, out_ids_prefix, cand, n_sample=2):
    ids = torch.tensor([pre_ids + out_ids_prefix + [cand]], device=DEV)
    acts = []
    for mode in ["greedy"] + ["sample"] * n_sample:
        g = model.generate(ids, max_new_tokens=180,
                           do_sample=(mode == "sample"), temperature=1.0, top_p=1.0,
                           pad_token_id=tok.eos_token_id)
        txt = tok.decode(g[0][ids.shape[1]:], skip_special_tokens=True)
        m = ACT_RE.search(txt)
        acts.append(m.group(1).lower() if m else "(none)")
    return acts

# rebuild the same eval pairs as ws_ccr_field (menu collisions, succ-succ)
gmean = float(np.mean([t["score"] for t in trajs]))
for t in trajs: t["adv"] = t["score"] - gmean
menu_rows = []
for tr in trajs:
    for s in tr["steps"]:
        resp = s.get("response", "")
        if not resp or resp == "(forced)": continue
        p, out_ids, pre_ids = dists(s["prompt"], resp)
        t5v, t5i = p.topk(5, dim=-1)
        t5i = t5i.cpu().numpy()
        idx = torch.arange(len(out_ids), device=DEV)
        py = p[idx, torch.tensor(out_ids, device=DEV)].cpu().numpy()
        for i in range(len(out_ids)):
            menu_rows.append(dict(k=tr["k"], adv=tr["adv"], t=s["t"], i=i,
                                  y=int(out_ids[i]), p_y=float(py[i]),
                                  menu=tuple(sorted(int(x) for x in t5i[i]))))
        del p; torch.cuda.empty_cache()

groups = defaultdict(list)
for r in menu_rows: groups[r["menu"]].append(r)
pairs, seen = [], set()
for mem in groups.values():
    if len({m["k"] for m in mem}) < 2 or len({m["y"] for m in mem}) < 2: continue
    for a in mem:
        if a["adv"] <= 0 or a["p_y"] > 0.95: continue
        for b in mem:
            if b["k"] == a["k"] or b["adv"] <= 0 or b["y"] == a["y"]: continue
            key = (a["k"], a["t"], a["i"], a["y"], b["y"])
            if key in seen: continue
            seen.add(key)
            pairs.append((a["k"], a["t"], a["i"], a["y"], b["y"]))
print(f"unique fork pairs: {len(pairs)}")

by_k = {t["k"]: t for t in trajs}
same_g, same_any, rows = [], [], []
for (k, t_, i, y_a, y_b) in pairs[:24]:
    tr = by_k[k]
    s = next(s for s in tr["steps"] if s["t"] == t_)
    _, out_ids, pre_ids = dists(s["prompt"], s["response"])
    aA = actions_after(pre_ids, out_ids[:i], y_a)
    aB = actions_after(pre_ids, out_ids[:i], y_b)
    g_same = aA[0] == aB[0]
    any_overlap = len(set(aA) & set(aB)) > 0
    same_g.append(g_same); same_any.append(any_overlap)
    rows.append(dict(k=k, t=t_, i=i, ya=tok.decode([y_a]), yb=tok.decode([y_b]),
                     actsA=aA, actsB=aB))
    print(f"k{k}t{t_}i{i} {tok.decode([y_a])!r}->|{tok.decode([y_b])!r}: "
          f"greedy_same={g_same}  A={aA}  B={aB}", flush=True)

print(f"\nSummary over {len(same_g)} fork pairs:")
print(f"  greedy action identical : {np.mean(same_g):.0%}")
print(f"  action-set overlap      : {np.mean(same_any):.0%}")
json.dump(rows, open(os.path.join(HERE, "ws_fork_consequence.json"), "w"))
print("WS_FORK_CONSEQ_DONE")

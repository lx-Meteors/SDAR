"""
Selector comparison: is teacher-student entropy GAP (delta_H, TECA) a *better*
key-token selector than plain student/teacher entropy?

On real successful (positive-advantage) webshop rollouts we compute, per response
token, four scalar signals:
  dh          = H_T^{\\y} - H_S^{\\y}     (TECA gap; teacher minus student, target-excluded)
  student_full= H_S      (student full-vocab entropy)          <- "directly filter by entropy"
  student_woy = H_S^{\\y}(student target-excluded entropy)     <- student-only, matched form
  teacher_woy = H_T^{\\y}(teacher target-excluded entropy)     <- teacher-only

Each selector keeps the per-trajectory top-20% of its score among top-1-eligible
tokens (same eligibility/quota as TECA). We then ask which selector's kept tokens
best coincide with the *task-relevant decision tokens*. Since webshop's key
decision is objectively "did you honor the instruction constraints (color / size /
price / material / category)", we use instruction-constraint-word hit-rate as an
objective proxy, plus content-word fraction, plus overlap with dh_pos.

Selectors:
  dh_pos      top-20% by  +dh   (TECA current: teacher more spread)
  dh_neg      top-20% by  -dh   (teacher more certain via skill)
  abs_dh      top-20% by |dh|
  student_full top-20% by H_S    (plain "filter by entropy")
  student_woy  top-20% by H_S^{\\y}
  teacher_woy  top-20% by H_T^{\\y}
  random       random 20%        (reference floor)

Real rollouts, real env rewards. Offline plausibility check only.
"""

import json
import os
import re
import sys

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/home/test/yyy/SDAR")
from verl.trainer.ppo.teca_utils import entropy_excluding_target  # noqa: E402
from verl.trainer.ppo.rlsd_utils import SkillProvider  # noqa: E402

MODEL_PATH = "/home/test/models/Qwen2.5-3B-Instruct"
RECORDS = os.environ.get("RECORDS", "/home/test/yyy/SDAR/tmp_scale/records_g16.json")
SKILLS_DIR = "/home/test/yyy/SDAR/skills/webshop"
OUT_PATH = os.environ.get(
    "OUT_PATH", os.path.join(os.path.dirname(__file__), "selector_compare.json")
)

TEMPERATURE = 1.0
TOP_FRAC = 0.20
EPS = 1e-6

STOPWORDS = set(
    "the a an of to in and or is are was were be been being it this that these those "
    "for on with as at by from into would will can could should i we you they he she "
    "there here so no not need needs want looking buy purchase price dollars dollar "
    "has have had do does did but however therefore since which who whom whose what "
    "when where why how their our its his her my your me us them if then than also "
    "one two more most less least very much many few new used item items product "
    "products option options choose select click search action think observation "
    "instruction current best why what next".split()
)


def is_function_tok(tok_str):
    s = tok_str.strip().lower()
    if s == "" or re.fullmatch(r"[^\w]+", s):
        return True
    return s in STOPWORDS


def build_constraint_set(instruction):
    """Task-relevant constraint words from the instruction (attrs/values/numbers)."""
    words = re.findall(r"[a-zA-Z][a-zA-Z\-]+|\d+(?:\.\d+)?", instruction.lower())
    cons = set()
    for w in words:
        w = w.strip("-")
        if len(w) >= 2 and w not in STOPWORDS:
            cons.add(w)
    return cons


def token_hits_constraint(tok_str, cons):
    """A response token hits if its alnum fragment (>=3 chars) matches a constraint
    word (either substring direction, to absorb subword tokenization)."""
    s = re.sub(r"[^a-z0-9]", "", tok_str.strip().lower())
    if len(s) < 3:
        return False
    for w in cons:
        if len(w) < 3:
            continue
        if s in w or w in s:
            return True
    return False


def build_user(instruction, history, obs):
    hist = ""
    for (a, o) in history:
        hist += f"Action: {a}\nObservation: {o}\n"
    return (
        "You are an expert shopping agent in the WebShop text environment.\n"
        f"Instruction: {instruction}\n\n"
        + (f"History so far:\n{hist}\n" if hist else "")
        + f"Current observation:\n{obs}\n\n"
        "First reason carefully inside <think> </think>: state what the instruction "
        "requires, what the current observation tells you, which candidate looks best "
        "and why, and what to do next (at least 2-3 sentences). Then output exactly one "
        "action inside <action> </action>. An action is either search[keywords] "
        "(only on the search page) or click[value]."
    )


@torch.no_grad()
def response_logits(model, prompt_ids, response_ids, device):
    input_ids = torch.cat([prompt_ids, response_ids]).unsqueeze(0).to(device)
    logits = model(input_ids=input_ids, use_cache=False).logits[0]
    n = response_ids.size(0)
    return logits[-n - 1 : -1].float() / TEMPERATURE


def full_entropy(logps):
    return -(logps.exp() * logps).sum(-1)


@torch.no_grad()
def traj_stats(model, tok, skill_provider, instruction, traj, device):
    """Per-token signals across a trajectory."""
    out = {"dh": [], "hs_full": [], "hs_woy": [], "ht_woy": [],
           "top1": [], "tok": [], "seg": [], "hit": []}
    cons = build_constraint_set(instruction)
    history = []
    for st in traj["steps"]:
        action = st.get("action")
        if not action:
            continue
        user = build_user(instruction, history, st["obs"])
        student_text = tok.apply_chat_template(
            [{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True
        )
        skill_text = skill_provider.get_privileged_info_from_prompt(user)
        teacher_text = f"[Privileged Skill Information]\n{skill_text}\n\n" + student_text
        think = st.get("think") or ""
        head = f"<think>\n{think}\n</think>\n\n<action>"
        resp_text = head + f"{action}</action>"
        action_start = len(tok.encode(head, add_special_tokens=False))

        sp_ids = torch.tensor(tok.encode(student_text, add_special_tokens=False))
        tp_ids = torch.tensor(tok.encode(teacher_text, add_special_tokens=False))
        r_ids = torch.tensor(tok.encode(resp_text, add_special_tokens=False))
        if r_ids.numel() < 2:
            history.append((action, st["obs"][:400]))
            continue

        s_lp = F.log_softmax(response_logits(model, sp_ids, r_ids, device), dim=-1)
        t_lp = F.log_softmax(response_logits(model, tp_ids, r_ids, device), dim=-1)
        r_dev = r_ids.to(device)
        hs_woy = entropy_excluding_target(s_lp, r_dev).cpu()
        ht_woy = entropy_excluding_target(t_lp, r_dev).cpu()
        hs_full = full_entropy(s_lp).cpu()
        dh = (ht_woy - hs_woy)
        top1 = ((t_lp.argmax(-1).cpu() == r_ids) & (s_lp.argmax(-1).cpu() == r_ids))
        for j in range(r_ids.numel()):
            out["dh"].append(float(dh[j]))
            out["hs_full"].append(float(hs_full[j]))
            out["hs_woy"].append(float(hs_woy[j]))
            out["ht_woy"].append(float(ht_woy[j]))
            out["top1"].append(bool(top1[j]))
            ts = tok.decode(r_ids[j : j + 1])
            out["tok"].append(ts)
            out["seg"].append("action" if j >= action_start else "think")
            out["hit"].append(token_hits_constraint(ts, cons))
        history.append((action, st["obs"][:400]))
    return {k: (torch.tensor(v) if k in ("dh", "hs_full", "hs_woy", "ht_woy")
                else (torch.tensor(v) if k in ("top1", "hit") else v))
            for k, v in out.items()}


def select_topfrac(score, elig, frac):
    """Boolean mask: within eligible, keep top `frac` by score."""
    keep = torch.zeros_like(elig)
    se = score[elig]
    if se.numel() == 0:
        return keep
    thr = torch.quantile(se.float(), 1.0 - frac)
    keep = elig & (score >= thr)
    return keep


def main():
    device = "cuda"
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map=device,
        attn_implementation="flash_attention_2",
    )
    model.eval()
    skill_provider = SkillProvider(skills_dir=SKILLS_DIR, skill_all=False)
    records = json.load(open(RECORDS))

    torch.manual_seed(0)
    selectors = ["dh_pos", "dh_neg", "abs_dh", "student_full", "student_woy", "teacher_woy", "random"]
    # accumulate: kept-token counts, hits, content, action-seg, mean student entropy
    acc = {s: {"kept": 0, "hit": 0, "content": 0, "action": 0, "hs_sum": 0.0} for s in selectors}
    overlap = {s: [] for s in selectors}  # per-traj jaccard vs dh_pos
    n_traj = 0

    for g in records:
        rewards = torch.tensor([float(t["reward"]) for t in g["trajs"]])
        if rewards.std() < 1e-8:
            continue
        grp_adv = (rewards - rewards.mean()) / (rewards.std() + EPS)
        for ti, traj in enumerate(g["trajs"]):
            if grp_adv[ti].item() <= 0:
                continue
            st = traj_stats(model, tok, skill_provider, g["instruction"], traj, device)
            elig = st["top1"]
            if elig.sum() < 3:
                continue
            n_traj += 1
            hit = st["hit"]
            n = elig.numel()

            masks = {
                "dh_pos": select_topfrac(st["dh"], elig, TOP_FRAC),
                "dh_neg": select_topfrac(-st["dh"], elig, TOP_FRAC),
                "abs_dh": select_topfrac(st["dh"].abs(), elig, TOP_FRAC),
                "student_full": select_topfrac(st["hs_full"], elig, TOP_FRAC),
                "student_woy": select_topfrac(st["hs_woy"], elig, TOP_FRAC),
                "teacher_woy": select_topfrac(st["ht_woy"], elig, TOP_FRAC),
            }
            # random: sample ~TOP_FRAC of eligible
            elig_idx = elig.nonzero(as_tuple=True)[0]
            k = max(1, int(round(TOP_FRAC * elig_idx.numel())))
            perm = elig_idx[torch.randperm(elig_idx.numel())[:k]]
            rnd = torch.zeros_like(elig)
            rnd[perm] = True
            masks["random"] = rnd

            dhpos = masks["dh_pos"]
            seg_action = torch.tensor([s == "action" for s in st["seg"]])
            for s, m in masks.items():
                acc[s]["kept"] += int(m.sum())
                acc[s]["hit"] += int((m & hit).sum())
                acc[s]["content"] += int(sum(1 for j in range(n) if m[j] and not is_function_tok(st["tok"][j])))
                acc[s]["action"] += int((m & seg_action).sum())
                acc[s]["hs_sum"] += float(st["hs_full"][m].sum())
                inter = int((m & dhpos).sum())
                uni = int((m | dhpos).sum())
                overlap[s].append(inter / max(uni, 1))
        print(f"gid={g['gid']} done ({n_traj} positive trajs)", flush=True)

    summary = {"n_positive_trajectories": n_traj, "top_frac": TOP_FRAC, "selectors": {}}
    for s in selectors:
        kept = max(acc[s]["kept"], 1)
        summary["selectors"][s] = {
            "kept_tokens": acc[s]["kept"],
            "constraint_hit_rate": round(acc[s]["hit"] / kept, 4),
            "content_word_frac": round(acc[s]["content"] / kept, 4),
            "action_seg_frac": round(acc[s]["action"] / kept, 4),
            "mean_student_entropy": round(acc[s]["hs_sum"] / kept, 4),
            "jaccard_vs_dh_pos": round(sum(overlap[s]) / max(len(overlap[s]), 1), 4),
        }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    with open(OUT_PATH, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()

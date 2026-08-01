"""
Effect survey across ALL successful (positive-advantage) trajectories in
records_g16.json. Answers:
  (1) Is the ~40% boost fraction (pure relu on target-excluded delta_H) typical?
  (2) Are boosted tokens meaningful content words, or mostly function words / punct?
  (3) Would a lightweight per-trajectory-centered variant
        delta_H' = relu(delta_H - mean_traj(delta_H over top-1 tokens))
      concentrate credit onto fewer, more decisive tokens?

Real webshop rollouts, real env rewards. Offline plausibility check only.
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
RECORDS = "/home/test/yyy/SDAR/tmp_scale/records_g16.json"
SKILLS_DIR = "/home/test/yyy/SDAR/skills/webshop"
OUT_PATH = os.path.join(os.path.dirname(__file__), "effect_survey.json")

TEMPERATURE = 1.0
BETA = 0.1
EPS = 1e-6
TOP_FRAC = 0.20  # strong selectivity: keep only the top 20% dh among eligible tokens

STOPWORDS = set(
    "the a an of to in and or is are was were be been being it this that these those "
    "for on with as at by from into would will can could should i we you they he she "
    "there here so no not need has have had do does did but however therefore since "
    "which who whom whose what when where why how their our its his her my your".split()
)


def is_function_tok(tok_str):
    s = tok_str.strip().lower()
    if s == "" or re.fullmatch(r"[^\w]+", s):  # whitespace or pure punctuation
        return True
    return s in STOPWORDS


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


@torch.no_grad()
def traj_stats(model, tok, skill_provider, instruction, traj, device):
    """Per-token dh, top1 flag, token string, seg across the trajectory."""
    dh_all, top1_all, toks, segs = [], [], [], []
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
        dh = (entropy_excluding_target(t_lp, r_dev) - entropy_excluding_target(s_lp, r_dev)).cpu()
        top1 = ((t_lp.argmax(-1).cpu() == r_ids) & (s_lp.argmax(-1).cpu() == r_ids))
        for j in range(r_ids.numel()):
            dh_all.append(float(dh[j]))
            top1_all.append(bool(top1[j]))
            toks.append(tok.decode(r_ids[j : j + 1]))
            segs.append("action" if j >= action_start else "think")
        history.append((action, st["obs"][:400]))
    return torch.tensor(dh_all), torch.tensor(top1_all), toks, segs


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

    per_traj = []
    # aggregate function-vs-content counts of boosted tokens
    agg = {"relu": {"content": 0, "func": 0},
           "centered": {"content": 0, "func": 0},
           "top20": {"content": 0, "func": 0}}
    content_examples, func_examples, top20_examples = [], [], []

    for g in records:
        rewards = torch.tensor([float(t["reward"]) for t in g["trajs"]])
        if rewards.std() < 1e-8:
            continue
        grp_adv = (rewards - rewards.mean()) / (rewards.std() + EPS)
        for ti, traj in enumerate(g["trajs"]):
            A = grp_adv[ti].item()
            if A <= 0:
                continue  # only positive samples are shaped
            dh, top1, toks, segs = traj_stats(model, tok, skill_provider, g["instruction"], traj, device)
            n = dh.numel()
            if n == 0:
                continue
            elig = top1  # positive sample already guaranteed
            # pure relu
            relu_boost = elig & (dh > 0)
            # per-trajectory centered relu (center on mean dh over eligible/top-1 tokens)
            center = dh[elig].mean() if elig.any() else torch.tensor(0.0)
            centered_boost = elig & ((dh - center) > 0)
            # strong selectivity: keep only the top TOP_FRAC of dh among eligible tokens
            dh_elig = dh[elig]
            if dh_elig.numel() > 0:
                thr = torch.quantile(dh_elig, 1.0 - TOP_FRAC)
                top20_boost = elig & (dh >= thr)
            else:
                top20_boost = torch.zeros_like(elig)

            for j in range(n):
                if relu_boost[j]:
                    key = "func" if is_function_tok(toks[j]) else "content"
                    agg["relu"][key] += 1
                if centered_boost[j]:
                    key = "func" if is_function_tok(toks[j]) else "content"
                    agg["centered"][key] += 1
                    if key == "content" and len(content_examples) < 40:
                        content_examples.append({"tok": toks[j], "dh": round(float(dh[j]), 3), "seg": segs[j]})
                if top20_boost[j]:
                    key = "func" if is_function_tok(toks[j]) else "content"
                    agg["top20"][key] += 1
                    if key == "content" and len(top20_examples) < 40:
                        top20_examples.append({"tok": toks[j], "dh": round(float(dh[j]), 3), "seg": segs[j]})

            per_traj.append({
                "gid": g["gid"], "rollout": int(traj["rollout"]),
                "reward": round(float(rewards[ti]), 3), "advantage": round(A, 3),
                "n_tokens": int(n),
                "relu_boost_frac": round(relu_boost.float().mean().item(), 4),
                "centered_boost_frac": round(centered_boost.float().mean().item(), 4),
                "top20_boost_frac": round(top20_boost.float().mean().item(), 4),
                "dh_top1_mean": round(dh[elig].mean().item(), 4) if elig.any() else None,
            })
        print(f"gid={g['gid']} done ({len(per_traj)} positive trajs so far)", flush=True)

    def frac(d):
        tot = d["content"] + d["func"]
        return round(d["content"] / max(tot, 1), 4)

    fracs_relu = [t["relu_boost_frac"] for t in per_traj]
    fracs_cent = [t["centered_boost_frac"] for t in per_traj]
    fracs_t20 = [t["top20_boost_frac"] for t in per_traj]
    summary = {
        "n_positive_trajectories": len(per_traj),
        "relu_boost_frac_mean": round(sum(fracs_relu) / max(len(fracs_relu), 1), 4),
        "relu_boost_frac_min_max": [round(min(fracs_relu), 4), round(max(fracs_relu), 4)] if fracs_relu else None,
        "centered_boost_frac_mean": round(sum(fracs_cent) / max(len(fracs_cent), 1), 4),
        "top20_boost_frac_mean": round(sum(fracs_t20) / max(len(fracs_t20), 1), 4),
        "top20_boost_frac_min_max": [round(min(fracs_t20), 4), round(max(fracs_t20), 4)] if fracs_t20 else None,
        "relu_boosted_content_word_frac": frac(agg["relu"]),
        "centered_boosted_content_word_frac": frac(agg["centered"]),
        "top20_boosted_content_word_frac": frac(agg["top20"]),
        "relu_counts": agg["relu"],
        "centered_counts": agg["centered"],
        "top20_counts": agg["top20"],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    with open(OUT_PATH, "w") as f:
        json.dump({"summary": summary, "per_trajectory": per_traj,
                   "centered_content_examples": content_examples,
                   "top20_content_examples": top20_examples}, f, indent=2, ensure_ascii=False)
    print(f"saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()

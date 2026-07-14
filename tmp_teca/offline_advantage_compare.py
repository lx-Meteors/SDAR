"""
Offline advantage-shaping comparison: SDAR vs TECA on REAL webshop rollouts.

TECA signal (per user's definition):
  candidate set = whole vocabulary EXCLUDING the realized/target token,
  delta_H = H_teacher^{\\y} - H_student^{\\y}   (target-excluded, renormalized).
Eligibility (top-1): realized token is the full-vocab argmax of BOTH teacher and
student. Positive samples only. Additive shaping, no threshold, no clipping:
  A'_{i,t} = A_{i,t} + beta * delta_H_t.

Data: tmp_scale/records_g16.json (real grouped rollouts, real env rewards).
Group advantages are the real GRPO advantages (r - mean_g)/(std_g + eps) -- exactly
what SDAR trains on. We call the training code path (verl.trainer.ppo.teca_utils)
so offline and training share identical shaping logic.
"""

import json
import os
import sys

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/home/test/yyy/SDAR")
from verl.trainer.ppo.teca_utils import (  # noqa: E402
    compute_teca_advantage,
    entropy_excluding_target,
)
from verl.trainer.ppo.rlsd_utils import SkillProvider  # noqa: E402

MODEL_PATH = "/home/test/models/Qwen2.5-3B-Instruct"
RECORDS = "/home/test/yyy/SDAR/tmp_scale/records_g16.json"
SKILLS_DIR = "/home/test/yyy/SDAR/skills/webshop"
OUT_PATH = os.path.join(os.path.dirname(__file__), "advantage_compare.json")

TEMPERATURE = 1.0
BETAS = [0.05, 0.1, 0.2, 0.5, 1.0]  # sweep to calibrate additive strength
EPS = 1e-6


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

    adv, dh, targ, sarg, resp, seg = [], [], [], [], [], []
    n_groups_used = 0

    for group in records:
        gid = group["gid"]
        instruction = group["instruction"]
        rewards = torch.tensor([float(t["reward"]) for t in group["trajs"]])
        if rewards.std() < 1e-8:
            continue
        n_groups_used += 1
        grp_adv = (rewards - rewards.mean()) / (rewards.std() + EPS)

        for ti, traj in enumerate(group["trajs"]):
            A = grp_adv[ti].item()
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

                # target-excluded, renormalized candidate entropy over the WHOLE vocab
                h_t = entropy_excluding_target(t_lp, r_dev)
                h_s = entropy_excluding_target(s_lp, r_dev)
                delta_h = (h_t - h_s).cpu()

                L = r_ids.numel()
                dh.append(delta_h)
                targ.append(t_lp.argmax(-1).cpu())
                sarg.append(s_lp.argmax(-1).cpu())
                resp.append(r_ids)
                adv.append(torch.full((L,), A))
                seg.append((torch.arange(L) >= action_start).long())

                history.append((action, st["obs"][:400]))
        print(f"group gid={gid}: cumulative tokens={sum(x.numel() for x in resp)}", flush=True)

    advantages = torch.cat(adv).unsqueeze(0)
    delta_h = torch.cat(dh).unsqueeze(0)
    teacher_argmax = torch.cat(targ).unsqueeze(0)
    student_argmax = torch.cat(sarg).unsqueeze(0)
    responses = torch.cat(resp).unsqueeze(0)
    seg_t = torch.cat(seg)
    mask = torch.ones_like(advantages)
    N = advantages.numel()

    # eligibility (independent of beta): top-1 agreement AND positive sample
    top1 = (teacher_argmax == responses) & (student_argmax == responses)
    elig = top1.squeeze(0) & (advantages.squeeze(0) > 0)
    adv_f = advantages.squeeze(0)
    dh_f = delta_h.squeeze(0)
    action_tok = seg_t.bool()

    def q(x):
        if x.numel() == 0:
            return None
        return [round(v, 4) for v in torch.quantile(x, torch.tensor([0.1, 0.5, 0.9, 0.99])).tolist()]

    # delta_H distribution (this drives beta calibration)
    dh_elig = dh_f[elig]
    dh_stats = {
        "delta_h_all_mean": round(dh_f.mean().item(), 4),
        "delta_h_eligible_mean": round(dh_elig.mean().item(), 4) if dh_elig.numel() else None,
        "delta_h_eligible_std": round(dh_elig.std().item(), 4) if dh_elig.numel() > 1 else None,
        "delta_h_eligible_q10_50_90_99": q(dh_elig),
        "frac_eligible_pos_dh": round((dh_elig > 0).float().mean().item(), 4) if dh_elig.numel() else None,
        "eligible_frac_of_positive": round(elig[adv_f > 0].float().mean().item(), 4) if (adv_f > 0).any() else None,
        "eligible_in_action_seg_frac": round(action_tok[elig].float().mean().item(), 4) if elig.any() else None,
    }

    # beta sweep with the training shaping function
    sweep = {}
    for beta in BETAS:
        shaped, metrics = compute_teca_advantage(
            advantages=advantages, delta_h=delta_h,
            teacher_argmax_ids=teacher_argmax, student_argmax_ids=student_argmax,
            responses=responses, response_mask=mask,
            beta=beta, positive_only=True, require_top1=True,
        )
        shp_f = shaped.squeeze(0)
        d = shp_f - adv_f
        sweep[f"beta={beta}"] = {
            "sign_flip_count": int(((adv_f * shp_f) < 0).sum()),
            "adv_abs_change_mean": round(metrics["teca/adv_abs_change_mean"], 5),
            "pos_mass_before": round(adv_f[adv_f > 0].sum().item(), 2),
            "pos_mass_after": round(shp_f[adv_f > 0].sum().item(), 2),
            "median_abs_change_on_eligible": round(d[elig].abs().median().item(), 4) if elig.any() else None,
        }

    summary = {
        "candidate_set": "whole_vocab_excluding_target",
        "shaping": "additive A + beta*deltaH (no threshold, no clip)",
        "n_groups_used": n_groups_used,
        "n_tokens_total": int(N),
        "n_positive_sample_tokens": int((adv_f > 0).sum()),
        "n_eligible_tokens": int(elig.sum()),
        "top1_agree_frac": round(top1.float().mean().item(), 4),
        **dh_stats,
        "beta_sweep": sweep,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    with open(OUT_PATH, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()

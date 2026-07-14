"""
Per-trajectory case study: WHERE does TECA put the extra credit?

Uses real webshop rollouts (records_g16.json, real env rewards). For a few task
groups that contain BOTH successful and failed rollouts, we:
  * compute per-token delta_H = H_teacher^{\\y} - H_student^{\\y}  (whole-vocab,
    target-excluded), the eligibility (top-1 agreement) and GRPO advantage,
  * apply the finalized TECA shaping  A' = A + beta * max(delta_H, 0)  on eligible
    tokens of positive samples (beta=0.1),
  * for each SUCCESSFUL (positive-advantage) trajectory, print the tokens that
    receive the most extra credit, WITH their local context, so we can judge by eye
    whether the boosted tokens are meaningful decisions / reasoning,
  * contrast with a FAILED trajectory in the same group.

This validates the *plausibility* of the credit assignment (offline). It does NOT
measure final task-score lift -- that needs a training run.
"""

import json
import os
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
OUT_PATH = os.path.join(os.path.dirname(__file__), "case_study.json")

TEMPERATURE = 1.0
BETA = 0.1
EPS = 1e-6
N_GROUPS = 2          # number of task groups to detail
TOP_BOOST = 10        # boosted tokens to show per trajectory
CTX = 8               # context tokens before the boosted token


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
def traj_token_stats(model, tok, skill_provider, instruction, traj, device):
    """Return per-token records across the whole trajectory."""
    recs = []
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
        t_arg = t_lp.argmax(-1).cpu()
        s_arg = s_lp.argmax(-1).cpu()
        top1 = (t_arg == r_ids) & (s_arg == r_ids)

        for j in range(r_ids.numel()):
            lo = max(0, j - CTX)
            recs.append({
                "step": int(st["step"]),
                "seg": "action" if j >= action_start else "think",
                "token": tok.decode(r_ids[j : j + 1]),
                "context": tok.decode(r_ids[lo:j]),
                "delta_h": round(float(dh[j]), 4),
                "top1": bool(top1[j]),
            })
        history.append((action, st["obs"][:400]))
    return recs


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

    # pick groups with reward variance, most-successful first
    cand_groups = []
    for g in records:
        rs = [float(t["reward"]) for t in g["trajs"]]
        if max(rs) - min(rs) > 1e-6:
            cand_groups.append((max(rs), g))
    cand_groups.sort(key=lambda x: -x[0])
    groups = [g for _, g in cand_groups[:N_GROUPS]]

    report = []
    for g in groups:
        rewards = torch.tensor([float(t["reward"]) for t in g["trajs"]])
        grp_adv = (rewards - rewards.mean()) / (rewards.std() + EPS)
        # best positive trajectory + one failed trajectory
        best_ti = int(torch.argmax(grp_adv))
        fail_ti = int(torch.argmin(grp_adv))

        group_entry = {"gid": g["gid"], "instruction": g["instruction"], "trajectories": []}
        for role, ti in [("success", best_ti), ("failure", fail_ti)]:
            A = grp_adv[ti].item()
            recs = traj_token_stats(model, tok, skill_provider, g["instruction"], g["trajs"][ti], device)
            elig = [r for r in recs if r["top1"] and A > 0 and r["delta_h"] > 0]
            elig.sort(key=lambda r: -r["delta_h"])
            n_tok = len(recs)
            boosted = elig[:TOP_BOOST]
            group_entry["trajectories"].append({
                "role": role,
                "rollout": int(g["trajs"][ti]["rollout"]),
                "reward": round(float(rewards[ti]), 3),
                "advantage": round(A, 3),
                "n_tokens": n_tok,
                "n_boosted": len(elig),
                "boost_frac": round(len(elig) / max(n_tok, 1), 4),
                "total_extra_credit": round(BETA * sum(r["delta_h"] for r in elig), 4),
                "top_boosted_tokens": [
                    {"seg": r["seg"], "delta_h": r["delta_h"], "d_adv": round(BETA * r["delta_h"], 4),
                     "context": r["context"], "token": r["token"]}
                    for r in boosted
                ],
            })
        report.append(group_entry)

    with open(OUT_PATH, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # human-readable print
    for ge in report:
        print("=" * 100)
        print(f"TASK gid={ge['gid']}: {ge['instruction'][:110]}")
        for tr in ge["trajectories"]:
            print(f"\n  [{tr['role'].upper()}] rollout={tr['rollout']} reward={tr['reward']} "
                  f"advantage={tr['advantage']}  tokens={tr['n_tokens']} boosted={tr['n_boosted']} "
                  f"({tr['boost_frac']*100:.1f}%)  total_extra_credit={tr['total_extra_credit']}")
            if tr["role"] == "failure":
                print("     (negative/zero advantage -> TECA leaves it untouched by construction)")
                continue
            for b in tr["top_boosted_tokens"]:
                ctx = b["context"].replace("\n", " ")[-60:]
                print(f"     dH={b['delta_h']:+.3f} d_adv={b['d_adv']:+.3f} [{b['seg']:6s}] "
                      f"...{ctx!r} >>{b['token']!r}<<")
    print(f"\nsaved -> {OUT_PATH}")


if __name__ == "__main__":
    main()

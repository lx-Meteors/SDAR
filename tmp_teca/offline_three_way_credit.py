"""
Three-way credit-assignment comparison on REAL webshop rollouts:

  GRPO/SDAR : trajectory-level z-scored reward, identical for every step of a
              trajectory (SDAR's advantage IS GRPO's; distillation is a separate
              loss term that does not change credit assignment).
  GiGPO     : episode advantage + step advantage from anchor-state grouping
              (exact-obs clusters within a task group), discounted step returns.
              Faithful to verl-agent gigpo/core_gigpo.py with the webshop
              defaults: gamma=0.95, step_advantage_w=1.0, mode=mean_norm.
  TECA      : GRPO advantage + beta * delta_H shaping on the per-response
              top-20% delta_H tokens (positive trajs, top-1 agreed tokens).
              A step's scalar credit = mean over its tokens of the shaped
              advantage = A + beta * sum(selected dh) / n_tokens.

Oracle step labels (same as tmp_scale/analyze_scale.py):
  click[gold_asin]        -> +1        click[other_asin] -> -1
  click[goal_option]      -> +1
  search[...] w/ gold in next obs -> +1  else -> -1
  everything else         ->  0 (unlabeled)

Metrics:
  within-trajectory pairwise AUC : for (good, bad) labeled step pairs of the
      SAME trajectory, fraction where credit(good) > credit(bad), ties = 0.5.
      This is the credit-assignment question: GRPO is 0.5 by construction.
  sign-acc / coverage            : across steps, does sign(credit) match oracle;
      coverage = fraction of labeled steps with non-zero credit.
  GiGPO live-anchor fraction     : fraction of labeled steps whose anchor group
      has >=2 rollouts (else GiGPO's step term is dead).

Real rollouts, real env rewards. Offline check only.
"""

import json
import os
import re
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/home/test/yyy/SDAR")
from verl.trainer.ppo.teca_utils import entropy_excluding_target  # noqa: E402
from verl.trainer.ppo.rlsd_utils import SkillProvider  # noqa: E402

MODEL_PATH = "/home/test/models/Qwen2.5-3B-Instruct"
RECORDS_LIST = os.environ.get(
    "RECORDS_LIST",
    "/home/test/yyy/SDAR/tmp_scale/records_g8_fresh7.json,"
    "/home/test/yyy/SDAR/tmp_scale/records_g8_fresh11.json",
).split(",")
SKILLS_DIR = "/home/test/yyy/SDAR/skills/webshop"
OUT_PATH = os.path.join(os.path.dirname(__file__), "three_way_credit.json")

TEMPERATURE = 1.0
BETA = 0.1
TOP_FRAC = 0.20
GAMMA = 0.95          # verl-agent webshop default
STEP_ADV_W = 1.0      # verl-agent webshop default
EPS = 1e-6

ASIN_RE = re.compile(r"^[A-Z0-9]{10}$")


def oracle(action, next_obs_has_gold, gold_asin, goal_options):
    a = (action or "").lower()
    m = re.match(r"click\[(.+)\]", a)
    if m:
        v = m.group(1).strip()
        if ASIN_RE.match(v.upper()):
            return +1 if v == gold_asin.lower() else -1
        if any(v == o for o in goal_options):
            return +1
    if a.startswith("search["):
        return +1 if next_obs_has_gold else -1
    return 0


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
def teca_step_boosts(model, tok, skill_provider, instruction, traj, device):
    """Per-step TECA boost: beta * sum(selected relu(dh)) / n_tokens of that step.

    Selection = top-1 agreed tokens, dh > 0, then per-TRAJECTORY top TOP_FRAC of dh
    (matching compute_teca_advantage's per-row quantile over the whole response).
    Returns list of (step_index_in_traj, boost) for steps with an action.
    """
    per_step = []       # (list index into traj["steps"], token dh tensor, elig mask, n_tokens)
    history = []
    for si, st in enumerate(traj["steps"]):
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
        resp_text = f"<think>\n{think}\n</think>\n\n<action>{action}</action>"

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
        per_step.append((si, dh, top1 & (dh > 0), dh.numel()))
        history.append((action, st["obs"][:400]))

    # per-trajectory top-frac quantile over eligible dh (like compute_teca_advantage rows)
    all_dh = torch.cat([dh[m] for (_, dh, m, _) in per_step]) if per_step else torch.tensor([])
    boosts = {}
    if all_dh.numel() > 0:
        thr = torch.quantile(all_dh.float(), 1.0 - TOP_FRAC)
        for (si, dh, m, n) in per_step:
            sel = m & (dh >= thr)
            boosts[si] = BETA * float(dh[sel].clamp(min=0).sum()) / max(n, 1)
    else:
        for (si, _, _, _) in per_step:
            boosts[si] = 0.0
    return boosts


def gigpo_credits(task):
    """Faithful offline GiGPO (mean_norm, gamma=0.95, w=1.0) for one task group.

    Returns {(rollout, step_list_idx): (credit, anchor_live)}.
    """
    rows = []
    for tr in task["trajs"]:
        steps = [(si, st) for si, st in enumerate(tr["steps"]) if st.get("action")]
        T = len(steps)
        for pos, (si, st) in enumerate(steps):
            step_ret = (GAMMA ** (T - 1 - pos)) * float(tr["reward"])
            rows.append(dict(rollout=tr["rollout"], si=si, obs=" ".join(st["obs"].split()),
                             R=float(tr["reward"]), step_ret=step_ret))
    if not rows:
        return {}
    # episode advantage, mean over step-rows (compute_mean_std_cross_steps=True)
    ep_mean = float(np.mean([r["R"] for r in rows]))
    # anchor grouping: exact obs within task
    grp = defaultdict(list)
    for r in rows:
        grp[r["obs"]].append(r)
    out = {}
    for v in grp.values():
        live = len(set(x["rollout"] for x in v)) >= 2
        m = float(np.mean([x["step_ret"] for x in v]))
        for x in v:
            step_adv = (x["step_ret"] - m) if live else 0.0
            credit = (x["R"] - ep_mean) + STEP_ADV_W * step_adv
            out[(x["rollout"], x["si"])] = (credit, live)
    return out


def pairwise_auc(items):
    """items: list of (credit, oracle_label in {+1,-1}) within one trajectory."""
    goods = [c for c, o in items if o > 0]
    bads = [c for c, o in items if o < 0]
    if not goods or not bads:
        return None
    wins, ties, tot = 0, 0, 0
    for g in goods:
        for b in bads:
            tot += 1
            if g > b + 1e-12:
                wins += 1
            elif abs(g - b) <= 1e-12:
                ties += 1
    return (wins + 0.5 * ties) / tot


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

    tasks = []
    for p in RECORDS_LIST:
        p = p.strip()
        if p and os.path.exists(p):
            tasks += json.load(open(p))
    print(f"loaded {len(tasks)} tasks from {len(RECORDS_LIST)} files", flush=True)

    methods = ["grpo_sdar", "gigpo", "teca"]
    aucs = {m: [] for m in methods}          # per-traj AUC (labeled good&bad both present)
    aucs_pos = {m: [] for m in methods}      # same, positive-advantage trajs only
    sign = {m: {"cov": 0, "ok": 0, "lab": 0} for m in methods}
    gig_sign_flips = 0                       # labeled steps where sign(gigpo) != sign(grpo)
    gig_live_lab, gig_lab = 0, 0
    n_pos_traj, n_traj = 0, 0

    for task in tasks:
        rewards = torch.tensor([float(t["reward"]) for t in task["trajs"]])
        if rewards.std() < 1e-8:
            continue
        grp_adv = (rewards - rewards.mean()) / (rewards.std() + EPS)
        gig = gigpo_credits(task)

        for ti, tr in enumerate(task["trajs"]):
            A = float(grp_adv[ti])
            n_traj += 1
            boosts = {}
            if A > 0:
                n_pos_traj += 1
                boosts = teca_step_boosts(model, tok, skill_provider,
                                          task["instruction"], tr, device)

            items = {m: [] for m in methods}
            for si, st in enumerate(tr["steps"]):
                if not st.get("action"):
                    continue
                o = oracle(st["action"], st.get("next_obs_has_gold", False),
                           task["gold_asin"], task["goal_options"])
                credit = {
                    "grpo_sdar": A,
                    "gigpo": gig.get((tr["rollout"], si), (0.0, False))[0],
                    "teca": A + boosts.get(si, 0.0),
                }
                if o != 0:
                    gig_lab += 1
                    gig_live_lab += int(gig.get((tr["rollout"], si), (0.0, False))[1])
                    if np.sign(credit["gigpo"]) != np.sign(credit["grpo_sdar"]):
                        gig_sign_flips += 1
                    for m in methods:
                        c = credit[m]
                        items[m].append((c, o))
                        sign[m]["lab"] += 1
                        if abs(c) > 1e-9:
                            sign[m]["cov"] += 1
                            if np.sign(c) == o:
                                sign[m]["ok"] += 1
            for m in methods:
                a = pairwise_auc(items[m])
                if a is not None:
                    aucs[m].append(a)
                    if A > 0:
                        aucs_pos[m].append(a)
        print(f"task gid={task['gid']} done", flush=True)

    summary = {
        "n_tasks": len(tasks), "n_trajs_scored": n_traj, "n_positive_trajs": n_pos_traj,
        "n_trajs_with_good_and_bad_steps": len(aucs["grpo_sdar"]),
        "gigpo_live_anchor_frac_on_labeled_steps": round(gig_live_lab / max(gig_lab, 1), 4),
        "gigpo_sign_flip_frac_vs_grpo": round(gig_sign_flips / max(gig_lab, 1), 4),
        "methods": {},
    }
    for m in methods:
        cov = sign[m]["cov"] / max(sign[m]["lab"], 1)
        sacc = sign[m]["ok"] / max(sign[m]["cov"], 1)
        summary["methods"][m] = {
            "within_traj_pairwise_auc_mean": round(float(np.mean(aucs[m])), 4) if aucs[m] else None,
            "within_traj_auc_n": len(aucs[m]),
            "within_traj_auc_positive_trajs": round(float(np.mean(aucs_pos[m])), 4) if aucs_pos[m] else None,
            "within_traj_auc_positive_n": len(aucs_pos[m]),
            "sign_coverage": round(cov, 4),
            "sign_acc_of_covered": round(sacc, 4),
        }
    # paired bootstrap over trajectories for AUC differences
    rng = np.random.RandomState(0)
    arr = {m: np.array(aucs[m]) for m in methods}
    n = len(arr["grpo_sdar"])
    boot = {"teca_minus_grpo": [], "teca_minus_gigpo": []}
    for _ in range(10000):
        idx = rng.randint(0, n, n)
        boot["teca_minus_grpo"].append(arr["teca"][idx].mean() - arr["grpo_sdar"][idx].mean())
        boot["teca_minus_gigpo"].append(arr["teca"][idx].mean() - arr["gigpo"][idx].mean())
    for k, v in boot.items():
        v = np.array(v)
        summary[k] = {
            "mean": round(float(v.mean()), 4),
            "ci95": [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)],
            "p_leq_0": round(float((v <= 0).mean()), 4),
        }
    summary["per_traj_auc"] = {m: [round(float(a), 4) for a in aucs[m]] for m in methods}

    print(json.dumps({k: v for k, v in summary.items() if k != "per_traj_auc"}, indent=2))
    with open(OUT_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()

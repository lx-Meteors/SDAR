"""
Decompose WHERE the dh<0 ("student entropy > teacher entropy") bonus's AUC gain
comes from. The "student good exploration" hypothesis requires the student to
actually be uncertain (H_S visibly > 0) at the boosted position. The competing
explanation is tail/copy noise: near-deterministic positions (H_S ~ H_T ~ 0,
ASIN digits, '>', '</') where the target-excluded residual entropy is numerically
dominated, and which happen to pile up on click steps.

Variants (all: A + beta*bonus, positive trajs, top-1 tokens, per-traj top-20%):
  pos            dh > 0                      (current TECA, reference)
  neg            dh < 0                      (all)
  neg_uncertain  dh < 0 and H_S_full >= 0.3  (student genuinely hesitating)
  neg_certain    dh < 0 and H_S_full <  0.1  (student collapsed = copy/tail noise)
  abs            |dh| over all top-1 tokens  (both directions)
  abs_uncertain  dh>0 OR (dh<0 & H_S>=0.3)   (pos + genuine-hesitation part)
  pos_plus_negc  dh>0 OR (dh<0 & H_S<0.1)    (pos + noise part only)

Also reports composition of the plain-neg selected set by student-entropy band.
Judge: oracle within-trajectory pairwise AUC, as in offline_neg_dh_check.py.
"""

import json
import os
import re
import sys

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
    "/home/test/yyy/SDAR/tmp_scale/records_g8_fresh11.json,"
    "/home/test/yyy/SDAR/tmp_scale/records_g8_fresh23a.json,"
    "/home/test/yyy/SDAR/tmp_scale/records_g8_fresh23b.json,"
    "/home/test/yyy/SDAR/tmp_scale/records_g8_fresh23c.json",
).split(",")
SKILLS_DIR = "/home/test/yyy/SDAR/skills/webshop"
OUT_PATH = os.environ.get(
    "OUT_PATH", os.path.join(os.path.dirname(__file__), "neg_dh_decompose.json"))

TEMPERATURE = 1.0
BETA = 0.1
TOP_FRAC = 0.20
EPS = 1e-6
HS_HI = 0.3   # "student genuinely hesitating"
HS_LO = 0.1   # "student collapsed"

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


def full_entropy(logps):
    return -(logps.exp() * logps).sum(-1)


@torch.no_grad()
def traj_token_stats(model, tok, skill_provider, instruction, traj, device):
    steps = []
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
        hs_full = full_entropy(s_lp).cpu()
        top1 = (t_lp.argmax(-1).cpu() == r_ids) & (s_lp.argmax(-1).cpu() == r_ids)
        steps.append(dict(si=si, dh=dh, top1=top1, hs=hs_full, n=r_ids.numel()))
        history.append((action, st["obs"][:400]))
    return steps


def make_variant(steps, elig_fn, score_fn):
    scores, masks = [], []
    for s in steps:
        masks.append(s["top1"] & elig_fn(s))
        scores.append(score_fn(s))
    all_sc = torch.cat([sc[m] for sc, m in zip(scores, masks)]) if steps else torch.tensor([])
    boosts = {}
    if all_sc.numel() > 0:
        thr = torch.quantile(all_sc.float(), 1.0 - TOP_FRAC)
        for s, sc, m in zip(steps, scores, masks):
            sel = m & (sc >= thr)
            boosts[s["si"]] = BETA * float(sc[sel].clamp(min=0).sum()) / max(s["n"], 1)
    else:
        for s in steps:
            boosts[s["si"]] = 0.0
    return boosts


def pairwise_auc(items):
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


VARIANTS = {
    "pos":           (lambda s: s["dh"] > 0,                        lambda s: s["dh"]),
    "neg":           (lambda s: s["dh"] < 0,                        lambda s: -s["dh"]),
    "neg_uncertain": (lambda s: (s["dh"] < 0) & (s["hs"] >= HS_HI), lambda s: -s["dh"]),
    "neg_certain":   (lambda s: (s["dh"] < 0) & (s["hs"] < HS_LO),  lambda s: -s["dh"]),
    "abs":           (lambda s: torch.ones_like(s["top1"]),         lambda s: s["dh"].abs()),
    "abs_uncertain": (lambda s: (s["dh"] > 0) | ((s["dh"] < 0) & (s["hs"] >= HS_HI)),
                      lambda s: s["dh"].abs()),
    "pos_plus_negc": (lambda s: (s["dh"] > 0) | ((s["dh"] < 0) & (s["hs"] < HS_LO)),
                      lambda s: s["dh"].abs()),
}


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
    print(f"loaded {len(tasks)} tasks", flush=True)

    names = list(VARIANTS.keys())
    aucs_pos_only = {m: [] for m in names}
    comp = dict(neg_sel=0, neg_sel_hs_lo=0, neg_sel_hs_hi=0)

    for task in tasks:
        rewards = torch.tensor([float(t["reward"]) for t in task["trajs"]])
        if rewards.std() < 1e-8:
            continue
        grp_adv = (rewards - rewards.mean()) / (rewards.std() + EPS)

        for ti, tr in enumerate(task["trajs"]):
            A = float(grp_adv[ti])
            if A <= 0:
                continue
            steps = traj_token_stats(model, tok, skill_provider,
                                     task["instruction"], tr, device)
            boosts = {m: make_variant(steps, *VARIANTS[m]) for m in names}

            # composition of the plain-neg selected set by student-entropy band
            sc = torch.cat([(-s["dh"])[s["top1"] & (s["dh"] < 0)] for s in steps]) \
                if steps else torch.tensor([])
            if sc.numel():
                thr = torch.quantile(sc.float(), 1 - TOP_FRAC)
                for s in steps:
                    sel = s["top1"] & (s["dh"] < 0) & (-s["dh"] >= thr)
                    comp["neg_sel"] += int(sel.sum())
                    comp["neg_sel_hs_lo"] += int((sel & (s["hs"] < HS_LO)).sum())
                    comp["neg_sel_hs_hi"] += int((sel & (s["hs"] >= HS_HI)).sum())

            items = {m: [] for m in names}
            for si, st in enumerate(tr["steps"]):
                if not st.get("action"):
                    continue
                o = oracle(st["action"], st.get("next_obs_has_gold", False),
                           task["gold_asin"], task["goal_options"])
                if o == 0:
                    continue
                for m in names:
                    items[m].append((A + boosts[m].get(si, 0.0), o))
            for m in names:
                a = pairwise_auc(items[m])
                if a is not None:
                    aucs_pos_only[m].append(a)
        print(f"task gid={task['gid']} done", flush=True)

    summary = dict(
        n_tasks=len(tasks),
        hs_hi=HS_HI, hs_lo=HS_LO, beta=BETA, top_frac=TOP_FRAC,
        neg_selected_composition=dict(
            total=comp["neg_sel"],
            frac_student_collapsed=round(comp["neg_sel_hs_lo"] / max(comp["neg_sel"], 1), 4),
            frac_student_hesitant=round(comp["neg_sel_hs_hi"] / max(comp["neg_sel"], 1), 4),
        ),
        auc_positive_trajs={m: (round(float(np.mean(v)), 4) if v else None)
                            for m, v in aucs_pos_only.items()},
        n_trajs={m: len(v) for m, v in aucs_pos_only.items()},
    )
    # paired bootstrap of each variant against pos
    rng = np.random.RandomState(0)
    base = np.array(aucs_pos_only["pos"])
    summary["vs_pos_bootstrap"] = {}
    for m in names:
        if m == "pos" or not aucs_pos_only[m]:
            continue
        a_o = np.array(aucs_pos_only[m])
        n = min(len(base), len(a_o))
        diffs = []
        for _ in range(10000):
            idx = rng.randint(0, n, n)
            diffs.append(a_o[idx].mean() - base[idx].mean())
        diffs = np.array(diffs)
        summary["vs_pos_bootstrap"][m] = dict(
            mean=round(float(diffs.mean()), 4),
            ci95=[round(float(np.percentile(diffs, 2.5)), 4),
                  round(float(np.percentile(diffs, 97.5)), 4)],
            p_leq_0=round(float((diffs <= 0).mean()), 4),
        )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    with open(OUT_PATH, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()

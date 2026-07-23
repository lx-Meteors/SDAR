"""
Does the OPPOSITE direction (student entropy > teacher entropy, i.e. dh < 0)
also deserve an advantage bonus ("student's good exploration")?

Three shaping variants, all additive A + beta * bonus, positive trajs only,
top-1 agreed tokens, per-trajectory top-20% quota (same as TECA):

  teca_pos : bonus tokens = dh > 0, ranked by  dh    (current TECA)
  teca_neg : bonus tokens = dh < 0, ranked by -dh    (the proposal to test)
  teca_abs : bonus tokens ranked by |dh| (both directions)

Judge: oracle step labels (click gold / click wrong / search recalls gold or not)
-> within-trajectory pairwise AUC of the step credit (does the shaped credit
rank oracle-good steps above oracle-bad steps of the SAME trajectory?).
GRPO baseline is 0.5 by construction.

Extra token-level diagnostics for the dh<0 population on positive trajs:
  - where do dh_neg-selected tokens sit (think vs action segment)?
  - teacher entropy at those positions (is teacher ~deterministic, i.e. is
    -dh just student single-side entropy in disguise?)
  - correlation(-dh, student entropy) among selected dh<0 tokens
  - dump top examples (token, context window) for eyeballing.

Real rollouts, real env rewards. Offline check only.
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
    "/home/test/yyy/SDAR/tmp_scale/records_g8_fresh23a.json,"
    "/home/test/yyy/SDAR/tmp_scale/records_g8_fresh23b.json,"
    "/home/test/yyy/SDAR/tmp_scale/records_g8_fresh23c.json",
).split(",")
SKILLS_DIR = "/home/test/yyy/SDAR/skills/webshop"
OUT_PATH = os.path.join(os.path.dirname(__file__), "neg_dh_check.json")

TEMPERATURE = 1.0
BETA = 0.1
TOP_FRAC = 0.20
EPS = 1e-6
N_EXAMPLES = 40

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
    """Per-step token stats for one trajectory. Returns list of dicts per action step."""
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
        ht_full = full_entropy(t_lp).cpu()
        dh = ht_woy - hs_woy
        top1 = (t_lp.argmax(-1).cpu() == r_ids) & (s_lp.argmax(-1).cpu() == r_ids)
        steps.append(dict(
            si=si, dh=dh, top1=top1, hs_full=hs_full, ht_full=ht_full,
            n=r_ids.numel(), r_ids=r_ids, action_start=action_start,
        ))
        history.append((action, st["obs"][:400]))
    return steps


def step_boosts(steps, direction):
    """Per-step additive boost under a shaping direction.

    direction: 'pos' -> eligible dh>0 ranked by dh; 'neg' -> dh<0 ranked by -dh;
               'abs' -> all top-1 tokens, ranked by |dh|.
    Per-trajectory top-frac quota over eligible scores (matches TECA's quantile).
    boost(step) = BETA * sum(score of selected tokens) / n_tokens_of_step
    """
    scores, masks = [], []
    for s in steps:
        if direction == "pos":
            elig = s["top1"] & (s["dh"] > 0)
            score = s["dh"]
        elif direction == "neg":
            elig = s["top1"] & (s["dh"] < 0)
            score = -s["dh"]
        else:
            elig = s["top1"]
            score = s["dh"].abs()
        scores.append(score)
        masks.append(elig)
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

    variants = ["teca_pos", "teca_neg", "teca_abs"]
    aucs = {m: [] for m in variants}
    aucs_pos_only = {m: [] for m in variants}

    # dh<0 vs dh>0 selected-token population diagnostics on positive trajs
    diag = dict(neg_sel_action=0, neg_sel_total=0, pos_sel_action=0, pos_sel_total=0,
                ht_full_on_neg_sel=[], hs_full_on_neg_sel=[],
                ht_full_on_pos_sel=[], hs_full_on_pos_sel=[],
                corr_pairs=[])
    examples = []

    n_pos_traj = 0
    for task in tasks:
        rewards = torch.tensor([float(t["reward"]) for t in task["trajs"]])
        if rewards.std() < 1e-8:
            continue
        grp_adv = (rewards - rewards.mean()) / (rewards.std() + EPS)

        for ti, tr in enumerate(task["trajs"]):
            A = float(grp_adv[ti])
            boosts = {m: {} for m in variants}
            if A > 0:
                n_pos_traj += 1
                steps = traj_token_stats(model, tok, skill_provider,
                                         task["instruction"], tr, device)
                boosts["teca_pos"] = step_boosts(steps, "pos")
                boosts["teca_neg"] = step_boosts(steps, "neg")
                boosts["teca_abs"] = step_boosts(steps, "abs")

                # ---- population diagnostics ----
                sc_neg = torch.cat([(-s["dh"])[s["top1"] & (s["dh"] < 0)] for s in steps]) \
                    if steps else torch.tensor([])
                sc_pos = torch.cat([s["dh"][s["top1"] & (s["dh"] > 0)] for s in steps]) \
                    if steps else torch.tensor([])
                thr_neg = torch.quantile(sc_neg.float(), 1 - TOP_FRAC) if sc_neg.numel() else None
                thr_pos = torch.quantile(sc_pos.float(), 1 - TOP_FRAC) if sc_pos.numel() else None
                for s in steps:
                    if thr_neg is not None:
                        seln = s["top1"] & (s["dh"] < 0) & (-s["dh"] >= thr_neg)
                        diag["neg_sel_total"] += int(seln.sum())
                        diag["neg_sel_action"] += int((seln & (torch.arange(s["n"]) >= s["action_start"])).sum())
                        diag["ht_full_on_neg_sel"] += s["ht_full"][seln].tolist()
                        diag["hs_full_on_neg_sel"] += s["hs_full"][seln].tolist()
                        for j in seln.nonzero(as_tuple=True)[0].tolist():
                            diag["corr_pairs"].append((float(-s["dh"][j]), float(s["hs_full"][j])))
                            if len(examples) < N_EXAMPLES:
                                lo, hi = max(0, j - 6), min(s["n"], j + 3)
                                ctx = tok.decode(s["r_ids"][lo:hi])
                                examples.append(dict(
                                    tok=tok.decode(s["r_ids"][j:j+1]),
                                    neg_dh=round(float(-s["dh"][j]), 3),
                                    hs=round(float(s["hs_full"][j]), 3),
                                    ht=round(float(s["ht_full"][j]), 3),
                                    seg="action" if j >= s["action_start"] else "think",
                                    ctx=ctx.replace("\n", " ")[:120],
                                ))
                    if thr_pos is not None:
                        selp = s["top1"] & (s["dh"] > 0) & (s["dh"] >= thr_pos)
                        diag["pos_sel_total"] += int(selp.sum())
                        diag["pos_sel_action"] += int((selp & (torch.arange(s["n"]) >= s["action_start"])).sum())
                        diag["ht_full_on_pos_sel"] += s["ht_full"][selp].tolist()
                        diag["hs_full_on_pos_sel"] += s["hs_full"][selp].tolist()

            items = {m: [] for m in variants}
            for si, st in enumerate(tr["steps"]):
                if not st.get("action"):
                    continue
                o = oracle(st["action"], st.get("next_obs_has_gold", False),
                           task["gold_asin"], task["goal_options"])
                if o == 0:
                    continue
                for m in variants:
                    items[m].append((A + boosts[m].get(si, 0.0), o))
            for m in variants:
                a = pairwise_auc(items[m])
                if a is not None:
                    aucs[m].append(a)
                    if A > 0:
                        aucs_pos_only[m].append(a)
        print(f"task gid={task['gid']} done", flush=True)

    corr = None
    if len(diag["corr_pairs"]) >= 3:
        arr = np.array(diag["corr_pairs"])
        corr = float(np.corrcoef(arr[:, 0], arr[:, 1])[0, 1])

    summary = dict(
        n_tasks=len(tasks), n_positive_trajs=n_pos_traj,
        n_trajs_auc=len(aucs["teca_pos"]),
        beta=BETA, top_frac=TOP_FRAC,
        methods={},
        neg_population=dict(
            action_seg_frac=round(diag["neg_sel_action"] / max(diag["neg_sel_total"], 1), 4),
            mean_teacher_full_entropy=round(float(np.mean(diag["ht_full_on_neg_sel"])), 4)
                if diag["ht_full_on_neg_sel"] else None,
            mean_student_full_entropy=round(float(np.mean(diag["hs_full_on_neg_sel"])), 4)
                if diag["hs_full_on_neg_sel"] else None,
            corr_negdh_vs_student_entropy=round(corr, 4) if corr is not None else None,
            n_selected=diag["neg_sel_total"],
        ),
        pos_population=dict(
            action_seg_frac=round(diag["pos_sel_action"] / max(diag["pos_sel_total"], 1), 4),
            mean_teacher_full_entropy=round(float(np.mean(diag["ht_full_on_pos_sel"])), 4)
                if diag["ht_full_on_pos_sel"] else None,
            mean_student_full_entropy=round(float(np.mean(diag["hs_full_on_pos_sel"])), 4)
                if diag["hs_full_on_pos_sel"] else None,
            n_selected=diag["pos_sel_total"],
        ),
        examples_neg_dh=examples,
    )
    for m in variants:
        summary["methods"][m] = dict(
            auc_all=round(float(np.mean(aucs[m])), 4) if aucs[m] else None,
            auc_positive_trajs=round(float(np.mean(aucs_pos_only[m])), 4) if aucs_pos_only[m] else None,
            n_pos=len(aucs_pos_only[m]),
        )
    # paired bootstrap: pos vs neg / pos vs abs on positive trajs
    rng = np.random.RandomState(0)
    a_pos = np.array(aucs_pos_only["teca_pos"]) if aucs_pos_only["teca_pos"] else np.array([0.5])
    for other in ["teca_neg", "teca_abs"]:
        a_o = np.array(aucs_pos_only[other]) if aucs_pos_only[other] else np.array([0.5])
        n = min(len(a_pos), len(a_o))
        if n >= 3:
            diffs = []
            for _ in range(10000):
                idx = rng.randint(0, n, n)
                diffs.append(a_pos[idx].mean() - a_o[idx].mean())
            diffs = np.array(diffs)
            summary[f"pos_minus_{other.split('_')[1]}"] = dict(
                mean=round(float(diffs.mean()), 4),
                ci95=[round(float(np.percentile(diffs, 2.5)), 4),
                      round(float(np.percentile(diffs, 97.5)), 4)],
                p_leq_0=round(float((diffs <= 0).mean()), 4),
            )

    print(json.dumps({k: v for k, v in summary.items() if k != "examples_neg_dh"},
                     indent=2, ensure_ascii=False))
    with open(OUT_PATH, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()

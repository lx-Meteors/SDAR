"""
TECA (Teacher-Entropy Credit Assignment) utilities.

Long-horizon credit assignment via teacher-guided token-level advantage shaping.

Signal (delta_H): at each response position t with realized token y_t, form the
candidate set as the WHOLE vocabulary EXCLUDING the target token itself,
    C_t = V \\ {y_t},
renormalize teacher / student distributions over C_t and take their entropies
    H_T^{\\y}(C_t), H_S^{\\y}(C_t),   delta_H_t = H_T^{\\y} - H_S^{\\y}.
delta_H measures how much *wider* the teacher's set of remaining alternatives is
than the student's, once the actually-taken token is set aside.

Eligibility (top-1): the realized token y_t must be the full-vocab argmax of BOTH
the teacher and the student distributions (i.e. both agree y_t is top-1).

Shaping (additive only, no gating threshold, no clipping):
    A'_{i,t} = A_{i,t} + beta * delta_H_t     for eligible tokens of positive samples.

Two ways to obtain H^{\\y}:
  * entropy_excluding_target            -- exact, from full-vocab log-probs (offline / small batch)
  * entropy_excluding_target_closed_form -- from full entropy + target log-prob (training, no need
                                            to materialize / store top-k candidates)
The closed form uses, with p_y = p(y_t) and H_full the full-vocab entropy:
    H^{\\y} = (H_full + p_y * log p_y) / (1 - p_y) + log(1 - p_y).
"""

import torch


def entropy_excluding_target(full_logps: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    """Entropy of the distribution renormalized over V \\ {target}, from full log-probs.

    Args:
        full_logps: (..., V) log-softmaxed full-vocabulary log-probabilities.
        target_ids: (...) realized token id to exclude at each position.

    Returns:
        (...,) entropy in nats over the target-excluded, renormalized distribution.
    """
    masked = full_logps.clone()
    masked.scatter_(-1, target_ids.unsqueeze(-1), float("-inf"))
    norm = masked - torch.logsumexp(masked, dim=-1, keepdim=True)
    p = norm.exp()
    term = torch.where(p > 0, p * norm, torch.zeros_like(p))
    return -term.sum(-1)


def entropy_excluding_target_closed_form(
    full_entropy: torch.Tensor, target_logp: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """H^{\\y} from full entropy and target log-prob (avoids storing the vocab).

    Args:
        full_entropy: (...) full-vocabulary entropy H_full (nats).
        target_logp: (...) log p(y_t) of the realized token.
    """
    py = target_logp.exp().clamp(max=1.0 - eps)
    z = (1.0 - py).clamp(min=eps)
    return (full_entropy + py * target_logp) / z + torch.log(z)


def compute_teca_advantage(
    advantages: torch.Tensor,
    delta_h: torch.Tensor,
    teacher_argmax_ids: torch.Tensor,
    student_argmax_ids: torch.Tensor,
    responses: torch.Tensor,
    response_mask: torch.Tensor,
    beta: float = 1.0,
    positive_only: bool = True,
    require_top1: bool = True,
    positive_dh_only: bool = True,
    top_frac: float | None = 0.2,
    variant: str = "pos",
) -> tuple[torch.Tensor, dict]:
    """Additive teacher-entropy advantage shaping.

    Args:
        advantages: (bs, L) GRPO sequence-level advantage broadcast per token.
        delta_h: (bs, L) teacher-minus-student target-excluded candidate entropy gap.
        teacher_argmax_ids: (bs, L) teacher full-vocab argmax token ids.
        student_argmax_ids: (bs, L) student full-vocab argmax token ids.
        responses: (bs, L) realized response token ids.
        response_mask: (bs, L) valid-token mask.
        beta: shaping strength.
        positive_only: only shape tokens in sequences with positive advantage.
        require_top1: require realized token to be the argmax of BOTH distributions.
        positive_dh_only: (only used when variant="pos") rectify delta_H to
            max(delta_H, 0) so shaping only ever *adds* credit; tokens where the
            teacher's alternative set is narrower than the student's are left
            unchanged.
        top_frac: if not None, strong selectivity -- within each row, only the tokens
            whose rank-score is in the top `top_frac` fraction (among that row's
            already eligible tokens) are shaped. Concentrates credit onto the few
            most decisive tokens instead of the ~40% that pass a plain relu. None
            disables. The rank-score depends on `variant`.
        variant: which side(s) of delta_H are eligible and how credit is signed.
            * "pos" (default, unchanged behaviour): eligible = delta_H>0; rank by
              delta_H; credit = beta*relu(delta_H). Only rewards positions where the
              teacher is MORE spread over the non-target candidates than the student.
            * "abs" (v2_abs, two-sided positive): eligible = delta_H!=0; rank by
              |delta_H|; credit = beta*|delta_H|. ALSO gives POSITIVE credit to
              student-high-entropy (delta_H<0) tokens -- any large teacher/student
              candidate-entropy gap in EITHER direction is treated as informative.
            * "neg": eligible = delta_H<0; rank by -delta_H; credit = beta*(-delta_H).
              Rewards only student-high-entropy positions.

    Returns:
        shaped advantages (bs, L), metrics dict.
    """
    with torch.no_grad():
        mask = response_mask.bool()

        eligible = mask.clone()
        if require_top1:
            top1_agree = (teacher_argmax_ids == responses) & (student_argmax_ids == responses)
            eligible &= top1_agree
        if positive_only:
            eligible &= advantages > 0

        # direction filter + per-token rank score + effective (always-additive) dh
        if variant == "abs":
            eligible &= delta_h != 0
            rank_score = delta_h.abs()
            dh_eff = delta_h.abs()
        elif variant == "neg":
            eligible &= delta_h < 0
            rank_score = -delta_h
            dh_eff = (-delta_h).clamp(min=0.0)
        else:  # "pos" -- original behaviour
            if positive_dh_only:
                eligible &= delta_h > 0
            rank_score = delta_h
            dh_eff = delta_h.clamp(min=0.0) if positive_dh_only else delta_h

        if top_frac is not None and 0.0 < top_frac < 1.0:
            # per-row (1 - top_frac) quantile over that row's eligible rank-score
            keep = torch.zeros_like(eligible)
            for r in range(eligible.size(0)):
                row_elig = eligible[r]
                if row_elig.any():
                    thr = torch.quantile(rank_score[r][row_elig].float(), 1.0 - top_frac)
                    keep[r] = row_elig & (rank_score[r] >= thr)
            eligible = keep

        shaped = advantages + beta * dh_eff * eligible
        shaped = shaped * response_mask

        n_valid = mask.sum().clamp(min=1).float()
        n_eligible = eligible.sum().clamp(min=1).float()
        metrics = {
            "teca/eligible_token_frac": (eligible.sum().float() / n_valid).item(),
            "teca/delta_h_mean": ((delta_h * mask).sum() / n_valid).item(),
            "teca/delta_h_eligible_mean": ((delta_h * eligible).sum() / n_eligible).item(),
            "teca/dh_eff_eligible_mean": ((dh_eff * eligible).sum() / n_eligible).item(),
            "teca/adv_abs_change_mean": (((shaped - advantages).abs() * mask).sum() / n_valid).item(),
        }
        if require_top1:
            metrics["teca/top1_agree_frac"] = (
                (((teacher_argmax_ids == responses) & (student_argmax_ids == responses) & mask).sum().float() / n_valid).item()
            )

    return shaped, metrics

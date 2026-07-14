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
        positive_dh_only: rectify delta_H to max(delta_H, 0) so shaping only ever
            *adds* credit (reward-only); tokens where the teacher's alternative set
            is narrower than the student's are left unchanged.
        top_frac: if not None, strong selectivity -- within each row, only the tokens
            whose delta_H is in the top `top_frac` fraction (among that row's already
            eligible tokens) are shaped. Concentrates credit onto the few most
            decisive tokens instead of the ~40% that pass a plain relu. None disables.

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
        if positive_dh_only:
            eligible &= delta_h > 0

        if top_frac is not None and 0.0 < top_frac < 1.0:
            # per-row (1 - top_frac) quantile over that row's eligible delta_H
            keep = torch.zeros_like(eligible)
            for r in range(eligible.size(0)):
                row_elig = eligible[r]
                if row_elig.any():
                    thr = torch.quantile(delta_h[r][row_elig].float(), 1.0 - top_frac)
                    keep[r] = row_elig & (delta_h[r] >= thr)
            eligible = keep

        dh_eff = delta_h.clamp(min=0.0) if positive_dh_only else delta_h
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

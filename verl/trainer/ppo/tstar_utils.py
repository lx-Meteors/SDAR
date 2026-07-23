"""
t* redistribution loss (skill-teacher-guided GRPO).

Single target distribution that fuses the scalar outcome signal (GRPO) with a
skill-conditioned teacher's directional signal:

    t*_v  ∝  p_v^{1-γ} · q_v^{γ} · exp(A · 1[v == y])

where
    p = student policy π_θ(·|x, y_<t)          (full vocab, requires grad)
    q = skill-teacher π_θ(·|skill, x, y_<t)     (full vocab, detached)
    A = sequence outcome advantage (GRPO)       (scalar per token)
    y = realized token at this position

Training objective is the forward KL toward the (detached) target:

    L = KL(t* ‖ p) = Σ_v t*_v (log t*_v − log p_v)

whose gradient wrt the student logits is exactly  g = p − t*  (per position),
i.e. the update direction is  t* − p.  γ=0 recovers GRPO ( t* = softmax(logits
+ A·1[y]) whose CE gradient reduces to the standard A·(e_y − p) direction up to
the log-partition, see unit test).

This module is pure-tensor and framework-agnostic so it can be unit-tested on
CPU without the FSDP / vLLM stack.
"""

from __future__ import annotations

import torch


def build_tstar_log_target(
    log_p: torch.Tensor,
    log_q: torch.Tensor,
    adv: torch.Tensor,
    y_index: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Construct the (normalized) log target log t* over the full vocab.

    Args:
        log_p: (..., V) student log-softmax (detached is fine here; the target
               is always treated as a constant).
        log_q: (..., V) teacher log-softmax (detached).
        adv:   (...,)   scalar outcome advantage A per position.
        y_index: (...,) realized token id per position (long).
        gamma: teacher trust in [0, 1]. 0 -> pure outcome (GRPO-like) target.

    Returns:
        log_tstar: (..., V) normalized log target distribution.

    Two-block factorized form (menu-factorized): the two degrees of freedom
    of the conserved gradient are owned by different information sources,
    with no cross-talk.

      y-block (outcome only, teacher-free):
          t*_y = p_y·e^A / (p_y·e^A + 1 − p_y)          [= t0_y exactly]
      menu-block (teacher only; outcome enters solely through β(|A|)):
          t*_v = (1 − t*_y) · softmax_menu((1−β)·log p̃ + β·log q̃)_v

        β(A) = γ·(1 − e^{−|A|})

    Rationale: e^{A·1y} moves a fraction (1 − e^{−|A|}) of the realized
    token's mass, so β makes the teacher steer exactly the mass the outcome
    force moves — the teacher never adds force of its own. The target token
    uses NO teacher (q_y never enters t*_y — the outcome/return, the longest
    -horizon signal, owns the target); the teacher owns only the menu-internal
    allocation, whose sign does not flip with the outcome (the coherent,
    non-cancelling field: coherence 0.556 vs GRPO-net 0.197 on mixed-outcome
    tokens, ds_cancel.py). Consequence: the correction c = t* − t0 has
    c_y == 0 identically — the aux loss cannot touch the target coordinate.
    Properties: β(0)=0 ⇒ t*≡p (stationary; no standing pull toward the
    flatter q, which caused the smoke-run entropy runaway 1.07→3.27 in 7
    steps); sign-symmetric; single hyperparameter γ; γ=0 ⇒ exact GRPO t0.
    """
    beta = gamma * (1.0 - torch.exp(-adv.abs()))        # (...,)
    g = beta.unsqueeze(-1)
    yexp = y_index.unsqueeze(-1)
    a = adv.unsqueeze(-1)

    log_py = log_p.gather(-1, yexp)
    # teacher-mixed menu block, y excluded
    mixed = (1.0 - g) * log_p + g * log_q
    mixed = mixed.scatter(-1, yexp, torch.finfo(mixed.dtype).min)
    log_Zm = torch.logsumexp(mixed, dim=-1, keepdim=True)

    # outcome-only mass split between y and the menu (teacher-free)
    log_1mpy = torch.log1p(-log_py.exp().clamp(max=1.0 - 1e-6))
    log_Z = torch.logaddexp(log_py + a, log_1mpy)
    log_ty = log_py + a - log_Z
    log_1mty = log_1mpy - log_Z          # 1 − t*_y = (1 − p_y)/Z

    log_t = log_1mty + mixed - log_Zm
    log_t = log_t.scatter(-1, yexp, log_ty)
    return log_t


def tstar_correction_loss(
    student_logits: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    y_index: torch.Tensor,
    adv: torch.Tensor,
    gamma: float,
) -> tuple[torch.Tensor, dict]:
    """Pure-redistribution auxiliary loss (outcome force excluded).

    The outcome credit A·(e_y − p) is carried by the clipped PG channel.
    This loss carries ONLY the teacher-steering correction between the GRPO
    target t0 = softmax(log p + A·1y) and the full target t*:

        c = t* − t0            (sums to 0 per position; c ≡ 0 when γ=0 or A=0)
        L = −Σ_v c_v · log p_v (c detached)

    Since Σ_v c_v = 0, the logit gradient of L is exactly −c, i.e. the update
    moves p along the conserved correction and nothing else. γ=0 therefore
    recovers GRPO *exactly* (clean control arm), and no unclipped outcome
    force leaks around the PPO trust region.

    Args / returns mirror tstar_kl_loss; loss is per-position (N,).
    """
    log_p = torch.log_softmax(student_logits, dim=-1)
    with torch.no_grad():
        log_p_d = log_p.detach()
        log_tstar = build_tstar_log_target(log_p_d, teacher_log_probs, adv, y_index, gamma)
        log_t0 = build_tstar_log_target(log_p_d, teacher_log_probs, adv, y_index, 0.0)
        c = log_tstar.exp() - log_t0.exp()               # (N, V), zero-sum

    loss = -(c * log_p).sum(-1)                          # grad wrt logits = -c

    with torch.no_grad():
        p = log_p_d.exp()
        cy = c.gather(-1, y_index.unsqueeze(-1)).squeeze(-1)
        stats = {
            "tstar/corr_l1_mean": c.abs().sum(-1).mean(),   # redistribution size
            "tstar/corr_on_y_mean": cy.mean(),
            "tstar/py_mean": p.gather(-1, y_index.unsqueeze(-1)).squeeze(-1).mean(),
        }
    return loss, stats


def tstar_kl_loss(
    student_logits: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    y_index: torch.Tensor,
    adv: torch.Tensor,
    gamma: float,
) -> tuple[torch.Tensor, dict]:
    """Per-position forward-KL loss toward t* (target detached).

    Args:
        student_logits: (N, V) student logits (already temperature-scaled),
                        REQUIRES GRAD.
        teacher_log_probs: (N, V) teacher log-softmax, detached.
        y_index: (N,) realized token ids.
        adv: (N,) outcome advantage per position.
        gamma: teacher trust.

    Returns:
        loss_per_pos: (N,) KL(t* ‖ p) per position (>= 0, target detached).
        stats: dict of scalars for logging (no grad).
    """
    log_p = torch.log_softmax(student_logits, dim=-1)
    with torch.no_grad():
        log_q = teacher_log_probs
        log_tstar = build_tstar_log_target(log_p.detach(), log_q, adv, y_index, gamma)
        tstar = log_tstar.exp()

    # forward KL with detached target: minimizing this == cross-entropy H(t*, p)
    # up to the (constant) target entropy; gradient wrt logits is p - t*.
    ce = -(tstar * log_p).sum(-1)                       # H(t*, p)
    neg_ent = (tstar * log_tstar).sum(-1)               # -H(t*) (constant)
    kl = ce + neg_ent                                   # KL(t* || p) >= 0

    with torch.no_grad():
        p = log_p.exp()
        # mass moved off the realized token menu (L1/2 between t* and p)
        tv = 0.5 * (tstar - p).abs().sum(-1)
        py = p.gather(-1, y_index.unsqueeze(-1)).squeeze(-1)
        ty = tstar.gather(-1, y_index.unsqueeze(-1)).squeeze(-1)
        stats = {
            "tstar/kl_mean": kl.mean(),
            "tstar/tv_mean": tv.mean(),
            "tstar/dy_on_target_mean": (ty - py).mean(),  # >0 sharpen, <0 soften
        }
    return kl, stats

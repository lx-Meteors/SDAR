"""Environment-step latent-flow distillation utilities.

The teacher and student share weights, but the teacher additionally observes
privileged skill information.  Instead of matching absolute hidden states, we
match how their decision representations move between consecutive environment
steps in the same trajectory.  A detached privilege-relevance gate suppresses
transitions whose privileged teacher does not improve the likelihood of the
student-sampled action.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F


VALID_LATENT_FLOW_LAYERS = ("last", "last4", "all", "even", "odd")
VALID_PRIVILEGE_GATE_MODES = ("positive_tanh", "sigmoid")


def validate_latent_flow_layers(layers: str) -> str:
    if layers not in VALID_LATENT_FLOW_LAYERS:
        raise ValueError(f"latent_flow_layers must be one of {VALID_LATENT_FLOW_LAYERS}, got {layers!r}")
    return layers


def get_hidden_state_indices(num_hidden_states: int, layers: str) -> list[int]:
    """Return HF hidden-state tuple indices (0 is the embedding output)."""
    validate_latent_flow_layers(layers)
    if num_hidden_states < 2:
        raise ValueError(f"expected embedding plus at least one transformer layer, got {num_hidden_states}")
    if layers == "last":
        return [num_hidden_states - 1]
    if layers == "last4":
        first_layer = max(1, num_hidden_states - 4)
        return list(range(first_layer, num_hidden_states))
    if layers == "all":
        return list(range(1, num_hidden_states))

    num_layers = num_hidden_states - 1
    parity = 0 if layers == "even" else 1
    return [1 + layer_idx for layer_idx in range(parity, num_layers, 2)]


def build_next_step_indices(
    traj_uids: Sequence[object],
    turn_steps: Sequence[object],
) -> tuple[torch.LongTensor, torch.Tensor]:
    """Pair each row with the next environment step from the same trajectory.

    Rows without a successor point to themselves and receive a zero flow mask.
    Duplicate rows introduced by batch padding are harmless: the first matching
    successor is used and contains identical trajectory data.
    """
    if len(traj_uids) != len(turn_steps):
        raise ValueError("traj_uids and turn_steps must have the same length")

    first_index: dict[tuple[str, int], int] = {}
    normalized: list[tuple[str, int]] = []
    for idx, (uid, step) in enumerate(zip(traj_uids, turn_steps)):
        key = (str(uid), int(step))
        normalized.append(key)
        first_index.setdefault(key, idx)

    next_indices = torch.arange(len(normalized), dtype=torch.long)
    flow_mask = torch.zeros(len(normalized), dtype=torch.float32)
    for idx, (uid, step) in enumerate(normalized):
        successor = first_index.get((uid, step + 1))
        if successor is not None:
            next_indices[idx] = successor
            flow_mask[idx] = 1.0

    return next_indices, flow_mask


def compute_privilege_relevance_gate(
    teacher_log_probs: torch.Tensor,
    student_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    beta: float = 5.0,
    mode: str = "positive_tanh",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute a detached per-environment-step privilege relevance score.

    ``positive_tanh`` is zero unless privileged context improves the mean
    likelihood of the sampled action.  ``sigmoid`` reproduces SDAR's legacy
    gate shape.  ``beta <= 0`` disables gating and returns uniform weights.

    Returns:
        gate: ``(batch,)`` detached weights in ``[0, 1]``.
        mean_gap: ``(batch,)`` detached teacher-minus-student log-prob gap.
    """
    if mode not in VALID_PRIVILEGE_GATE_MODES:
        raise ValueError(f"gate mode must be one of {VALID_PRIVILEGE_GATE_MODES}, got {mode!r}")
    if teacher_log_probs.shape != student_log_probs.shape or teacher_log_probs.shape != response_mask.shape:
        raise ValueError(
            "teacher_log_probs, student_log_probs, and response_mask must have identical shapes; "
            f"got {teacher_log_probs.shape}, {student_log_probs.shape}, {response_mask.shape}"
        )

    mask = response_mask.to(dtype=teacher_log_probs.dtype)
    denom = mask.sum(dim=-1).clamp_min(1.0)
    mean_gap = (((teacher_log_probs.detach() - student_log_probs.detach()) * mask).sum(dim=-1) / denom).detach()

    if beta <= 0:
        gate = torch.ones_like(mean_gap)
    elif mode == "positive_tanh":
        gate = torch.tanh(beta * mean_gap).clamp_min(0.0)
    else:
        gate = torch.sigmoid(beta * mean_gap)
    return gate.detach(), mean_gap


def compute_latent_flow_loss(
    student_current_repr: torch.Tensor,
    student_next_repr: torch.Tensor,
    teacher_flow: torch.Tensor,
    flow_mask: torch.Tensor,
    privilege_gate: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Align environment-step decision-flow directions with gated cosine loss.

    Representations may be ``(B, D)`` for one layer or ``(B, L, D)`` for a
    layer subset.  The loss first averages over layers, then over valid source
    environment steps.  The denominator is the number of valid transitions,
    so the relevance gate controls both selection and auxiliary-loss strength.
    """
    if student_current_repr.shape != student_next_repr.shape:
        raise ValueError(
            "current and next student representations must have identical shapes; "
            f"got {student_current_repr.shape} and {student_next_repr.shape}"
        )
    if student_current_repr.shape != teacher_flow.shape:
        raise ValueError(
            "student representations and teacher flow must have identical shapes; "
            f"got {student_current_repr.shape} and {teacher_flow.shape}"
        )
    if student_current_repr.dim() not in (2, 3):
        raise ValueError(f"expected (B,D) or (B,L,D) representations, got {student_current_repr.shape}")

    student_flow = student_next_repr.float() - student_current_repr.float()
    teacher_flow_fp32 = teacher_flow.detach().float()

    student_norm = student_flow.norm(dim=-1)
    teacher_norm = teacher_flow_fp32.norm(dim=-1)
    cosine = F.cosine_similarity(student_flow, teacher_flow_fp32, dim=-1, eps=eps)
    valid_norm = (student_norm > eps) & (teacher_norm > eps)

    if cosine.dim() == 2:
        layer_valid = valid_norm.float()
        per_sample_cosine = (cosine * layer_valid).sum(dim=-1) / layer_valid.sum(dim=-1).clamp_min(1.0)
        per_sample_has_signal = valid_norm.any(dim=-1)
    else:
        per_sample_cosine = cosine
        per_sample_has_signal = valid_norm

    mask = flow_mask.float() * per_sample_has_signal.float()
    gate = privilege_gate.detach().float().clamp(0.0, 1.0)
    per_sample_loss = 1.0 - per_sample_cosine
    valid_count = mask.sum().clamp_min(1.0)
    loss = (per_sample_loss * gate * mask).sum() / valid_count

    with torch.no_grad():
        effective_weight = gate * mask
        metrics = {
            "latent_flow/loss": loss.detach().item(),
            "latent_flow/cosine": ((per_sample_cosine * mask).sum() / valid_count).item(),
            "latent_flow/gated_cosine": (
                (per_sample_cosine * effective_weight).sum() / effective_weight.sum().clamp_min(1.0)
            ).item(),
            "latent_flow/gate_mean": ((gate * mask).sum() / valid_count).item(),
            "latent_flow/gate_active_ratio": (((gate > 0).float() * mask).sum() / valid_count).item(),
            "latent_flow/valid_transitions": mask.sum().item(),
            "latent_flow/student_flow_norm": (
                (student_norm.mean(dim=-1) if student_norm.dim() == 2 else student_norm) * mask
            ).sum().div(valid_count).item(),
            "latent_flow/teacher_flow_norm": (
                (teacher_norm.mean(dim=-1) if teacher_norm.dim() == 2 else teacher_norm) * mask
            ).sum().div(valid_count).item(),
        }

    return loss, metrics

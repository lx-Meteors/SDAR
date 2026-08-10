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


def left_pad_student_inputs(
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    target_sequence_length: int,
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Left-pad a student batch without changing any valid token or RoPE ID.

    The extra tensor slots let a teacher prepend privileged tokens while the
    original prompt, decision anchor, and response remain at identical tensor
    indices in both forwards.
    """
    if input_ids.dim() != 2 or attention_mask.shape != input_ids.shape:
        raise ValueError("input_ids and attention_mask must be same-shaped rank-2 tensors")
    if position_ids.dim() != 2 or position_ids.shape != input_ids.shape:
        raise ValueError("text position_ids must have the same rank-2 shape as input_ids")

    pad_length = target_sequence_length - input_ids.size(1)
    if pad_length < 0:
        raise ValueError(
            f"target_sequence_length={target_sequence_length} is shorter than "
            f"student sequence length {input_ids.size(1)}"
        )
    if pad_length == 0:
        return input_ids, attention_mask, position_ids

    batch_size = input_ids.size(0)
    id_padding = input_ids.new_full((batch_size, pad_length), pad_token_id)
    mask_padding = attention_mask.new_zeros((batch_size, pad_length))
    position_padding = position_ids.new_zeros((batch_size, pad_length))
    return (
        torch.cat([id_padding, input_ids], dim=1),
        torch.cat([mask_padding, attention_mask], dim=1),
        torch.cat([position_padding, position_ids], dim=1),
    )


def shift_valid_position_ids(
    position_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    """Shift every valid token in each row by one nonnegative RoPE offset."""
    if position_ids.dim() != 2 or attention_mask.shape != position_ids.shape:
        raise ValueError("position_ids and attention_mask must be same-shaped rank-2 tensors")
    if offsets.dim() != 1 or offsets.size(0) != position_ids.size(0):
        raise ValueError(
            f"offsets must have shape ({position_ids.size(0)},), got {tuple(offsets.shape)}"
        )
    if (offsets < 0).any():
        raise ValueError("position offsets must be nonnegative")

    return position_ids + offsets.to(position_ids.device, position_ids.dtype).unsqueeze(1) * (
        attention_mask != 0
    ).to(position_ids.dtype)


def validate_decision_slot_alignment(
    *,
    student_input_ids: torch.Tensor,
    student_attention_mask: torch.Tensor,
    student_position_ids: torch.Tensor,
    teacher_input_ids: torch.Tensor,
    teacher_attention_mask: torch.Tensor,
    teacher_position_ids: torch.Tensor,
    response_length: int,
) -> int:
    """Require full student content to occupy identical teacher tensor slots."""
    named_tensors = {
        "student_input_ids": student_input_ids,
        "student_attention_mask": student_attention_mask,
        "student_position_ids": student_position_ids,
        "teacher_input_ids": teacher_input_ids,
        "teacher_attention_mask": teacher_attention_mask,
        "teacher_position_ids": teacher_position_ids,
    }
    reference_shape = student_input_ids.shape
    for name, tensor in named_tensors.items():
        if tensor.dim() != 2 or tensor.shape != reference_shape:
            raise ValueError(
                f"{name} must have shape {reference_shape} for latent position alignment, "
                f"got {tensor.shape}"
            )

    sequence_length = reference_shape[1]
    if response_length <= 0 or response_length >= sequence_length:
        raise ValueError(
            f"response_length must be in [1, {sequence_length - 1}], got {response_length}"
        )
    decision_index = sequence_length - response_length - 1

    student_valid = student_attention_mask[:, decision_index].bool()
    teacher_valid = teacher_attention_mask[:, decision_index].bool()
    invalid_rows = (~student_valid | ~teacher_valid).nonzero(as_tuple=True)[0]
    if invalid_rows.numel() > 0:
        raise ValueError(
            "decision anchor is padding for teacher or student at rows "
            f"{invalid_rows[:8].tolist()}"
        )

    token_mismatch = (
        student_input_ids[:, decision_index] != teacher_input_ids[:, decision_index]
    ).nonzero(as_tuple=True)[0]
    if token_mismatch.numel() > 0:
        raise ValueError(
            "teacher/student decision token IDs differ at rows "
            f"{token_mismatch[:8].tolist()}"
        )

    # Every valid student prompt token must remain in the same tensor slot in
    # the teacher.  Teacher-only skill tokens may occupy student padding slots.
    student_prompt_mask = student_attention_mask[:, : decision_index + 1].bool()
    teacher_prompt_mask = teacher_attention_mask[:, : decision_index + 1].bool()
    missing_prompt_tokens = (student_prompt_mask & ~teacher_prompt_mask).any(dim=-1).nonzero(
        as_tuple=True
    )[0]
    if missing_prompt_tokens.numel() > 0:
        raise ValueError(
            "teacher masked valid student prompt slots at rows "
            f"{missing_prompt_tokens[:8].tolist()}"
        )
    prompt_token_mismatch = (
        (student_input_ids[:, : decision_index + 1] != teacher_input_ids[:, : decision_index + 1])
        & student_prompt_mask
    ).any(dim=-1).nonzero(as_tuple=True)[0]
    if prompt_token_mismatch.numel() > 0:
        raise ValueError(
            "teacher changed student prompt tokens at rows "
            f"{prompt_token_mismatch[:8].tolist()}"
        )

    response_mask_mismatch = (
        student_attention_mask[:, decision_index + 1 :]
        != teacher_attention_mask[:, decision_index + 1 :]
    ).any(dim=-1).nonzero(as_tuple=True)[0]
    if response_mask_mismatch.numel() > 0:
        raise ValueError(
            "teacher/student response masks differ at rows "
            f"{response_mask_mismatch[:8].tolist()}"
        )

    response_token_mismatch = (
        student_input_ids[:, decision_index + 1 :] != teacher_input_ids[:, decision_index + 1 :]
    ).any(dim=-1).nonzero(as_tuple=True)[0]
    if response_token_mismatch.numel() > 0:
        raise ValueError(
            "teacher/student response token slots differ at rows "
            f"{response_token_mismatch[:8].tolist()}"
        )

    # Masked student padding deliberately has no meaningful RoPE ID. Every
    # actual student token, however, must use exactly the teacher's position ID.
    valid_position_mismatch = (
        (student_position_ids != teacher_position_ids) & student_attention_mask.bool()
    ).any(dim=-1).nonzero(as_tuple=True)[0]
    if valid_position_mismatch.numel() > 0:
        raise ValueError(
            "teacher/student valid-token position IDs differ at rows "
            f"{valid_position_mismatch[:8].tolist()}"
        )

    return decision_index


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

import pytest
import torch

from verl.trainer.ppo.latent_flow_utils import (
    build_next_step_indices,
    compute_latent_flow_loss,
    compute_privilege_relevance_gate,
    get_hidden_state_indices,
    validate_decision_semantic_alignment,
)


def test_build_next_step_indices_handles_interleaved_trajectories_and_terminals():
    next_indices, flow_mask = build_next_step_indices(
        traj_uids=["a", "b", "a", "b", "a"],
        turn_steps=[0, 0, 1, 1, 2],
    )

    assert next_indices.tolist() == [2, 3, 4, 3, 4]
    assert flow_mask.tolist() == [1.0, 1.0, 1.0, 0.0, 0.0]


def test_positive_tanh_gate_selects_only_teacher_improvements():
    teacher = torch.tensor([[-0.5, -0.5], [-2.0, -2.0], [-1.0, -9.0]])
    student = torch.tensor([[-1.0, -1.0], [-1.0, -1.0], [-2.0, -3.0]])
    mask = torch.tensor([[1.0, 1.0], [1.0, 1.0], [1.0, 0.0]])

    gate, gap = compute_privilege_relevance_gate(teacher, student, mask, beta=2.0)

    assert gap.tolist() == pytest.approx([0.5, -1.0, 1.0])
    assert gate[0].item() == pytest.approx(torch.tanh(torch.tensor(1.0)).item())
    assert gate[1].item() == 0.0
    assert gate[2].item() == pytest.approx(torch.tanh(torch.tensor(2.0)).item())


def test_nonpositive_beta_disables_relevance_gating():
    values = torch.zeros(2, 3)
    gate, _ = compute_privilege_relevance_gate(values, values, torch.ones_like(values), beta=0.0)
    assert torch.equal(gate, torch.ones(2))


def test_latent_flow_loss_is_zero_for_aligned_flow_and_masks_terminals():
    current = torch.tensor([[0.0, 0.0], [10.0, 10.0]])
    next_repr = torch.tensor([[1.0, 0.0], [0.0, 10.0]], requires_grad=True)
    teacher_flow = torch.tensor([[2.0, 0.0], [0.0, 1.0]])
    flow_mask = torch.tensor([1.0, 0.0])
    gate = torch.tensor([1.0, 1.0])

    loss, metrics = compute_latent_flow_loss(current, next_repr, teacher_flow, flow_mask, gate)

    assert loss.item() == pytest.approx(0.0, abs=1e-6)
    assert metrics["latent_flow/valid_transitions"] == 1.0
    loss.backward()
    assert next_repr.grad is not None


def test_multilayer_flow_loss_averages_layers_and_applies_gate_strength():
    current = torch.zeros(2, 2, 2)
    next_repr = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [1.0, 0.0]],
        ]
    )
    teacher_flow = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.0, 1.0], [0.0, 1.0]],
        ]
    )

    loss, _ = compute_latent_flow_loss(
        current,
        next_repr,
        teacher_flow,
        flow_mask=torch.ones(2),
        privilege_gate=torch.tensor([1.0, 0.5]),
    )

    # Sample 0 loss is 0; sample 1 has cosine 0 on both layers and gate 0.5.
    assert loss.item() == pytest.approx(0.25, abs=1e-6)


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("last", [4]),
        ("last4", [1, 2, 3, 4]),
        ("all", [1, 2, 3, 4]),
        ("even", [1, 3]),
        ("odd", [2, 4]),
    ],
)
def test_get_hidden_state_indices(mode, expected):
    assert get_hidden_state_indices(5, mode) == expected


def test_last4_uses_every_layer_when_model_has_fewer_than_four_layers():
    assert get_hidden_state_indices(3, "last4") == [1, 2]


def test_semantic_alignment_accepts_different_slots_and_rope_ids():
    student_mask = torch.tensor([[0, 0, 1, 1, 1, 1]])
    teacher_mask = torch.tensor([[0, 0, 1, 1, 1, 1, 1, 1]])
    student_ids = torch.tensor([[0, 0, 7, 8, 9, 10]])
    teacher_ids = torch.tensor([[0, 0, 5, 6, 7, 8, 9, 10]])
    student_positions = torch.tensor([[0, 0, 0, 1, 2, 3]])
    teacher_positions = torch.tensor([[0, 0, 0, 1, 2, 3, 4, 5]])

    indices = validate_decision_semantic_alignment(
        student_input_ids=student_ids,
        student_attention_mask=student_mask,
        student_position_ids=student_positions,
        teacher_input_ids=teacher_ids,
        teacher_attention_mask=teacher_mask,
        teacher_position_ids=teacher_positions,
        response_length=2,
    )
    assert indices == (3, 5)


def test_semantic_alignment_rejects_changed_decision_token():
    student_ids = torch.tensor([[0, 7, 8, 9]])
    teacher_ids = torch.tensor([[5, 6, 7, 3, 9]])

    with pytest.raises(ValueError, match="decision token IDs differ"):
        validate_decision_semantic_alignment(
            student_input_ids=student_ids,
            student_attention_mask=torch.tensor([[0, 1, 1, 1]]),
            student_position_ids=torch.tensor([[0, 0, 1, 2]]),
            teacher_input_ids=teacher_ids,
            teacher_attention_mask=torch.ones(1, 5),
            teacher_position_ids=torch.tensor([[0, 1, 2, 3, 4]]),
            response_length=1,
        )

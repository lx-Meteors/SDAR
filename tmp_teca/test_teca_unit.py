"""Unit tests for TECA: target-excluded entropy math, additive shaping, dp_actor forward."""

import sys

import torch
from omegaconf import OmegaConf

sys.path.insert(0, "/home/test/yyy/SDAR")

from verl.trainer.ppo.teca_utils import (  # noqa: E402
    compute_teca_advantage,
    entropy_excluding_target,
    entropy_excluding_target_closed_form,
)


def test_entropy_excluding_target():
    """Direct mask-based H^{\\y} must match the naive renormalized reference,
    and the closed form (from full entropy + target logp) must match it too."""
    torch.manual_seed(0)
    logits = torch.randn(4, 7, 200)
    logps = torch.log_softmax(logits, dim=-1)
    target = torch.randint(0, 200, (4, 7))

    h = entropy_excluding_target(logps, target)

    # naive reference: zero out target prob, renormalize, entropy
    p = logps.exp().clone()
    p.scatter_(-1, target.unsqueeze(-1), 0.0)
    p = p / p.sum(-1, keepdim=True)
    h_ref = -(p * p.clamp_min(1e-30).log()).sum(-1)
    assert torch.allclose(h, h_ref, atol=1e-4), (h - h_ref).abs().max()

    # closed form from full entropy + target logp
    full_entropy = -(logps.exp() * logps).sum(-1)
    target_logp = torch.gather(logps, -1, target.unsqueeze(-1)).squeeze(-1)
    h_cf = entropy_excluding_target_closed_form(full_entropy, target_logp)
    assert torch.allclose(h, h_cf, atol=1e-3), (h - h_cf).abs().max()
    print("test_entropy_excluding_target OK; max|mask-cf| =",
          round((h - h_cf).abs().max().item(), 6))


def test_compute_teca_advantage():
    bs, L = 2, 6
    # delta_H: positive at some positions, negative at others
    delta_h = torch.tensor([
        [0.0, 0.8, 0.4, -0.5, 0.2, 0.6],
        [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    ])
    responses = torch.zeros(bs, L, dtype=torch.long)
    t_arg = torch.zeros(bs, L, dtype=torch.long)
    s_arg = torch.zeros(bs, L, dtype=torch.long)
    t_arg[0, 1] = 9  # break top-1 agreement at (0,1) -> not eligible

    adv = torch.ones(bs, L)
    adv[1] = -1.0  # negative sample
    mask = torch.ones(bs, L)
    mask[0, -1] = 0  # padded

    shaped, metrics = compute_teca_advantage(
        advantages=adv, delta_h=delta_h,
        teacher_argmax_ids=t_arg, student_argmax_ids=s_arg, responses=responses,
        response_mask=mask, beta=0.5, positive_only=True, require_top1=True,
        positive_dh_only=True, top_frac=None,
    )
    # (0,1): top-1 broken -> unchanged
    assert shaped[0, 1] == 1.0, shaped[0]
    # (0,2): eligible, dh=0.4 -> 1 + 0.5*0.4 = 1.2
    assert abs(shaped[0, 2].item() - 1.2) < 1e-5, shaped[0, 2]
    # (0,3): dh<0, relu -> unchanged
    assert shaped[0, 3] == 1.0, shaped[0, 3]
    # (0,0): dh=0 -> unchanged
    assert shaped[0, 0] == 1.0
    # (0,5): masked
    assert shaped[0, 5] == 0.0
    # negative sample untouched
    assert (shaped[1] == -1.0).all(), shaped[1]

    # without relu, negative dh subtracts
    shaped2, _ = compute_teca_advantage(
        advantages=adv, delta_h=delta_h,
        teacher_argmax_ids=t_arg, student_argmax_ids=s_arg, responses=responses,
        response_mask=mask, beta=0.5, positive_only=True, require_top1=True,
        positive_dh_only=False, top_frac=None,
    )
    assert abs(shaped2[0, 3].item() - (1 + 0.5 * -0.5)) < 1e-5, shaped2[0, 3]

    # top_frac selectivity: row 0 eligible dh values are {0.0(pos0 excluded, dh=0 not>0),
    # 0.4(pos2), 0.2(pos4), 0.6(pos5 masked-> excluded)}. Eligible (top1 & A>0 & dh>0)
    # positions of row 0 (mask drops pos5): pos2(0.4), pos4(0.2). top_frac=0.5 keeps the
    # higher one only (pos2=0.4).
    shaped3, _ = compute_teca_advantage(
        advantages=adv, delta_h=delta_h,
        teacher_argmax_ids=t_arg, student_argmax_ids=s_arg, responses=responses,
        response_mask=mask, beta=0.5, positive_only=True, require_top1=True,
        positive_dh_only=True, top_frac=0.5,
    )
    assert abs(shaped3[0, 2].item() - 1.2) < 1e-5, shaped3[0, 2]  # top dh kept
    assert shaped3[0, 4].item() == 1.0, shaped3[0, 4]             # lower dh dropped by top_frac
    print("test_compute_teca_advantage OK  metrics:", {k: round(v, 4) for k, v in metrics.items()})


@torch.no_grad()
def test_dp_actor_teca_stats():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from verl.workers.actor.dp_actor import DataParallelPPOActor

    model_path = "/home/test/models/Qwen2.5-3B-Instruct"
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="flash_attention_2"
    )
    model.eval()

    texts = ["The capital of France is Paris. The capital of Germany is",
             "1+1=2. 2+2=4. 3+3="]
    resp_texts = [" Berlin, and the capital of Italy is Rome.", "6. 4+4=8."]

    prompt_len, resp_len = 24, 16
    input_ids, attn, resp = [], [], []
    for t, r in zip(texts, resp_texts):
        p_ids = tok.encode(t)[:prompt_len]
        r_ids = tok.encode(r)[:resp_len]
        pad_p = prompt_len - len(p_ids)
        pad_r = resp_len - len(r_ids)
        ids = [tok.pad_token_id] * pad_p + p_ids + r_ids + [tok.pad_token_id] * pad_r
        m = [0] * pad_p + [1] * (len(p_ids) + len(r_ids)) + [0] * pad_r
        input_ids.append(ids)
        attn.append(m)
        resp.append(r_ids + [tok.pad_token_id] * pad_r)

    input_ids = torch.tensor(input_ids, device="cuda")
    attn = torch.tensor(attn, device="cuda")
    resp = torch.tensor(resp, device="cuda")
    pos = torch.clip(torch.cumsum(attn, dim=-1) - 1, min=0)
    micro = {"input_ids": input_ids, "attention_mask": attn, "position_ids": pos, "responses": resp}

    cfg = OmegaConf.create({
        "use_remove_padding": True, "use_fused_kernels": False,
        "ulysses_sequence_parallel_size": 1, "use_torch_compile": False, "grad_clip": 1.0,
    })
    actor = DataParallelPPOActor(config=cfg, actor_module=model)
    out = actor._forward_micro_batch_teca_stats(micro, temperature=1.0)

    # reference: per-sample unpadded HF forward (training uses rmpad only)
    valid = attn[:, -resp_len:].bool()
    max_ent_diff, max_realized_diff, argmax_agree, n_pos = 0.0, 0.0, 0, 0
    for i in range(input_ids.size(0)):
        m = attn[i].bool()
        ids_i = input_ids[i][m].unsqueeze(0)
        logits_i = model(input_ids=ids_i).logits.float()[0]
        n_r = int(valid[i].sum())
        ref_logits = logits_i[-n_r - 1: -1]  # (n_r, V)
        ref_logps = torch.log_softmax(ref_logits, dim=-1)

        ref_entropy = -(ref_logps.exp() * ref_logps).sum(-1)
        max_ent_diff = max(max_ent_diff, (ref_entropy - out["full_entropy"][i, :n_r]).abs().max().item())

        realized_ref = torch.gather(ref_logps, -1, resp[i, :n_r].unsqueeze(-1)).squeeze(-1)
        max_realized_diff = max(max_realized_diff, (realized_ref - out["realized_log_probs"][i, :n_r]).abs().max().item())

        argmax_agree += (ref_logps.argmax(-1) == out["argmax_ids"][i, :n_r]).sum().item()
        n_pos += n_r

    agree = argmax_agree / n_pos
    assert max_ent_diff < 0.2, f"entropy: {max_ent_diff}"
    assert max_realized_diff < 0.2, f"realized: {max_realized_diff}"
    assert agree > 0.9, agree
    print(f"test_dp_actor_teca_stats OK; entropy diff {max_ent_diff:.4f}, "
          f"realized diff {max_realized_diff:.4f}, argmax agreement {agree:.3f}")


@torch.no_grad()
def test_fused_teca_stats_path():
    """The fused path (_forward_micro_batch with calculate_teca_stats=True, used by
    compute_log_prob to spare a dedicated student forward) must return exactly the
    same stats as the standalone _forward_micro_batch_teca_stats, and identical
    log_probs to the plain path."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from verl.workers.actor.dp_actor import DataParallelPPOActor

    model_path = "/home/test/models/Qwen2.5-3B-Instruct"
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="flash_attention_2"
    )
    model.eval()

    texts = ["The capital of France is Paris. The capital of Germany is",
             "1+1=2. 2+2=4. 3+3="]
    resp_texts = [" Berlin, and the capital of Italy is Rome.", "6. 4+4=8."]

    prompt_len, resp_len = 24, 16
    input_ids, attn, resp = [], [], []
    for t, r in zip(texts, resp_texts):
        p_ids = tok.encode(t)[:prompt_len]
        r_ids = tok.encode(r)[:resp_len]
        pad_p = prompt_len - len(p_ids)
        pad_r = resp_len - len(r_ids)
        ids = [tok.pad_token_id] * pad_p + p_ids + r_ids + [tok.pad_token_id] * pad_r
        m = [0] * pad_p + [1] * (len(p_ids) + len(r_ids)) + [0] * pad_r
        input_ids.append(ids)
        attn.append(m)
        resp.append(r_ids + [tok.pad_token_id] * pad_r)

    input_ids = torch.tensor(input_ids, device="cuda")
    attn = torch.tensor(attn, device="cuda")
    resp = torch.tensor(resp, device="cuda")
    pos = torch.clip(torch.cumsum(attn, dim=-1) - 1, min=0)
    micro = {"input_ids": input_ids, "attention_mask": attn, "position_ids": pos, "responses": resp}

    cfg = OmegaConf.create({
        "use_remove_padding": True, "use_fused_kernels": False,
        "ulysses_sequence_parallel_size": 1, "use_torch_compile": False, "grad_clip": 1.0,
    })
    actor = DataParallelPPOActor(config=cfg, actor_module=model)

    ref_stats = actor._forward_micro_batch_teca_stats(micro, temperature=1.0)
    ent_plain, lp_plain = actor._forward_micro_batch(micro, temperature=1.0, calculate_entropy=True)
    ent_fused, lp_fused, fused_stats = actor._forward_micro_batch(
        micro, temperature=1.0, calculate_entropy=True, calculate_teca_stats=True
    )

    valid = attn[:, -resp_len:].bool()
    for key in ("full_entropy", "realized_log_probs"):
        d = ((ref_stats[key] - fused_stats[key]).abs() * valid).max().item()
        assert d < 1e-3, f"{key}: fused vs standalone max diff {d}"
    assert (ref_stats["argmax_ids"][valid] == fused_stats["argmax_ids"][valid]).all()

    d_lp = ((lp_plain - lp_fused).abs() * valid).max().item()
    assert d_lp < 1e-6, f"log_probs changed by fused path: {d_lp}"
    # fused entropy is the fp32 TECA entropy; must agree with the plain bf16 one
    d_ent = ((ent_plain.float() - ent_fused).abs() * valid).max().item()
    assert d_ent < 0.05, f"entropy fused vs plain: {d_ent}"
    # realized_log_probs must equal the actual old_log_probs (same forward)
    d_real = ((fused_stats["realized_log_probs"] - lp_fused.float()).abs() * valid).max().item()
    assert d_real < 1e-2, f"realized vs log_probs: {d_real}"
    print(f"test_fused_teca_stats_path OK; lp diff {d_lp:.2e}, ent diff {d_ent:.4f}, "
          f"realized-vs-lp {d_real:.2e}")


if __name__ == "__main__":
    test_entropy_excluding_target()
    test_compute_teca_advantage()
    test_dp_actor_teca_stats()
    test_fused_teca_stats_path()
    print("ALL TESTS PASSED")

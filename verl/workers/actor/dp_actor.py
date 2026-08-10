# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process Actor
"""

import itertools
import time
import logging
import os
from typing import Tuple

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, compute_policy_loss, compute_policy_loss_gspo, kl_penalty
from verl.utils.debug import GPUMemoryLogger
from verl.utils.device import get_device_name, get_torch_device, is_cuda_available, is_npu_available
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outpus_and_unpad, ulysses_pad_and_slice_inputs, ulysses_pad
from verl.workers.actor import BasePPOActor

if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input


__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOActor(BasePPOActor):
    def __init__(self, config, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        print(f"Actor use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        print(f"Actor use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        self.compute_entropy_from_logits = (
            torch.compile(verl_F.entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  #  use torch compile by default
            else verl_F.entropy_from_logits
        )
        self.device_name = get_device_name()

    def _extract_decision_repr(
        self,
        hidden_states,
        *,
        response_length: int,
        layers: str,
        indices=None,
        batch_size: int | None = None,
        seqlen: int | None = None,
        pad_size: int = 0,
        remove_padding: bool | None = None,
    ) -> torch.Tensor:
        """Extract hidden states immediately before the first action token."""
        from verl.trainer.ppo.latent_flow_utils import get_hidden_state_indices

        if remove_padding is None:
            remove_padding = self.use_remove_padding
        layer_indices = get_hidden_state_indices(len(hidden_states), layers)
        prompt_end = seqlen - response_length - 1
        if prompt_end < 0:
            raise ValueError(
                f"cannot extract decision state: seqlen={seqlen}, response_length={response_length}"
            )

        decision_reprs = []
        for layer_idx in layer_indices:
            layer_hidden = hidden_states[layer_idx]
            if remove_padding:
                if indices is None or batch_size is None or seqlen is None:
                    raise ValueError("remove-padding decision extraction requires indices/batch_size/seqlen")
                if layer_hidden.dim() == 3:
                    layer_hidden = layer_hidden.squeeze(0)
                if self.use_ulysses_sp:
                    layer_hidden = gather_outpus_and_unpad(
                        layer_hidden,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                layer_hidden = pad_input(
                    hidden_states=layer_hidden,
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )
            decision_reprs.append(layer_hidden[:, prompt_end, :])

        if len(decision_reprs) == 1:
            return decision_reprs[0]
        return torch.stack(decision_reprs, dim=1)

    @staticmethod
    @torch.no_grad()
    def _teca_stats_from_logits(logits_2d, rolled_ids, chunk_size=4096):
        """Chunked fp32 per-row statistics for TECA from (already temperature-scaled)
        logits of shape (N, vocab): full-vocab entropy, argmax id, realized log-prob
        of `rolled_ids`. Single exp pass per chunk (max/argmax fused, no softmax
        materialization):
            H = lse - sum(e * x) / se,  lse = mx + log(se),  e = exp(x - mx), se = sum(e)
        """
        entropy_lst, argmax_lst, realized_lst = [], [], []
        total = logits_2d.size(0)
        for st in range(0, total, chunk_size):
            chunk = logits_2d[st : st + chunk_size].float()
            mx, amax = chunk.max(dim=-1)
            e = torch.exp(chunk - mx.unsqueeze(-1))
            se = e.sum(dim=-1)
            pxs = (e * chunk).sum(dim=-1)
            lse = mx + se.log()
            entropy_lst.append(lse - pxs / se)
            argmax_lst.append(amax)
            realized_lst.append(
                torch.gather(chunk, -1, rolled_ids[st : st + chunk_size].unsqueeze(-1)).squeeze(-1) - lse
            )
            del chunk, e
        return torch.cat(entropy_lst), torch.cat(argmax_lst), torch.cat(realized_lst)

    def _forward_micro_batch(
        self,
        micro_batch,
        temperature,
        calculate_entropy=False,
        calculate_teca_stats=False,
        return_decision_repr=False,
        latent_flow_layers="all",
        force_no_remove_padding=False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch["responses"].size(-1)
        if return_decision_repr and calculate_teca_stats:
            raise ValueError("decision representation and TECA statistics cannot be requested together")
        if return_decision_repr and self.use_fused_kernels:
            raise NotImplementedError("latent-flow decision representations do not support fused kernels")
        decision_repr = None
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch:
            for key in micro_batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat([inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0)

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            use_remove_padding = self.use_remove_padding and not force_no_remove_padding
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

            if use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices).transpose(0, 1).unsqueeze(1)  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(0, 1)

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                pad_size = 0
                if self.use_ulysses_sp:
                    is_vlm_model = "multi_modal_inputs" in micro_batch
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True
                if return_decision_repr:
                    extra_args["output_hidden_states"] = True

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if return_decision_repr:
                    decision_repr = self._extract_decision_repr(
                        output.hidden_states,
                        response_length=response_length,
                        layers=latent_flow_layers,
                        indices=indices,
                        batch_size=batch_size,
                        seqlen=seqlen,
                        pad_size=pad_size,
                        remove_padding=use_remove_padding,
                    )

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)
                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # TECA: fp32 full-vocab stats from the SAME forward (saves a whole
                    # extra student forward pass). Must run before logprobs_from_logits
                    # which may modify logits in place.
                    if calculate_teca_stats:
                        if self.use_ulysses_sp:
                            raise NotImplementedError("TECA stats does not support ulysses sp > 1")
                        teca_ent_rm, teca_amax_rm, teca_real_rm = self._teca_stats_from_logits(
                            logits_rmpad, input_ids_rmpad_rolled
                        )

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy (reuse the fp32 TECA entropy if already computed)
                    if calculate_entropy:
                        if calculate_teca_stats:
                            entropy_rmpad = teca_ent_rm
                        else:
                            entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outpus_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outpus_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

                if calculate_teca_stats:
                    def _pad_back_resp(x_rmpad):
                        return pad_input(
                            hidden_states=x_rmpad.unsqueeze(-1), indices=indices,
                            batch=batch_size, seqlen=seqlen,
                        ).squeeze(-1)[:, -response_length - 1 : -1]

                    teca_stats = {
                        "full_entropy": _pad_back_resp(teca_ent_rm),
                        "argmax_ids": _pad_back_resp(teca_amax_rm),
                        "realized_log_probs": _pad_back_resp(teca_real_rm),
                    }

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                if return_decision_repr:
                    extra_args["output_hidden_states"] = True
                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if return_decision_repr:
                    decision_repr = self._extract_decision_repr(
                        output.hidden_states,
                        response_length=response_length,
                        layers=latent_flow_layers,
                        batch_size=batch_size,
                        seqlen=seqlen,
                        remove_padding=use_remove_padding,
                    )

                if self.use_fused_kernels:
                    if calculate_teca_stats:
                        raise NotImplementedError("TECA stats does not support fused kernels")
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    if calculate_teca_stats:
                        ent2d, amax2d, real2d = self._teca_stats_from_logits(
                            logits.reshape(-1, logits.size(-1)),
                            micro_batch["responses"].reshape(-1),
                        )
                        teca_stats = {
                            "full_entropy": ent2d.view(logits.shape[:2]),
                            "argmax_ids": amax2d.view(logits.shape[:2]),
                            "realized_log_probs": real2d.view(logits.shape[:2]),
                        }
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        if calculate_teca_stats:
                            entropy = teca_stats["full_entropy"]
                        else:
                            entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)

            if calculate_teca_stats:
                return entropy, log_probs, teca_stats
            if return_decision_repr:
                return entropy, log_probs, decision_repr
            return entropy, log_probs

    @torch.no_grad()
    def _forward_micro_batch_teca_stats(self, micro_batch, temperature):
        """
        No-grad forward pass returning per-response-token statistics for TECA:
          full_entropy      : (bs, L) full-vocabulary entropy H_full (nats)
          argmax_ids        : (bs, L) full-vocab argmax token id (for the top-1 gate)
          realized_log_probs: (bs, L) log p(y_t) of the realized token

        The target-excluded candidate entropy H^{\\y} is recovered downstream in
        closed form from (full_entropy, realized_log_probs); no top-k candidate
        tensors need to be materialized or stored.
        """
        if self.use_ulysses_sp:
            raise NotImplementedError("TECA stats does not support ulysses sp > 1")
        if self.use_fused_kernels:
            raise NotImplementedError("TECA stats does not support fused kernels")

        response_length = micro_batch["responses"].size(-1)

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)
                position_ids_rmpad = index_first_axis(
                    rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                ).transpose(0, 1)
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1).squeeze(0)  # (total_nnz,)

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    use_cache=False,
                )
                logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab), bf16
                logits_rmpad.div_(temperature)

                full_entropy_rmpad, argmax_ids_rmpad, realized_logps_rmpad = self._teca_stats_from_logits(
                    logits_rmpad, input_ids_rmpad_rolled
                )
                del logits_rmpad

                def _pad_back(x_rmpad):
                    x_rmpad = x_rmpad.unsqueeze(-1)
                    return pad_input(x_rmpad, indices, batch_size, seqlen).squeeze(-1)[:, -response_length - 1 : -1]

                result = {
                    "full_entropy": _pad_back(full_entropy_rmpad),
                    "argmax_ids": _pad_back(argmax_ids_rmpad),
                    "realized_log_probs": _pad_back(realized_logps_rmpad),
                }
            else:
                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=False,
                )
                logits = output.logits.float()
                logits.div_(temperature)
                logits = logits[:, -response_length - 1 : -1, :]  # (bs, L, vocab)
                lse = torch.logsumexp(logits, dim=-1)
                probs = torch.softmax(logits, dim=-1)
                full_entropy = lse - (probs * logits).sum(dim=-1)
                argmax_ids = logits.argmax(dim=-1)
                realized_logps = torch.gather(
                    logits, -1, micro_batch["responses"].unsqueeze(-1)
                ).squeeze(-1) - lse
                del logits, probs

                result = {
                    "full_entropy": full_entropy,
                    "argmax_ids": argmax_ids,
                    "realized_log_probs": realized_logps,
                }

            return result

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_candidate_stats(self, data: DataProto):
        """Compute per-token TECA statistics over a full batch.

        meta_info: micro_batch_size, temperature.
        Returns full_entropy, argmax_ids, realized_log_probs (each (bs, L)).
        """
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        batch = data.select(batch_keys=select_keys).batch
        micro_batches = batch.split(micro_batch_size)

        outputs = []
        for micro_batch in micro_batches:
            outputs.append(self._forward_micro_batch_teca_stats(micro_batch, temperature=temperature))

        return {key: torch.concat([o[key] for o in outputs], dim=0) for key in outputs[0].keys()}

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        calculate_teca_stats = data.meta_info.get("calculate_teca_stats", False)
        return_decision_repr = data.meta_info.get("return_decision_repr", False)
        latent_flow_layers = data.meta_info.get("latent_flow_layers", "all")

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        if return_decision_repr and has_multi_modal_inputs:
            raise NotImplementedError(
                "latent-flow decision representations do not yet support multi-modal inputs"
            )

        if has_multi_modal_inputs:
            num_micro_batches = data.batch.batch_size[0] // micro_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
        elif use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        teca_stats_lst = []
        decision_repr_lst = []
        for micro_batch in micro_batches:
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                if calculate_teca_stats:
                    entropy, log_probs, teca_stats = self._forward_micro_batch(
                        micro_batch, temperature=temperature,
                        calculate_entropy=calculate_entropy, calculate_teca_stats=True,
                    )
                    teca_stats_lst.append(teca_stats)
                elif return_decision_repr:
                    entropy, log_probs, decision_repr = self._forward_micro_batch(
                        micro_batch,
                        temperature=temperature,
                        calculate_entropy=calculate_entropy,
                        return_decision_repr=True,
                        latent_flow_layers=latent_flow_layers,
                    )
                    decision_repr_lst.append(decision_repr)
                else:
                    entropy, log_probs = self._forward_micro_batch(micro_batch, temperature=temperature, calculate_entropy=calculate_entropy)
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        decision_repr = None
        if return_decision_repr:
            decision_repr = torch.concat(decision_repr_lst, dim=0)
        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]
            if entropys is not None:
                entropys = entropys[revert_indices]
            if decision_repr is not None:
                decision_repr = decision_repr[revert_indices]
            if calculate_teca_stats:
                raise NotImplementedError("TECA stats with use_dynamic_bsz is not supported")

        if calculate_teca_stats:
            teca_stats = {
                key: torch.concat([s[key] for s in teca_stats_lst], dim=0) for key in teca_stats_lst[0].keys()
            }
            return log_probs, entropys, teca_stats

        if return_decision_repr:
            return log_probs, entropys, decision_repr

        return log_probs, entropys

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        multi_turn = data.meta_info.get("multi_turn", False)

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "old_log_probs", "advantages"]
        if multi_turn:
            select_keys.append("loss_mask")
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        if self.config.get("use_sdl_loss", False) or self.config.get("use_sdar_loss", False):
            select_keys.append("teacher_log_probs")
        use_latent_flow = self.config.get("use_latent_flow_loss", False)
        if use_latent_flow:
            select_keys.extend(
                [
                    "teacher_flow",
                    "privilege_gate",
                    "flow_mask",
                    "next_responses",
                    "latent_input_ids",
                    "latent_attention_mask",
                    "latent_position_ids",
                    "next_latent_input_ids",
                    "next_latent_attention_mask",
                    "next_latent_position_ids",
                ]
            )
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        if use_latent_flow and has_multi_modal_inputs:
            raise NotImplementedError(
                "latent-flow actor updates do not yet support multi-modal next-step inputs"
            )

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        if has_multi_modal_inputs:
            num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
        else:
            dataloader = batch.split(self.config.ppo_mini_batch_size)

        metrics = {}
        for epoch in range(self.config.ppo_epochs):
            for batch_idx, data in enumerate(dataloader):
                # split batch into micro_batches
                mini_batch = data
                if has_multi_modal_inputs:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    num_micro_batches = mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
                elif self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    # split batch into micro_batches
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for data in micro_batches:
                    # Support all hardwares
                    if isinstance(data, DataProto):
                        data = {**data.batch.to(get_torch_device().current_device()), **data.non_tensor_batch}
                    else:
                        data = data.to(get_torch_device().current_device())  # actor device is cpu when using offload
                    responses = data["responses"]
                    response_length = responses.size(1)
                    attention_mask = data["attention_mask"]
                    if multi_turn:
                        response_mask = data["loss_mask"][:, -response_length:]
                    else:
                        response_mask = attention_mask[:, -response_length:]

                    old_log_prob = data["old_log_probs"]
                    advantages = data["advantages"]

                    clip_ratio = self.config.clip_ratio
                    clip_ratio_low = self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
                    clip_ratio_high = self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
                    clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True
                    if use_latent_flow:
                        latent_micro_batch = {
                            **data,
                            "input_ids": data["latent_input_ids"],
                            "attention_mask": data["latent_attention_mask"],
                            "position_ids": data["latent_position_ids"],
                        }
                        entropy, log_prob, current_decision_repr = self._forward_micro_batch(
                            micro_batch=latent_micro_batch,
                            temperature=temperature,
                            calculate_entropy=calculate_entropy,
                            return_decision_repr=True,
                            latent_flow_layers=self.config.get("latent_flow_layers", "all"),
                            force_no_remove_padding=True,
                        )
                    else:
                        entropy, log_prob = self._forward_micro_batch(
                            micro_batch=data,
                            temperature=temperature,
                            calculate_entropy=calculate_entropy,
                        )
                    
                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                    if loss_mode == "vanilla":
                        policy_loss_fn = compute_policy_loss
                    elif loss_mode == "gspo":
                        policy_loss_fn = compute_policy_loss_gspo
                    else:
                        raise ValueError(f"Unsupported loss_mode: {loss_mode}")

                    pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = policy_loss_fn(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        cliprange=clip_ratio,
                        cliprange_low=clip_ratio_low,
                        cliprange_high=clip_ratio_high,
                        clip_ratio_c=clip_ratio_c,
                        loss_agg_mode=loss_agg_mode,
                    )

                    pg_loss_coef = self.config.get("pg_loss_coef", 1.0)
                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        # compute policy loss
                        policy_loss = pg_loss * pg_loss_coef - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss * pg_loss_coef

                    if self.config.use_kl_loss:
                        ref_log_prob = data["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type)
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        metrics["actor/kl_loss"] = kl_loss.detach().item()
                        metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if self.config.get("use_sdl_loss", False):
                        from verl.trainer.ppo.skillsd_utils import compute_sdl_loss
                        teacher_log_probs = data["teacher_log_probs"]
                        sdl_loss = compute_sdl_loss(
                            student_log_probs=log_prob,
                            teacher_log_probs=teacher_log_probs,
                            old_log_probs=old_log_prob,
                            response_mask=response_mask,
                            loss_agg_mode=loss_agg_mode,
                        )
                        sdl_coef = self.config.get("sdl_loss_coef", 0.1)
                        policy_loss = policy_loss + sdl_loss * sdl_coef
                        metrics["actor/sdl_loss"] = sdl_loss.detach().item()
                        metrics["actor/sdl_coef"] = sdl_coef

                    if self.config.get("use_sdar_loss", False):
                        from verl.trainer.ppo.sdar_utils import compute_sdar_loss
                        teacher_log_probs = data["teacher_log_probs"]
                        sdar_loss, sdar_metrics = compute_sdar_loss(
                            student_log_probs=log_prob,
                            teacher_log_probs=teacher_log_probs,
                            response_mask=response_mask,
                            gate_beta=self.config.get("sdar_gate_beta", 5.0),
                            loss_agg_mode=loss_agg_mode,
                        )
                        sdar_coef = self.config.get("sdar_loss_coef", 0.1)
                        policy_loss = policy_loss + sdar_loss * sdar_coef
                        metrics.update(sdar_metrics)
                        metrics["sdar/coef"] = sdar_coef

                    if use_latent_flow:
                        next_micro_batch = {
                            "responses": data["next_responses"],
                            "input_ids": data["next_latent_input_ids"],
                            "attention_mask": data["next_latent_attention_mask"],
                            "position_ids": data["next_latent_position_ids"],
                        }
                        _, _, next_decision_repr = self._forward_micro_batch(
                            micro_batch=next_micro_batch,
                            temperature=temperature,
                            calculate_entropy=False,
                            return_decision_repr=True,
                            latent_flow_layers=self.config.get("latent_flow_layers", "all"),
                            force_no_remove_padding=True,
                        )

                        from verl.trainer.ppo.latent_flow_utils import compute_latent_flow_loss

                        latent_flow_loss, latent_flow_metrics = compute_latent_flow_loss(
                            student_current_repr=current_decision_repr,
                            student_next_repr=next_decision_repr,
                            teacher_flow=data["teacher_flow"],
                            flow_mask=data["flow_mask"],
                            privilege_gate=data["privilege_gate"],
                        )
                        latent_flow_coef = self.config.get("latent_flow_loss_coef", 0.1)
                        policy_loss = policy_loss + latent_flow_coef * latent_flow_loss
                        latent_flow_metrics["latent_flow/coef"] = latent_flow_coef
                        append_to_dict(metrics, latent_flow_metrics)


                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * (len(data) / self.config.ppo_mini_batch_size)
                    else:
                        loss = policy_loss / self.gradient_accumulation
                    loss.backward()

                    data = {
                        "actor/pg_loss": pg_loss.detach().item(),
                        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                        "actor/ppo_kl": ppo_kl.detach().item(),
                        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                    }
                    append_to_dict(metrics, data)

                grad_norm = self._optimizer_step()
                data = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, data)
        self.actor_optimizer.zero_grad()
        return metrics

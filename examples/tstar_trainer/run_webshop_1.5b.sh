#!/bin/bash
# GRPO + t* redistribution (skill-teacher-guided) on Webshop.
#
# Training settings mirror BEACON examples/migpo_trainer/run_webshop.sh EXACTLY,
# except for the method and the deviations REQUIRED to run it:
#   (method)  adv_estimator: migpo -> grpo               (GRPO base, as requested)
#             + t* forward-KL toward  t*_v ∝ p^{1-γ} q^{γ} e^{A·1[y]}
#   (required) actor entry: main_ppo -> main_tstar        (skill-teacher scaffold)
# use_remove_padding stays TRUE (== baseline): the t* loss now gathers the
# response-position full-vocab logits directly from the PACKED rmpad tensor
# (searchsorted over unpad indices), so no dense padded attention is needed.
# All other settings (batch sizes, lr, KL, group size, gamma, penalties,
# epochs, prompt/response lengths, memory settings) are IDENTICAL.
#
# NOTE: run from the SDAR repo root. GPUs 0-3 are used for training; if the
# probe vLLM servers on GPU 4-7 are still up they do not conflict.
set -x
ENGINE=${1:-vllm}
export VLLM_ATTENTION_BACKEND=XFORMERS
export RAY_TMPDIR=/data1/test/ray_tmp

# Ignore ~/.local/lib/python3.10/site-packages so user-level pip packages
# (e.g. a stray numpy 2.x) cannot shadow the conda env's packages.
export PYTHONNOUSERSITE=1
export WANDB_API_KEY=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

export CUDA_VISIBLE_DEVICES=4,5,6,7
N_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

num_cpus_per_env_worker=0.1

train_data_size=16
val_data_size=128
group_size=8

# t* hyperparameters (single knob: teacher trust gamma; coef scales the aux loss)
# beta(A) = gamma*(1-exp(-|A|)); gamma=0.8 puts |A|=1 at beta~0.5, matching the
# operating point of the offline audits (which used exponent 0.5 at A=+-1).
tstar_gamma=0.8
tstar_coef=1.0
skill_all=false

# Batch sizes (identical to the BEACON migpo webshop reference).
ppo_mini_batch_size=64
ppo_micro_batch_size_per_gpu=16
log_prob_micro_batch_size_per_gpu=16

# mfac = menu-factorized target: t*_y = outcome-only (teacher-free, == t0_y);
# menu block = teacher mixture of mass 1-t*_y, beta(A)=gamma*(1-exp(-|A|)).
# corr = aux loss carries only c = t* - t0 (c_y == 0; outcome force stays
# solely in the clipped PG channel; gamma=0 is EXACTLY GRPO).
experiment_name="tstar_qwen2.5_1.5b_mfac${tstar_gamma}_corr_c${tstar_coef}"

python3 -m examples.data_preprocess.prepare \
    --mode 'text' \
    --train_data_size $train_data_size \
    --val_data_size $((val_data_size * 2))

python3 -m verl.trainer.main_tstar \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/verl-agent/text/train.parquet \
    data.val_files=$HOME/data/verl-agent/text/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=7000 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=/home/test/models/Qwen2.5-1.5B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$log_prob_micro_batch_size_per_gpu \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$log_prob_micro_batch_size_per_gpu \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=1 \
    algorithm.use_kl_in_reward=False \
    algorithm.gamma=0.95 \
    +algorithm.tstar.gamma=$tstar_gamma \
    +algorithm.tstar.coef=$tstar_coef \
    +algorithm.tstar.skills_dir=skills/webshop \
    +algorithm.tstar.skill_all=$skill_all \
    env.env_name=Webshop \
    env.seed=0 \
    env.max_steps=15 \
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    trainer.critic_warmup=0 \
    trainer.resume_mode=auto \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_agent_webshop' \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=10 \
    trainer.total_epochs=200 \
    trainer.val_before_train=True $@

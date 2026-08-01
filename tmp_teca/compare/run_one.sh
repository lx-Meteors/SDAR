#!/bin/bash
# Matched single-method launcher for the webshop 3B comparison (single-GPU capable).
#   Usage: METHOD=sdar|teca|grpo GPUS=4 HORIZON=30 bash run_one.sh
# SDAR and TECA share IDENTICAL data / seed / batch / lr / kl / horizon; the only
# difference is TECA adds teacher-entropy advantage shaping on positive samples.
set -x
cd /data1/test/yyy/SDAR
source /home/test/miniconda3/etc/profile.d/conda.sh
conda activate verl-webshop

METHOD=${METHOD:-teca}
GPUS=${GPUS:-4}
HORIZON=${HORIZON:-30}
TEST_FREQ=${TEST_FREQ:-5}
VAL_BEFORE=${VAL_BEFORE:-True}
GMEM=${GMEM:-0.40}                 # vllm gpu_memory_utilization (single GPU shares with FSDP)
OFFLOAD=${OFFLOAD:-True}           # param/optimizer offload (needed on single GPU for 3B)
MICRO=${MICRO:-4}                  # actor micro batch per gpu

export CUDA_VISIBLE_DEVICES=$GPUS
n_gpus=$(echo $GPUS | tr ',' '\n' | grep -c .)

num_cpus_per_env_worker=0.1
train_data_size=16
val_data_size=128
group_size=8
ppo_mini_batch_size=64

sdar_coef=0.01
gate_beta=5.0
skill_all=false
teca_beta=${TECA_BETA:-0.1}
teca_top_frac=${TECA_TOP_FRAC:-0.2}

case "$METHOD" in
  grpo)
    ENTRY="verl.trainer.main_ppo"; EXTRA=""
    experiment_name="cmp_grpo_h${HORIZON}" ;;
  sdar)
    ENTRY="verl.trainer.main_sdar"
    EXTRA="+algorithm.sdar.sdar_coef=$sdar_coef +algorithm.sdar.gate_beta=$gate_beta +algorithm.sdar.skills_dir=skills/webshop +algorithm.sdar.skill_all=$skill_all"
    experiment_name="cmp_sdar_h${HORIZON}" ;;
  teca)
    ENTRY="verl.trainer.main_teca"
    EXTRA="+algorithm.sdar.sdar_coef=$sdar_coef +algorithm.sdar.gate_beta=$gate_beta +algorithm.sdar.skills_dir=skills/webshop +algorithm.sdar.skill_all=$skill_all \
      +algorithm.teca.beta=$teca_beta +algorithm.teca.positive_only=true \
      +algorithm.teca.require_top1=true +algorithm.teca.positive_dh_only=true \
      +algorithm.teca.top_frac=$teca_top_frac"
    experiment_name="cmp_teca_b${teca_beta}_tf${teca_top_frac}_h${HORIZON}" ;;
  *) echo "unknown METHOD=$METHOD"; exit 1;;
esac

export WANDB_API_KEY=${WANDB_API_KEY:-your_key_here}

python3 -m examples.data_preprocess.prepare --mode 'text' \
    --train_data_size $train_data_size --val_data_size $val_data_size

python3 -m $ENTRY \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/verl-agent/text/train.parquet \
    data.val_files=$HOME/data/verl-agent/text/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=4096 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=/home/test/models/Qwen2.5-3B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=$OFFLOAD \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$OFFLOAD \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=$GMEM \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    $EXTRA \
    env.env_name=Webshop \
    env.seed=0 \
    env.max_steps=15 \
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_agent_webshop_cmp' \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=$n_gpus \
    trainer.ray_wait_register_center_timeout=600 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=$TEST_FREQ \
    trainer.total_epochs=$HORIZON \
    trainer.val_before_train=$VAL_BEFORE $@

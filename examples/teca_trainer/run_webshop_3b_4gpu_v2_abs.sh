set -x


ENGINE=${1:-vllm}

num_cpus_per_env_worker=0.1

# SDAR distillation hyperparameters (loss unchanged vs SDAR baseline)
sdar_coef=0.01
gate_beta=5.0
skill_all=false

# TECA v2_abs advantage-shaping hyperparameters
# delta_H = target-excluded candidate entropy gap (teacher - student), whole vocab.
# v2_abs = TWO-SIDED, always-positive credit: any position with a LARGE candidate-
# entropy gap in EITHER direction is treated as informative. Relative to the
# production "pos" variant, +algorithm.teca.variant=abs changes three things:
#   eligible   : top1_agree & (delta_H != 0)          (was: & delta_H > 0)
#   rank/select: per-row top-`top_frac` by |delta_H|  (was: by delta_H)
#   credit     : A'_{i,t} = A_{i,t} + beta * |delta_H_t|   (was: beta * relu(delta_H))
# i.e. student-high-entropy (delta_H<0) tokens ALSO receive POSITIVE credit.
# (see verl/trainer/ppo/teca_utils.py::compute_teca_advantage)
teca_beta=${TECA_BETA:-0.1}
teca_top_frac=${TECA_TOP_FRAC:-0.2}
teca_variant=${TECA_VARIANT:-abs}

# Everything below is IDENTICAL to examples/teca_trainer/run_webshop_3b_4gpu.sh
# (and to sdar/grpo/gigpo 3b_4gpu) so TECA-v2_abs vs TECA-pos vs SDAR vs GRPO is an
# apples-to-apples comparison.
n_gpus=4
train_data_size=16
val_data_size=128
group_size=8
ppo_mini_batch_size=64
experiment_name="teca_v2abs_qwen2.5_3b_beta${teca_beta}_top${teca_top_frac}_4gpu"
export WANDB_API_KEY=${WANDB_API_KEY:-your_key_here}

python3 -m examples.data_preprocess.prepare \
    --mode 'text' \
    --train_data_size $train_data_size \
    --val_data_size $val_data_size

python3 -m verl.trainer.main_teca \
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
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    +algorithm.sdar.sdar_coef=$sdar_coef \
    +algorithm.sdar.gate_beta=$gate_beta \
    +algorithm.sdar.skills_dir=skills/webshop \
    +algorithm.sdar.skill_all=$skill_all \
    +algorithm.teca.beta=$teca_beta \
    +algorithm.teca.variant=$teca_variant \
    +algorithm.teca.positive_only=true \
    +algorithm.teca.require_top1=true \
    +algorithm.teca.positive_dh_only=true \
    +algorithm.teca.top_frac=$teca_top_frac \
    env.env_name=Webshop \
    env.seed=0 \
    env.max_steps=15 \
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_agent_webshopv1' \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=$n_gpus \
    trainer.ray_wait_register_center_timeout=600 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=5 \
    trainer.total_epochs=${TOTAL_EPOCHS:-150} \
    trainer.val_before_train=${VAL_BEFORE_TRAIN:-True} $@

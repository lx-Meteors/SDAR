set -x


ENGINE=${1:-vllm}

num_cpus_per_env_worker=0.1

# GiGPO hyperparameters (verl-agent webshop defaults). This is the ONLY thing that
# differs from examples/sdar_trainer/run_webshop_3b_4gpu.sh:
#   entry           : main_sdar -> main_ppo   (no SDAR distillation loss)
#   adv_estimator   : grpo      -> gigpo      (episode adv + anchor-state step adv)
#   + algorithm.gamma / gigpo.step_advantage_w / gigpo.mode
#   - all +algorithm.sdar.* args are removed
# Every other training hyperparameter (model, batch sizes, lr, KL, parallelism,
# rollout, env, trainer schedule) is IDENTICAL to the SDAR/TECA 3B-4GPU scripts,
# so GRPO(=SDAR credit) / GiGPO / TECA is an apples-to-apples comparison.
gamma=0.95
step_advantage_w=1.0
mode="mean_norm"  # "mean_norm" or "mean_std_norm"

n_gpus=4
train_data_size=16
val_data_size=128
group_size=8
ppo_mini_batch_size=64
experiment_name="gigpo_qwen2.5_3b_gamma${gamma}_w${step_advantage_w}_${mode}_4gpu"
export WANDB_API_KEY=${WANDB_API_KEY:-your_key_here}

python3 -m examples.data_preprocess.prepare \
    --mode 'text' \
    --train_data_size $train_data_size \
    --val_data_size $val_data_size

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gigpo \
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
    algorithm.gamma=$gamma \
    algorithm.gigpo.step_advantage_w=$step_advantage_w \
    algorithm.gigpo.mode=$mode \
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

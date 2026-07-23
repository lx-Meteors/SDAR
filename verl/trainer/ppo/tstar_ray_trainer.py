"""
t* redistribution trainer.

Reuses the SkillSD training loop (which keeps the standard GRPO sequence-level
advantages and performs a skill-teacher forward pass) but, instead of the
realized-token SDL/SDAR distillation, it stashes the skill-augmented teacher
INPUT tensors into the batch. The actor then runs its own no-grad teacher
forward to obtain the full-vocabulary teacher distribution q and optimizes the
forward KL toward the single target

    t*_v ∝ p_v^{1-γ} q_v^{γ} exp(A · 1[v == y])

See verl/trainer/ppo/tstar_utils.py for the (unit-tested) loss math and
verl/workers/actor/dp_actor.py (_tstar_kl_from_logits) for the in-actor pass.
"""

from verl.trainer.ppo.rlsd_utils import SkillProvider
from verl.trainer.ppo.rlsd_ray_trainer import build_teacher_batch
from verl.trainer.ppo.skillsd_ray_trainer import SkillSDRayTrainer


class TstarRayTrainer(SkillSDRayTrainer):
    """SkillSD loop with the teacher INPUT tensors forwarded to the actor.

    The teacher realized-token log-probs are still computed for logging the
    teacher-student gap (cheap sanity signal); the actual t* target is built
    inside the actor from a full-vocab teacher forward.
    """

    def _compute_teacher_log_probs(self, batch):
        teacher_batch = build_teacher_batch(
            batch=batch,
            skill_provider=self.skill_provider,
            tokenizer=self.tokenizer,
            max_prompt_length=self.config.data.max_prompt_length,
            truncation=self.config.data.get("truncation", "left"),
        )

        # Stash teacher inputs so the actor can run its own no-grad full-vocab
        # teacher forward for the t* target. Ordering matches `batch` because
        # build_teacher_batch iterates samples in order and no reordering
        # happens between here and update_actor.
        batch.batch["teacher_input_ids"] = teacher_batch.batch["input_ids"]
        batch.batch["teacher_attention_mask"] = teacher_batch.batch["attention_mask"]
        batch.batch["teacher_position_ids"] = teacher_batch.batch["position_ids"]

        # NOTE: no compute_log_prob here. The t* loss builds q from its own
        # no-grad teacher forward inside the actor; the trainer-level realized
        # -token teacher log-probs were only feeding the skillsd/* gap logging
        # (a whole wasted forward pass per step, ~60s). Return zeros so the
        # SkillSD fit() loop's gap metrics stay defined (they read as -student
        # logprob and should be ignored for t* runs; tstar/* metrics from the
        # actor are the real signal).
        import torch

        return torch.zeros_like(batch.batch["old_log_probs"])

# Privileged Environment-Step Latent-Flow Distillation

`verl.trainer.main_sdar` now defaults to environment-step latent-flow
distillation.  The student generates the existing on-policy agent trajectories;
the same model then reads the skill-augmented prompt as a no-gradient teacher.

For each trajectory, rows are paired by `(traj_uid, turn_step)` and the hidden
state immediately before the first action token is used as the decision
representation.  For every non-terminal environment step the auxiliary target
is

```text
teacher_flow[t] = teacher_decision[t + 1] - teacher_decision[t]
student_flow[t] = student_decision[t + 1] - student_decision[t]
```

The loss is a cosine-direction loss gated by the privileged teacher's mean
log-probability improvement on the action sampled at step `t`:

```text
gap[t]  = mean_response(log p_teacher - log p_student)
gate[t] = max(0, tanh(beta * gap[t]))
loss[t] = gate[t] * (1 - cosine(student_flow[t], stopgrad(teacher_flow[t])))
```

Setting `gate_beta <= 0` disables the gate and gives every valid transition a
weight of one.  Terminal steps and missing successors are always masked.

## Configuration

```bash
+algorithm.sdar.mode=latent_flow \
+algorithm.sdar.flow_coef=0.1 \
+algorithm.sdar.gate_beta=5.0 \
+algorithm.sdar.gate_mode=positive_tanh \
+algorithm.sdar.flow_layers=all
```

`flow_layers` accepts `last`, `last4`, `all`, `even`, or `odd`.  The experiment
default is `all`, matching OPRD's released representation-distillation setup.
Multi-layer targets scale CPU/Ray transfer approximately linearly with the
number of selected layers.

For latent-flow forwards, the student receives extra left-padding equal to the
longest privileged prefix in the rollout batch. The teacher uses those slots
for the complete privileged prefix. Neither the skill nor the student prompt is
truncated, and the original prompt, decision anchor, and response occupy
identical teacher/student tensor slots. Every valid student position ID is then
shifted by that row's privileged-prefix length, giving corresponding prompt,
decision, and response tokens identical teacher/student RoPE IDs. These aligned
student representation forwards keep padding masked and bypass the packed
remove-padding path; rollout generation is unchanged. The applied shift and
post-alignment offset are logged as `latent_flow/student_position_shift` and
`latent_flow/decision_rope_offset` (the latter must be zero).

For the old output-space SDAR ablation, use:

```bash
+algorithm.sdar.mode=logprob \
+algorithm.sdar.sdar_coef=0.01 \
+algorithm.sdar.gate_beta=5.0
```

Current latent-flow support is limited to text models with FSDP/FSDP2,
Ulysses sequence parallel size 1, and non-fused actor kernels.

## Metrics

- `latent_flow/loss`: gated auxiliary loss before `flow_coef`.
- `latent_flow/cosine`: ungated mean teacher/student environment-flow cosine.
- `latent_flow/gated_cosine`: cosine over privilege-selected transitions.
- `latent_flow/gate_mean`: mean gate over valid transitions.
- `latent_flow/gate_active_ratio`: fraction selected by the gate.
- `latent_flow/pair_ratio`: fraction of rollout rows with a successor.
- `latent_flow/student_flow_norm`, `latent_flow/teacher_flow_norm`: flow scale diagnostics.

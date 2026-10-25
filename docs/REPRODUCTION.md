# Reproduction contract: SDAR's pure GRPO baseline

## Identity and immutable source

Source: https://github.com/ZJU-REAL/SDAR at
`29dcb12a55d8fc17ba6ae03fd0a69048f43051f3`.
Native launcher: `examples/grpo_trainer/run_alfworld.sh`.
Native entry: `verl.trainer.main_ppo.run_ppo`.
Native loop: `verl.trainer.ppo.ray_trainer.RayPPOTrainer.fit`.

This is **pure GRPO with its original reference KL/entropy/invalid-action
handling**, not GRPO+OPSD at beta=0 or with a zero distillation coefficient.
There is no privileged-context Teacher, skill retrieval, skill input, OPD,
SDAR loss, RLSD advantage reweighting, HINT-SD, RETHINK, Analyzer, SFT or API.
The shared upstream checkout contains code for other algorithms, but this
entry never instantiates their trainers. A fixed base reference remains for
the original reference KL; it is not a distillation Teacher.

## Exact recipe

Both pinned Qwen2.5 Instruct backbones start independently from their
corresponding original public model. One update globally samples 16 task
groups, eight episodes each. Episodes contain a variable number of decision
rows; retain native adjustment/grouping and advantage statistics.

Retain native GRPO advantage normalization, clipped policy loss, response
mask and token-mean. No extra reasoning/action mask is introduced. LR=1e-6,
PPO global decision-row mini-batch256, micro32/GPU, one PPO epoch;
reference KL=0.01 low_var_kl; entropy=0.001; invalid-action penalty=0.1.

Interaction: max50 environment steps, history2, prompt2048, response512,
training temperature1, top_p1, top_k=-1. Each size has eight FSDP ranks and
four TP=2 vLLM rollout replicas, memory0.6, eager=false, free_cache=false.
The global batch does not increase with GPU count. env.rollout.n=8 while
actor_rollout_ref.rollout.n remains one, as required by upstream.

## Explicit adaptations

- Add the public 1.5B backbone and fixed public revisions for both sizes.
- Portable public source/model/environment preparation, project-relative paths.
- Fix validation to the included seen140 game identities, SHA hashes and
  seed2026+i, at update0 and every ten updates. Official launcher defaults
  to128 validation entries and every five updates; this difference is explicit.
- Stop at150 updates, save20/final150, keep newest two complete checkpoints,
  local JSON metrics and final HF export. Official launcher disables saves.
- Pool16 independent original text environments per Ray actor; preserve
  environment seeds, transitions, identities and result order.
- Read-only evaluation/logging subclass, with the original training loop
  called through super. Only its import alias is changed in main_ppo.

No source implementing advantages, policy gradients, Actor backward, FSDP,
reward, response generation, observation history, prompt or action projection
is modified. Twelve core file hashes and the complete small runtime patch
are checked before launch. No authored machine's conda path, proxy, wandb key
or private filesystem is inherited from the official shell script.

## Clean-machine preparation and runtime acceptance

Use README installation steps in a fresh Linux/Python3.12 environment, with
the pinned public downloads and dependency constraints. `setup.py --framework`
only requires Git and Python's standard library; it does not require the
author's installed environments or caches. No training trajectories need
collection or download. The parquet records only declare modality/batch size;
the native ALFWorld environment supplies the actual online tasks.

Run source verification, `scripts/check_release.py`, CPU tests, and both
dry-run commands before GPU work. On actual hardware confirm:

1. Eight 80GB GPUs, eight FSDP ranks, TP2 rollout, intended model/revision.
2. Real environment reset, model response, parsed action, execution and feedback.
3. Finite rewards/advantages/loss/grad norm and actual parameter updates.
4. No Teacher scoring, skill injection or auxiliary distillation metrics.
5. First save contains every rank's model/optimizer/extra state plus data.pt
   and Actor configuration; reject partial or foreign checkpoints.

CPU CI executes native tensor-level GRPO advantage/loss/backward tests and
configuration/source contracts. It is not proof of GPU memory fit, NCCL,
vLLM startup, full RL training or benchmark scores. Any runtime adaptation
must be recorded; do not silently alter sampling count, loss or task budget.

## Evaluation and recovery

The same seen140 manifest is used for both sizes and the public comparison
repos. One rollout/task, temperature0.4, same limits and seeds. Native success
and action execution are untouched. Additional reasoning/strict-format metrics
are telemetry, never gates. Preserve token IDs and special-token text.

Report initial base performance and update150 with successes/140, success
rate, task-level outcomes and actual training/validation curves. Do not use
the test set to select the best checkpoint as the final result. An independently
launched exported-model evaluation starts controller numbering at0; record
its actual training source step through the export manifest.

Resume requires the same model, source, config, task-manifest and eight-rank
binding. Checkpoints include optimizer, scheduler/RNG extra and dataloader,
but do not snapshot TextWorld environment iterators. Exact uninterrupted vs
resumed rollout identity is not promised. First-save failure has no complete
intermediate resume point; do not silently call a new run a continuation.

Budget at least300GiB free, preferably400GiB, for both models, newest two
complete checkpoints/size, public resources and save/export peaks. This is
an estimate, not a measured minimum. No checkpoints or run logs ship publicly.

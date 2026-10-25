# Sources and modifications

- **SDAR**, ZJU-REAL: https://github.com/ZJU-REAL/SDAR , Apache-2.0.
  Fixed commit `29dcb12a55d8fc17ba6ae03fd0a69048f43051f3`.
  This release uses the public **GRPO baseline** launcher
  `examples/grpo_trainer/run_alfworld.sh`, not the SDAR method trainer.
  Based on veRL and verl-agent; fetched source retains their notices.
  Limited adapters modify `agent_system/environments/env_manager.py`,
  `agent_system/environments/env_package/alfworld/envs.py`, and
  `verl/trainer/main_ppo.py`: environment scheduling, fixed evaluation, and
  read-only logging. A helper selects existing validation games.
  Native GRPO, RayPPOTrainer.fit, Actor optimization, reward, prompt, memory
  and action projection are unchanged; hashes are in `configs/upstream.json`.
- **Qwen2.5-1.5B-Instruct**: https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct , Apache-2.0.
- **Qwen2.5-3B-Instruct**: https://huggingface.co/Qwen/Qwen2.5-3B-Instruct , Qwen Research License.
  Weights are obtained from pinned public revisions, not redistributed here.
- **ALFWorld**: https://github.com/alfworld/alfworld . Use the official downloader
  and comply with the source/data licenses.
- The fixed **seen140 manifest** contains public game-relative paths, game
  hashes and environment seeds. It is shared with the public SDAR/RLSD/RETHINK
  comparisons, not private trajectories or a private environment snapshot.

Please credit GRPO, ALFWorld, Qwen, veRL, verl-agent and the SDAR public
implementation in derivative research. SDAR paper: https://arxiv.org/abs/2605.15155 .
GRPO reference: https://arxiv.org/abs/2402.03300 .

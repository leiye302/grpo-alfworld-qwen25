# GRPO on ALFWorld — Qwen2.5-1.5B / 3B, 8×A100 80GB

独立运行 [SDAR公开仓库](https://github.com/ZJU-REAL/SDAR) 中的 **纯GRPO基线**，固定源码版本 `29dcb12a55d8fc17ba6ae03fd0a69048f43051f3`。官方入口是 `examples/grpo_trainer/run_alfworld.sh`，实际使用 `verl.trainer.main_ppo` 和原生 `RayPPOTrainer`，**不是把OPSD／SDAR系数设为零来模拟GRPO**。

两个尺寸分别从公开的 **Qwen2.5-1.5B-Instruct / Qwen2.5-3B-Instruct** 基模开始，各训练 **150 updates**。每个实验用满 **8×A100 80GB**，默认先1.5B、再3B顺序运行，不把全局采样量乘八。

| 实验 | 初始化 | 算法 | 默认硬件 |
| --- | --- | --- | --- |
| GRPO 1.5B | Qwen/Qwen2.5-1.5B-Instruct | 原生多轮Agent GRPO | 八卡FSDP，rollout TP=2 |
| GRPO 3B | Qwen/Qwen2.5-3B-Instruct | 相同配方 | 八卡FSDP，rollout TP=2 |

**无Route-SFT、RETHINK、Teacher蒸馏、skill、Analyzer或外接API。** 保留官方的固定reference KL、entropy与非法动作惩罚；“纯GRPO”不等于删除这些原本就在GRPO脚本里的项。所有代码、模型、ALFWorld资源均通过下文公开来源取得，**不依赖维护者本机、私有服务器、私人数据或旧训练断点**。

## 1. 保留的实现和固定配方

观察 → Student生成reasoning/action → 原生解析与环境执行 → 环境反馈 → 同任务八条轨迹的原生GRPO优势 → 原生PPO clipped更新。没有Teacher前向、反馈生成、skill检索或优势蒸馏重加权。固定reference只用于KL约束，不是带特权信息的Teacher。

\[
\rho_{i,t}=\exp\big(\log\pi_\theta(y_{i,t}\mid h_{i,t})-\log\pi_{\rm old}(y_{i,t}\mid h_{i,t})\big).
\]

\[
\mathcal L=\mathcal L_{\rm GRPO}^{\rm native}
 +0.01\mathcal L_{\rm reference\ KL}-0.001\mathcal H.
\]

这里的GRPO优势、跨决策步统计、clipping、response mask、token-mean和非法动作惩罚，**全部以锁定源码为准，未另写一套GRPO公式替换实现**。尤其不新增SFT的reasoning mask，原生有效response token照常参与RL。

| 项目 | 两个尺寸共用 |
| --- | --- |
| 训练池 | 原生ALFWorld train中的3553个可用任务，在线交互采样 |
| 每update采样 | 全局16任务 × 每任务8条轨迹 = 128条episode |
| 更新数 / LR | 150 updates；1e-6 |
| PPO | 全局决策行mini-batch256；每卡micro32；PPO epoch1；token-mean |
| Reference KL | use_kl_loss=true；coefficient0.01；low_var_kl |
| Entropy / 非法动作惩罚 | 0.001 / 0.1，保留上游实现 |
| 交互 | max steps50；history2；prompt2048；response512 |
| 训练采样 | temperature1、top_p1、top_k=-1 |
| FSDP / rollout | 八个rank；四个TP=2 rollout副本 |
| vLLM | 显存比例0.6；enforce_eager=false；free_cache_engine=false |
| 验证 | update0和每10步；固定seen140；一任务一条rollout；temperature0.4 |
| 保存 | 每20步及最终150；最近两份完整断点，含最终150 |

`env.rollout.n=8`才是原生Agent的分组采样；`actor_rollout_ref.rollout.n=1`保持官方要求。128条episode是全局量，不是每卡的量。两个尺寸除模型固定版本、实验名和输出路径外，使用同一份 [experiment.json](configs/experiment.json)。

## 2. 新服务器准备

需要Linux x86-64、Python3.12、八张已获配的 **A100 80GB**、兼容CUDA12.8的驱动。建议64个可用逻辑CPU、256–512GiB主机内存、至少300GiB空闲磁盘，400GiB更稳；这是包含两种尺寸、最近两份完整断点及存盘/导出峰值的预算，**不是实测最低要求**。本配方未按A100 40GB准备。

```bash
git clone https://github.com/leiye302/grpo-alfworld-qwen25.git
cd grpo-alfworld-qwen25
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel packaging ninja psutil

# 仅准备锁定的公开源码，不占GPU，不下载模型。
python src/setup.py --framework
python -m pip install -r configs/requirements.txt
python -m pip install flash-attn==2.7.4.post1 --no-build-isolation
python -m pip install -e .deps/sdar -c configs/requirements.txt
python -m pip check

cp .env.example .env
# 编辑自己获配的八张GPU、CPU数和ALFWORLD_DATA路径。
set -a
source .env
set +a

# 阅读各模型LICENSE后下载固定revision，无需作者的权重或轨迹。
python src/setup.py --download-model both
alfworld-download
python src/setup.py --verify
python scripts/check_release.py
python -m pytest tests -q
```

FlashAttention需要匹配的wheel，或CUDA Toolkit / nvcc现场编译。依赖约束锁定Torch/vLLM等版本；不要另跑上游未约束的全量requirements安装而升级环境。网络镜像或Hugging Face凭据由运行者自行安全配置，模型ID/revision仍须保持不变。官方下载器读取 `ALFWORLD_DATA`，需准备 `json_2.1.1` 与 `logic`。

仓库不包含大模型或ALFWorld数据副本。`configs/seen140_tasks.json`只包含公开游戏相对路径、文件SHA及seed。模型固定版本见 [models.json](configs/models.json)。

## 3. 开跑：两个尺寸顺序使用全部八卡

先看配置，不占GPU、不读大权重、不启动训练：

```bash
bash scripts/train.sh --size 1.5b --dry-run
bash scripts/train.sh --size 3b --dry-run
```

确认八卡归自己使用、没有别人的进程，在 `tmux` 或调度器内执行：

```bash
# 1.5B150步 → 导出；之后3B150步 → 导出。
bash scripts/run_both.sh
```

单独运行：

```bash
bash scripts/train.sh --size 1.5b
# 或
bash scripts/train.sh --size 3b
```

**不要在同一组八张卡上同时执行两个尺寸。** 不加外层torchrun，由原生Ray启动八个FSDP rank和TP=2的vLLM副本。每次使用独立短路径Ray临时目录与本地集群，不连接其他作业，不执行整机 `ray stop`。默认拒绝覆盖已有输出，同输出目录有控制器锁。

第一次真机运行要确认八rank/TP2、环境reset/action/feedback、奖励与优势有限、grad norm有限、Actor真实backward/update，以及一次完整存盘。CPU CI不能替代CUDA/NCCL/vLLM验证。遇到OOM先核对模型、实际显存、配置和日志；任何工程调整需记录，不擅自改全局batch、group size、loss或生成预算。

## 4. 统一评测、结果和曲线

固定 [seen140_tasks.json](configs/seen140_tasks.json) 与现有SDAR/RLSD/RETHINK对比仓库相同：valid_seen，140个唯一任务，环境seed=2026+i；每任务一次rollout，max steps50、history2、prompt2048、response512、temperature0.4、top_p1、top_k=-1。

**这是统一主实验的seen140协议，不冒充官方脚本128条验证原样复现。** 保留原生动作解析与环境reward；额外的strict protocol指标不改变执行、奖励或训练过滤。若比较对象用了不同执行解析器，需统一执行器复测，不能只看同名成功率。

update0的初始验证就是该尺寸基模成绩，之后0/10/.../150均做真实环境评测。最终报告固定150步，不挑测试集最高断点。没有Teacher、skill、hint或Analyzer参与任何评测。

```text
runs/grpo_1.5b/                         # 另一尺寸对应grpo_3b
  binding.json                         # 模型/源码/配方/任务集身份
  resolved_config.yaml
  metrics.jsonl                        # 原生全部训练和验证指标
  training_pool.json                    # 训练任务身份，与seen140互斥
  evaluation/update_000/summary.json    # 成功数/140、reasoning、格式与动作有效率
  evaluation/update_000/decisions.json  # 生成文本、token IDs和逐步记录
  evaluation/update_150/summary.json
  checkpoints/global_step_150/          # 八rank模型、optimizer、extra及data.pt
  hf150/                               # 最终推理导出
```

```bash
python src/run.py export --size 1.5b
bash scripts/eval.sh --size 1.5b --model "$PWD/runs/grpo_1.5b/hf150" \
  --run-dir "$PWD/runs/grpo_1.5b_final_eval"
python src/report.py
```

3B替换size及对应路径；独立评测同样使用八张空闲卡。`report.py`从真实验证输出生成 `results/seen140.csv` 与PNG，不预填成绩。独立评测的实际训练来源步数以模型导出清单为准，其新控制器内部update从0计数。

## 5. 中断、续跑与存盘

先确认旧进程已退出、最新断点完整，再使用：

```bash
bash scripts/train.sh --size 1.5b --resume
# 或 --size 3b --resume
```

原生保存model、optimizer、scheduler/rank RNG相关extra_state与dataloader。启动检查全部八rank文件、原始模型/配方/源码/评测绑定；缺文件或另一配方的断点会拒绝。TextWorld环境迭代器本身不在原生断点里，不能承诺续跑与不中断训练逐位一致。首个20步存盘若未完成，不存在可完整恢复的中途断点。

`run_both.sh`是首次执行队列；中断后用单尺寸resume完成当前训练并export，再单独启动下一尺寸。不会隐式覆盖旧输出或从0重跑。每尺寸只保留最近两份完整断点及最终导出；不因最终体积小而忽略第三份保存时的峰值空间。

## 6. 官方代码与必要适配的边界

12个未修改核心文件有SHA锁，涵盖GRPO优势/损失、原生 `RayPPOTrainer.fit`、Actor backward、FSDP、rollout、reward、prompt、history和动作解析。`patches/runtime.patch`只包含：

1. 固定seen140选择及每任务seed；
2. 每16个独立原生文字环境共用一个Ray进程，保留原seed、状态和返回顺序；
3. 将main_ppo的Trainer别名指向只记录指标/固定评测的子类。其 `fit` 调用原生super，不复制或重写训练循环。

另外只增加公开下载、项目相对路径、固定1.5B版本、150步预算、本地JSON日志、10步验证和20步/最终保存。原八卡、TP2、vLLM显存0.6与graph设置不变。详见 [docs/REPRODUCTION.md](docs/REPRODUCTION.md)。原脚本中的作者本机conda激活、代理、路径和wandb占位配置不会执行或继承。

## 7. 验证范围与许可

GitHub Actions在干净Linux检出中直接从公开上游准备源码、对照官方GRPO脚本、测试两尺寸配方、Teacher/蒸馏禁入、真实GRPO tensor梯度、固定任务、环境池、八rank断点及shell入口。**没有附带训后成绩，也未在8×A100上实际训练；运行者仍须验收首轮GPU运行。**

包装代码和上游框架采用Apache-2.0。Qwen2.5-1.5B-Instruct是Apache-2.0；Qwen2.5-3B-Instruct为Qwen Research License，不能把3B权重一概称为Apache许可。ALFWorld资源遵守其来源许可。见 [THIRD_PARTY.md](THIRD_PARTY.md)。

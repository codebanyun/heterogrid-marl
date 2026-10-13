# MAPPO 与 R-MAPPO

本接入复用外部 MAPPO 仓库中的 `R_MAPPOPolicy`、`R_MAPPO` 和 `SharedReplayBuffer`，不复制算法实现，也不依赖 EPyMARL。共享 actor 读取局部观测，集中式 critic 读取全局状态；使用合法动作掩码和死亡节点掩码。通信沿用环境配置，没有额外的学习消息头。

以下命令均从 HeteroGrid 项目根目录执行。

## 依赖与外部算法

先安装环境与可选训练依赖，再准备原算法仓库：

```bash
python -m pip install -e '.[mappo]'
git clone https://github.com/marlbenchmark/on-policy.git ../on-policy
export MAPPO_ROOT="$(cd ../on-policy && pwd)"
```

`MAPPO_ROOT` 应指向包含 `onpolicy/config.py` 的目录。也可在每条训练命令中传入 `--mappo-root /path/to/on-policy`，其优先级高于环境变量。PowerShell 用户可使用 `$env:MAPPO_ROOT = 'C:\path\to\on-policy'` 或直接传入参数。

无需安装 StarCraft、wandb 或 TensorBoard。PyTorch 的 CPU / CUDA 构建应按机器条件选择；`.[mappo]` 不安装外部 MAPPO 源码，也不替代其版本管理。团队复现实验时应固定外部仓库提交和依赖版本，原始源码快照保存在各次实验的 `_sources/`。

## 单种子训练

```bash
python -m algorithms.mappo.train \
  --config configs/scenarios/6v6_rule_distance.json \
  --steps 65536 --seed 42 --eval-episodes 16 \
  --output output/mappo/6v6_seed42
```

默认场景即 `6v6_rule_distance.json`：6v6、规则蓝方、半径 10 的距离通信、双方自动目标报告、超时失败、时间成本及有上限的情报奖励。基础手动报告场景为 `configs/scenarios/default_6v6.json`；紧凑流程检查场景为 `configs/scenarios/3v3_compact.json`，两者难度不同，不能混为同一任务比较。

`--steps` 必须是 `--num-envs × --rollout-length` 的整数倍，默认分别为 4 和 128。步数是累计团队环境步，不乘节点数量、不含评估步数。`--device cuda` 可选择 GPU；默认使用 CPU。输出目录必须尚不存在。

R-MAPPO 使用相同入口：

```bash
python -m algorithms.mappo.train --algorithm rmappo --steps 65536
```

R-MAPPO 的 rollout 长度必须是 16 的整数倍。本接入将任务时限视为已评分的有限时域终局，不跨回合 bootstrap；这不改变 PettingZoo 的 truncation 标记。

## 多种子与后台实验

Linux 等 POSIX 系统可使用内置 `nohup` 管理器：

```bash
python -m algorithms.mappo.launch \
  --steps 10000000 --seeds 42 43 44 \
  --output output/mappo/6v6_rule_distance_10m
```

管理器独立运行，默认从头训练三个前馈 MAPPO 种子；场景要求规则蓝方与距离通信。预算向上对齐到完整 rollout：10M 对齐后为 10,000,384 步。默认约每 250k 步评估 64 个固定验证场景、约每 2M 步保存模型，结束后在 128 个独立测试场景上评估。模型保存与评估频率独立。

常用参数为 `--eval-interval`、`--save-interval`、`--eval-episodes`、`--heldout-episodes` 和 `--replay-episodes`。记录文件：

```text
output/mappo/<实验名>/
  README.md                 本次记录说明
  experiment.json           算法、预算、种子、间隔和启动信息
  env.json                  本次完整环境配置
  status.json               管理进程和各训练进程的状态、PID
  nohup.log                 合并日志
  seed_<种子>/              独立模型、日志与评估记录
  figures/                  均值与 SEM 曲线、CSV
  RESULTS.md                完成后的聚合报告
```

```bash
tail -f output/mappo/6v6_rule_distance_10m/nohup.log
```

停止整组实验时，对 `status.json` 中的 `manager_pid` 执行 `kill -TERM <PID>`。中断记录保持 `INTERRUPTED`；只保证已保存的模型可用，停止时未保存的权重不会自动保留。当前不支持精确恢复中断训练。

前台多种子诊断可用于不同平台：

```bash
python -m algorithms.mappo.study \
  --steps 262144 --seeds 42 43 44 --workers 3 \
  --output output/mappo/learning_study
```

该诊断每 16,384 步评估 8 局，最终独立测试 32 局。以上均是示例预算，不是任务收敛所必需的固定步数。

## 环境因素批量实验

Linux 上可使用独立队列管理器运行地图大小、队伍规模和通信方式实验：

```bash
python -m algorithms.mappo.matrix \
  --steps 5000000 --seeds 42 43 44 --workers 12 --device cpu \
  --output output/mappo/sensitivity_5m
```

默认共 13 个配置、39 次独立训练，每次向上对齐到 5,000,192 环境步。`--workers` 限制同时运行的训练数，剩余任务自动排队；应按实际 CPU、内存和吞吐选择。`--device cuda` 可启用 GPU，但小型网络的端到端速度仍需实测。管理器通过 `nohup` 运行，退出终端不影响训练。

| 配置 | 相对于 24×24×6、6v6、距离通信半径 10 的基准 |
|---|---|
| B0 | 基准 |
| M1、M2、M3 | 地图分别为 12×12×3、36×36×9、48×48×12 |
| N1、N2、N3 | 双方分别为 3、9、12 个节点；侦察、打击、混合节点各占三分之一 |
| C1、C2 | 无通信、全通信 |
| C3、C4 | 距离通信半径 5、15 |
| C5、C6 | 随机通信，节点离线概率 0.25、0.50 |

通信配置同时影响红蓝双方；随机通信是在线节点全连接，不是距离通信叠加丢包。奖励、固定规则对手、网络与 PPO 参数一致，不按胜率提前停止。主要比较固定预算结束后的模型，验证集最优模型作为补充；独立测试集不参与选择。

输出根目录保存 `experiment.json`、`status.json`、`nohup.log` 和 `configs/`；每个配置目录内保存 `seed_42/` 等独立记录及 `logs/`。完成后自动审计并为每个配置生成均值 ±SEM 曲线、CSV 和报告。默认不创建 `trajectories/` 和 `evaluations/`，周期评估指标仍保留，最终评估保留少量完整回放。磁盘低于安全阈值会停止整组任务，避免写满磁盘。停止方式仍为向 `status.json` 中的管理 PID 发送 SIGTERM。

## 输出与评估

| 位置 | 用途 |
|---|---|
| `models/` | 初始、周期、最终及验证集最优模型；`latest.pt` 只在保存时更新 |
| `config.json`、`run.json` | 完整参数、依赖版本、源码摘要和运行状态 |
| `metrics.jsonl` | 每次 PPO 更新的损失、累计步数和近期训练统计 |
| `training_episodes.jsonl` | 每个完成对局的回报、胜负与奖励分量 |
| `metrics.json`、`info.json` | `steps/timestamps/values` 指标序列及兼容导出 |
| `cout.txt` | 当前种子的控制台日志 |
| `_sources/` | 环境、接入和外部算法源码快照 |
| `evaluations/` | 周期验证及初始独立测试的统计，不是完整回放 |
| `trajectories/` | 压缩训练数组，不是浏览器回放；单次训练默认保存每批，后台实验默认抽样保存 |
| `replays/` | 完整 `episode_*.json` 和动作序列，浏览器可直接打开 |
| `summary.json` | 完成训练、模型重载及最终评估后的汇总；中断时可能不存在 |

`--trajectory-interval` 控制训练数组保存频率，`--log-interval` 控制兼容指标导出频率。`--no-trajectories` 可完全禁用训练数组；`--no-evaluation-files` 可不生成周期评估目录，同时保留指标。`--replay-episodes` 只限制完整回放数，不减少参与统计的评估局数。各次实验的可选训练数组和评估统计可在不再需要审计时人工归档或删除；不要删除模型、指标和源码证据。

默认不在项目外创建结果目录。需要编号索引时显式传入 `--results-root /path/to/results`，生成指向原始输出目录的链接；Windows 上需具备创建符号链接的权限。这是兼容格式导出，不是 Sacred 执行。

单独评估保存模型，不进行训练：

```bash
python -m algorithms.mappo.train \
  --checkpoint output/mappo/6v6_seed42/models/best.pt \
  --eval-episodes 32 --output output/mappo/6v6_seed42_eval
```

模型文件含环境配置与网络参数。仅加载可信来源的 PyTorch 检查点；不同任务形状的模型不能直接混用。

## 奖励消融与审计

```bash
python -m algorithms.mappo.ablate --steps 32768 --seeds 42 43 44 --workers 3
python -m algorithms.mappo.audit --recompute output/mappo/6v6_seed42
```

消融依次比较对称共享、加入终局成本、加入情报奖励。地图、阵容、对手与通信条件相同，配置见[配置说明](../../configs/README.md)。奖励不同，不应直接用塑形回报判断优劣，应比较胜率、超时率、伤害等行为指标。

审计仅用于完成且保留原始记录的实验，会核对模型更新、指标、源码摘要、训练数组及回放；`--recompute` 逐帧复算每个评估分组的一局。已中断或人工删除审计材料的实验不满足该完整性检查。

聚合曲线只显示多训练种子的均值和 ±SEM 阴影，SEM 为样本标准差除以种子数量的平方根，不是独立地图上的置信区间。验证曲线用于学习诊断，独立测试结果另行报告。

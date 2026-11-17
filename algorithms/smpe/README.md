# SMPE

15 配置、三个 seed 的通信/规模/角色配比实验见 [统一实验协议](../common/COMM_SCALE_ROSTER.md)，由共享 CPU/GPU 队列与 EPyMARL MAPPO 并行运行，输出分别管理。

本接入使用外部 SMPE 仓库的 `BasicMAC`、`ActorCriticVLearner`、`VAEController`、`EpisodeBatch` 和 `ReplayBuffer`。保留原始策略梯度、状态 VAE、过滤器、哈希探索奖励和模型训练过程，不用 MAPPO 替代 SMPE，也不复制外部网络实现。

命令均从 HeteroGrid 项目根目录执行。

## 依赖

```bash
python -m pip install -e '.[smpe]'
export SMPE_ROOT=/path/to/smpe_raw
```

外部目录应包含 `learners/actor_critic_learner.py`、`model_learners/vae_controller.py` 和 `config/algs/smpe_lbf.yaml`。也可显式传入 `--smpe-root`。接入不调用原仓库的环境注册器，不需要 StarCraft、LBF、Sacred 或 TensorBoard。旧版算法依赖 Gym 和 NumPy 1.x；它们仅属于可选算法依赖，不改变 HeteroGrid 的 Gymnasium 接口。

## 训练设置

默认合并原仓库的 `config/default.yaml` 与 `config/algs/smpe_lbf.yaml`：

- 共享 GRU actor，隐藏维度 128；潜变量维度 32；集中式 critic 和过滤 critic。
- 每批 10 个完整对局，训练批次与回放缓冲区均为 10 个对局，保留原生完整批次更新方式。
- 策略、状态模型、过滤器和内在奖励均启用；VAE 每次仍训练 15 轮、每轮抽取 15 个对局批次。
- 模型更新沿用原生 LBF 的计数器：每次策略更新加 250，每 2,000 个计数单位训练 VAE，每 50,000 个计数单位更新目标过滤器。这些计数单位不是环境步。

HeteroGrid 的全局状态不是局部观测的拼接，因此内部设置 `env="sc2"`，使用原代码已有的完整状态建模分支。这只是数据布局适配，实际环境仍是 HeteroGrid；执行 actor 只读取局部观测和由局部观测推断的潜变量，不读取全局状态。

评分后的超时按有限时域终局处理，设置 `add_value_last_step=False`，不跨终局 bootstrap。死亡槽位只允许等待动作，策略遵守环境动作掩码。内在奖励只供训练使用，训练对局回报、验证和测试回报均单独统计环境奖励。

关闭原仓库 VAE 的共享路径自动存盘，改由接入层统一保存完整模型。接入层的 `TargetSyncActorCriticVLearner` 补齐原始 learner 遗漏的 `target_critic_w` 同步：硬更新复制过滤 critic，软更新使用与普通目标 critic 相同的系数和触发时机。策略梯度、价值目标、VAE 与探索奖励公式不变，外部仓库和历史实验快照不改动。

修复后的运行在 `config.json` 和 `summary.json` 中记录 `learner_correction=filtered-critic-target-sync-v1`，并保存过滤目标更新次数及参数变化。`network.json` 保存实际网络结构、输入输出维度和参数量。旧检查点仍可加载评估，但其权重不会被自动修改，不能视为修复后重训结果。具体源码版本以实验的 `_code/` 和 `_sources/` 为准，不将本接入等同于论文全部实验的复现。

## 单种子训练

```bash
python -m algorithms.smpe.train \
  --config configs/scenarios/6v6_rule_distance.json \
  --steps 16384 --seed 42 --eval-episodes 4 --heldout-episodes 4 \
  --eval-interval 8192 --replay-episodes 1 \
  --output output/smpe/short_check
```

步数是累计团队环境步，不乘节点数量，不含评估步数。达到目标后完成当前对局批次，实际步数最多比目标多不到 `batch_size_run × max_steps`，默认少于 2,000 步，不能把未完成对局伪装成终局以强行对齐预算。

`--device cpu` 为默认值，`--device cuda` 使用 GPU。应实测完整训练吞吐再选择设备，不能仅按显存容量决定。短训练只能验证执行链路，不能证明收敛。

## 复用 MAPPO 实验矩阵

若只研究红方通信、固定蓝方通信，请使用[固定蓝方通信实验](../common/COMMUNICATION.md)的联合入口。该入口会重新训练新独立通信实现下的 B0，而不是复用旧矩阵的基准。

密度受控的 D0–D3 规模实验使用[独立 D 组入口](../common/DENSITY.md)，开启参考奖励归一化及双方诊断，并无损压缩逐局日志。

```bash
python -m algorithms.smpe.matrix \
  --reference output/mappo/sensitivity_5m \
  --workers 12 --device cpu \
  --output output/smpe/sensitivity_5m
```

参考实验支持已完成的 13 配置 MAPPO/SMPE 敏感性实验，或已完成的 MAPPO/SMPE 四配置 D 组。默认直接继承其三个种子、5M 对齐预算、验证与独立测试种子、评估局数、评估和模型保存间隔，不重新生成近似的环境配置。启动前会校验环境核心源码与参考实验一致。

对 D 组单独重跑修复后的 SMPE：

```bash
python -m algorithms.smpe.matrix \
  --reference output/smpe/<原D组实验> \
  --smpe-root "$SMPE_ROOT" --workers 12 --replay-episodes 1 \
  --output output/smpe/<新的修复验证实验>
```

可用 `--conditions D0_baseline D3_12v12` 选择参考实验的子集。D 组保持参考奖励归一化及双方诊断，训练逐局记录自动无损压缩。验证修复影响时应保持原网络和超参数，从头训练，不同时扩大网络或更改探索奖励。

补齐 B/C/M/N 时，可以组合两个来源，避免重新运行旧版同时干扰蓝方的通信对照：

```bash
python -m algorithms.smpe.matrix \
  --reference output/smpe/<原sensitivity实验> \
  --communication-reference output/smpe/<原blue_fixed实验> \
  --allow-compatible-environment --compact-episode-log \
  --external-status output/smpe/<正在运行的D组>/status.json \
  --workers 12 --save-interval 5000192 --replay-episodes 1 \
  --output output/smpe/<新的BCMN修复实验>
```

B0、C1–C6 使用固定蓝方来源，M1–M3、N1–N3 保留 sensitivity 配置及旧随机流语义。环境源码变化时，`--allow-compatible-environment` 会先逐帧复算各配置的一局历史测试回放，检查状态、奖励和终局；结果记录在清单中，不表示穷尽验证所有状态。默认仍要求源码严格一致。

`--external-status` 只接受已没有排队任务的实验；新增队列扣除其存活训练进程数，和它共享 `--workers` 总上限。该选项适合一个补充队列接续已有实验，不用于多个彼此不协调的新队列。上例不额外保留中间步数快照，只保留初始、验证最优和最终模型。

管理器通过 `nohup` 后台运行，退出终端不影响实验。`--workers` 控制并发数；较大队伍优先启动，其余任务自动排队。正式训练从头初始化，不复用 MAPPO 或短检查权重。

`--steps`、`--seeds`、`--eval-episodes`、`--heldout-episodes`、`--eval-interval`、`--save-interval` 和 `--log-interval` 可用于较小的流程检查。修改后不能再将实验称为与原参考预算、评估协议完全一致。

```text
output/smpe/<实验名>/
  README.md                 本次记录说明
  experiment.json           来源、预算、协议、配置摘要与算法参数
  status.json               队列、状态、PID 和实际步数
  nohup.log                 管理日志
  configs/                  与参考实验一致的完整环境配置
  _code/                    冻结的项目与外部算法源码
  <配置>/
    logs/                   每个 seed 的控制台日志
    seed_<种子>/            独立训练记录
    figures/                完成后生成的均值 ±SEM 曲线和 CSV
    RESULTS.md              完成后的独立测试报告
  summary.json              全部配置完成后的汇总
```

停止整组任务：向 `status.json` 中的 `manager_pid` 发送 SIGTERM。管理器会终止全部工作进程并记录中断。磁盘低于安全阈值也会停止整组任务；失败不会静默重试。当前不支持精确断点续训。

## 模型、评估与审计

每个 seed 保存 `metrics.jsonl`、`training_episodes.jsonl`、兼容结果格式的 `metrics.json` / `info.json`、完整设置、运行状态和源码快照。不生成 `trajectories/`、`evaluations/`。最终评估保留少量完整回放，保存数量不影响参与胜率统计的局数。

`models/` 包含初始、约 2M / 4M、最终和验证最优模型。完整检查点包括 actor、两个 critic 及目标网络、各智能体 VAE、过滤器、归一化统计、优化器状态和哈希计数。仅保存 actor 不能恢复 SMPE 的推理行为。

验证分别统计采样策略和贪心策略，最优模型按贪心验证胜率、回报依次选择。独立测试不参与选择，主要结果使用最终预算模型，最优模型作为补充。

对局批次长度不同，实际评估步数无法保证逐 seed 相同。曲线按统一预算阈值对齐，实际评估在达到阈值后的完整批次结束时执行；每个实际步数另记在 `test_actual_environment_steps`。偏移少于一个批次的最大步数，没有插值或外推。阴影为三个训练种子的 SEM，不是 128 个测试场景上的置信区间。

```bash
python -m algorithms.smpe.audit --recompute output/smpe/short_check
python -m algorithms.smpe.train \
  --checkpoint output/smpe/short_check/models/best.pt \
  --eval-episodes 32 --output output/smpe/short_check_eval
```

审计检查参数更新、状态模型更新、预算偏移、逐局步数、奖励分量、指标、源码摘要、检查点完整性与回放；`--recompute` 逐帧复算每个回放分组的一局。仅加载可信来源的检查点。

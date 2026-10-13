# SMPE

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

关闭原仓库 VAE 的共享路径自动存盘，改由接入层统一保存完整模型。未改写或修正外部 learner 的内部更新公式；具体源码版本以实验的 `_code/` 和 `_sources/` 为准，不将本接入等同于论文全部实验的复现。

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

```bash
python -m algorithms.smpe.matrix \
  --reference output/mappo/sensitivity_5m \
  --workers 12 --device cpu \
  --output output/smpe/sensitivity_5m
```

参考实验必须是已完成的 13 配置 MAPPO 敏感性实验。默认直接继承其三个种子、5M 对齐预算、验证与独立测试种子、评估局数、评估和模型保存间隔，不重新生成近似的环境配置。启动前会校验环境核心源码与参考实验一致。

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

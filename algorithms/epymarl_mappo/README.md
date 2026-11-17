# EPyMARL MAPPO

15 配置、三个 seed 的通信/规模/角色配比实验见 [统一实验协议](../common/COMM_SCALE_ROSTER.md)，与 SMPE 共享资源调度，结果仍独立于原 MAPPO 目录。

本目录直接调用外部 EPyMARL 的 `PPOLearner`、`BasicMAC`、`EpisodeBatch`、`ReplayBuffer` 和 `mappo.yaml`，不复用 on-policy/MAPPO 的网络或 PPO learner。与原接入并列管理：原实现仍为 `algorithms/mappo/`、`output/mappo/`；本实现使用 `algorithms/epymarl_mappo/`、`output/epymarl_mappo/`。

## 安装与单次训练

命令从项目根目录执行。外部算法仓库不纳入环境安装包。

```bash
python -m pip install -e '.[actor-critic]'
export EPYMARL_ROOT=/path/to/epymarl/src
python -m algorithms.epymarl_mappo.train \
  --config configs/scenarios/6v6_rule_distance.json \
  --steps 5000192 --seed 42 \
  --output output/epymarl_mappo/baseline_seed42
```

也可用 `--epymarl-root /path/to/epymarl` 或其 `src/`。支持 `--device cpu` 和 `--device cuda`。先实测吞吐再选择设备；输出目录必须不存在，不覆盖历史运行。

## 原生训练设置

| 项目 | 设置 |
|---|---|
| 策略 | 参数共享 GRU，隐藏维度 128 |
| critic | 集中式两层 MLP，隐藏维度 128 |
| actor 输入 | 局部观测、合法通信报告及 agent ID，不读取全局状态 |
| critic 输入 | 全局状态与 agent ID |
| 完整对局批次 | 10 个环境、10 个对局；缓冲区容量也是 10 |
| PPO 更新 | 每批 4 轮，裁剪系数 0.2 |
| 学习率、熵系数 | 0.0003、0.001 |
| 价值目标 | 原生 5-step，折扣 0.99 |
| 奖励处理 | 保留原生奖励标准化，不额外标准化回报 |
| 目标 critic | 原生软更新系数 0.01 |

训练时团队奖励每步只记录一次。评分后的 200 步超时是有限时域终局，设置 `add_value_last_step=False`，不额外跨终局 bootstrap。死亡槽位只允许 WAIT。数据转换层将数组转换为 EPyMARL `EpisodeBatch` 要求的张量，不修改 PPO、价值目标或奖励标准化公式。

达到步数预算后完成当前对局批次，实际超出小于 `batch_size_run × max_steps`，默认少于 2,000 个团队步。不乘智能体数，不计验证和测试步数。每次采集新鲜完整批次，不复用旧策略产生的历史对局。

这些设置与 on-policy/MAPPO 接入不同；比较结果是两套具体实现与原生配置的对比，不应归因于某一个超参数。

## B/C/M/N/D 实验

```bash
python -m algorithms.epymarl_mappo.campaign \
  --epymarl-root "$EPYMARL_ROOT" \
  --communication-reference output/smpe/blue_fixed_5m \
  --sensitivity-reference output/smpe/sensitivity_5m \
  --density-reference output/smpe/density_5m \
  --workers 12 \
  --output output/epymarl_mappo/bcmnd_5m
```

三个参考实验必须已完成；这里只读取其配置和评估协议，不加载任何模型或训练数据。

- B0、C1–C6 复制固定蓝方通信配置，蓝方始终 distance、半径 10、无失联，仅改变红方通信。
- M1–M3、N1–N3 保留 sensitivity 的地图、人数、奖励和旧通信随机流兼容设置。
- D0–D3 保留密度受控地图、参考奖励归一化及双方诊断。诊断不进入 actor。

默认 17 个配置，seed 42/43/44，共 51 次独立从头训练，每次目标 5,000,192 团队步。验证 64 局、独立测试 128 局，分别使用参考实验的验证和测试场景种子。最优模型只按验证结果选择，主要结果使用最终固定预算模型。

管理器自动通过 nohup 启动，统一调度单线程工作进程。`--external-status /path/to/status.json` 可与一个已全部启动、没有排队任务的实验共享 `--workers` 总上限。`--conditions B0_baseline D3_12v12 --seeds 42 --steps 16384 --eval-episodes 2 --heldout-episodes 2` 可做小规模检查，不能作为学习或收敛结论。

## 输出与检查

```text
output/epymarl_mappo/<实验名>/
  experiment.json           配置来源、算法参数、预算和评估协议
  status.json               队列、进度、PID 与实际步数
  nohup.log                 管理日志
  _code/                    本次项目及 EPyMARL 源码快照
  configs/                  每个场景的完整原始配置
  <场景>/seed_<种子>/
    config.json             单次运行元数据
    network.json            实际网络结构、维度和参数量
    metrics.jsonl           更新与训练统计
    metrics.json            兼容结果格式的指标序列
    training_episodes.jsonl.gz  无损压缩的逐局统计
    models/                 initial.pt、best.pt、latest.pt
    replays/                少量完整回放和评估汇总
    audit.json              完成后生成的检查结果
  <场景>/figures/            多种子均值与 SEM 曲线及 CSV
  RESULTS.md                全部场景的最终胜率汇总
```

检查点包括 actor、旧策略 actor、critic、目标 critic、优化器、奖励标准化统计和计数器。当前不支持精确断点续训。仅加载可信来源的检查点。

不生成 `trajectories/` 或 `evaluations/`，每组最终评估默认只保留一局可视化回放，不减少参与统计的测试局数。回放文件为 `replays/heldout_latest_greedy/episode_000.json`。

```bash
python -m algorithms.epymarl_mappo.audit --recompute \
  output/epymarl_mappo/bcmnd_5m/B0_baseline/seed_42
EPYMARL_ROOT=/path/to/epymarl/src python -m unittest \
  tests.integrations.epymarl_mappo.test_backend -v
```

检查覆盖 PPO 更新次数、旧策略同步、奖励标准化保存恢复、团队奖励与实际步数、独立测试种子、参数变化和完整回放复算。不同外部算法含同名顶层 Python 包，后端测试应分别在独立进程执行。

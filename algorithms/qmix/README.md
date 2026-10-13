# QMIX 接入与灵敏度实验

本接入复用外部 [MARL-Algorithms](https://github.com/starry-sky6688/MARL-Algorithms) 的 `policy/qmix.py`、共享 GRU 和混合网络，不复制算法损失。环境规则不改变。执行网络读取局部观测、上一动作和 agent ID；全局状态只进入训练混合网络。通信为环境自动目标报告，没有学习消息头。

## 准备与单次训练

使用满足项目要求的 Python，安装环境和 PyTorch，再准备外部源码：

```bash
python -m pip install -e '.[value-decomposition]'
git clone https://github.com/starry-sky6688/MARL-Algorithms.git ../MARL-Algorithms
export MARL_ALGORITHMS_ROOT=/path/to/MARL-Algorithms
python -m algorithms.qmix.train \
  --config configs/scenarios/6v6_rule_distance.json \
  --steps 5000000 --seed 42 --output output/qmix/baseline_seed42
```

也可传 `--marl-root`。输出必须为新目录。核心测试通过不等于完整训练已收敛。

## 固定训练参数

| 参数 | 值 |
|---|---:|
| 共享 GRU 隐藏维度 | 64 |
| QMIX 隐藏维度 | 32 |
| 超网络 | 单层，候选隐藏维度64（单层时不用） |
| 优化器 / 学习率 | 原实现 RMSprop / 0.0005 |
| 折扣因子 | 0.99 |
| 环境数 | 4（批量执行网络，环境按固定顺序推进） |
| 回放容量 / 批量 | 1000完整回合 / 32回合 |
| 预热 | 32完整回合 |
| 更新频率 | 每200团队环境步更新一次 |
| 目标网络更新 | 每200学习更新，沿用原实现计数方式 |
| 探索 | ε从1在50k团队步内线性降至0.05，之后固定 |
| 梯度裁剪 | 10 |

跨场景固定这些参数，仅观测、状态、动作与agent-ID维度按任务变化。QMIX/QTRAN没有 PPO 更新轮数。原实现回放为5000局且重复双精度保存相邻观测；接入统一改为1000局、float32无损存储，相邻观测共享序列。完整回合从首帧训练，保留全长 BPTT，不随机截断RNN历史。

原 QMIX 目标 RNN 从 `o_1` 开始，遗漏 `o_0`；接入层补上目标网络首帧预热，与 QTRAN 原有逻辑一致。目标网络不构建梯度图，损失与优化器仍调用外部 `learn`。死亡槽位保留、局部观测置零且仅WAIT合法；团队奖励只计一次。超时已按任务终局评分，不进行价值bootstrap。预算到达时未完成回合不加入回放，不把预算截断伪造成任务终止。

## 13场景×3种子的后台实验

共享任务矩阵在 `algorithms/common/sensitivity.py`，B0/M1–M3/N1–N3/C1–C6 对应地图、人数、通信对照。保存的JSON均完整展开，双方角色1:1:1，保持当前角色能力、规则蓝方、200步上限、奖励和自动报告。

```bash
python -m algorithms.common.value_decomposition.campaign \
  --marl-root /path/to/MARL-Algorithms \
  --name sensitivity_5m --workers 8
nohup setsid -f python -u -m algorithms.common.value_decomposition.campaign \
  --run output/campaigns/sensitivity_5m_qmix_qtran \
  > output/campaigns/sensitivity_5m_qmix_qtran/nohup.log 2>&1 < /dev/null
```

准备命令创建 QMIX 和 QTRAN 各39个独立任务，并冻结环境、接入与外部算法源码。`--workers` 是两算法合计并发数；每个工作进程PyTorch/BLAS线程均固定为1。`setsid -f` 使管理器脱离当前终端会话，实际PID读取 `status.json`，不使用shell的 `$!`。可以用 `--algorithms qmix` 仅创建本算法，算法根目录仍独立保存。管理器每5秒更新状态、约每5分钟更新报告。磁盘不足2GiB或三次工作进程异常时停止，避免持续失败；不按胜率提前停止。

默认每次精确训练5M团队环境步，不乘智能体数、不含评估。0步和每250k步验证64个固定场景；2M、4M保存周期模型，结束保存 `final.pt`，另存验证最佳 `best.pt`。最终与最优模型均重新加载后，在128个独立测试场景上分别测试贪心策略和 ε=0.05 采样策略。验证与测试场景种子各自固定、彼此不交叉，训练回合使用单独种子区间。评估不进入回放，也不消耗训练探索随机数。

## 记录与汇总

```text
output/qmix/<实验名>/
  experiment.json、configs/        固定参数、完整13场景配置
  status.json、run_status.csv      后台进度、PID、吞吐量
  B0/seed_42/                    其余任务与种子同结构
    config.json、env.json、run.json
    _sources/                    带SHA256的源码快照
    metrics.jsonl、metrics.json、info.json、cout.txt
    training_episodes.jsonl、progress.json
    models/initial.pt、best.pt、step_*.pt、final.pt
    evaluations/validation_*/summary.json
    replays/heldout_{final,best}_{greedy,sampled}/
    summary.json
  figures/                       三组胜率图(PNG/SVG/PDF)、源CSV、QA说明
  independent_test_per_seed.csv、aggregate.json、RESULTS.md
```

主报告模型为固定预算 `final.pt`，`best.pt`仅补充。AUC用包含初始验证点的梯形积分，归一化AUC再除以训练预算；首达80%仅按离散验证点判断，未达到为null。接敌时间以接敌局为条件，击杀比例为击杀数/蓝方总人数。吞吐量区分采集+学习与包含评估、保存的端到端时间。不比较不同人数的绝对伤害/回报来排名。

曲线使用所有种子共同完成的真实验证点，不插值、不外推、不平滑。阴影为独立训练种子的样本标准差/√n；测试地图在种子间共享，SEM不是地图置信区间，阴影重叠不作显著性判定。

```bash
python -m algorithms.common.value_decomposition.report output/qmix/<实验名>
```

对 `status.json` 中 `manager_pid` 发送TERM可停止整组。工作进程保存 `interrupted.pt`，状态为INTERRUPTED；不自动覆盖或重跑已有目录。检查点不含回放，不支持精确续训。正式实验不保存所有训练数组，仅保留逐局指标、源码、模型和有限完整回放，控制磁盘开销。

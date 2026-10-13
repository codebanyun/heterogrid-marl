# MAA2C：EPyMARL 接入与灵敏度实验

复用 EPyMARL 的 `ActorCriticLearner`、`BasicMAC`、共享 GRU actor 和 `CentralVCritic`。不重写算法损失，不改变环境规则。actor 只读取局部观测与 agent ID；全局状态只进入训练 critic。死亡槽位观测为零且仅 WAIT 合法。团队奖励每步只计一次。

## 单次训练

从项目根目录执行，使用已安装环境依赖、PyTorch、PyYAML 的 Python：

```bash
python -m pip install -e '.[actor-critic]'
export EPYMARL_ROOT=/path/to/epymarl/src
python -m algorithms.maa2c.train \
  --config configs/scenarios/6v6_rule_distance.json \
  --steps 5000000 --seed 42 --output output/maa2c/baseline_seed42
```

也可传 `--epymarl-root /path/to/epymarl` 或其 `src/`。输出必须为新目录。无需 Sacred、SMAC/星际客户端、GPU 或 EPyMARL 的其他算法依赖；只导入本任务的 learner 与相应模块。

## 固定参数与 on-policy 契约

| 参数 | MAA2C | COMA |
|---|---:|---:|
| actor GRU / critic 两层隐藏维度 | 64 | 64 |
| 优化器 | Adam | Adam |
| actor 与 critic 学习率 | 0.0005 | 0.0003 |
| gamma / 梯度裁剪 | 0.99 / 10 | 0.99 / 10 |
| 环境数 / 更新批量 | 4 / 4完整回合 | 4 / 4完整回合 |
| 每批更新次数 | 1 | 1 |
| n-step | 5 | 10 |
| 熵系数 | 0.01 | 0.001 |
| target 更新 | 每次更新软更新 tau=0.01 | 每200次学习更新硬同步 |
| 奖励/回报标准化 | 关闭 | 关闭 |
| 上一动作输入 | 关闭 | 关闭 |
| agent ID | 开启 | 开启 |

基础参数读取外部 `config/default.yaml` 和对应算法 YAML，再显式应用本接入的固定设置。跨场景只自动适配观测、状态、动作和 ID 维度，不随地图/人数调整超参数。MAA2C 原 YAML 的 hidden128改为64、原4以外批量统一改为4；COMA保留原学习率/熵/n-step/目标更新方式。双方不额外标准化奖励，避免改变任务奖励及原实现将 padding 计入奖励统计的影响。

四个环境同步开始，各自结束后等待其余环境，全部完成后一次更新。下一批全部重新采集；无回放，无跨更新保留的旧策略轨迹。完整回合从初始隐状态训练，不截断 BPTT。训练采样为合法动作 masked-softmax categorical 策略，与外部 controller 一致；补充测试也是 categorical 采样，不用 QMIX/QTRAN 的 epsilon-greedy。

超时是任务定义的已评分终局，关闭 `add_value_last_step`，终局不 bootstrap。`filled` 只标记真实转移，末帧/padding不进入损失；padding有合法WAIT，避免softmax无合法动作。COMA critic屏蔽被评估智能体的当前动作，保留其他智能体当前动作，使用原反事实基线。

预算和验证/保存点精确按团队环境步计数；必要时拆分向量环境的一步，未推进槽位的 RNN 隐状态也不推进。预算末尾若不足完整一批，则整批不更新，不把预算截断伪装成终局；未用于更新的采集步数单独记录，最多不足800步。检查点用于评估，不支持精确续训。

## 相同的13场景×3种子实验

```bash
python -m algorithms.common.actor_critic.campaign \
  --epymarl-root /path/to/epymarl \
  --name sensitivity_5m --workers 48
nohup setsid -f python -u -m algorithms.common.actor_critic.campaign \
  --run output/campaigns/sensitivity_5m_maa2c_coma \
  > output/campaigns/sensitivity_5m_maa2c_coma/nohup.log 2>&1 < /dev/null
```

默认创建 MAA2C、COMA 各39次独立训练（13场景×42/43/44），每次从头训练5M团队环境步，共390M步。共享任务矩阵在 `algorithms/common/sensitivity.py`，场景与 QMIX/QTRAN 保持一致。双方角色1:1:1、规则蓝方、200步时限、视野/射程/奖励/自动共享不变。无通信时自动报告也不能跨链路。

环境数为每个实验4个、工作进程线程为1；`--workers`控制两算法合计并发，机器资源只影响任务排队，不改变训练超参数。任务启动前冻结环境、接入、配置和EPyMARL源码。管理器每5秒记录状态、每约5分钟更新报告。磁盘不足2GiB或三个worker失败时中止，不按胜率提前停止。向 `status.json` 的 `manager_pid` 发TERM可转发停止；不使用 shell `$!` 判断脱离会话后的PID。

0步和每250k步验证64局，固定种子100042–100105；2M、4M、结束保存模型，另存验证最优。主模型为 `final.pt`，`best.pt`仅补充，按验证贪心胜率、再按回报选择。两模型重载后各做128局独立贪心和128局采样测试，种子200042–200169，不参与选模型。训练回合种子使用 `1000000 + training_seed*100000000 + episode_index`，与评估分离；评估不消耗训练采样随机数。

## 输出

```text
output/maa2c/<实验名>/             COMA在output/coma/下同结构
  experiment.json、configs/       完整任务与固定算法参数
  status.json、run_status.csv     队列状态、PID、步数、吞吐量
  B0/seed_42/                    13任务×3seed
    config.json、env.json、run.json、_sources/
    training_episodes.jsonl、metrics.jsonl、metrics.json、info.json、cout.txt
    progress.json、summary.json
    models/initial.pt、best.pt、step_002000000.pt、step_004000000.pt、final.pt
    evaluations/validation_*/summary.json
    replays/heldout_{final,best}_{greedy,sampled}/
  figures/                       三组PNG/SVG/PDF/TIFF曲线、原始CSV、QA说明
  independent_test_per_seed.csv、aggregate.json、RESULTS.md
```

保留逐局训练统计、逐局验证/独立测试、完整配置与SHA256源码快照、模型和有限完整回放，不持久化训练张量。曲线复用已验证的Python报告流程，只画三个seed共有的实际验证点，均值±样本标准差/√3，不平滑、不插值、不外推。AUC含初始验证，首达80%未达到记null；不同人数的绝对回报/伤害不直接排名，SEM重叠不作显著性判定。

```bash
python -m algorithms.common.value_decomposition.report output/maa2c/<实验名>
```

短程检查只验证训练链路，不证明收敛；测试产物应放临时目录，不混入正式输出。

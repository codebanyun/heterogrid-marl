# 可选的任务迁移接口与MAPPO基线

本功能独立启用。旧`EnvConfig`、单场景JSON、`parallel_env`、AEC、`SMACAdapter`及旧训练入口的默认含义不变；旧实验不自动转换。

## 问题与边界

固定蓝方规则代码和参数，在多个源场景中在线学习同一策略，再到留出的地图尺度、人数、角色组成中零样本评估；也可显式允许目标任务继续训练。每回合内部配置固定。

新增`TaskSuite`管理源/验证/测试划分，`TransferSMACAdapter`在回合边界选择底层环境，并向算法提供固定容量接口。补位槽不是真实单位，不会参与环境物理或蓝方决策。

## 已提供的任务集合

配置位于`configs/transfer/*.suite`。内容是JSON，使用独立扩展名，是因为旧`configs/**/*.json`契约是可单独加载的`EnvConfig`；任务集合不能冒充单场景JSON。

| 集合 | 源任务 | 验证 | 目标任务 |
|---|---|---|---|
| `scale_fixed.suite` | 3v3、6v6；固定24×24×6 | 源配置的独立种子 | 9v9、12v12；同一地图 |
| `scale_density.suite` | 3v3、6v6；近似固定密度 | 源配置的独立种子 | 9v9、12v12；同时放大地图 |
| `map.suite` | 6v6，20×20×6、24×24×6 | 22×22×6 | 28×28×6、32×32×6 |
| `composition.suite` | 红方角色配比2:2:2、1:2:3、3:2:1 | 2:1:3 | 1:3:2、3:1:2 |
| `fct_main.suite` | S1–S6，多地图/人数/配比 | V1–V6，源配置独立种子 | T1–T8，地图/人数/配比及联合变化 |
| `fct_extra_map_blue.suite` | 无（仅测试集合） | 无 | REF、额外地图及蓝方配比测试，以文件 splits 为准 |

角色比例顺序为scout:strike:hybrid；组成组固定蓝方6人及其组成。其他条件从`configs/scaling/density_6v6.json`完整展开，包括固定规则蓝方、通信、奖励归一化和诊断；没有更改该旧文件。密度组不同比放大视野、通信范围或时限，不等同于纯人数变化。

生成自己的集合：

```bash
python -m heterogrid.transfer_scenarios \
  --base configs/scaling/density_6v6.json --family scale_fixed \
  --output output/my_scale.suite
```

支持`scale_fixed`、`scale_density`、`map`、`composition`。输出文件必须不存在。任务定义是展开后的完整配置，不依赖隐式继承；可用`TaskSuite.load`检查或以Python显式构建。

## 环境接口

```python
import numpy as np
from heterogrid import TaskSuite, TransferSMACAdapter

suite = TaskSuite.load("configs/transfer/scale_fixed.suite")
env = TransferSMACAdapter(suite, split="train")
obs, state = env.reset(seed=42)
done = False
while not done:
    actions = [int(np.flatnonzero(mask)[0]) for mask in env.get_avail_actions()]
    reward, done, info = env.step(actions)
print(info["task_id"])
# 无task_id时，在当前允许的任务集合中按回合均匀采样。
env.reset()
env.close()
```

`reset(options={"task_id": "train_6v6"})`显式选择任务；不同任务仅能在上一回合结束后切换。重复reset当前任务仍可重开该任务。默认训练实例不能选择测试任务。只有构造时显式提供`task_ids=[...]`才改变允许集合，用于目标微调；不要把该能力用于零样本测试的数据收集。

任务采样与环境随机数分开管理，固定seed可复现任务序列与场景。显式seed原样传给底层环境，因此可对照相同配置、seed、合法动作下的旧接口物理轨迹。任务名称只出现在元数据/日志，不作为actor特征。

### 容量、槽位和掩码

`TransferCapacity(max_red, max_blue, message_dim)`确定统一接口。默认从整个任务集合推得，正式实验应明确声明容量；这只是公开的接口上界，不是目标任务交互数据。超容量任务报错，不截断、不丢弃单位。

- 红方动作列表始终有`max_red`个槽，不存在的红方槽只能输出wait。
- 动作编号0–6保持旧移动/wait语义；第7+j项始终对应底层blue_j。
- 超出实际蓝方人数的动作被屏蔽，错误提交会直接报错，不转换成另一个真实目标。
- `ally_slots`与`target_slots`表示公开的初始阵容槽位，不表示存活或可见性。旧可变长度接口本来也暴露初始人数。
- 目标特征中的known、direct-visible、age和reported沿用底层本地记录，隐藏敌方的角色、坐标或生命值不会从全局state补入。
- `get_active_mask()`供learner排除死亡与补位样本，不附加全局同队存活真值给actor。
- `map_inverse_span`是公开地图尺寸的倒数，为局部尺度校准提供信息；只有新迁移协议增加该字段，所有新协议基线都获得相同信息。

`get_entity_obs_agent(i)`返回结构化本地观测；`get_obs_agent(i)`按`observation_layout()`顺序展开，适合固定长度基线。自定义实体网络应使用结构化字段及其感知掩码，不能把slot存在误当成敌情已知。

全局state按红方容量、蓝方容量分别补位，末尾追加时间和初始存在槽位，只供critic/混合器使用。`get_comm_mask`仍为通信运输/诊断接口，不应直接放入actor。

12人双方容量、消息长度8时，展开本地输入为379维、state为289维、动作数19；不同实际人数的这些维度相同。旧6v6接口仍是178/133/13，不自动兼容新检查点。

## 在线MAPPO参考流程

新入口为`algorithms.mappo.transfer`。它使用外部MAPPO的actor、critic、PPO更新器，仅增加迁移数据接口、有效样本采集和独立评估流程。**当前可运行基线是前馈MAPPO，不是下述研究模块的完整训练实现，也不支持直接用逐行过滤替代R-MAPPO的序列采样。** 旧R-MAPPO入口保持原状。

源任务在线训练：

```bash
python -m algorithms.mappo.transfer --mappo-root /path/to/on-policy \
  --suite configs/transfer/scale_fixed.suite --output output/transfer/source \
  --steps 1048576 --num-envs 4 --rollout-length 128 \
  --eval-interval 65536
```

默认仅在train任务中采样、在validation任务上评估。各回合均匀抽任务，不保证长短不同任务的环境步数相等，因此日志记录每个任务的实际环境步数。研究中的规模均衡优化是另一个显式机制，不能把此基线声称为已实现该优化。

长训练可以加 `--checkpoint-minutes 30`。该模式在更新边界按间隔覆盖保存 `models/latest.pt`，训练结束时立即保存最终模型。逐轮指标和回合日志使用缓冲，进度以已写盘的记录为准。加 `--skip-initial-evaluation` 可以跳过零步评估。GPU 和 CPU 分别使用 `--device cuda` 与 `--device cpu`，可用 `--threads` 调整 PyTorch 的 CPU 线程数。

冻结模型到测试任务评估：

```bash
python -m algorithms.mappo.transfer --mappo-root /path/to/on-policy \
  --suite configs/transfer/scale_fixed.suite --output output/transfer/test \
  --checkpoint output/transfer/source/models/latest.pt \
  --eval-split test --eval-episodes 100
```

目标任务微调：

```bash
python -m algorithms.mappo.transfer --mappo-root /path/to/on-policy \
  --suite configs/transfer/scale_fixed.suite --output output/transfer/adapt_9 \
  --init-checkpoint output/transfer/source/models/latest.pt \
  --adapt-task test_9v9 --steps 65536
```

三种运行状态明确区分：`online_source_training`、`evaluation`、`target_adaptation`。微调只初始化actor、critic和值归一化统计，使用新优化器；它不是完整resume。源任务和目标任务预算分别记录。

新入口不会用检查点里的源场景覆盖当前suite；但会检查容量、特征布局与网络形状。可在相同接口下指定新地图/角色配置，不支持把旧单场景模型的矩阵强行加载到新接口。旧`algorithms.mappo.train --checkpoint`仍按旧含义运行。

### PPO补位处理

仅在loss乘mask不足以完整处理补位：外部value normalizer可能在计算loss前读取返回值，小批次也可能全为无效槽。新入口的feed-forward buffer先过滤无效样本再采样，避免这两种情况。RNN序列不能这样逐行打散，需单独实现保留时序的采样器后再接入。

## EPyMARL 的循环 MAPPO 对照

`algorithms.mappo.epymarl_transfer` 读取外部 EPyMARL 的 `mappo.yaml`，使用其原生 BasicMAC、GRU、集中式 Critic 和 PPOLearner。HeteroGrid 适配器每轮从各源任务采相同数量的完整回合。训练沿用 EPyMARL 的奖励标准化和团队时间步掩码，补位和死亡槽只能等待。评估统一为贪心动作，配置时限按任务终止处理。

```bash
python -m algorithms.mappo.epymarl_transfer --epymarl-root /path/to/EPyMARL \
  --suite configs/transfer/fct_main.suite --output output/epymarl_mappo_5m_seed42 \
  --device cuda --seed 42 --steps 5000000 --episodes-per-task 2 \
  --checkpoint-minutes 30 --eval-interval 250000 \
  --eval-episodes 10 --eval-seed 50001 --test-episodes 100 --test-seed 100001
```

每半小时在整轮更新边界覆盖保存 `models/latest.pt`，训练结束时保存最终检查点并评估验证集和测试集。源训练预算计团队环境步，最后一轮采满完整回合。EPyMARL 对照与 FCT 和前馈 MAPPO 共用场景与评估种子，优化器、归一化和损失掩码使用各自实现。

## 记录和评估

输出目录必须新建。保存完整suite、接口schema、训练参数、源检查点路径、源码快照、每任务环境步数、回合记录、模型和按任务分组的评估。默认验证种子起点100042、测试起点200042，可显式覆盖；不按测试成绩选择best模型。评估不更新模型，且恢复训练随机数状态。

主指标包括胜率、回报、接触率、首次接触时间和超时率。短训练只验证链路，不证明收敛或泛化效果。请将地图、人数、组成和联合变化分别报告，不用总体平均掩盖某一组失败。

## 研究算法边界

研究算法放在独立仓库，本仓库提供环境接口、任务配置和标准基线适配。

远端环境新增的终止选项默认关闭。已有研究应继续使用冻结的环境、完整套件和检查点协议；不要混用不同版本自动序列化的配置进行无记录续训。

## 合并验证记录（2026-11-17）

在远端 `a321eb1` 基础上集成迁移适配。Windows/Python 3.11/PyTorch 2.6 环境下，环境回归 101 项通过；补位过滤、日志续写回滚、原生 EPyMARL learner/RNG 恢复和 COMA 动作 one-hot 专项 4 项通过。

同一机器、相同可选后端设置下，远端原始版本全量 272 项测试与集成后 283 项测试均为 17 failures、35 errors，失败用例集合相同，没有新增失败。原有失败涉及 Windows 路径、符号链接权限、Linux 专用 fcntl 和测试后端路径等条件；这不代表全量测试通过。EPyMARL 专项使用实际可用的 src 路径单独验证通过。

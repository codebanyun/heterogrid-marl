# robust_comm_v1：本环境定向适配

这是经用户授权的整合方法变体，不是原始 DIB 的等价计算优化，也不是单因素消融。
不要把其结果归入原生 DIB、原48次实验或先前2.5M筛查。

## 不变的实验条件

环境与执行观测、动作空间、目标记录攻击限制、奖励、角色能力、蓝方 rule 均不变。
蓝方固定 distance / radius10 / dropout0；C1/C5/C6只改红方通信。
hidden/embed/AHS/teacher=128，z=64、K=4、4环境、PPO4轮、lr3e-4不变。
执行只使用自己的局部观测、历史和上一动作。训练监督能访问其他节点观测和本节点未来，
但这些信息不传入执行actor，不生成虚拟报告，不用隐藏位置评分或开火。

## 紧凑结构化监督

每个其他智能体的目标由完整观测转为：

- 自身：存活标签、绝对位置xyz、HP，共5维；不重建可由固定ID确定的角色/能力与公共时间。
- 队友槽：可见标签、相对位置xyz、HP，每槽5维。
- 敌目标记录：有记录标签、直接可见标签、相对位置xyz、HP、年龄、报告来源标签，每槽8维。
- 不重建未使用的通用消息content、消息mask和年龄。自动目标共享仍在目标记录中，未被删除。

6v6每观测178维不变；每个其他节点的辅助目标78维，5个节点共390维，而非890维。
这是训练目标压缩，绝不是缩短执行观测。网络的probe/decoder输出头因此变小。
当前实现明确要求人数对等、message_dim=8；其他布局报错，不静默套错索引。

分类使用BCE logits，正标签权重固定3，负标签全部保留。主体存活标签始终学习；
主体阵亡后不回归其属性。队友属性仅可见时回归；敌属性/可见/来源仅记录存在时监督。
坐标的真实零值保留，不能按数值是否为零屏蔽。连续量使用sigmoid/tanh对应其物理范围，
用字段组尺度标准化后Smooth L1；组均衡避免高维空槽按维数压倒少数有效字段。
位置/HP/年龄尺度分别为.2/.25/.5（归一化观测单位），跨场景固定。
AHS probe连续不确定性logstd限定[-2,1]，使用相对NLL（只加常数使值非负），
分类不确定性用Bernoulli熵，而非未训练的分类logstd。
teacher重建与hypothesis混合监督使用相同结构化目标；distillation目标值detach。

## MHF证据与假设

动力学输出本节点下一步的紧凑实体观测，不再回归所有178维raw字段。
在线只用自身完成的transition评分：自身HP、队友可见性/有效属性、目标记录与有效属性。
滤波评分排除主要由自身动作决定的位置；正负可见性/记录标签均保留。
无记录时的属性不参与似然，不能利用训练时的其他节点mask在执行消歧。

采用稳定的 `softmax(log(pi)-grouped_error/0.25)`；候选相同且证据相同则保持原相对权重，
首步无历史时直接用proposal。没有乘前一时刻权重，也不对每次候选误差强制标准化来制造胜者。
这是分组BCE/Huber能量评分，不声称与原稿各向同性高斯似然完全等价。
假设混合目标同样使用分组能量。分离距离基于可解释的解码概率/属性、训练标签的有效字段，
按有效维数取RMS，margin=.15；不会强迫假设采取不同动作。

## 残差安全接入与训练

在原基础策略与信念分支之外加257参数的局部信任头：
`sigmoid(Linear([base_hidden.detach(), context.detach()]))`，权重零初始化、bias=-2。
此头只受PPO任务目标训练，不接受特权标签；不按后验熵强制抑制不确定场景。
残差为 `0.5 * schedule_gate * trust * (2*tanh(raw_residual/2))`，单动作修正绝对值不超过1。
原PPO剪切/KL/critic目标不变；不能保证这个保守界限就是最优，效果由新训练验证。

调度按采集批次起点，和old-MAC/行为策略严格一致：

| 时段 | 辅助学习 | 残差门控 |
|---|---|---|
| 0–1M | 关闭 | 0 |
| 1–1.5M | 从0渐入1 | 0，先训练信念 |
| 1.5–3M | 按3.5M衰减周期缓降 | 从0渐入1 |
| 3–5M | 继续衰减，5M到0.2 | 1，但仍乘学习得到的trust与有界残差 |

移除按PPO标量绝对值限制辅助loss的原cap；保留原lambda_ahs/bbd/hyp=.2，lambda_fwd=.1。
基础策略、信念分支分别做norm5梯度裁剪，norm以float64计算，避免FP32平方和溢出，
也避免辅助梯度缩小基础策略更新。非有限梯度直接异常中止，不静默跳过。
actor/辅助用动作前存活mask；团队critic仍学习完整团队后续回报。
熵系数从.01渐变到.003（4M），不使用原.001下限；AHS取消长度惩罚、sharpness=.001。
这些都是明确的训练协议变更，不能宣称原始算法、同参数容量或单项因果结论。

## 配置与启动

原CLI默认仍为active_mask_1m；仅显式 `--training-profile robust_comm_v1` 启用本变体。
旧native backend、旧profile、两个已有冻结campaign均不改写；不在运行进程上热切换本变体。

```bash
/root/miniconda3/envs/smpe-py39/bin/python -m algorithms.dib_mappo.robust_campaign \
  --dib-root /root/research/DIB-MAPPO/src --workers 9
# 使用返回coordinator的_runtime作为cwd，以nohup运行（路径不可指向旧实验）
nohup /root/miniconda3/envs/smpe-py39/bin/python -u -m algorithms.dib_mappo.robust_campaign \
  --run <coordinator> > <coordinator>/nohup.log 2>&1 < /dev/null &
```

默认C1/C5/C6×seed42/43/44，9次从头5M，共45M团队步，不继承模型。
每250k步64局验证；每模型128局贪心与128局采样独立测试；主模型final，best补充。
验证500042起、测试600042起；测试不用于选择配置。不因胜率早停。
输出 `output/dib_mappo/20261023_robust_comm_v1_5m/`，单任务`<场景>/seed_<seed>/`。
保存压缩原始训练数据、权重、配置、验证/测试、诊断，源码及其SHA另行冻结。
管理器共用2GiB磁盘预留和3次异常停止派发保护；不要用权重检查点声称精确续训。

新增训练诊断：`mhf_weight_update_l1`、`mhf_posterior_entropy`、`hyp_policy_tv`、
`belief_trust_mean`、`ahs_rho_mean`、`structured_other_alive_fraction`、
`base_actor_grad_norm`与`belief_actor_grad_norm`。低重建loss或尖锐权重本身不等于有效。

测试入口：`python -m unittest tests.integrations.test_dib_robust -v`。
涵盖真实零坐标梯度、负标签保留、无效属性零直接梯度、阵亡目标语义、
首步/等证据过滤、局部执行隔离、门控与预热、巨大有限梯度，以及native/序列完整PPO更新对照。

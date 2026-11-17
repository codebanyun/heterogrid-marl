# DIB Tactical-v2：C组优化与验证协议

本变体针对 C1（none）、C3（distance5）、C5（节点失联25%）、C6（节点失联50%）。
目标是超过本仓库已有算法的共同测试结果，不承诺达到全领域SOTA或统计显著性。
这是新的 DIB 方法变体，不是原始DIB等价计算优化，不能混入原48次、2.5M筛查或robust_comm_v1。

## 实验与信息边界

仍用24×24×6、6v6、三角色1:1:1、max_steps200、原奖励、自动目标共享。
蓝方rule、distance10、dropout0固定。环境/角色能力/观测/动作/开火掩码均不修改。
actor仅读自身当前局部观测、历史、上一动作、ID，不查全局state、隐藏敌情或其他节点实时输入。
不加手写执行策略，不以推断代替目标报告，不改变战斗命中规则。
新增攻击头bias=1只是可训练初始化偏好，不是“有攻击就强制开火”。
全局state仍仅供critic；其他节点的观测/动作是训练监督，未来上下文仅供训练teacher。
没有继承旧模型、其他场景模型或其他训练seed的模型。

## 方法变更

### 实体化基础策略与动作头

基础编码器分别处理自身、可见队友、有效敌记录，并按mask聚合实体。
输入保留自身上一动作。GRU128递推；移动/等待7个logits单独生成，攻击logits按敌实体共享头生成。
不再让一个扁平输出层从稀疏178维输入分别学习每个目标ID的攻击映射。
共享攻击线性层按上下文/目标拆分投影，保持同一层权重和输出，仅避免重复投影。
缺失实体不进入聚合；最终仍严格使用原环境可用动作mask。

### 流式AHS与轨迹BBD

AHS在最近4–16个**因果GRU状态**上做自适应窗口注意力，不再对每一步的16个原始token重新跑窗口GRU。
历史状态本身携带因果前缀，不读未来；执行与训练重放有一致性测试。
完整有效episode的双向GRU作为训练teacher，替代原8步过去+4步未来窗口。
用packed sequence隔离padding，死亡后的真实零观测与回合结束后的padding区别对待。
teacher监督仍为紧凑实体观测；额外预测其他存活节点当前实际动作，给信念增加协同意图任务。
其他节点动作仅在辅助训练中使用，不输入actor、在线滤波或执行记忆。
这两项改变了原始AHS/BBD定义，论文复现必须保留旧profile。

### MHF专门化与安全接入

K=4、z=64。候选使用正交初始化的可训练锚点加有界残差，学生std下限.25，避免起点近乎重合。
结构化目标保持robust_comm_v1的字段语义：记录/可见/存活标签保留负样本，属性仅有定义时回归，真实零坐标不屏蔽。
混合监督包括best-of-K预测、赢家proposal分配损失、软混合项和弱批次负载平衡；不强制每个状态权重均匀或尖锐。
在线权重只由自身已完成transition的实体证据与proposal计算，不读取监督标签，也不递推未对齐的旧候选权重。
有界信任残差继续保留：基础策略可在信念不可靠时减少其影响。
不以“假设距离大/权重变化大”代替胜率验证。

### PPO与价值估计

统一gamma=.99、GAE lambda=.95，回合内计算一次returns与存活样本归一化优势，PPO4轮不重新移动优势目标。
公共团队奖励使用一个共享团队V，避免同一团队回报因agent-ID产生不同价值噪声。
value clip=.2、policy clip=.2、KL=.03、Adam lr5e-4，4个新完整回合一批，不用回放。
存活mask取动作前HP，死亡发生当步仍学习；critic不因单个节点死亡截断团队回报。
基础策略与信念分支分别用float64 norm裁剪，非有限更新中止，不静默跳过。
环境奖励不变、没有额外塑形；GAE和auxiliary是明确的算法训练变更。

### 5M内调度与搜索参数

- 0–50k：辅助关闭、残差关闭。
- 50k–250k：辅助从0渐入1；基础策略仍不接收残差。
- 250k–1M：残差从0渐入1，仍受trust与有界幅度限制。
- 之后：辅助按4.75M周期渐降；熵系数.01到.003，3M完成退火。

这是为5M重新分配预算，不沿用旧版1M/2M预热；旧版不被改写。
网络隐藏/embedding/teacher128、K4、z64、4环境、PPO4轮对所有C场景固定。
三个整合参数preset只改变辅助总强度：policy_first=.03、balanced=.10、belief_first=.30。
不是逐模块消融，不允许按场景/seed分别挑测试最优配置拼接。

## 固定预算、选择与测试

默认3 preset×4场景×3训练seed（42/43/44），**36次从头5M，共180M团队训练步**。
这是显式搜索总成本；单模型预算5M不能掩盖参数搜索成本。
每约250k（下一个完整批次边界，偏移<800步）验证64局，种子700042–700105。
2M/4M附近和结束存模型；预算尾部精确停在5M，不将部分回合伪造成终局用于更新。

全部训练结束后，只根据各seed最后3个验证点的均值选一套全场景统一preset：
先求每场景均值，按目标C1=.05/C3=.75/C5=.40/C6=.05归一化，上限1.5，再取四场景平均；平分用验证平均回报。
规则在训练前冻结。缺失场景/seed的preset不能只用成功样本参与选择。
然后才对选定12个模型的final/best分别测试128局贪心、128局采样，种子800042–800169。
主报告是final；best仅补充。独立测试不参与配置或模型选择。

QMIX、QTRAN、MAA2C、COMA和可用的原DIB fixed5M final模型，在同一800042测试集只读重评。
缺失的原DIB C6 seed43不填0、不重命名4M模型为final。原模型与原结果都不改写。
结果表必须披露容量不匹配、额外搜索成本和缺失重复；均值领先不等于显著或全球SOTA。

## 后台启动与恢复

```bash
/root/miniconda3/envs/smpe-py39/bin/python -m algorithms.dib_mappo.tactical_campaign \
  --name 20261103_tactical_v2_search_5m --workers 36 --matched-baselines

cd output/campaigns/20261103_tactical_v2_search_5m_dib_mappo/_runtime
nohup /root/miniconda3/envs/smpe-py39/bin/python -u -m algorithms.dib_mappo.tactical_campaign \
  --run /root/heterogrid-marl/output/campaigns/20261103_tactical_v2_search_5m_dib_mappo \
  > ../nohup.log 2>&1 < /dev/null &
```

状态：coordinator/status.json、结果root/status.json、每任务progress.json。
root为 `output/dib_mappo/20261103_tactical_v2_search_5m/`；子目录 `<preset>/<C场景>/seed_<seed>/`。
选择后生成 `selection.json` 和 `selected` 链接；测试完成自动更新 `RESULTS.md`。
基准重评在 `matched_baselines/`，provenance包含原final模型SHA。

每50k附近的完整rollout/update边界写原子resume.pt，并保留resume_previous.pt。
检查点含actor、old actor、critic、两个Adam、torch/NumPy/Python RNG、下一episode seed索引、计数、验证历史和日志段清单。
恢复严格核对源码、环境、参数、预算、版本身份；不支持把权重检查点冒充精确续训。
失去检查点之后尚未提交的轨迹需重算；原不完整日志保留归档，不混入canonical记录。
训练过程中有效封存段在training_parts/，完成后合并为training_episodes.jsonl.gz。

重启机器后，在原冻结_runtime显式恢复：

```bash
nohup /root/miniconda3/envs/smpe-py39/bin/python -u -m algorithms.dib_mappo.tactical_campaign \
  --resume-run <同一coordinator> >> <同一coordinator>/nohup.log 2>&1 < /dev/null &
```

文件锁和命令行/PID校验防止重复启动；source SHA不一致时拒绝继续。
磁盘2GiB预留、3次异常停止派发；SIGTERM在完整批次边界退出。没有安装系统服务或修改启动配置。
不得直接改冻结运行时或在旧任务上热切换本变体。

## 测试与局限

测试覆盖因果前缀不变、其他节点输入隔离、实体攻击置换、padding与终局GAE、
native/序列完整更新一致、优化器与随机数逐位恢复、完整矩阵验证选择不读heldout。
功能测试和短训练不证明胜率改善。C1仍受本地目标记录限制；增加候选不能创造从未观测过的信息。
若独立结果没有超过基准，必须报告未达到目标，不能改测试条件、挑seed或按测试结果换preset。

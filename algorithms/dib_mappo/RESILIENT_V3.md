# DIB Resilient-v3：C1/C6后续研究

本版本基于Tactical-v2，是新的整合算法变体，不是原始DIB等价优化，也不改写旧结果。
目标是提高C1胜率、降低C6跨seed失败率；没有承诺所有seed达到80%。

## 证据与信息边界

对旧belief_first模型在910042起32个开发场景的只读行为复查发现：

- C1三个seed的战斗单位全部阵亡时间均值约32.04、45.58、44.84步。
  许多失败局随后只剩无攻击能力的侦察单位，仍运行到200步。
- C1打击/一体单位有攻击机会时的开火比例约98.5%–100%，不是普遍不会开火。
- C6成功seed42在开发集胜率84.375%；失败seed43为0%，seed44为3.125%。
  seed43许多角色持续上下移动，seed44多向远端边界移动；并不是把43/44换名成42即可解决。
- 旧训练末段多假设动作分布TV通常很小，潜变量不同不等于有用的策略分歧。

这些是诊断关联，不是单因素因果证明。910042起开发场景可用于调参，不作为最终独立测试。
此前已公开800042起测试结果，不能继续将其当成新方法未见过的测试集。

仍固定蓝方rule、distance10、dropout0；红方C1为none，C6为full+节点失联50%。
地图24×24×6，6v6、三角色1:1:1、原奖励、原动作/观测/射击限制、200步上限全部保持。
没有调用专家动作，没有加载旧模型，没有插入集结/追击/优先开火脚本。
actor仍仅接收自身合法观测、上一动作、ID及自己的历史。
训练时critic/teacher可集中使用数据，不把其他节点实时观测或隐藏敌情送入执行策略。

## 整合改动

### 1. 战斗能力耗尽后的训练样本掩码

动作前所有红方有伤害能力的单位均已阵亡时，之后所有actor/auxiliary样本权重设为0。
之前的所有存活角色都仍训练，包括侦察单位；导致最后战斗单位死亡的动作仍保留。
判断只在训练batch上进行，执行不读队友隐藏存活状态。

不会提前终止环境、修改winner或伪造terminal。完整环境步仍计入5M，完整奖励仍记录。
critic/GAE仍使用真实完整轨迹和真实200步超时结算。
teacher有效未来上下文也只到最后战斗能力耗尽前的有效前缀。
这是明确的学习样本重加权，不声称与旧奖励优化目标严格等价；省略了只能影响失败结束时机的后段actor决策。

### 2. 按episode均衡的PPO权重

每个非空episode的actor权重总和为1，再在batch内求均值。
加权优势均值/方差及PPO KL也按同样权重计算。
避免200步失败局仅因长度和存活侦察样本多，就压过几十步的成功局。
辅助目标仍按有效actor样本平均，不按失败局长度刻意放大。

### 3. 长时胜负信用与批量稳定

gamma=.997、GAE lambda=.99、Adam lr3e-4、8个新完整episode/更新、PPO4轮。
对照Tactical-v2为gamma=.99、lambda=.95、lr5e-4、4个episode。
指数GAE尺度约从16.8步提高到77.1步，200步terminal折扣约从.134提高到.548。
这是改变折扣/训练目标，不是新增或修改环境奖励。
熵系数从.02退火至.005，4M完成；clip=.2、KL=.03维持。

### 4. 局部几何候选动作编码

为等待和六个单格移动动作计算候选位移、移动后自身坐标、
相对于合法可见队友/有效敌记录的移动前后距离及相对方向。
按真实地图轴尺度恢复几何单位，缺失/已记录死亡实体不进入聚合。
所有特征只由同一节点合法局部观测与公开地图/动作几何产生。
共享MLP输出三种角色的候选分数，再由自身角色选择；没有固定偏好任何轴、边界、集结点或敌人。
攻击头保持Tactical-v2实体共享头；最终仍使用环境原有动作mask。
DIB的AHS/BBD/MHF、在线证据滤波和有界信任残差继续工作。
没有额外强迫假设动作多样性的损失；也不把猜测写入环境目标记录。

## 正式协议与输出

只用一套整合配置：辅助总强度belief_first=.30；所有场景/seed同一超参数。
C1/C6 × seed42/43/44，6次从头5M，共30M团队环境步，不能用成功seed权重初始化失败seed。
每约250k附近64局开发验证，8回合完整批次导致偏移<1600步。
开发验证种子910042–910105；新独立测试1800042–1800169。
单一preset不再进行多配置挑选，final为主模型、验证best仅补充。
不按胜率提前停止或延长个别seed预算。
训练完成后每模型128局贪心+128局采样；逐seed完整呈现，特别核查C6是否每个seed均达到80%。
旧四基准、原DIB及Tactical-v2 belief_first final在同一新测试集只读重评。
本轮是针对已公开失败结果的自适应后续研究，额外训练成本单列，不能混作原始预注册比较。

```bash
/root/miniconda3/envs/smpe-py39/bin/python -m algorithms.dib_mappo.tactical_campaign \
  --name 20261107_resilient_v3_c1_c6_5m --training-profile resilient_v3 \
  --scenarios C1 C6 --presets belief_first --seeds 42 43 44 --workers 6 \
  --eval-seed 910042 --heldout-seed 1800042 --matched-baselines \
  --reference-root /root/heterogrid-marl/output/dib_mappo/20261103_tactical_v2_search_5m/belief_first

cd output/campaigns/20261107_resilient_v3_c1_c6_5m_dib_mappo/_runtime
nohup /root/miniconda3/envs/smpe-py39/bin/python -u -m algorithms.dib_mappo.tactical_campaign \
  --run /root/heterogrid-marl/output/campaigns/20261107_resilient_v3_c1_c6_5m_dib_mappo \
  > ../nohup.log 2>&1 < /dev/null &
```

结果位于output/dib_mappo/20261107_resilient_v3_c1_c6_5m/；内部仍为preset/场景/seed。
source SHA冻结、每50k附近完整边界保存Adam/RNG、原子检查点和--resume-run机制沿用Tactical-v2。
源码不一致拒绝续训。磁盘保留2GiB，三个异常时停止派发；没有自动重启系统服务。
本机启动前磁盘仅余约4.3GiB，因此本轮收敛为6次整合实验，不铺开新的36次搜索，也不删除旧数据。

功能测试、短训练、候选几何特征或非零梯度都不证明胜率提升。
若C1仍低、C6部分seed仍失败，应完整报告，不能按独立测试结果改配置、挑seed或换best代替final。

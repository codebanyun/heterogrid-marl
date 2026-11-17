# DIB-MAPPO v2

通过 `SMACAdapter` 接入外部 `/root/research/DIB-MAPPO/src` 的原生
`DIBMAC`、`DIBRecurrentAgent` 和 `DIBPPOLearner`，启用 AHS、BBD、MHF。
不是用普通 MAPPO 替换 DIB，也不改蓝方 rule 策略、物理通信或角色能力。

2026-11-07新增 `resilient_v3`：针对C1失败局战斗能力提前耗尽、C6跨seed不稳定，
整合局部几何动作编码、战斗能力训练掩码、episode均衡PPO、长时GAE和8回合批次。
这是明确的新训练目标/网络变体，环境和执行信息不变，不加载成功seed的权重。
只跑C1/C6各3seed、固定辅助.30，共6次从头5M（30M总预算）；使用新独立测试种子。
方法、诊断依据和完整命令见[RESILIENT_V3.md](RESILIENT_V3.md)。

2026-11-03新增明确方法变体 `tactical_v2`：实体动作头、流式AHS、完整有效轨迹teacher、
锚定MHF、固定GAE优势与共享团队V。新CLI和精确续训协议见[TACTICAL_V2.md](TACTICAL_V2.md)。
默认C1/C3/C5/C6、三个训练seed、三个辅助强度preset，共36次5M；只按验证选一套配置后再独立测试。
这是180M步显式参数搜索，不是原算法等价优化，也不能预先宣称SOTA。旧默认profile和冻结任务不变。

2026-11-07对上轮已选定的 `tactical_v2 / belief_first=.30` 做[全组固定配置扩展](FULL_GROUPS_V2.md)：
B/M/N/C2/C4/D1-D3共36次新5M训练；C3/C5复用旧结果，D0复用B0，C1/C6留给独立v3。
用 `python -m algorithms.dib_mappo.full_campaign --matched-baselines` 准备，再nohup运行。
计算源码直接复制原冻结版本，不混入v3；compact存储不改变训练计算，细节见链接。

2026-10-23新增独立 `robust_comm_v1`：针对无通信/受干扰的结构化目标、局部证据滤波、
有界信任残差和训练稳定性整合变体。**不是上游等价优化**，仅显式profile启用，
不覆盖原48次或2.5M筛查；完整配置、信息边界、启动命令见[ROBUST_COMM.md](ROBUST_COMM.md)。

## 环境与固定参数

环境配置仍使用 [环境说明](../../docs/environment.md) 的 `EnvConfig` JSON。
通信实验固定蓝方 `distance / radius=10 / drop_prob=0`，只改变红方；原有
B/M/N/C 共13场景保留旧奖励。D1/D0/D2/D3 分别为 3/6/9/12 人，地图
19×19×5 / 24×24×6 / 27×27×7 / 30×30×8，使用 `reference` 奖励归一化。
全部场景启用双方诊断，不改变观测、奖励或动力学。

保留上游 `dib_mappo_v2.yaml` 的网络和优化设置：hidden/embed/AHS/teacher128、
z64、4个假设、lr=3e-4、gamma=.99、PPO4轮、clip=.2、KL阈值=.03，
以及固定2M步渐入。原始 `legacy` 为2M预热；2026-10-21起新运行默认
`active_mask_1m`，为1M预热并启用动作前存活掩码。各场景只自动适配输入输出维度。
**该原生128维结构不是此前64维基线的等参数量对照。**

统一适配为4环境，采集4个新的完整回合后调用一次原生 learner，无经验回放。
与上游默认运行器的明确差异：关闭奖励/return二次标准化，200步超时视为任务终局，
不做末步 bootstrap；与本仓库此前实验的环境奖励和终局定义保持一致。
调度使用当前采集批次的起始团队步，整批残差门控固定，保证 PPO old-MAC
重放与实际采集策略一致；最大不足800步的预算尾部不更新，不伪造终局。

执行阶段始终调用原生局部 actor，只使用自身历史，不读取其他节点观测或未来。
teacher/其他智能体信息仅用于原生训练辅助目标，不绕过环境通信链路。
训练重放将原生滑动历史窗口和前馈层批处理，AHS使用固定256窗口分块和反向重计算
降低激活内存（不切断梯度）；基础 GRU 使用相同参数的单层序列核，仍按时间因果递推，
非有限轨迹回退到原生逐步恢复路径。序列优化保留原生损失方程、PPO/KL逻辑及优化器时钟；
预热和有效样本掩码由显式训练profile决定，不把这两项变更声称为等价性能优化。
`tests/integrations/test_dib_mappo.py` 对照原生输出、辅助损失、PPO参数更新、
预热到联合训练的 Adam 时钟，并检查局部执行与未来观测隔离。

### 等价计算优化与独立测速

当前工作区优化只改变计算组织，不改变网络参数、损失定义、PPO轮数、n-step、
环境数、预热调度或训练预算：

- 历史分块用 `split`，时序访问用序列核／`unbind`，避免重复构造整段零梯度张量。
- Q/K/V按token投影后复用历史窗口，不对同一token重复做最多16次线性投影；
  零历史槽仍带原生线性层bias，不把无效槽偷偷改成零向量。
- 原生教师只使用窗口中心输出。前向GRU只需到中心，反向GRU只需从末尾到中心；
  后续不影响中心的状态不计算，不改变教师可见的过去／未来或任何有效梯度。
- 复用完整辅助张量及已计算的student解码；old-MAC不计算策略未使用的重建输出。
- 自身两两距离保留原生 `cdist` 前向值，合并两个绑定输入的梯度累加；保留零距离子梯度，
  通过双精度 `gradcheck`，不改间隔、损失权重或假设数。
- n-step按相同的逐项加法顺序批处理起始时间，终局mask及末步bootstrap语义不变。

独立测速不写正式实验目录，不修改存活worker。例：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /root/miniconda3/envs/smpe-py39/bin/python -m algorithms.dib_mappo.benchmark \
  --implementation optimized --dib-root /root/research/DIB-MAPPO/src \
  --agents 6 --length 200 --repeats 2 --output /tmp/dib_speed_6.json
```

`--implementation frozen --frozen-runtime <旧快照目录>` 对照旧运行版本；
`--implementation native` 对照完全原生重放；优化版再加
`--compare-frozen --frozen-runtime <旧快照目录>`，会在同一固定批次上检查
完整PPO更新后的actor、critic、target-critic与Adam状态。
默认4环境×200步、PPO4轮、单线程、固定模型初始化；输出实际执行轮数、耗时、
进程累计峰值RSS和优化源码hash。测试用固定合法WAIT轨迹，只反映计算成本，
不是胜率或正式训练结果。背景负载会影响墙钟耗时，不承诺正式训练等比例加速。
工作区优化不会自动进入已冻结的campaign；不要直接改写正式运行快照。
2026-10-19的测量结果、资源口径和数值容差见 [PERFORMANCE.md](PERFORMANCE.md)。

### 已运行队列的显式切换

检查点没有Adam状态，不能用停止后重载模型的方式声称“无损续训”。
`deploy_upgrade.py` 提供本机CPython/GDB专用的一次性切换工具：
先复制原冻结运行时，再只覆盖DIB适配目录，保留环境代码和外部后端；
源码hash、PID启动时间和命令行均校验后，提交Python pending call。
运行中的任务在下一批四个完整回合结束后安装等价内核，旧批次更新不被中断。
网络、参数对象、两个Adam优化器及其矩/时钟、全部随机数状态、采集批次、
环境对象、计数及门控在安装前后做精确hash/身份审计，首次新更新还检查有限值和计数。
安装审计失败会恢复旧方法；不要跳过小型真实队列预检和单任务试切。

```bash
python -m algorithms.dib_mappo.deploy_upgrade <coordinator> --prepare
python -m algorithms.dib_mappo.deploy_upgrade <coordinator> --inject <manager_pid>
python -m algorithms.dib_mappo.deploy_upgrade <coordinator> --inject <pilot_worker_pid>
# pilot 的 optimization_switch.json 达到 VERIFIED 后才切换其余任务
python -m algorithms.dib_mappo.deploy_upgrade <coordinator> --inject-workers
# 汇总每个任务的实际切换步数到管理器状态
python -m algorithms.dib_mappo.deploy_upgrade <coordinator> --inject <manager_pid>
```

不要直接修改旧 `_runtime` 或训练源快照，也不要重新启动管理器。
管理器内存中的排队任务改指向新快照；已运行任务保持原PID和状态。
总清单为 `optimization_upgrade.json`，单任务证据为 `optimization_switch.json`；
`VERIFIED` 且 `state_before == state_after` 才表示切换审计通过。
这保留切换瞬间状态，但浮点运算组织改变后不保证后续训练轨迹逐位相同。

## 单次运行

新环境安装可用 `pip install -e '.[dib-mappo]'`，外部 DIB 源码另行提供。
本机复用已有 `smpe-py39` 环境，不升级其共享依赖。

```bash
conda run --no-capture-output -n smpe-py39 python -m algorithms.dib_mappo.train \
  --dib-root /root/research/DIB-MAPPO/src \
  --config configs/scenarios/6v6_rule_distance.json \
  --output output/dib_mappo/single_run --seed 42 --training-profile active_mask_1m
```

默认精确5M团队环境步（不乘人数、不含评估）；初始及每250k步64局固定验证，
2M/4M/结束保存模型，验证最优另存 `best.pt`。主报告是固定预算 `final.pt`。
两模型重载后各做128局独立贪心测试和128局 categorical 采样测试。
训练种子42/43/44；验证地图100042–100105，测试地图200042–200169。
只按异常中止，胜率不触发提前结束；原生 KL 仅限制单批 PPO 更新轮次。

### 存活掩码、1M预热和MHF审计

`--training-profile` 是训练协议，不改变动作空间、观测、通信、奖励或角色能力：

| profile | actor/辅助掩码 | 辅助与残差起点 | 渐入长度 |
|---|---|---:|---:|
| `legacy` | 原始回合mask | 2M | 2M |
| `active_mask` | 回合mask×动作前存活 | 2M | 2M |
| `warmup1m` | 原始回合mask | 1M | 2M |
| `active_mask_1m` | 回合mask×动作前存活 | 1M | 2M |

存活取自HeteroGrid动作前自身HP（局部观测index3），不是下一步是否存活。
死亡发生当步保留；阵亡后不参与actor/辅助loss、优势归一化及策略统计。
critic学习的是团队后续回报，因此仍用回合mask，不因某个节点死亡而截断n-step。
当前适配不能直接用于其他环境的观测布局。全无效批次安全跳过，正常采集每批初态均有存活节点。

1M版本在3M时门控全开；仍沿用原4M辅助衰减周期，因此5M的辅助scale为0.6，
legacy在5M为0.8。调度时钟仍为采集批次起点，整批行为策略与old-MAC一致。
Python API `settings(root)`／未指定profile的 `build_policy` 保持legacy兼容；
新的train/campaign命令行默认active_mask_1m，复现旧版请显式指定legacy。
现有48任务的冻结代码和排队命令不会因此改变。

原稿 `/root/research/DIB/main.tex` 的 `eq:dynamics-filter` 定义
`w_t ∝ pi_t * exp(-epsilon_t/tau_d)`，没有 `w_(t-1)`。
所以 `filter_weights` 保存但不被下一步读取是冗余状态，不是缺失贝叶斯递推的bug；
保持原公式，不擅自乘历史后验或改变假设对齐。详见[适配审计](ADAPTATION.md)。

独立2×2筛查（C1/C3、seed42、每次2.5M，不替代三seed的5M正式比较）：

```bash
python -m algorithms.dib_mappo.ablation --dib-root /root/research/DIB-MAPPO/src
# 使用上条命令输出的coordinator；在其_runtime中以nohup启动：
nohup /root/miniconda3/envs/smpe-py39/bin/python -u -m algorithms.dib_mappo.ablation \
  --run <coordinator> > <coordinator>/nohup.log 2>&1 < /dev/null &
python -m algorithms.dib_mappo.ablation --report <coordinator>
```

四个profile×两场景共8次；验证种子300042起，独立测试400042起，与原实验分离。
只按共同验证点对齐；不按独立测试选参数，单训练seed不报告SEM。
筛查源码另行冻结，输出 `<筛查名>/<profile>/<场景>/seed_42/`；不覆盖原结果。

## 整组后台运行

```bash
conda run --no-capture-output -n smpe-py39 python -m algorithms.dib_mappo.campaign \
  --dib-root /root/research/DIB-MAPPO/src \
  --name new_active_mask_warmup1m_5m --training-profile active_mask_1m --workers 32

cd output/campaigns/new_active_mask_warmup1m_5m_dib_mappo/_runtime
nohup /root/miniconda3/envs/smpe-py39/bin/python -u -m algorithms.dib_mappo.campaign \
  --run /root/heterogrid-marl/output/campaigns/new_active_mask_warmup1m_5m_dib_mappo \
  > ../nohup.log 2>&1 < /dev/null &
```

先冻结本仓库运行时代码、场景配置及外部源码，再启动48个新训练任务：
原13场景×3种子=39，加D1–D3×3=9，共240M团队步。
D0 是该算法新 B0 的符号链接，不借用其他算法模型、不重复计训练预算。
默认6v6的 reference 归一化因子均为1，因此其奖励和诊断与 B0 完全一致；
具体配置及来源见 `configs/D0.json` 和 `D0_provenance.json`。

管理器的 `status.json` 保存任务状态与进度；3个工作进程失败或磁盘剩余低于2GiB时
停止派发并向剩余进程发送 TERM。并发数必须按联合训练阶段的内存峰值决定，
不能仅根据预热阶段的低内存占用设置。

输出 `output/dib_mappo/<实验名>/`：原13场景结果在 `RESULTS.md`、`figures/`，
D组独立在 `D_report/`；单任务仍为 `<场景>/seed_<种子>/`。
保存逐局双边诊断、训练/验证曲线、AUC、首次80%步数、独立测试、吞吐量与耗时。
`training_episodes.jsonl.gz` 为无损压缩原始训练记录。
模型包含 actor/critic/target-critic、实际残差门控、步数和配置；为节约磁盘
不保存优化器，因此**只支持重载评估，不支持精确续训**。异常任务须从头重跑，
不覆盖原输出。训练前后都不要把预检数据混入正式结果目录。

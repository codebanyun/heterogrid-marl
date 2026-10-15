# MARL 算法接入

本目录存放环境的可选算法接入，不属于环境核心安装包。命令均从项目根目录执行。

## 目录约定

- `common/`：公共记录、评估、审计、队列和多种子汇总，不放环境规则或某个算法的网络结构。
- `mappo/`：[MAPPO 与 R-MAPPO 使用说明](mappo/README.md)。
- `smpe/`：[SMPE 使用说明](smpe/README.md)，复用外部仓库的循环策略、状态模型和内在探索奖励。
- `qmix/`：[QMIX 接入、后台任务矩阵与结果说明](qmix/README.md)。
- `qtran/`：[QTRAN-base 接入说明](qtran/README.md)。
- `maa2c/`：[EPyMARL MAA2C 接入与同预算实验](maa2c/README.md)。
- `coma/`：[EPyMARL COMA 接入说明](coma/README.md)。
- `common/actor_critic/`：MAA2C/COMA共用的on-policy采集、外部learner接入与队列；复用公共报告格式。
- `common/value_decomposition/`：QMIX/QTRAN共用的训练、回放、后台队列与报告工具。

新增算法时创建 `algorithms/<算法名>/`，包含独立的中文 `README.md`、训练入口和必要的适配代码。只为已实现的算法创建目录，不预建空占位目录。

## 接口选择

- 使用动态存活节点和字典观测：接入 `heterogrid.parallel_env`。
- 使用逐节点 API：接入 `heterogrid.env`，一轮动作齐备后推进环境。
- 使用固定节点槽位、全局状态和合法动作掩码：接入 `heterogrid.SMACAdapter`。

环境规则留在 `heterogrid/`，算法特有的观测转换、RNN 状态和训练流程留在算法子目录。禁止把全局状态、隐藏坐标或诊断快照泄漏给执行 actor。

## 实验约定

1. 配置按任务命名，使用 `configs/` 中共享场景；优化器、网络和训练预算由算法入口管理。
2. 默认输出到 `output/<算法名>/<实验名>/`，多种子实验再分 `seed_<种子>/`。输出目录必须是新目录，避免覆盖。
3. 保存完整环境配置、算法参数、训练种子、评估种子、依赖版本与源码快照。外部仓库路径通过参数或环境变量指定，不写死在代码中。
4. 验证种子用于模型选择，独立测试种子不参与模型选择；步数按环境时间步计，不乘智能体数量。
5. 聚合曲线使用多训练种子的均值与 SEM；注明种子数量、评估局数和是否平滑，不外推尚未训练的区间。
6. 将单元测试放入 `tests/integrations/<算法名>/`。短训练只能验证链路，不能作为收敛证明。

训练和评估的终止处理、奖励复制、死亡掩码等契约见[环境文档](../docs/environment.md)。

## 固定蓝方通信的四算法实验

下面创建四算法合计156个任务（各13场景×42/43/44），每次从头5M团队环境步。
蓝方在全部场景固定 `distance`、半径10、失联0；C1–C6只改变红方通信。
M/N场景仍按原矩阵改变地图/双方人数，不固定蓝方人数或对战中的行为轨迹。
C5/C6沿用旧实验的 `full` 拓扑叠加25%/50%节点失联，并非距离拓扑叠加失联。
红蓝使用独立通信随机流。角色能力、规则策略、奖励、自动共享和算法超参数不变。

```bash
python -m algorithms.common.fixed_blue_campaign \
  --marl-root /path/to/MARL-Algorithms --epymarl-root /path/to/epymarl \
  --name fixed_blue_sensitivity_5m --workers 48
nohup setsid -f python -u -m algorithms.common.fixed_blue_campaign \
  --run output/campaigns/fixed_blue_sensitivity_5m_all \
  > output/campaigns/fixed_blue_sensitivity_5m_all/nohup.log 2>&1 < /dev/null
```

一个共享管理器跨四算法动态填满 `--workers` 个槽位；每个进程一个CPU线程，
每次训练仍4个环境，不随排队或资源改变超参数。
`output/<算法名>/<实验名>/` 保存各自数据与自动汇总；`experiment.json` 的
`communication_protocol` 和展开的 `configs/` 标明本轮为单边通信干预。
`output/campaigns/<实验名>_all/status.json` 是统一队列状态与实际PID。
另外两个算法家族目录只保存冻结源码与原始计划（`DELEGATED`），不启动额外管理器。
验证/独立测试与模型保存预算沿用各算法原设置，不按胜率提前停止，不覆盖历史数据。
也可以在原来的双算法准备命令中加 `--fixed-blue`，只运行对应家族。

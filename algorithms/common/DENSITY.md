# 密度受控的规模实验

本入口为 MAPPO 和 SMPE 各运行 D0–D3，每个场景默认三个独立训练种子，共 24 次训练。复用公共队列、冻结源码、原算法训练入口、评估和审计，不修改算法更新公式。

## 场景协议

| 场景 | 双方人数 | 地图 | 相对 D0 的单队密度 |
|---|---|---|---:|
| D0 | 6v6 | 24×24×6 | 1.0000 |
| D1 | 3v3 | 19×19×5 | 0.9573 |
| D2 | 9v9 | 27×27×7 | 1.0159 |
| D3 | 12v12 | 30×30×8 | 0.9600 |

密度定义为初始本队人数除以地图网格数 `X×Y×Z`。双方均为侦察、打击、混合角色 1:1:1，蓝方为规则策略，200 步。双方通信均显式指定 `distance`、半径 10、失联概率 0，使用独立通信随机流。

以完成的 MAPPO 实验 B0 为环境参数参考，保留原角色能力、自动目标共享和学习奖励设置。启用 `reward_normalization="reference"`，参考人数 6、参考总 HP 480，仅缩放伤害与击杀，不缩放终局、时间和情报奖励。启用 `diagnostics_enabled=true`，诊断仅写入日志，不能输入执行 actor。

地图只近似保持密度，视野、射程、移动、通信半径和时间上限不随地图缩放。初始距离、边界、通信度数和任务难度不保证一致。D0 从头训练，不合并旧 B0，不直接把 D 组与旧 N 组之差解释为纯密度效果。

## 启动

```bash
python -m algorithms.common.density_study \
  --reference output/mappo/<已完成的敏感性实验> \
  --mappo-root /path/to/MAPPO --smpe-root /path/to/smpe_raw \
  --name density_5m --workers 12
```

默认种子 42、43、44，每次目标 5,000,192 团队环境步，不乘智能体数量、不含评估。MAPPO 精确到预算结束；SMPE 完成当前对局批次后停止，记录实际超额步数。各算法保持原有网络、优化器和更新参数，所有训练从头初始化，不复用旧模型。

默认每约 250k 步验证 64 局，独立测试 128 局，场景种子继承参考实验。主要结果使用最终预算模型；验证最优模型作为补充，独立测试集不参与选择。约 2M、4M 保存阶段模型，另保留初始、最终和验证最优模型。

共享管理器通过 `nohup` 运行，两个算法合计最多 `--workers` 个单线程 CPU 任务，较大队伍优先。项目和外部算法源码均被冻结。默认剩余磁盘低于 2GB 时停止整组任务，不支持精确断点续训。

## 输出与诊断

- `output/mappo/<名称>/`、`output/smpe/<名称>/`：各算法实际数据。
- `output/density/<名称>/`：共享队列、冻结源码及对比汇总；算法子目录为链接，不是重复数据。
- 各算法的 `status.json` 和 `nohup.log` 链接到共享的 24 项任务状态，各自 `experiment.json` 描述单算法 12 项。
- 每项 `training_episodes.jsonl.gz` 无损保存全部逐局奖励和双方诊断，使用流式压缩，不同时保存 `training_episodes.json` 整数组。MAPPO 内存只保留近期 32 局用于原有滚动指标。
- `metrics.jsonl`、`metrics.json` 和 `info.json` 仍为原来的未压缩进度与指标格式。
- 最终 `summary.json` 包含逐局独立测试及双方诊断，回放数量默认每分组 1 局，不减少参与统计的 128 局。
- 各场景 `diagnostics.json` 和 `diagnostics.csv` 汇总最终贪心独立测试的双方诊断，保留有效 seed 数和有效局数；未定义指标保持空值，不补零。
- 各场景 `figures/` 保存训练种子均值 ±SEM 曲线和 CSV，`RESULTS.md` 保存结果；不生成 `trajectories/` 和 `evaluations/`。

压缩日志可用公共读取接口：

```python
from algorithms.common.episode_log import episode_rows

for row in episode_rows("output/smpe/density_5m/D3_12v12/seed_42"):
    red = row["team_episode_metrics"]["red"]["diagnostics"]
    blue = row["team_episode_metrics"]["blue"]["diagnostics"]
```

诊断重点包括双方通信度、侦察覆盖、侦察与打击节点连通性、命中率、过量伤害、存活比例和初始最近敌人距离。原始伤害和击杀仍是物理量，奖励分量才经过参考归一化。

## 短程检查

同一入口可加 `--steps 16384 --seeds 42 --eval-episodes 1 --heldout-episodes 1 --eval-interval 8192 --save-interval 8192 --log-interval 8192`，并指定独立 `--output-root`，检查两个算法各四个场景。短检查只验证执行、记录和审计，不证明学习效果。

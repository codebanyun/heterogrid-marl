# 固定蓝方通信实验

此入口联合调度 MAPPO 和 SMPE，只重跑 B0 与 C1–C6，每个算法 7 个配置、每个配置三个训练种子，共 42 次训练。适用于隔离红方通信条件的影响，不重跑地图和阵容变化实验。

## 环境设置

以已完成 MAPPO 实验冻结的 B0 为基础，保留地图、阵容、角色参数、奖励和规则对手。所有配置均显式指定双方通信：

| 配置 | 红方模式 | 红方半径 | 红方失联概率 |
|---|---|---:|---:|
| B0 | distance | 10 | 0 |
| C1 | none | 10 | 0 |
| C2 | full | 10 | 0 |
| C3 | distance | 5 | 0 |
| C4 | distance | 15 | 0 |
| C5 | full | 10 | 0.25 |
| C6 | full | 10 | 0.50 |

**蓝方始终固定为 distance、半径 10、失联概率 0。** `full` 和 `none` 模式不使用半径判断链路。C5/C6 沿用原随机通信实验的在线节点全连接定义，不叠加距离约束。

独立通信配置使用红蓝独立随机流，与旧共享采样实现不等价。B0 必须从头重跑，只与本次 C1–C6 比较，不合并或复用旧 B0。

## 启动

```bash
python -m algorithms.common.communication_study \
  --reference output/mappo/<已完成的敏感性实验> \
  --mappo-root /path/to/MAPPO --smpe-root /path/to/smpe_raw \
  --name communication_blue_fixed_5m --workers 12
```

命令从项目根目录执行，需要两个算法的可选依赖；外部路径也可通过 `MAPPO_ROOT` 和 `SMPE_ROOT` 设置。入口要求支持 `nohup` 和符号链接的 POSIX 系统。

默认每个任务目标为 5,000,192 环境步，种子为 42、43、44；MAPPO 精确结束，SMPE 完成当前对局批次后结束并记录实际步数。两个算法沿用各自原训练参数，不加载既往权重。验证 64 局、独立测试 128 局，评估种子继承参考实验；主要结果使用最终预算模型，验证最优模型只作补充。

共享队列通过 `nohup` 后台运行，两个算法合计最多占用 `--workers` 个单线程 CPU 槽位。一个任务完成后立即从队列补充，不为某算法预留空闲槽位。项目和外部算法源码均在启动时冻结，后续编辑不影响排队任务。

## 输出

- `output/mappo/<名称>/`：MAPPO 配置和原始数据。
- `output/smpe/<名称>/`：SMPE 配置和原始数据。
- `output/communication/<名称>/`：共享队列、冻结源码和最终对比报告；`mappo/`、`smpe/` 是指向实际结果目录的链接，不是额外数据副本。

各算法目录中的 `status.json`、`nohup.log` 链接到共享队列，进度统计两个算法合计 42 项，而各自 `experiment.json` 描述单算法 21 项。

每项保存完整环境配置、模型、指标、逐局统计、源码快照和少量最终回放；不生成 `trajectories/` 和 `evaluations/`。约 2M、4M 步保存阶段模型，另保留初始、最终和验证最优模型。结束后自动审计、复算回放并生成各配置训练种子均值 ±SEM 曲线和结果报告。

停止整组任务：向共享 `status.json` 中的 `manager_pid` 发送 SIGTERM。磁盘低于安全阈值会停止队列。当前不支持精确断点续训，失败不会静默重试。

## 短程检查

```bash
python -m algorithms.common.communication_study \
  --reference output/mappo/<已完成的敏感性实验> \
  --mappo-root /path/to/MAPPO --smpe-root /path/to/smpe_raw \
  --output-root /tmp/communication-check --name smoke \
  --steps 16384 --seeds 42 --workers 12 \
  --eval-episodes 1 --heldout-episodes 1 --replay-episodes 1 \
  --eval-interval 8192 --save-interval 8192 --log-interval 8192
```

短检查包含两个算法各 7 个配置，检查训练、参数更新、模型重载、队列、审计和报告流程，不用于判断学习效果或收敛。

# QMIX + MA2E

本接入将 MA2E 掩码轨迹预训练与 QMIX 价值分解结合，不是普通 QMIX，也不复用 MAPPO 的网络或优化器。外部源码通过 `--ma2e-root` 或 `MA2E_ROOT` 指定，可以是仓库根目录或 `src/`。后端负责记录原生配置、实际网络及必要兼容修复。

执行阶段每个智能体仅使用自身局部观测、合法通信报告及自身动作历史。训练阶段的集中式混合器和预训练信息不得直接进入分散执行策略；不能把真实队友历史当作可用输入。

## 适配修正与保留项

适配层针对上游推理中的槽位和时间对齐问题采用以下修正，不能将未适配的上游路径直接视为正确实现：

- 自身历史放入该智能体的实际槽位，不统一放入第 0 槽。执行掩码、当前观测覆盖和输出提取（gather）均使用同一实际槽位，不注入真实队友历史。
- 动作历史严格因果对齐：时刻 `t` 的窗口最多包含 `a_(t-1)`，`t=0` 的动作输入为零，不读取尚未选择的 `a_t`。MAE 重建使用的动作窗口也截止于 `a_(t-1)`，与执行路径一致。
- MAE 有效样本依据 `filled` 和有效性掩码选择，包含真实终止转移，排除仅供 bootstrap 的槽位及填充，避免把终局与 padding 混为一谈。
- RL 阶段冻结 MAE 参数并保持 `MAE.eval()`；冻结参数本身不能替代评估模式，不能让 dropout 随策略训练模式重新启用。
- 回放缓冲区采用紧凑的原始数据模板，按需构造模型输入，不为每条历史常驻保存膨胀后的 MAE 输入模板。

原生 QMIX 的 TD-lambda 目标、mixer 和优化器保留；原生 `MT_train` 的 SGD、MSE 及按轨迹长度缩放的方式也保留，不将适配修正描述为新的损失或优化算法。

位置编码保留上游 `both` 路径及其限制。本接入不声称已修复为正确的联合时间/智能体位置编码；槽位对齐修正不等于位置编码设计得到修复。上述行为的实现位置和验证证据由 backend/train/audit 维护，不以本文替代测试。

## 单次运行

以下命令从项目根目录执行；输出目录必须不存在。

```bash
export MA2E_ROOT=/path/to/MA2E
python -m algorithms.ma2e.launch \
  --ma2e-root "$MA2E_ROOT" \
  --config configs/scenarios/6v6_rule_distance.json \
  --output output/ma2e/baseline_seed42 \
  --steps 5000192 --seed 42 --device auto \
  --gpu-min-free-mb 10000 --gpu-max-workers 2
```

`train` 接收 `--ma2e-root --config --output --steps --seed --device`，以及与 EPyMARL 接入一致的 `--eval-seed --heldout-seed --eval-episodes --heldout-episodes --eval-interval --log-interval --replay-episodes`。自动设备选择和失败重启由独立的 `launch` 包装层负责，而不是由 campaign 固定所有任务的设备。

## 完整实验

```bash
python -m algorithms.ma2e.campaign \
  --ma2e-root "$MA2E_ROOT" \
  --communication-reference output/mappo/20261014_blue_fixed_5m \
  --sensitivity-reference output/mappo/20261010_sensitivity_5m \
  --density-reference output/mappo/20261018_density_5m \
  --workers 3 --gpu-max-workers 2 --device auto \
  --output output/ma2e/bcmnd_5m
```

上述三个参考路径也是默认值。只读取已完成实验的配置及评估协议，不读取模型、训练数据或回放。

- B0、C1-C6 保留显式固定的蓝方 distance 通信，半径 10、失联率 0。
- M1-M3、N1-N3 原样保留 sensitivity 配置及通信随机流兼容设置。
- D0-D3 保留密度控制、reference 奖励归一化及双方诊断，诊断不进入策略。

默认 17 个配置、seed 42/43/44，共 51 次独立从头训练，每次目标 5,000,192 个团队环境步，不乘智能体数。预训练采集交互包含在总预算内，验证和独立测试不计入；达到预算后仅完成当前采集批次，实际步数和超出量必须记录并审计。

评估种子及局数沿用参考实验，默认验证 64 局、独立测试 128 局，每 250,368 步验证。评估初始、最终及验证最优模型；最优模型只按验证结果选择。MA2E 的 sampled 评估使用最终探索率的 epsilon-greedy，不等同于 PPO 的策略分布采样；主表统一使用贪心测试。最终主表使用固定预算的 latest 模型，报告训练种子均值及 SEM，不以测试结果选模型。显式覆盖局数或间隔会写入 `protocol_overrides`，仅适合检查，不能称为完全匹配的正式比较。`--conditions`、`--seeds`、`--steps` 可缩小检查范围。

## 调度与磁盘

管理器通过 nohup 使用冻结项目启动，复用 `algorithms.common.study_queue.run_queue`，不修改共享队列。默认总并发为 3；`--external-status` 指定的已有实验必须没有待启动任务，其存活 worker 会占用同一个总并发额度。每个 worker 的 OMP、OpenBLAS、MKL 和 NumExpr 线程数为 1，绘图使用 Agg。

`supervisor.py` 另提供资源感知队列和运行中 launcher 接管。它读取 `resource_control.json` 的总并发上限、GPU 上限、CPU 阵容上限及内存余量，预留完整回放缓冲区、float32 重建缓存和临时计算空间后才启动任务；CPU 优先选择小阵容。资源估算不是操作系统硬配额。接管记录必须验证 PID 启动标识和实际模块命令，不重新初始化正在训练的模型，也不能用于恢复已终止的训练。运行中控制文件应以原子替换方式更新。

控制文件可用 `priority_jobs` 指定优先任务 ID（例如 `B0_baseline/seed_43`），用 `gpu_only_jobs` 指定不得退回 CPU 的任务。优先级不会绕过内存准入；GPU 专用任务在 GPU 槽位已满时不会被调度到 CPU。将失败任务补跑前，必须归档旧目录和同种子的设备记录及日志，避免覆盖旧结果或误读旧 PID。不要直接修改运行中管理器维护的 `status.json`；需通过经过身份校验的管理器接管流程修改离线队列状态。

`launch` 的必要契约如下，具体实现由训练包装层提供：

- 接收训练参数和 `--device auto --gpu-min-free-mb 10000 --gpu-max-workers 2`；GPU 上限可在吞吐、显存实测后调为 3。
- 每个任务启动时查询 `nvidia-smi`，检查空闲显存并用跨进程共享锁限制同时持有 GPU 的任务数。显存检查不是硬配额，实际峰值仍可能触发 OOM 回退。无 GPU、显存不足或租约满时回退 CPU，不因 GPU 忙而阻塞整个队列。
- 仅在识别为 CUDA OOM 时执行一次全新 CPU 重启，重设随机种子、模型、优化器、回放缓冲区及计数器，不能从失败权重续训。不得删除旧文件；失败尝试移至独立保留目录，并记录实际设备、失败原因和重启来源，防止失败指标混入有效训练。
- 显式加 `--no-cpu-fallback` 时，无可用 GPU 或发生 CUDA OOM 都会记录失败并退出，不启动 CPU 尝试；适用于 CPU 内存峰值无法安全容纳的任务。该选项仅由包装层处理，不传给训练后端。
- 租约目录必须跨 campaign 共享，不能仅在单个输出目录内计数。包装进程必须转发终止信号、等待训练子进程并释放租约，不能留下孤立训练进程。

启动前要求可用空间至少覆盖源码快照、预计新增数据和安全水位。预计新增量取 `--projected-storage-mb`（默认且至少 1500 MiB）与 `任务数 × (3 × --checkpoint-estimate-mb + --logs-replays-estimate-mb)` 的较大值；默认每份完整检查点估计 8 MiB，每次运行的稀疏压缩日志与少量回放估计 5 MiB。估计必须包含目标网络、优化器和预训练状态，正式运行前根据完整状态检查点的实测大小调整；它不是硬配额或容量保证。`--minimum-free-disk-gb` 默认且至少为 0.4 GiB，队列持续检查该水位，触及后停止并保留记录，不删除历史实验。发生 OOM 后保留的失败尝试也占用磁盘，不能假定自动清理。

## 输出与审计

`experiment.json` 记录配置哈希、评估协议、预算、设备策略及来源。`_code/project/` 冻结项目的 algorithms、heterogrid、configs，`_code/ma2e/src/` 冻结完整外部源码，包括 learner 导入所需的 `envs/matrix_game`。`_code/provenance.json` 记录源仓库路径、commit、remote、工作区状态及冻结文件的 SHA-256；存在本地修改时，以冻结内容哈希为准。

仅保存 initial、best、latest 三类模型检查点，包含在线/目标网络、优化器和计数器，但不含回放缓存及随机流状态，不支持精确断点续训。逐局统计无损压缩；不生成 `trajectories/` 或 `evaluations/`。默认每个回放阶段保留 1 局，不减少实际评估局数。`status.json` 记录进度，`nohup.log` 和场景 `logs/` 记录管理器及 worker 日志。

冻结 MAE 的 decoder latent 按回放对局以 float32 缓存，在线网络和目标网络复用同一重建结果；每次 MAE 微调、模型加载和回放覆盖均使相关缓存失效。重建后的观测仍使用原解码层计算，不缓存可训练注意力层的输出。循环更新使用激活重算降低峰值内存，不改变损失或批次大小。检查点中完全相同的目标张量共享序列化存储，加载后仍是独立网络参数。

队列完成后对每次运行执行 `audit(path, recompute=True)`，要求训练输出兼容共享的 `config.json`、`metrics.json`、`summary.json` 契约。审计通过后复用 `summarize_runs` 生成场景汇总和均值/SEM 曲线；存在双方诊断时复用密度诊断汇总。顶层 `RESULTS.md` 列出全部场景最终贪心胜率的均值、SEM 和种子数。单种子不估计 SEM。

campaign 不实现 learner、预算计数或检查点恢复；这些保证必须由 backend/train/launch/audit 联合实现并验证。准备 campaign 本身不会启动训练，只有显式执行上述命令才会启动管理器。

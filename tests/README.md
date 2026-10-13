# 自动化测试

`tests/` 是环境开发和算法接入的回归测试，不是训练数据。整理目录时保留原有测试；新增功能应在所属分组补充测试，避免不同机器或不同算法接入时破坏已有行为。

## 分组

| 目录 | 检查内容 | 依赖 |
|---|---|---|
| `environment/` | 对抗、观测信息边界、通信、奖励、配置、PettingZoo API、SMAC 固定槽位与回放 | 环境核心依赖；离屏渲染测试在缺少 matplotlib 时跳过 |
| `environment/test_team_communication.py` | 红蓝独立通信、距离叠加失联、蓝方随机流隔离、自动/手动共享、旧采样兼容与空间维度 | 环境核心依赖 |
| `integrations/mappo/` | MAPPO 入口、预算对齐、进度读取、路径配置、缓存观测与死亡掩码 | 观测测试可选 PyTorch，其余不需要外部 MAPPO 源码 |
| `integrations/test_value_decomposition.py` | QMIX/QTRAN任务矩阵、回放终局与补齐、目标RNN首帧对齐、缺失种子曲线处理 | RNN测试可选PyTorch，不需要外部算法源码 |
| `integrations/test_actor_critic.py` | MAA2C/COMA固定参数、终局n-step、padding、执行与原MAC一致、COMA反事实输入、真实批量更新 | 可选PyTorch、PyYAML与EPyMARL源码，`EPYMARL_ROOT`指向其src目录 |
| `integrations/smpe/` | 冻结配置、源码隔离、原生模型更新、终局处理、动作掩码和完整检查点 | 矩阵测试只需核心依赖；后端测试需 PyTorch，并用 `SMPE_ROOT` 指定外部仓库 |
| `common/` | 日志、指标、源码快照、可选结果索引和运行状态 | 环境核心依赖 |
| `viewer/` | HTTP 控制、快照只读、配置原子性；另有可选浏览器检查 | Python 测试无前端依赖，浏览器检查需 Node.js 与 Playwright |

## Python 回归

从项目根目录执行：

```bash
python -m unittest discover -s tests -t . -v
python -m unittest discover -s tests/environment -t . -v
python -m unittest discover -s tests/integrations -t . -v
python -m unittest discover -s tests/viewer -t . -v
```

不需要安装 pytest。PettingZoo 对包含动作掩码的 Dict 观测可能给出提示，只要断言通过就不是测试失败。单元测试不会启动长期训练。

## 可选浏览器检查

准备 Node.js、Playwright 和 Chromium，先生成回放并启动测试服务：

```bash
python -m heterogrid.rollout --episodes 1 --replay-dir output/replays
python -m heterogrid.viewer --host 127.0.0.1 --port 8765
```

另一个终端运行：

```bash
node tests/viewer/check_viewer.cjs
```

可通过 `VIEWER_URL` 指定服务地址，通过 `PLAYWRIGHT_MODULE` 指定 Playwright 模块位置，通过 `PLAYWRIGHT_EXECUTABLE` 指定浏览器可执行文件。脚本会改变演示场景，应使用独立测试服务，不能连接正在查看的实验会话。截图保存到 `output/viewer-preview.png`。

## 训练链路检查

修改算法入口后，还应按该算法 README 执行一次短训练，验证参数更新、合法动作、模型重载和日志保存。短训练结果不等于性能或收敛证据；多种子实验应另行记录。

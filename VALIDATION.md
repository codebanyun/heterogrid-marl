# 本地验证记录

日期：2026-10-09。运行环境：Windows，conda `petzoo`，Python 3.11.11、NumPy 1.26.4、Gymnasium 1.0.0、PettingZoo 1.24.3。

- 37 项 unittest 测试通过。包含 PettingZoo 官方 Parallel、AEC、seed 测试；以及早期死亡的 AEC 轮次、同时伤害、超时、视野外报告、缓存失效、消息传输、固定槽位、预训练策略回调、回放和 HTTP 服务。
- 三个独立接口示例运行通过：PettingZoo、SMAC 适配器、侦察报告后视野外打击。
- 默认 SMAC 接口：6 个红方 agent，13 个动作，178 维展平局部观测，133 维全局状态。
- 默认规则对抗完整运行 5 局；seed 42、43 红方胜利，44、45、46 超时平局。这仅是环境运行检查，不代表学习算法效果。
- 无渲染基准：2000 个环境时间步耗时 2.799 秒，约 714.4 步／秒。包含脚本敌方与局部观测生成，不包含神经网络推理、训练、文件发布或视频渲染。此数据来自本地 Windows，不是用户 Linux/V100 服务器测量。
- Chrome 无头浏览器交互测试通过：页面、单步、开始／暂停、重置、节点详情、图层、配置变更、回放；JavaScript 错误数为 0。
- 已检查 `output/viewer-preview.png` 页面截图。示例回放在 `output/replays/`。

Linux 使用路径参见 README。已采用可移植 Python 标准库／NumPy／Gymnasium／PettingZoo 实现，但尚未在用户 Linux 服务器执行测试。没有训练或附带预训练 MARL 网络。

可重复验证：

```bash
python -m unittest discover -s tests -v
python -m examples.parallel_usage
python -m examples.smac_usage
python -m examples.communication_usage
python -m examples.benchmark --steps 2000
```

可选浏览器测试需要 Node.js 和 Playwright，仅用于开发验证，不是环境或控制台运行依赖。启动 `python -m heterogrid.viewer` 后执行 `node tests/check_viewer.cjs`。可通过 PLAYWRIGHT_MODULE 指定模块路径，通过 PLAYWRIGHT_EXECUTABLE 指定已安装浏览器。

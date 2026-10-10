# 环境使用示例

本目录只放与具体训练算法无关的示例。先安装项目，然后从项目根目录运行：

```bash
python -m tutorials.parallel_usage
python -m tutorials.smac_usage
python -m tutorials.communication_usage
python -m tutorials.benchmark --steps 2000
```

| 文件 | 内容 |
|---|---|
| `parallel_usage.py` | PettingZoo Parallel 的重置、合法动作采样和终局处理 |
| `smac_usage.py` | 固定槽位、全局状态和团队奖励接口 |
| `communication_usage.py` | 侦察报告如何为视野外打击节点提供合法目标 |
| `pretrained_numpy.py` | NumPy 格式对手策略的加载与工厂封装，不附带已训练模型 |
| `benchmark.py` | 环境步吞吐量，不含网络推理、训练和渲染 |

预训练对手工厂示例：

```bash
python -m heterogrid.rollout --opponent pretrained \
  --opponent-factory tutorials.pretrained_numpy:make_policy \
  --checkpoint /path/to/actor.npz
```

该命令需要自行提供匹配的检查点，不能直接当作附带的预训练模型使用。MAPPO 等训练入口位于 `algorithms/<算法名>/`。

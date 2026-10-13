# COMA：EPyMARL 接入

复用 EPyMARL 的 `COMALearner`、`COMACritic`、`BasicMAC` 和共享 GRU actor。critic使用全局状态、其他智能体当前动作与agent ID，屏蔽本智能体当前动作；actor执行只用局部观测与ID。保留原实现反事实基线、n-step目标和Adam更新。

完整参数、on-policy/终局契约、后台任务矩阵、评估和输出结构见 [MAA2C/COMA共用说明](../maa2c/README.md)。COMA固定hidden64、lr0.0003、gamma0.99、n-step10、entropy0.001、target每200次更新同步、4环境4回合每批更新一次；跨场景不调整。没有PPO重复更新或旧经验回放。

```bash
export EPYMARL_ROOT=/path/to/epymarl/src
python -m algorithms.coma.train \
  --config configs/scenarios/6v6_rule_distance.json \
  --steps 5000000 --seed 42 --output output/coma/baseline_seed42
```

正式任务在 `output/coma/<实验名>/`，各39次从头训练。主报告固定预算结束模型的独立贪心测试，categorical采样和验证最优模型作为补充。

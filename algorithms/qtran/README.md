# QTRAN-base 接入与灵敏度实验

本接入复用外部 MARL-Algorithms 的 `policy/qtran_base.py`、`QtranQBase`、`QtranV` 和共享 GRU。选择 **QTRAN-base**，不混用 QTRAN-alt。损失仍为原实现的TD、最优和非最优约束组合，`lambda_opt=lambda_nopt=1`；联合Q和V隐藏维度均64。

执行网络只读局部观测、上一动作与agent ID；联合Q/V只在训练读取全局状态。目标RNN从初始观测预热，保持外部QTRAN逻辑；目标网络不构建梯度图。角色、奖励、通信和规则蓝方保持环境定义。

## 单次训练

```bash
python -m pip install -e '.[value-decomposition]'
export MARL_ALGORITHMS_ROOT=/path/to/MARL-Algorithms
python -m algorithms.qtran.train \
  --config configs/scenarios/6v6_rule_distance.json \
  --steps 5000000 --seed 42 --output output/qtran/baseline_seed42
```

可用 `--marl-root` 指定外部源码，输出必须为新目录。

共享训练参数、13场景×3种子后台管理、独立验证/测试划分、记录格式、停止方式与图表解释见 [完整接入说明](../qmix/README.md)。两算法均固定RNN64、学习率0.0005、折扣0.99、4环境、1000局回放、32局批量、32局预热、每200团队步一次更新、每200更新同步目标网络、50k步ε退火至0.05、梯度裁剪10。每个算法内部13个场景只适配任务维度，不随人数或地图改变超参数。

正式实验保存到 `output/qtran/<实验名>/<场景>/seed_<种子>/`。主报告用5M结束模型；验证最佳仅补充，最终独立测试各128局。`figures/` 自动生成地图、人数、通信三组均值±SEM图及原始CSV。状态、首达80%、AUC、吞吐量与测试行为指标采用与QMIX相同定义。

```bash
python -m algorithms.common.value_decomposition.report output/qtran/<实验名>
```

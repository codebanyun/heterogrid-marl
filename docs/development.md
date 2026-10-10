# 开发与团队协作

## 职责边界

环境规则、观测和 API 放在 `heterogrid/`；可选算法接入放在 `algorithms/<算法名>/`；接口教学放在 `tutorials/`；测试按对应职责分组。不要为新增算法修改既有场景的含义，或把算法框架依赖引入环境核心。

项目组织参考了 [PettingZoo](https://github.com/Farama-Foundation/PettingZoo) 的环境、文档、测试与教程分离，以及 [SMAC](https://github.com/oxwhirl/smac) 的环境与外部训练框架边界。本项目保留现有 `heterogrid/` 包布局，不为整理目录额外引入 `src/` 迁移。

## 提交前检查

```bash
python -m unittest discover -s tests -t . -v
python -m tutorials.parallel_usage
python -m tutorials.smac_usage
python -m tutorials.communication_usage
git diff --check
```

环境 API 回归由 GitHub Actions 在多个 Python 版本上运行；可选算法训练不属于核心 CI。训练入口迁移后检查所有子进程的 `python -m` 路径、源码快照路径和配置默认值。

## 依赖与安装产物

`pyproject.toml` 声明核心依赖与 `render`、`mappo` 可选依赖。只安装环境不需要 PyTorch。算法脚本和配置从源码仓库运行，不进入环境 wheel。

`heterogrid_marl.egg-info/` 是 setuptools 在安装或构建时生成的元数据，含包名、依赖和源码清单。它不是环境数据，也不是必需提交的源码，已加入忽略规则。可在不运行安装任务时删除，后续安装会重新生成；不要把它当作需要跨机器同步的目录。

`__pycache__/`、`build/`、`dist/` 和测试缓存也是生成物，不提交。`output/` 中已有实验不因代码整理而移动；对需要共享的结果另外选择存储位置，并保留完整配置、模型、指标和来源证据。

## 复现与变更

场景配置中的缺省项来自 `EnvConfig`，所以实验开始时必须展开并保存完整配置。历史实验的元数据、源码快照与模型不改写；文档入口可更新。环境规则或默认值的改变会影响学习问题，应单独记录，而不能与目录整理混在一起。

中文 README 面向所有机器与团队成员，不记录某台机器的 conda 环境名、绝对目录或特定机器上的验证结论。环境验证方法放在 `tests/README.md`，算法训练方法放在各算法 README，实验成绩只放在相应实验目录。

## 路径迁移

旧 `examples.train_mappo`、`launch_mappo`、`long_mappo`、`ablate_mappo`、`audit_mappo` 分别替换为 `algorithms.mappo.train`、`launch`、`study`、`ablate`、`audit`。

算法无关的 `examples.*_usage`、`examples.benchmark` 和 `examples.pretrained_numpy` 迁至 `tutorials/`。旧 `VALIDATION.md` 中机器相关的一次性验证记录不再作为项目使用文档，测试方法统一见 `tests/README.md`。配置路径迁移见 `configs/README.md`。

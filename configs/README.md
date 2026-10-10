# 环境配置

配置只描述任务，不包含算法网络、优化器或训练预算。同一任务配置可用于不同 MARL 算法。

## 保留的配置

| 文件 | 场景与用途 |
|---|---|
| `scenarios/default_6v6.json` | 基础 24×24×6、6v6、规则蓝方、距离通信；手动目标报告、基础奖励、超时平局，与 `EnvConfig()` 等价 |
| `scenarios/6v6_rule_distance.json` | 同一地图与阵容，双方自动报告；超时失败、时间成本、发现与有效共享奖励，适合作为学习基线 |
| `scenarios/3v3_compact.json` | 6×6×3、3v3、静止但可还击的蓝方、全连接通信，适合快速流程检查 |
| `ablations/6v6_shared.json` | 6v6 基础任务，双方自动共享，保留基础奖励 |
| `ablations/6v6_terminal.json` | 在共享设置上增加超时失败与每步 0.005 时间成本，不加情报奖励 |

`6v6_rule_distance.json` 同时作为消融的情报奖励组，不再保存重复的 `shaped` 配置。各配置独立加载，不存在文件继承；省略的字段使用 `EnvConfig` 默认值。实验记录必须保存 `config.to_dict()` 的完整展开结果，避免未来默认值变化影响复现。

## 加载已有场景

从项目根目录加载：

```python
from dataclasses import replace
from heterogrid import EnvConfig, parallel_env

config = EnvConfig.load('configs/scenarios/6v6_rule_distance.json')
env = parallel_env(config=config)

# 修改配置后创建新环境，不在运行中改变观测和动作空间。
no_comm = replace(config, communication='none', target_sharing='manual',
                  scripted_target_sharing=False)
random_links = replace(config, communication='random', dropout_probability=0.25)
```

保存自定义配置时使用 JSON，并给出任务含义明确的文件名。角色与平台、观测维度、通信和奖励字段详见[环境文档](../docs/environment.md)。

## 自定义 JSON 场景

总 README 中的 4v3 示例可写成以下 JSON。每个阵容元素是一名节点，不另设 `num_agents` 或角色数量字段；红蓝阵容必须分别填写。JSON 使用数组表示地图尺寸，用 `true` / `false` 表示布尔值：

```json
{
  "map_size": [16, 16, 4],
  "red": [
    {"role": "scout", "platform": "uav"},
    {"role": "strike", "platform": "robot"},
    {"role": "strike", "platform": "robot_dog"},
    {"role": "hybrid", "platform": "uav"}
  ],
  "blue": [
    {"role": "scout", "platform": "uav"},
    {"role": "strike", "platform": "robot"},
    {"role": "hybrid", "platform": "usv"}
  ],
  "max_steps": 300,
  "opponent": "rule",
  "communication": "distance",
  "comm_radius": 8.0,
  "target_sharing": "auto",
  "scripted_target_sharing": false
}
```

将自定义文件作为 `EnvConfig.load(...)`、演示命令或训练入口的 `--config` 参数即可。训练参数另由算法管理，详见各算法子目录；这里不包含训练预算。

## 参数速查

| 字段 | 默认值 | 设置说明 |
|---|---|---|
| `map_size` | `[24, 24, 6]` | XYZ 网格数量，三个正整数；Z 可为 1 |
| `red`、`blue` | 各 6 名节点 | 阵容列表，支持不同人数、不同组成；缺省的一方仍用默认阵容 |
| `role` | 每名节点必填 | `scout` / `strike` / `hybrid`，决定角色能力 |
| `platform` | `uav` | `uav` / `usv` / `robot` / `robot_dog`，当前仅作平台标签 |
| `max_steps` | `200` | 单局环境步上限，正整数 |
| `opponent` | `rule` | `dummy` / `static` / `random` / `rule` / `pretrained`，控制蓝方 |
| `communication` | `distance` | `none` / `full` / `distance` / `random`，四种互斥链路模式 |
| `comm_radius` | `10.0` | 非负，距离模式的三维欧氏通信半径 |
| `dropout_probability` | `0.25` | 0 到 1，仅随机模式生效；每个节点每步的失联概率，不是逐链路丢包率 |
| `target_sharing` | `manual` | `manual` 由调用者报告；`auto` 为双方经合法链路自动报告 |
| `scripted_target_sharing` | `true` | 手动模式下是否允许脚本蓝方自动报告；不控制红方显式报告 |
| `target_ttl` | `2` | 正整数；记录年龄不超过该值时有效，超过后清除 |
| `message_dim` | `8` | 自定义消息向量长度，正整数；通信内容需为有限值并位于 `[-1, 1]` |
| `timeout_as_loss` | `false` | 是否在超时时给红方失败奖励；PettingZoo 仍标记 truncation |
| `reward_mode` | `dense` | `dense` 包含伤害、击杀和已配置的情报奖励；`sparse` 关闭这些分量 |
| `time_penalty` | `0.0` | 每个环境步的团队成本，非负；显式设置时稀疏模式也生效 |
| `discovery_reward`、`sharing_reward` | 各 `0.0` | 首次发现与有效报告的团队奖励，非负、按整局去重 |
| `information_reward_cap` | `1.0` | 每队每局情报奖励合计上限，非负 |

`role` 和 `platform` 位于 `red` / `blue` 的节点对象内，其余表中字段位于 JSON 顶层。角色统计及基础奖励字段见下文和[环境规则](../docs/environment.md)。未知字段会报错，不应在环境 JSON 中加入算法参数。

## 调整角色能力

默认能力：

| 角色 | `hp` 生命值 | `vision` 视野 | `attack_range` 射程 | `damage` 伤害 |
|---|---:|---:|---:|---:|
| `scout` | 60 | 7 | 0 | 0 |
| `strike` | 100 | 3 | 6 | 20 |
| `hybrid` | 80 | 5 | 4 | 12 |

`role_stats` 按角色设置，并同时应用于红蓝双方的所有同角色节点，不是逐节点配置。生命值必须大于零，其余数值必须非负且有限。若在 JSON 中提供 `role_stats`，必须同时定义三个角色的全部四个字段，不能只填要修改的一项。

Python 中可保留原有角色，再替换侦察角色的能力：

```python
from dataclasses import replace
from heterogrid import EnvConfig, RoleStats

config = EnvConfig.load("configs/scenarios/6v6_rule_distance.json")
config = replace(config, role_stats={
    **config.role_stats,
    "scout": RoleStats(hp=60, vision=10, attack_range=0, damage=0),
})
```

改变视野、射程和奖励会改变任务难度，跨算法比较时必须使用相同配置。平台标签不会自动选择另一套能力或运动规则。

## 生效范围与优先级

1. `EnvConfig.load(path)` 使用文件中明确给出的值，其余项取类默认值；不会读取另一个场景作为基底。
2. Python 的 `replace(config, ...)` 返回新配置，覆盖指定字段，保留其余字段。仅创建配置不会修改已存在的环境。
3. `heterogrid.rollout` 的 `--opponent` 和 `--communication` 覆盖文件中的对应字段，不连带修改 `target_sharing` 等其他参数；地图、阵容和通信半径需要在文件中设置。
4. `heterogrid.viewer --config ...` 设置演示会话的初始配置。网页上的配置变更会重置当前演示，但不写回 JSON，不影响独立训练。
5. `opponent="pretrained"` 还需提供策略对象或命令行策略工厂；文件中只写该字符串不能自动找到检查点。

环境配置在一次训练中应保持固定，特别是阵容和消息维度会影响空间形状。随机初始化由 `env.reset(seed=...)` 或演示命令的 `--seed` 控制，`seed` 不是 `EnvConfig` 字段。

## 旧路径迁移

| 旧文件 | 新文件或替代方式 |
|---|---|
| `default.json` | `scenarios/default_6v6.json` |
| `mappo_6v6_shaped.json`、`mappo_6v6_rule_distance.json` | 合并到 `scenarios/6v6_rule_distance.json`，展开后的配置相同 |
| `mappo_compact_3v3.json` | `scenarios/3v3_compact.json` |
| `mappo_6v6_shared.json` | `ablations/6v6_shared.json` |
| `mappo_6v6_terminal.json` | `ablations/6v6_terminal.json` |
| `auto_broadcast.json` | 基础配置加 `target_sharing='auto'` |
| `no_communication.json` | 基础配置加 `communication='none'`、`target_sharing='manual'`、`scripted_target_sharing=False` |
| `random_outage.json` | 基础配置加 `communication='random'`，按需要指定失联概率 |

旧实验中的 `env.json`、`config.json` 与源码快照保持原样，不随当前目录重组改写。

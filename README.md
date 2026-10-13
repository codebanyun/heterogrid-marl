# HeteroGrid 三维异构多智能体环境

HeteroGrid 是一个面向多智能体强化学习（MARL）的轻量三维网格对抗环境。红方由外部算法控制，蓝方可使用固定规则、随机、静止策略或用户提供的预训练策略。双方包含侦察、打击和查打一体节点，支持局部感知、受限通信和目标报告。

环境与训练算法解耦，提供 PettingZoo Parallel、AEC 和 SMAC 风格固定槽位接口。核心运行不依赖 PyTorch、GPU 或游戏客户端，默认不渲染、不写文件、不发布网络数据。

## 安装

要求 Python 3.10 或更新版本。进入项目根目录后安装：

```bash
python -m pip install -e .
```

核心依赖为 NumPy、Gymnasium 和 PettingZoo，由安装工具自动处理。需要 `rgb_array` 离屏渲染时安装可选依赖：

```bash
python -m pip install -e '.[render]'
```

浏览器可视化不需要上述渲染依赖，也不需要 Node.js、前端构建工具或桌面环境。

## 快速开始

```python
import numpy as np
from heterogrid import parallel_env

env = parallel_env()
rng = np.random.default_rng(42)
observations, infos = env.reset(seed=42)
while env.agents:
    actions = {
        agent: int(rng.choice(np.flatnonzero(observations[agent]['action_mask'])))
        for agent in env.agents
    }
    observations, rewards, terminations, truncations, infos = env.step(actions)
env.close()
```

默认任务为 `24 × 24 × 6` 地图、6 对 6、200 步上限、规则蓝方、半径 10 的距离通信。`EnvConfig()` 保留基础奖励和手动报告模式；训练场景可通过[配置目录](configs/README.md)选择，不由算法名称决定。

红方各节点收到相同的团队奖励，统计团队回报时只计一次，不能对节点奖励再次求和。执行策略只能使用局部观测；全局 `state()` 供集中式训练使用。

## 环境设置

通过 `EnvConfig` 设置任务，然后传给 `parallel_env(config=...)`；AEC 接口和 `SMACAdapter` 也接受同一配置。下面创建一个 **16×16×4、红方 4 个节点对蓝方 3 个节点、规则蓝方、距离通信** 的环境：

```python
from heterogrid import EnvConfig, UnitSpec, parallel_env

config = EnvConfig(
    map_size=(16, 16, 4),
    red=[
        UnitSpec(role="scout", platform="uav"),
        UnitSpec(role="strike", platform="robot"),
        UnitSpec(role="strike", platform="robot_dog"),
        UnitSpec(role="hybrid", platform="uav"),
    ],
    blue=[
        UnitSpec(role="scout", platform="uav"),
        UnitSpec(role="strike", platform="robot"),
        UnitSpec(role="hybrid", platform="usv"),
    ],
    max_steps=300,
    opponent="rule",
    communication="distance",
    comm_radius=8.0,
    target_sharing="auto",
    scripted_target_sharing=False,
)

env = parallel_env(config=config)
observations, infos = env.reset(seed=42)
print(env.possible_agents)  # red_0、red_1、red_2、red_3
env.close()
```

### 地图与阵容

- `map_size=(X, Y, Z)`：三个方向的网格数量，均为正整数；上述坐标范围为 `0..15`、`0..15`、`0..3`。平面场景可设置 `Z=1`。
- `red`、`blue`：每个列表元素代表一个节点，列表长度就是该队节点数量。两队可以人数不同、组成不同，但都不能为空。
- 节点 ID 按列表顺序固定为 `red_0...` 和 `blue_0...`。只修改 `red` 不会自动改变 `blue`，未指定的一方仍使用默认六节点阵容。
- `max_steps`：单局最多推进多少个环境时间步，不是算法训练总步数。默认超时平局；`timeout_as_loss=True` 会在超时时对红方施加失败奖励，但终止原因仍为 `timeout`，不会改为蓝方全灭红方获胜。

`UnitSpec` 的 **角色** 决定能力，**平台** 当前只作为类别标签：

| 字段 | 可选值 | 含义 |
|---|---|---|
| `role` | `scout` | 侦察，默认视野大、不能攻击 |
| `role` | `strike` | 打击，默认视野小、射程远 |
| `role` | `hybrid` | 查打一体，同时具备观察与攻击能力 |
| `platform` | `uav`、`usv`、`robot`、`robot_dog` | 无人机、无人艇、机器人、机器狗；当前均可沿 XYZ 移动，不额外限制高度或地形 |

可重复添加同一角色，不能直接使用未定义的新角色名称。生命值、视野、射程、伤害通过 `role_stats` 按角色统一设置，见[配置说明](configs/README.md)。

### 通信与目标共享

| `communication` | 链路规则 | 相关参数 |
|---|---|---|
| `none` | 所有同队节点间均不能通信 | 无 |
| `full` | 所有存活同队节点间可通信，不受距离限制 | 无 |
| `distance` | 三维欧氏距离不超过半径时可通信 | `comm_radius`，默认 10 |
| `random` | 每个决策步独立采样各节点是否在线；失联节点不能发也不能收，在线同队节点间不受距离限制 | `dropout_probability`，默认 0.25 |

以上是旧版共享字段的语义：这四种模式互斥，`random` 不是“距离链路再随机丢包”，顶层 `dropout_probability` 不影响 `distance` 模式。通信仅限同队、单跳；未指定独立配置时，红蓝双方使用相同规则，保留原有实验行为。

现在支持 `red_communication` / `blue_communication` 分别设置两队。下面固定蓝方半径10、无失联，只改变红方：

```python
from dataclasses import replace
from heterogrid import CommunicationConfig, EnvConfig

base = EnvConfig.load("configs/scenarios/6v6_rule_distance.json")
base = replace(base,
    red_communication=CommunicationConfig(mode="distance", radius=10, dropout_probability=0),
    blue_communication=CommunicationConfig(mode="distance", radius=10, dropout_probability=0))
red_none = replace(base, red_communication=CommunicationConfig(mode="none"))
red_full = replace(base, red_communication=CommunicationConfig(mode="full"))
red_radius15 = replace(base, red_communication=CommunicationConfig(mode="distance", radius=15))
red_outage50 = replace(base, red_communication=CommunicationConfig(
    mode="distance", radius=10, dropout_probability=0.50))
```

独立配置将拓扑（`none/full/distance`）和节点失联概率分开，因此可以表达“半径10叠加50%节点失联”。两队使用独立通信随机数流；默认自动共享阶段仍对双方执行，但投递分别受各自链路约束。设置优先级、JSON示例见[配置说明](configs/README.md#固定蓝方只改变红方通信)。

链路存在不代表敌情必然共享，还需要选择报告方式：

- `target_sharing="auto"`：双方通过合法链路自动共享当前直接看到的敌情；适合固定通信机制下的训练，但不代表智能体学会了通信。
- `target_sharing="manual"`：红方需要在选动作前显式调用 `exchange_messages(...)` 或 `share_visible_targets()`；示例见[通信接口](docs/environment.md)。
- `scripted_target_sharing`：只控制手动模式下脚本蓝方是否自行报告，默认 `True`。需要双方都不自动报告时设为 `False`；自动模式下由统一自动阶段处理，不受该开关影响。

例如关闭通信，或改为随机失联：

```python
from dataclasses import replace

no_comm = replace(config, communication="none", target_sharing="manual",
                  scripted_target_sharing=False)
random_comm = replace(config, communication="random", dropout_probability=0.25)
```

两行代码得到的是新配置，需要另建环境才生效。`target_ttl` 控制目标与消息保留时长，`message_dim` 控制自定义消息维度；链路断开不会立即清除已收到且尚未过期的记录。

### 蓝方策略

`opponent` 控制整支蓝方，红方动作仍由调用者或训练算法提交：

| `opponent` | 行为 |
|---|---|
| `dummy` | 不移动、不攻击，用作检查靶标 |
| `static` | 不移动，但会攻击合法目标；不是无反应靶标 |
| `random` | 在各节点当前合法动作中随机选择，可能移动或攻击 |
| `rule` | 优先攻击合法低生命目标，否则接近已知目标或巡逻；不读取隐藏敌方坐标 |
| `pretrained` | 加载用户提供的策略对象，而非项目附带的已训练模型 |

切换脚本对手可用 `replace(config, opponent="random")`。选择 `pretrained` 时还必须向环境传入 `opponent_policy`，实现 `reset(seed=...)` 与 `act(observations, infos)`；模型加载、网络结构和循环状态由该封装负责，详见[预训练对手示例](tutorials/README.md)。

### JSON 配置与命令行

已有场景可以直接加载，再局部修改；省略的字段使用 `EnvConfig` 默认值，不会继承另一个 JSON 文件：

```python
from dataclasses import replace
from heterogrid import EnvConfig

config = EnvConfig.load("configs/scenarios/6v6_rule_distance.json")
config = replace(config, map_size=(32, 32, 8), comm_radius=12.0,
                 opponent="random")
```

用 `config.to_dict()` 保存完整展开后的配置，便于跨机器复现：

```python
import json
from pathlib import Path

path = Path("output/custom_scenario.json")
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(config.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8")
```

演示与浏览器可加载同一份配置：

```bash
python -m heterogrid.rollout --config output/custom_scenario.json --episodes 5
python -m heterogrid.viewer --config output/custom_scenario.json
```

`rollout` 还支持 `--opponent random`、`--communication none` 覆盖 JSON 中的对应字段；地图和阵容通过配置文件设置，没有同名命令行开关。该命令运行脚本策略对局，且显式执行红方目标广播，不是 MARL 训练或手动通信算法。浏览器里的设置只改变当前演示会话，不会改写配置文件或其他训练进程。

人数、地图和消息维度可能影响观测或动作空间；一次训练中保持配置固定，变更后创建新环境，已有模型也需检查输入输出维度及任务设置是否匹配。更多字段和 JSON 示例见[配置说明](configs/README.md)。

## 可视化与回放

```bash
python -m heterogrid.viewer --host 127.0.0.1 --port 8765
```

浏览器访问 <http://127.0.0.1:8765>，可运行规则策略演示、调整配置或打开完整 JSON 回放。

生成回放：

```bash
python -m heterogrid.rollout --episodes 5 --replay-dir output/replays
```

远程机器可通过 SSH 转发端口：

```bash
ssh -N -L 8765:127.0.0.1:8765 user@server
```

控制台没有用户认证，默认只监听本机，不应直接暴露到公网。快照发布和回放格式见[环境文档](docs/environment.md)。

## 项目结构

```text
heterogrid/           环境核心、接口适配器、策略与可视化
configs/             算法无关的场景配置和奖励消融配置
algorithms/          按算法分组的可选训练接入
  common/            公共实验记录工具
  mappo/             MAPPO 接入及独立使用说明
  qmix/              QMIX 接入与灵敏度实验
  qtran/             QTRAN-base 接入与灵敏度实验
  maa2c/             EPyMARL MAA2C 接入与灵敏度实验
  coma/              EPyMARL COMA 接入与灵敏度实验
tutorials/           接口、通信和预训练对手示例
tests/               按职责分组的自动化回归测试
docs/                环境细节与开发维护约定
output/              本地生成的实验结果，不提交到 Git
```

环境安装包只包含 `heterogrid/` 及其静态资源；训练接入、配置和教程从源码仓库使用，算法依赖不进入核心依赖。

## 文档导航

- [环境接口、通信、奖励与胜负规则](docs/environment.md)
- [场景配置与旧路径迁移](configs/README.md)
- [算法接入规范](algorithms/README.md)
- [接口使用示例](tutorials/README.md)
- [测试分组与运行方式](tests/README.md)
- [开发与团队协作约定](docs/development.md)

## 测试

在项目根目录执行：

```bash
python -m unittest discover -s tests -t . -v
```

测试覆盖对抗结算、信息边界、通信、奖励、PettingZoo API、固定槽位、回放与可视化服务。没有 PyTorch 时，相关算法观测测试会跳过；环境核心测试仍可运行。

## 建模边界

当前没有障碍物、碰撞、连续动力学、弹道、治疗、复活或在线敌方训练。平台类别作为标签，均可沿 XYZ 移动；双方同一步攻击同时结算。固定规则对手用于可控实验，战胜该对手不等价于泛化到未知策略。

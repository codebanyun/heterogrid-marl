# HeteroGrid 三维异构 MARL 环境

轻量三维网格红蓝对抗环境。红方由外部算法控制，蓝方使用静止、随机、固定规则策略，或加载用户提供的预训练 actor。没有障碍物、连续动力学、弹道、治疗、复活或在线敌方训练。

环境独立于训练算法，提供真正的 **PettingZoo ParallelEnv**、AEC 转换及 **SMAC 风格固定槽位适配器**。训练默认不渲染、不写文件、不发送网络数据。浏览器控制台单独运行。

## 安装与本地启动

已在本机现有 conda 环境 `petzoo` 验证，Python 3.11.11、NumPy 1.26.4、Gymnasium 1.0.0、PettingZoo 1.24.3。环境核心不需要 PyTorch 或 GPU。

```powershell
conda activate petzoo
cd D:\research\heterogrid-marl
python -m pip install -e . --no-deps --no-build-isolation
python -m unittest discover -s tests -v
python -m heterogrid.viewer
```

浏览器打开 <http://127.0.0.1:8765>。点击“开始”运行规则策略演示；拖动旋转战场，滚轮缩放。可修改地图、双方数量、蓝方策略、通信方式与随机种子。配置变更会重置演示环境。

新机器／Linux 可以安装到独立环境：

```bash
conda create -n heterogrid python=3.11 pip -y
conda activate heterogrid
cd /path/to/heterogrid-marl
python -m pip install -e .
# 仅 rgb_array 离屏绘图需要，浏览器无需此依赖
python -m pip install -e '.[render]'
```

从项目根目录执行示例：

```bash
python -m examples.parallel_usage
python -m examples.smac_usage
python -m examples.communication_usage
python -m heterogrid.rollout --episodes 5 --replay-dir output/replays
python -m examples.benchmark --steps 2000
```

## 场景与动作

默认地图 `24×24×6`，坐标范围分别为 `0..23, 0..23, 0..5`。每方两个侦察、两个打击、两个查打一体节点，最大 200 步。

| 角色 | role | 生命值 | 视野 | 射程 | 单次伤害 |
|---|---|---:|---:|---:|---:|
| 侦察 | scout | 60 | 7 | 0 | 0 |
| 打击 | strike | 100 | 3 | 6 | 20 |
| 查打一体 | hybrid | 80 | 5 | 4 | 12 |

距离使用三维欧氏距离。平台 `uav/usv/robot/robot_dog` 分别表示无人机、无人艇、机器人、机器狗；初版只作为标签显示，所有平台均可在 XYZ 方向移动。同一网格允许多个单元重叠，不模拟碰撞。

| 动作编号 | 含义 |
|---|---|
| 0 | WAIT，原地／死亡槽位 no-op |
| 1 / 2 | X 正／负方向移动一格 |
| 3 / 4 | Y 正／负方向移动一格 |
| 5 / 6 | Z 正／负方向移动一格 |
| 7+j | 攻击敌方固定槽位 j |

侦察节点的攻击动作始终被掩码禁用。超出动作空间的值报错；处于动作空间内但被掩码禁用的动作按 WAIT 执行，在 info 中标记 `invalid_action`。

**同时结算顺序**：使用当前步开始时的观测和报告决定动作 → 双方从同一个状态快照开火 → 汇总并同时扣血 → 存活且选择移动的单元移动 → 时间步加一 → 为下一决策步采样通信状态并更新本地感知。

攻击使用本地记录的目标坐标。如果坐标过期、目标已不在该格，攻击落空；不会读取隐藏的实时坐标来修正瞄准。目标本步选择移动时，仍可能先被同一步的攻击命中。

## PettingZoo Parallel 接口

```python
import numpy as np
from heterogrid import EnvConfig, parallel_env

env = parallel_env(config=EnvConfig())
obs, infos = env.reset(seed=42)
while env.agents:
    actions = {
        a: int(np.random.choice(np.flatnonzero(obs[a]['action_mask'])))
        for a in env.agents
    }
    obs, rewards, terminations, truncations, infos = env.step(actions)
env.close()
```

- `possible_agents`：固定红方 ID 列表，例如 `red_0..red_5`。
- `agents`：当前仍参与决策的红方节点；死亡即移除，回合结束后为空。
- `reset(seed, options)`：返回 `(observations, infos)`。可通过 `options['positions']` 指定初始位置；未知 options 被忽略。配置变更应构造新环境，避免改变既有 spaces。
- `step(actions)`：标准五元组。当前步结束的 agent 也会收到最终观测、奖励和终止标记；下一步只为 `env.agents` 提交动作。
- `observation_space(agent)`、`action_space(agent)`：返回固定 Gymnasium spaces。
- `state()`：全局训练状态；**只能给中央 critic 或训练监督使用**。
- `render_mode=None`：无绘图。`ansi` 返回文本，`rgb_array` 返回离屏 RGB 数组。
- `from heterogrid import env`：构造 AEC 版本，按 `agent_iter()` 调用；同一轮所有 agent 提交完动作后才推进物理。死亡 agent 必须提交 `None`。通信实验建议直接使用 Parallel 接口，避免中途交换消息造成不一致的决策轮次。

没有 LBF 专用接口；它与 LBF 一样通过标准 Gymnasium spaces 表述观测／动作，同时采用适合多智能体同步行动的 PettingZoo API。

### 每个 agent 的观测字典

| 字段 | shape | 含义 |
|---|---|---|
| observation | `(12+8*(N_red-1)+11*N_blue,)` | 自身、可见友军、目标记录 |
| action_mask | `(7+N_blue,)` | 当前合法动作，0/1 |
| messages | `(N_red, message_dim)` | 按发送者固定槽位排列的自定义消息 |
| message_mask | `(N_red,)` | 收件箱槽位是否有有效消息 |
| message_age | `(N_red,)` | 消息年龄 / target_ttl |

默认特征向量长度 118，自定义消息维度 8，默认消息矩阵 `6×8`。空记录填零；浮点字段为 float32，mask 为 int8。观测不包含全局存活数量或隐藏队友的实时状态。

特征布局：

- 自身 12 维：归一化 XYZ、生命比例、角色 one-hot 3 维、视野、射程、伤害、时间步比例、自身通信可用标记。
- 每个其他友军 8 维：当前可见标记、相对 XYZ、生命比例、角色 one-hot。视野外全部为零，不能凭此判断它是死亡还是远离。
- 每个敌方槽位 11 维：记录有效标记、当前直接可见标记、记录中的相对 XYZ、已知生命比例、角色 one-hot、记录年龄、是否来源于队友报告。
- 坐标按 `max(map_size-1,1)` 归一化；视野与射程按地图对角线归一化并截断到 1，伤害按配置中的最大伤害归一化。
- `env.observation_layout()` 返回分块偏移和目标 ID 顺序。友军块按固定 ID 顺序排除自身；messages 块包含全部红方槽位。

默认全局 state 为 `11*(N_red+N_blue)+1` 维，默认 133 维。每个节点依次为阵营 one-hot 2 维、XYZ 3 维、生命比例、存活标记、角色 one-hot 3 维、通信在线标记；最后是时间步比例。不把 state 或可视化 snapshot 传给执行 actor。

### 通信接口

`communication` 支持 `none/full/distance/random`。距离模式默认半径 10。random 模式每个决策步对节点独立采样，默认失联概率 0.25；失联节点该步既不能发也不能收。每步只采样一次，重复观察或交换消息不会改变链路。只允许同队通信，不做多跳。

默认 `target_sharing='manual'`，环境不会自动把红方敌情共享给所有节点。显式通信阶段放在标准 step **之前**：

```python
obs, infos = env.exchange_messages({
    'red_0': {
        'content': np.zeros(env.config.message_dim, dtype=np.float32),
        'targets': ['blue_0'],
        'recipients': ['red_1'],
    }
})
# 根据更新后的 obs 选择动作，再执行 env.step(actions)
```

- `content` 可省略；提供时必须是指定维度、范围 `[-1,1]` 的有限浮点向量。
- `targets` 可省略；只能报告发送者**当前直接看到的存活敌人**。环境根据合法观测生成报告，不能伪造真实位置或免费转发远端敌情。
- `recipients` 可省略，表示向其他同队节点广播；实际投递受物理通信条件约束。
- `exchange_messages` 不推进时间、不产生奖励，返回更新后的观测与 info。建议每个决策步调用一次，联合提交各发送者消息。
- 目标与消息默认在记录年龄 `0,1,2` 时有效，年龄大于 2 才清除。链路随后断开不撤销已收到的记忆。
- `get_comm_mask()` 是传输／诊断接口，矩阵方向为 `[sender, receiver]`；不要直接作为全局 actor 输入。actor 获取的是自己实际收到的消息。
- `share_visible_targets()` 是**显式广播基线**。`target_sharing='auto'` 可开启自动广播，但此时不适合比较算法是否学会选择通信结构。
- 脚本蓝方共享合法目标报告；预训练蓝方默认不自动共享，需要显式设置 auto 模式或由封装自行利用合法收到的消息。论文比较应统一双方与各方法的通信设定。

## SMAC 风格固定槽位接口

```python
from heterogrid import SMACAdapter
env = SMACAdapter()
obs, state = env.reset(seed=42)
print(env.get_env_info())
reward, terminated, info = env.step([0] * env.n_agents)
```

提供 `get_obs/get_obs_agent/get_obs_size/get_state/get_state_size/get_avail_actions/get_avail_agent_actions/get_total_actions/get_env_info/reset/step/seed/render/close/save_replay/get_stats`。

适配器始终保留 N_red 个槽位。死亡槽位观测为零，只允许动作 0。观测把局部向量、自定义消息、message_mask 与 message_age 展平，默认每个节点 **178 维**；动作默认 13 个。`step` 返回**一次团队奖励**，不会把相同的 agent 奖励求和。`info['episode_limit']` 用于区分时间限制并正确 bootstrap。

这是接口适配器，不依赖安装 SMAC/StarCraft，也不自动注册到某个 PyMARL 或 MAPPO 仓库的环境注册表。算法只需在自身注册表中创建该类；不需要修改环境核心。

## 对手与预训练模型

- `dummy`：不移动、不攻击的检查靶标。
- `static`：不移动，攻击合法目标中生命比例最低的目标。
- `random`：从当前合法动作中随机采样。
- `rule`：优先攻击合法低生命目标；否则接近已知目标；侦察节点在视野内保持观察；没有目标时按公开地图坐标巡逻。**不会使用隐藏敌方位置寻路**。
- `pretrained`：传入具有 `reset(seed=...)` 和 `act(observations, infos)` 的对象。act 只接收蓝方局部观测，返回每个存活蓝方节点的动作。架构、检查点加载、RNN 隐藏状态维护由封装负责，环境不假设是 MAPPO 还是其他算法。

```python
env = parallel_env(config=EnvConfig(opponent='pretrained'), opponent_policy=my_saved_actor)
```

命令行也支持工厂：

```bash
python -m heterogrid.rollout --opponent pretrained \
  --opponent-factory examples.pretrained_numpy:make_policy --checkpoint /path/to/actor.npz
```

`examples/pretrained_numpy.py` 演示 NumPy checkpoint 封装，**未提供任何已训练网络**。使用 PyTorch/MAPPO 时替换为自己的加载器即可。

## 奖励、胜负与复现

默认共享团队奖励：有效伤害每点 +0.01，首次击败单元 +1，获胜 +10，失败 -10。过量伤害按目标剩余生命值截断。双方同时全灭时为平局；稠密模式仍保留该步实际伤害和击败奖励。`reward_mode='sparse'` 时只保留终局胜负奖励。

消灭对方全部节点且己方仍存活才算胜利，包括必须击败敌方侦察节点。如果只剩不能攻击的节点，会一直到超时。超时默认平局；可用 `timeout_as_loss=true` 为红方添加失败奖励。即使采用这一评分方式，PettingZoo 时间限制仍返回 truncation，SMAC info 仍标记 episode_limit。

死亡为 termination，时间限制为 truncation，终局全灭为 termination。随机性由 `reset(seed)` 控制，不使用全局 NumPy 随机数；同一 seed、同一动作和同一消息顺序可以复现轨迹。跨实验固定配置、种子与对手版本。

配置示例在 `configs/`：

```python
config = EnvConfig.load('configs/default.json')
env = parallel_env(config=config)
```

支持自定义 role_stats，但一次实验内阵容固定，不动态增援／改变观测维度。不同阵容创建不同环境。

## Linux 训练与本地浏览器查看

**推荐独立评估进程**，定期加载训练检查点，运行环境并发布快照。发布器必须显式启用，默认训练路径完全没有可视化开销：

```python
from heterogrid.io import SnapshotWriter
publisher = SnapshotWriter('output/latest.json', every=10)
# 在评估循环中：
publisher.write(env)  # 每 10 次显式调用才写一次，使用原子文件替换
```

独立查看进程只读取快照，不控制训练或评估状态：

```bash
# 服务器上
python -m heterogrid.viewer --snapshot output/latest.json --host 127.0.0.1 --port 8765
# 本地电脑上
ssh -N -L 8765:127.0.0.1:8765 user@server
```

本地浏览器打开 <http://127.0.0.1:8765>。服务器无需桌面、显示器或 WebGL；绘图由本地浏览器 Canvas 完成，无 CDN、npm 或前端构建依赖。默认绑定 localhost，通过 SSH 连接；直接使用 `--host 0.0.0.0` 应限制在可信网络内，控制台没有用户认证。

也可以先运行一个有节奏的演示评估：

```bash
python -m heterogrid.rollout --episodes 10 --snapshot output/latest.json --publish-every 1 --fps 5
```

`--fps` 只用于演示，训练不要人为限速。写快照／运行评估仍然有资源开销；独立进程不会让浏览器渲染阻塞训练，但共享 CPU/GPU 时应控制评估频率。

## 回放与测试

`output/` 用于保存运行生成的文件，包括回放、最新快照、截图、日志和打包产物，不只用于回放。整个目录已加入 `.gitignore`，这些本地生成文件不会提交到 GitHub；源码、配置、示例和测试会纳入版本管理。

`parallel_env(record_replay=True)` 会记录每一步；结束后调用 `env.save_replay('output/episode.json')`。SMACAdapter 同样接受 record_replay。浏览器点击“打开回放”选取 JSON，支持拖动进度条、单步和倍速。回放在浏览器本地读取，不上传服务器。

测试使用标准库 unittest，不要求额外安装 pytest：

```bash
python -m unittest discover -s tests -v
```

覆盖 PettingZoo 官方 Parallel/AEC/种子测试、信息边界、同时全灭、过量伤害、消息有效期、通信模式、固定槽位、预训练入口、回放、独立快照和 HTTP 控制台。AEC 测试可能提示 Dict 观测不是单个数组，这是包含动作掩码的字典空间提示，不是失败。本机 PettingZoo 的 dead-step helper 含对只读 num_agents 的赋值，项目用自包含 AEC 转换处理，不修改已安装包。

API 依据：[PettingZoo Parallel 文档](https://pettingzoo.farama.org/api/parallel/)、[PettingZoo 官方测试说明](https://pettingzoo.farama.org/tutorials/custom_environment/4-testing-your-environment/)、[SMAC MultiAgentEnv 接口](https://github.com/oxwhirl/smac/blob/master/smac/env/multiagentenv.py)。

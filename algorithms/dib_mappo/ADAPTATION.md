# 2026-10-21：存活掩码、预热调度与MHF核对

这不是2026-10-19的等价计算优化。新的训练profile改变有效样本和调度，
必须与旧结果分开标识，现有48任务未做热替换，不停止或重训。

## 1. 存活掩码

原DIBPPOLearner只把回合filled/terminated复制为每个agent的mask。
HeteroGrid中已阵亡节点仍保留固定槽位，直到整局结束；这些节点不应继续
贡献actor损失、优势统计或自身辅助学习样本。

`alive.py` 适配原生train函数，只有以下行为不同：

- actor/辅助mask = 原回合mask × (动作前本地观测HP>0)。
- actor优势归一化、PPO/KL与logit诊断使用相同有效样本。
- 新记录active_actor_fraction、active_actor_samples、dead_actor_samples。
- 空活跃批次安全跳过，不产生除零。

团队critic仍使用原回合mask、原奖励、原n-step及终局定义。
自身死亡当步仍学习；不因为动作后的死亡而丢弃该动作的收益或代价。
其他节点的观测目标、死亡状态仍按原算法构造，不把未知和死亡私自改成同一语义。

## 2. MHF不是递归后验遗漏

依据用户本地原稿 `/root/research/DIB/main.tex`，Multi-Hypothesis Filter节，
`eq:dynamics-filter`：

```text
epsilon_(t,k) = mean squared local-transition prediction error
w_(t,k) = pi_(t,k) exp(-epsilon_(t,k)/tau)
          / sum_l pi_(t,l) exp(-epsilon_(t,l)/tau)
```

文稿明确说candidate来自当前student posterior，并使用最近一个已完成局部转移
给当前proposal重新加权。没有乘 `w_(t-1)`；candidate本身随当前context变化，
擅自按索引相乘还会引入未定义的跨时刻模式对齐。

因此先前仅凭filter_weights保存却不读取而怀疑“递推bug”的判断依据不足，
这里明确更正：它是兼容性保留的冗余隐状态字段，不是当前公式的实现缺陷。
维持actor、序列重放、旧MAC和辅助目标原样，不加入递归滤波。

本次核对的原稿SHA256：
`0b5734a0cfd32b4504218c4f672ec4d3a2e212df1de3b76f85f8da7a328bf02e`。

测试直接核对proposal×likelihood公式、初始步回退，并验证改变历史权重
不改变下一步结果。这是在保护原方法定义，不是用无变化测试证明性能。

## 3. 1M预热

`warmup1m`／`active_mask_1m` 将dib_warmup_steps与dib_actor_warmup_steps
同时设为1,000,000；原2,000,000渐入、4,000,000衰减周期不变。

| 采集批次起点 | 阶段 | 残差gate | 辅助scale |
|---|---|---:|---:|
| <1M | warmup | 0 | 0 |
| 1M | joint开始 | 0 | 0 |
| 2M | joint | 0.5 | 0.5 |
| 3M | joint | 1 | 1 |
| 5M | joint | 1 | 0.6 |

边界以完整采集批次起点为准，不在采集或旧策略重放中途改门控。
legacy保持2M预热，active_mask只改样本mask，两者可独立消融。

## 4. 测试口径

新增测试覆盖：动作前HP、保留死亡转移、阵亡后辅助latent零直接梯度、
padding、空批次、真实四轮PPO、团队critic/n-step不截断、全员存活时与旧版更新一致、
新profile原生/优化重放一致、1M阶段转换、MHF原公式与历史状态语义。

2026-10-21执行结果：全仓库154项，150通过、4可选SMPE跳过、0失败。
独立C1/C3短流程（4profile×2场景，每次2400步、2局验证/测试）8任务全部完成，
覆盖采集、验证、保存、重载、最终测试及队列接续。
短流程最后更新的active fraction：C1为0.648953，C3为0.555351，
对应更新排除了436和723个阵亡agent-step；未出现非有限更新或跳过更新。

另在C1上限200步的真实战斗中采集4个完整回合，实际173团队步、24个红方节点阵亡，
在独立诊断时钟1.1M执行完整4轮联合PPO：actor最大参数差1.150183e-7，
critic/target差0，原生与加速重放对照通过。active fraction为0.687861；
teacher/distill均非零、没有跳过更新。该时钟是检查阶段逻辑，不是声称真的已训练1.1M步。
这些检查只证明实现按定义工作，不能据此声称胜率提高；短流程原始数据位于/tmp。

独立筛查采用C1/C3、四个profile、seed42、每次固定2.5M（8次，共20M）；
验证300042起、测试400042起，不使用旧正式测试集选参。
2.5M用于筛查学习效率，尚未覆盖1M版在3M后的全门控阶段；
正面/负面结果均需后续三seed、5M预算确认，不按筛查胜率提前终止正式任务。

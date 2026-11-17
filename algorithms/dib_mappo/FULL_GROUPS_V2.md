# DIB-v2固定配置全组扩展（2026-11-07）

本轮使用20261103实验由验证集统一选出的 **Tactical-v2 / belief_first=.30**。
不是原始DIB，也不是Resilient-v3，更不是按C3挑弱辅助、按C5挑强辅助的组合。
不在新增场景重新搜索参数，不加载其他场景或seed的训练权重。

## 场景与预算

新增训练12场景：B0、M1/M2/M3、N1/N2/N3、C2/C4、D1/D2/D3。
每场景训练seed42/43/44，每次从头5M，36次新训练，共180M团队环境步。
C3/C5直接链接上轮belief_first的3seed完整数据、模型及独立测试，不重新训练。
C1/C6本轮排除，后续Resilient-v3单独训练和报告，不混入本轮v2表格。

D0复用B0，不增加3次“独立训练”。D0声明配置为reference奖励归一化，
但6v6下伤害/击杀缩放系数均严格为1；与B0其他设置、动力学、有效奖励、观测与诊断完全等价。
目录belief_first/D0是B0的符号链接；其中config.json仍忠实显示B0原配置，
而configs/D0.json保存D组声明配置。reused_results.json解释两者等价关系。
D1/D2/D3分别为3v3/19×19×5、9v9/27×27×7、12v12/30×30×8，使用原D组reference奖励协议。
与N组比较时，同时存在密度与奖励协议差异，不能把差异全部归因于地图密度。

最终v2表共15个场景（原17场景排除C1/C6），其中D0不是新的独立重复。

## 确保还是同一个算法

full_campaign从原20261103 coordinator的冻结_runtime及_backend复制代码，
验证原source SHA，并逐字节验证backend.py、tactical.py、robust.py、sequence.py、
auxiliary.py、alive.py、returns.py、train.py与原版本一致。
只覆盖full_campaign/tactical_campaign/tactical_train/resume/matched_eval运行及存储管理文件。
不带入v3的几何动作头、战斗能力掩码、episode均衡、长时GAE或8环境设置。

保持hidden/embed/teacher128、z64、K4、lr5e-4、gamma=.99、GAE=.95、PPO4轮、4环境，
原Tactical辅助及残差调度和奖励不变，仅按人数自动适配输入输出/目标维度。
蓝方始终rule、distance10、无失联；红方按对应场景配置；200步、射击限制及自动目标共享机制不变。

## 验证、测试与复用

约250k附近64局验证，700042–700105；2M/4M附近、结束、验证best保存模型。
全部新增训练结束后，final/best分别128局贪心+128局采样，800042–800169。
固定预算final为主报告，best仅补充；不按胜率提前停止、不延长个别seed预算。
C3/C5直接用原同协议独立测试，不重复测后再挑更好的结果。
旧QMIX/QTRAN/MAA2C/COMA和可用原DIB final在新增场景统一重评；缺失模型不补零。
原DIB部分N2/N3/D3未完成，表中可能缺失，但这不代表本轮训练失败。

这是在C3/C5结果公开后的固定配置扩展，不能声称所有场景都属于新的盲测方法选择。
均值领先不等于显著性或全领域SOTA；三个训练seed适合初步比较。

## 存储与恢复

磁盘紧张，因此本轮显式使用 `--storage-policy compact`：

- 不降精度、不量化权重；训练中保存完整Adam/RNG/模型/计数，一个原子resume.pt。
- 若actor与old actor（或critic与target critic）快照完全相同，序列化时共享同一不可变副本，加载后仍是独立网络。
- 不保留resume_previous.pt备份，因此单文件损坏时的恢复冗余较少；原子写入仍保留。
- 达到固定预算后不能继续训练，resume.pt改为评估专用状态，不再保留Adam；同预算评估可恢复。
- 不额外保存initial.pt；仍保留约2M、4M、final、best模型及全部验证与测试数据。
- training_parts/保留全部有效压缩逐局记录，最终用training_episodes.manifest.json列出有序段，
  不再复制拼接一份同样的training_episodes.jsonl.gz。按manifest顺序用gzip.open逐段读取即可。

训练期间可精确续训；不得改源码、预算或超参数后冒充精确恢复。
两个原有v2/v3冻结任务不热更新，也不压缩或删除旧正式数据。
自建短测试缓存迁到/dev/shm/heterogrid_preflight_archive_20261107（内存盘，重启可能丢失）；正式实验不放内存盘。

## 命令与目录

```bash
/root/miniconda3/envs/smpe-py39/bin/python -m algorithms.dib_mappo.full_campaign --matched-baselines

cd output/campaigns/20261107_tactical_v2_full_groups_5m_dib_mappo/_runtime
nohup /root/miniconda3/envs/smpe-py39/bin/python -u -m algorithms.dib_mappo.full_campaign \
  --run /root/heterogrid-marl/output/campaigns/20261107_tactical_v2_full_groups_5m_dib_mappo \
  > ../nohup.log 2>&1 < /dev/null &
```

结果位于output/dib_mappo/20261107_tactical_v2_full_groups_5m/。
36个worker，加上独立v3的6个worker，共42个训练进程（50逻辑CPU机器）。
status.json、每seed/progress.json记录状态；RESULTS.md由完成后的真实数据生成。
磁盘2GiB保护及三个异常停止派发机制保持；中断后在冻结_runtime执行同一模块的 `--resume-run <coordinator>`。
没有新增系统自启动服务，不会自动推送Git。

# GeoRoute：全参数版本执行契约

状态：实现与准备阶段，尚无本轮正式训练或测评成绩。关联的私有稿件不进入公开仓库；本文不复用稿件中的结果数字。

2026-09-21 数据修订优先于下文旧五源表述：增加 MindCube train10K，VLM3R改用GeoThinker的VSI205456＋VST132053（32帧版本）而非100K上限；六源共享协议为`shared-six-source-geothinker-vlm3r-mindcube-v4-openspatial-waiver`。VST按官方errata修订问题和标签并记录，排除2350条；MindCube完整集包含训练题，不作为独立测试。用户允许OpenSpatial 100K在缺少场景映射时加入训练，全部train-only，重叠风险未知，不能声称全数据无污染。最终训练规模以媒体、其余五源场景隔离与合并验收为准。排程在主RFT后补两个奖励消融，见[奖励协议](../../docs/RFT_REWARD_ABLATION_PLAN.md)。

## 研究问题与公平比较

本轮比较五种几何结构/训练方式。用户已取消新增 matched RGB SFT，因此不能严格回答几何是否优于同数据普通 SFT；历史 RGB 与原始基座仅作非匹配参考。强度诊断和无 LoRA RFT 的新决策见[补充契约](../../docs/GEOMETRY_NO_LORA_AND_STRENGTH.md)。

主线：released Qwen3-VL-2B-Instruct → 路由 TIP 初始化 → 全参数空间 SFT（含 TIP replay）→ 固定最终权重测评。不是从前一篇已 RFT 的模型继续训练。前一篇仅是资源队列依赖。

| 实验 | 唯一目标变化 | 必须控制的因素 |
|---|---|---|
| full | 四 pre-merger 出口、两 STB、TIP | 主协议 |
| no TIP | 去除初始化与 replay | 指令预算相同；额外 TIP 计算不匹配，单独报告 |
| one STB | 每出口一个 STB | 相同 TIP/SFT 与帧数；参数/FLOPs不同 |
| final only | 仅最终出口路由 | 相同数据与预算 |
| post merger | 改在 merger 后路由 | 图必须映射/去重到合并网格；维度/图结构改变需记录 |

不能通过推理时关模块来替代以上重新训练。训练失败或尚未实现的组保留待测，不填0分。

## 已知条件与本次补足

| 项目 | 原描述足够明确的部分 | 本轮补足/改动 |
|---|---|---|
| 模型 | Qwen3-VL-2B、VGGT，visual exits 5/11/17/23 | 权重与源码 revision 固定；不能复用不等价的 GeoWire 模块 |
| 更新范围 | TIP初始化路由，随后空间SFT | 用户指定不用LoRA；SFT全量更新语言、原生视觉、merger、路由；离散图教师冻结 |
| 数据 | SPAR234K/Hound64K/VSI590K/VLM3R/OpenSpatial五源 | 精确子集、许可、源scene去重、1%验证划分见资产协议；未就绪不得训练 |
| 图 | tracking、可见性/置信度过滤、入边topK、归一 | visibility≥0.5、confidence≥0.5、K8、重复边max、同帧边删除；这些是补充值 |
| STB | RMSNorm/down/SiLU/up、残差、两层 | bottleneck256、alpha初始0；孤立点严格identity |
| TIP | recovery/substitution/preservation | masked MSE＋margin0.1对比＋unmasked MSE；系数0.1/0.1；mask15%；同源帧非邻居负例，禁止跨样本 |
| TIP预算 | 未交代完整步数/数据清单 | 先对五源训练多帧清单一遍；有效支持不足需单列分母、停止检查，不能伪造边 |
| SFT预算 | LR1e-5/global64/cosine/3%warmup/bf16 | 一遍指令数据、AdamW weight_decay0、seed3407；每15个指令**optimizer update**后1个TIPupdate，不以microstep计数 |
| 输入 | 每视图独立image、32帧主设置 | 最多32真实帧、短序列不复制；固定448 letterbox，所有模型相同预处理；保存裁剪/缩放/帧ID |
| checkpoint | 未完整说明恢复 | 每100step含optimizer/scheduler/RNG；代码/配置/manifest不可变，完整重载后才accepted |
| 算力 | 与现有40GB卡不同 | ZeRO-3 SFT；TIP DDP；micro1/2/4真实测速、global64不变，GA=64/(卡数×micro) |

### 关键语义校准

- Qwen patch16 与 VGGT patch14 不同。在同448画布中，VGGT查询的是Qwen patch中心的**像素坐标**，不是把VGGT patch ID当Qwen ID。
- Qwen pre-merger张量采用merge-block排序，不是普通row-major；四个出口必须共享同一图索引。
- 中间出口只修改供DeepStack merger使用的分支，不覆盖继续向后运行的视觉主干。generation不能在每个decode token重新算图。
- VGGT原生tracking从第一帧查询。多anchor实现必须重排输入并还原帧身份；不得拿其它帧坐标冒充第一帧查询。
- 只缓存冻结教师产生的图。全参数Qwen特征随训练变化，不得复用旧视觉特征缓存；旧GeoWire缓存不能直接复用。
- 负例只能从允许的相同样本/源视图挑选；同一目的地的真实支持、confidence与mask不变。单视图无跨视图图，不能补self-loop制造训练监督。

## 评测与统计

VSI-Bench完整5130条为主；同时保留ReVSI。补充MMSI、**MindCube-Tiny**、ViewSpatial、SITE、CV。先固定各官方revision/评测split/题目ID，保留完整原提示及我们的格式适配层；未能复用官方metric的benchmark不得用通用字符串相等替代。

统一每帧独立image、最多32真实帧、448letterbox、greedy、non-thinking、512新token。固定金标不可见答案提取；报告官方算术的strict/extracted、解析/截断率、逐题与逐类分母。此输入/格式适配协议不是官方leaderboard完全等价推理。VSI数值MRA与多选准确率分开，按官方任务宏平均，不按题目数量加权冒充Avg。

对应关系诊断需另一个scene级heldout清单，含真实depth/intrinsics/pose及坐标约定；VGGT自身预测不能同时充当方法和真值。指标为精确patch R@1、0.1×max(H,W)像素阈值PCK、归一round-trip误差。场景/点/遮挡/无效深度/匹配特征层与距离函数均需锁定；当前不足时只记未完成，不输出稿件表格。

主表用固定最终checkpoint；不在test上选超参数、帧数或最好checkpoint。首轮seed3407；保留逐题输出，以scene/视频为单位配对bootstrap。需要独立seed重复后才作稳定性主张。记录TIP额外GPU小时、图生成缓存命中率、有效tokens/s、显存与训练时间。

## 阶段门与当前边界

实现入口：`spatial_intelligence/georoute.py`、`georoute_inputs.py`、`scripts/train-georoute-stage.py`。

1. 几何图、独立分支、无入度identity、负例/答案泄漏、真实tiny Qwen+保存重载CPU回归。
2. 完整五源媒体/标签/场景审计与benchmark adapters准备。
3. 前序论文结束后，真实多帧VGGT tracking/全模型forward-backward、组件更新、32帧长样本与重载/推理验收。
4. 锁定micro并从正式初始权重训练TIP、SFT与结构对照，诊断权重不进入主表。
5. 所有GeoRoute必选实验与评测成功，才释放GeoFits。

当前worker保存为`trained_pending_acceptance`，不自动声称`accepted`。全模型验收/完整benchmark桥接未通过前，训练队列必须停在准备门。源码函数存在、CPU测试通过或下载完成都不是正式复现成功。

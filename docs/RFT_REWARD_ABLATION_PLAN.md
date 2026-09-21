# RFT 奖励消融与后续论文顺序

日期：2026-09-21。最新修订：用户批准恢复旧词表和任务模板；代码已增加 `geopsro-lexicon-strict-v2`，保留v1仅作历史回归。部署仍须使用新不可变快照，不能热改旧队列。

## 执行顺序与选择

现有 SFT → RFT 数据/模型候选 → 固定验证集选择 → 两个奖励消融 → 全部配对测评 → GeoRoute 五组 → GeoFits。

当前 RFT 已排的候选实际上是同一个 downsample-trainable SFT 初始化上的 mixed（4DRL/SpatialLadder=70/30）和 four_d_only。不是所有几何模型都已安排 RFT。只有完成同协议训练和验证的组合才能进入候选集。

预先约定选择依据：4DRL heldout 场景的题型宏平均正确率为主，SpatialLadder heldout 分项成绩为辅助；主指标相同则选验证生成 token 更少的组合，再按固定候选名称排序。必须保存候选清单、逐题预测、验证分数和选择原因。不得使用 DSR-Bench、ReVSI 或 VSI 最终测试成绩选择训练组合。

## 三组奖励契约

| 组 | answer 权重 | format 权重 | words 权重 |
|---|---|---|---|
| answer_only | 1 | 0 | 0 |
| answer_format | 1 | 0.5 | 0 |
| full | 1 | 0.5 | 0.05 |

三组都从胜出组合的**同一 SFT 初始化**开始，不从 full-RFT 权重续训。共享数据 manifest、采样顺序、种子3407、G8、global prompt batch16、更新数、学习率、KL、512生成上限、prompt、答案提取和验证/测试清单。仅关闭奖励项，不改提示模板。answer-only 表示无额外格式奖励，仍使用相同最终答案提取契约。

若 full 主实验已使用完全相同的锁定契约，可复用它；若词表、prompt、匹配算法或计分改变，必须重跑 full，不能只补两个消融后横比。

## 已发现的词表与 prompt 不一致

迁移前 `spatial_intelligence/geometry_rft_reward.py` 是简化三组单词匹配、tau3；`geometry_rft.py` 使用新写短结构化 prompt。不能将旧运行标注为使用原 GeoPSRO 完整任务。新配方显式选择v2词奖励及原模板的numeric-safe适配版本。

已在本地保留的 GeoPSRO 交接源码中找到 `geopsro4d/reward/reward_fn.py`：

- OBSERVATION_WORDS：163；TRANSITION_WORDS：154；DERIVATION_WORDS：88。
- SPATIAL_RELATION_WORDS：101；EVIDENCE_GEOMETRY_WORDS：82；UNSUPPORTED_RISK_WORDS：21。
- 各集合有重叠，不能把相加值当作独立词总数；包含多词短语。
- 原 words 计分含五类加权覆盖、正确答案门控、重复/堆词防刷规则；风险词命中在所检查版本仅作统计，不能宣称已扣分。
- 原 format 是分项计分，当前实现为严格结构判定；恢复词表不等于恢复完整 reward。
- `geopsro4d/data/formatters.py::psro_prompt` 保存了原模板；本次找到的这个模板本身并非很长，尚未确认用户所指的大 prompt 的确切版本。

新版本采用旧源码五类加权公式、完整六类词表及防刷逻辑，保留严格格式判定和金标不可见答案解析，不恢复旧宽松答案解析或旧数值容差评分。单词和短语均加边界检查，避免frameshift被算作frames。数值题仅answer reward达到1时才开启words，部分正确保留数值分但不发词奖励。原prompt增加纯数值与单位适配说明，不将整个奖励词表塞进输入。论文三字段饱和公式与此次五类加权公式仍有差异，需在稿件中同步描述；不是旧计分代码原封不动回退。

## 验收与指标

先人工核对正常答案、错误答案、格式缺失、短语、重复词、词表堆砌、冲突答案、数值边界与截断案例。三组相同输出的 answer 分数必须一致；关闭项贡献严格为0；错误答案不能获得 words 奖励。通过真实 G8 rollout、非恒定组内奖励、有效梯度及保存重载后才扩量。

主表报告 DSR 分题型/宏平均、ReVSI、SpatialLadder 数值与选择题分开成绩；辅助记录解析成功率、格式完整率、词表覆盖、堆词比例、答案长度、截断率、token和GPU小时。用场景级配对 bootstrap；单 seed 不能支持稳定性结论。words提高而准确率不升，不能解释为推理能力提升。

## 后续论文契约及部署边界

GeoRoute/GeoFits 从 released Qwen3-VL-2B 各自初始化，不继承本轮 RFT；顺序仅表示资源依赖。共享六源数据：SPAR234K、Hound64K、VSI590K、GeoThinker对应32帧VLM3R VSI205456＋VST132053、MindCube train10000、OpenSpatial中ARKitScenes来源100K条。取消先前错误的VLM3R100K上限；VST另按官方errata修订。最终数量以媒体及授权数据协议后的manifest为准；MindCube完整21154清单包含训练数据，不能拿它充当独立测试。用户明确允许OpenSpatial无场景映射训练，记录其污染风险unknown，不伪造leakage_checked=true。

GeoRoute：full、no_tip、one_stb、final_only、post_merger。GeoFits：full优先，3D-only/4D-only/dense/no_gate/single_layer为待补实现和验收的后续消融，不声称全部已经可自动运行。

全参数替代论文LoRA；GeoFits还解冻原论文冻结的native RGB。外部VGGT/Pi3冻结。共享1epoch/global64/LR1e-5/seed3407，最多32真实帧，独立image输入。精确子集、TIP步数/权重、图阈值及部分模块宽度为显式本地选择。它们是受控全参数改编，不是论文数字的严格复现。

部署记录：为衔接用户批准的global64 SFT，followup v10 已替换旧等待器，依赖主RFT v7与奖励消融v3完成，再依次执行GeoRoute、GeoFits。两组主RFT共同使用新global64 downsample-trainable SFT初始化；RFT自身prompt batch16、G8、奖励及采样预算没有改变。每次更新创建不可变新版本，没有热改armed计划；旧等待器仅在确认尚未运行训练时停止。该状态表示排队生效，不代表正式训练已通过验收：数据、真实教师、全模型和补充benchmark验收未齐之前继续保留阶段门。

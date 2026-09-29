# GAP-4D：结构化推理冷启动与后续 RFT

状态：代码和候选清单准备；教师下载可独立运行。**未授权标注，不能启动标注、冷启动训练或新 RFT。**

## 实验契约

样本规模说明：下表保留已有4,000条准备方案；约1,000条冷启动实验应在新私有
plan 显式设置 `target_count: 1000`，并单独声明候选与验证规模。代码目前缺省
target_count 为4,000，不能省略此字段后声称运行的是1K。发布这些脚本不授权扩大
训练规模，也不修改任何现有清单或运行计划。1,000条、global64对应16次更新，
尾部补齐24条；实际合格样本和补齐量以运行回执为准。

问题：从成熟 Qwen3-VL-2B + VGGT SFT 初始化，增加少量视觉证据支持的三段式监督，能否改善格式遵循，并让后续 GSPO 中的奖励项真实生效？格式学会不等于空间能力提高。

教师是固定 revision 的 `Qwen/Qwen3.5-9B` 后训练版，见
[下载配置](../configs/rft-coldstart-assets.json)。教师不带学生的 VGGT，也不替换学生 backbone。
学生是已完成 SPAR/Hound 两阶段、下采样接口、解冻 VGGT 的成熟 Qwen3-VL SFT 权重；不是目前在训的 Qwen3.5 alignment，也不是已训练的 RFT 权重。

原始“35K + 15K”是近似名称。当前已审计 RFT train 实际为 32,558 条 4DRL、14,517 条 SpatialLadder，共 47,075 条。保留历史 RFT manifests，不用新的冷启动筛选覆盖它们。

| 集合 | 4DRL | SpatialLadder | 用途 |
|---|---:|---:|---|
| 初始标注候选 | 4,200 | 1,800 | 为正确性/格式过滤预留余量 |
| 首批人工核查 | 140 | 60 | 候选内的200条，不额外重复计入训练 |
| 合格冷启动目标 | 2,800 | 1,200 | 4,000条，单轮监督 |
| 独立验证 | 280 | 120 | 从既有来源组隔离 validation 取样，不进入 SFT/RFT |

70:30 与已声明 mixed RFT 抽样一致，并接近当前69.16:30.84的自然组成；这是受控起点，不是“最优比例”结论。源内按题型、答案类型、帧数比例分层；分层内对视频/场景组轮转，降低单个来源的重复占比。固定 seed3407。

每个分层的最终配额以候选清单预先锁定，不能按教师正确率重新分配到容易题。若6,000条不够筛出4,000条，停止并报告各层缺口，另建补充版本；不降低标准凑数。标注耗时和预算应在200条真实输出后估算。

## 数据语义与限制

- 保留学生 RFT 原来的 RGB 帧、视频帧索引/FPS或多图顺序；不为教师增加学生看不到的帧。
- 原始 CoT、标准答案、raw_annotation 均不进入教师输入；采用字段白名单，而不是仅删一个 answer 字段。
- SpatialLadder 多视角图像不自动解释为物体运动；未知时间顺序不能虚构。
- 发现部分4DRL问题中的秒数超过对应片段时长。当前冷启动筛选将其隔离为 `question_time_exceeds_clip_duration_unresolved_origin`。这只是时间原点/时长异常标记，不足以证明全部标注错误；禁止猜测 offset 或缩放时间。
- 当前候选筛选排除10,839条训练、183条原验证时间疑点；保留排除ID及原因。未知的未标记时间错误和采样过稀仍可能存在。
- 原 RFT 数据暂未改变。完整新 RFT 前必须有 `rft-timebase-review.json` 的证据化复核；若修复数据，使用新 `verified_data_receipt`，重新报告训练分母与预算，历史分数不再冒充严格对照。
- 教师筛选会产生选择偏差，即使保持题型配比也不等于原始难度分布。基础模型预训练暴露及未知来源别名仍未排除。

## 标注与接受条件

教师关闭额外隐式 thinking，显式输出仓库既有的三个标题：`Spatial Observation`、`Spatial Transition`、`Answer Derivation`，以及 `<think>` / `<answer>` 标签。目标是简洁、可核验的视觉解释，而非冗长思维文本。保存原始生成 token、文本、教师版本、输入清单、实际视觉 grid 和完整 prompt。

自动接受必须同时满足：无截断、三个段落齐全、选择题答案正确、数值题标量与金标相等、学生tokenizer下含EOS不超过512token。数值训练样本不使用“相对误差5%也算满分”来放行。禁止把错误教师答案替换成金标后仍把解释当正确。

前200条必须由具名审查者检查实际画面依据，包括自动不合格的样本。`human-review.json` 包含 `approved`、`reviewer`、`reviewed_ids`、`rejected_ids`、`notes`。自动正确性不能代替这一语义检查。冷启动 validation 保留所有固定问题，而不是只评测教师答对的子集。

## 训练与 RFT

冷启动：1 epoch，global batch64，四卡 micro1、GA16，LR1e-5，3% warmup、cosine、sample-mean completion loss。4,000条约63次更新，末尾重复补齐32条并记录。语言全参数及几何 projector 训练，RGB和已训练VGGT冻结，无LoRA。冻结视觉是为了先学习短程格式/解释，不是重做全量视觉 SFT。该设置是新实验配方，不声称官方 RoboRefer 配置。

使用与 RFT **完全相同**的 `prompt_inputs` 和 `response_logps` 路径，避免用普通多图 SFT Collator 悄悄替代视频时间编码。FP32 可训练参数与 AdamW CPU 状态，bf16 autocast。保存模型、optimizer、scheduler、各rank RNG和精确world/batch/数据契约；新运行需实际 GPU loss、参数更新与重载一致性验收。

最终冷启动权重为 FP32 `full_trainable.pt`，**依赖原成熟 SFT 基座，不是独立合并模型**。后续 RFT actor和冻结reference都从“同一SFT基座 + 同一冷启动权重”初始化，不能误用旧SFT作为reference。冷启动和RFT的overlay血缘检查、CPU小模型精度回归均有测试。

后续 mixed RFT 复用已有全语言+projector GSPO：G8、global prompt batch16（每步128条response，不等于SFT batch64）、LR1e-6、KL0.02、clip0.0003/0.0004、512输出token、70:30 source sampling。保留原大词表、原三段式prompt和奖励权重；不是新写一份简化词表。若旧数据原样验收，2-pass预算为5,885步、94,160次prompt抽样；若数据被修正，预算必须重新计算。

先执行真实rollout/update/reload gate；词表奖励须有正值并影响组内advantage。格式在冷启动后可能所有回答都正确，因此允许显式报告 `format_already_satisfied_no_advantage_effect`；不能将这种饱和视作格式奖励带来RL梯度的证据，更不能把全零奖励当饱和。旧实验默认门槛保持不变。

测评固定 DSR-Bench、ReVSI、VSI-Bench 及源内validation，比较成熟SFT / 冷启动后SFT / 冷启动后RFT三列。保存逐题完整回答、相同样本/帧/提示/输出预算、解析与截断率；不把换prompt后的成绩混进旧表。只有真实测评能回答能力是否改善，不保证提升。

## 操作

实际机器路径只放私有plan；`$ROOT`是持久存储，`$PLAN`指新实验plan。不要把这些示例变量写成实际主机信息提交。

```bash
python scripts/fetch-study-assets.py --root "$ROOT" \
  --catalog configs/rft-coldstart-assets.json --workers 1 --detach
python scripts/prepare-rft-coldstart.py --plan "$PLAN"
```

以上只下载和准备。**现在不要执行下面两条命令。**

收到用户明确的标注启动命令后，先200条：

```bash
python scripts/run-rft-coldstart.py --plan "$PLAN" --execute-authorized --pilot --detach
```

完成具名视觉复核后，继续剩余标注并自动衔接训练：

```bash
python scripts/run-rft-coldstart.py --plan "$PLAN" --execute-authorized --detach
```

队列先等现有GPU任务的终态回执和显存释放，不抢占健康训练。标注、语义复核、样本配额、冷启动重载/validation或RFT时间语义验收失败时明确停止，不自动放宽协议。

## 当前验收边界

CPU测试验证分层配比、隔离、金标白名单、数值严格过滤、初始化血缘和FP32保存恢复；静态检查验证源码解析。尚未授权真实教师输出，因此不能声称9B标注质量、GPU标注吞吐、冷启动首步、RFT奖励修复或下游效果已经通过。下载完成、脚本就绪与训练成功是三种不同状态。

2026-09-28验收：目标Linux环境中8个相关测试文件共 **80 passed**（含旧PEFT兼容测试的6条警告，不代表本配方使用LoRA）。候选及验证集的38,892个不同媒体路径存在且非空；真实8帧视频、8图多视角的CPU processor检查通过，学生tokenizer保留三段式字面标签。未调用教师模型forward。该检查不等于所有媒体均已完整解码，也不等于GPU显存/吞吐验收。

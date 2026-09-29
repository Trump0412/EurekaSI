# Geometry RFT：训练、评测与归档子项目

状态：正式主实验和收尾评测进行中，**尚未结项**。本页是可复用入口，
不是“任意服务器安装即已验收”的声明。机器映射和上传凭据仅放私有配置。

2026-09-28 审计：历史完整奖励与两个奖励消融的 structure/words 均未形成
有效奖励信号，不能把分数差异解释为这两项的因果贡献。旧权重和原始输出
保留；新版增加奖励活性门，但不代表修复后的正式实验已经完成。详见
[奖励审计](../../docs/GAP4D_REWARD_ACTIVITY_REPAIR.md) 与
[权重及证据备份指南](../../docs/WEIGHT_BACKUP_AND_RESTORE.md)。

2026-09-29 源码补充：新增[结构化冷启动流程](../../docs/RFT_COLDSTART_RUNBOOK.md)
与[可选动态采样 pilot](../../docs/RFT_DYNAMIC_SAMPLING.md)。包含数据筛选、
教师输入白名单、人工复核门、冷启动权重血缘及 actor/reference 初始化。
这些是待实际 GPU 验收的实验入口，不是已完成的奖励修复结果；发布代码不启动队列。
冷启动样本量必须在独立 plan 中显式设置，不能把文档示例当作已批准的实验规模。

本次发布检查：静态解析/文档链接与release检查通过；独立Linux源码快照、
屏蔽GPU的9个相关测试文件共74 passed（旧PEFT兼容用例6条警告）。覆盖冷启动
筛选与权重血缘、动态采样、奖励活性、队列、worker和保存恢复；不代表真实教师
标注质量、多卡冷启动或正式RFT验收。私有计划、权重、审稿材料不随源码发布。

## 研究对象

Qwen3-VL-2B + VGGT，下采样空间接口。共同 SFT 初始化使用
SPAR234K + Hound64K。两种 RFT 主实验为 4DRL-only 与
4DRL + SpatialLadder 多帧混合；固定 prompt 抽样预算与 G=8。
奖励消融沿用已部署的 answer-only、answer+format、完整奖励三个版本。
以实际配置和 receipt 为准，不能把历史 LoRA pilot 混入正式结果。

正式 `language_full_geometry` 更新语言模型全参数及几何接口，冻结
RGB 与 checkpoint-specific VGGT；这是**非 LoRA**，不是所有视觉参数也更新。
运行实现为本仓库 HF 几何策略/GSPO，不声称是 VERL/vLLM 几何模型注册。

## 代码地图

| 环节 | 入口 | 验收 |
|---|---|---|
| 科学配置 | `configs/geometry-rft.json` | 数据比例、预算、prompt/reward 版本 |
| 数据 | `scripts/prepare-geometry-rft-data.py` | 媒体、场景隔离、固定 manifest |
| 模型/保存加载 | `spatial_intelligence/geometry_rft.py` | actor/reference、固定视觉、精确恢复 |
| 目标与奖励 | `geometry_rft_objective.py`、`geometry_rft_reward.py`（公共模块） | 组内差异、金标不进入输入、非零更新 |
| 训练/原配对评测 | `scripts/train-geometry-rft.py` | 正式完成与独立重载 receipt |
| 持久队列 | `scripts/run-geometry-rft-queue.py` | gate → train → evaluate |
| 奖励消融 | `scripts/run-rft-reward-ablation.py` | 只用源 heldout 选分支，不能用 test 选模型 |
| DSR 四 SFT 对照 | `scripts/run-dsr-sft-queue.py` | 四份原 checkpoint、逐题输出 |
| 新收尾评测 | `scripts/run-rft-closeout.py` | benchmark 独立失败、先格式 smoke |
| 结果/权重清单 | `scripts/summarize-rft-project.py` | JSON 证据表、Markdown、文件大小清单 |

## 队列顺序

现有 SFT/RFT及其评测 → DSR四权重补测 + 奖励消融（不同节点可并行）
→ 三模型扩展评测 → GeoRoute → GeoFits。

三模型固定为 SFT backbone、mixed-RFT final、4D-only-RFT final。
新评测共12条 benchmark lane（SPBench拆SI/MV），即36个模型×测试集。
每条 lane 独立保存失败原因；一条失败不会取消后续测试。
`queue_finished=true` 仅代表所有项已尝试/终止，不代表所有指标成功。
GeoRoute 等所有节点的完成/释放凭据，不由“某一个评测结束”提前触发。

## 新机器运行步骤

先按 [矩阵指南](../../docs/GEOMETRY_MATRIX_RUNBOOK.md) 建独立 Conda 环境，
保留已有 Torch/Transformers 配套，不向运行中的训练环境安装新包。
本项目复用 geometry-matrix 环境的模型加载器、PyArrow、Pillow、OpenCV。

```bash
python scripts/fetch-study-assets.py --root "$ASSET_ROOT" \
  --catalog configs/rft-closeout-assets.json --workers 4 --detach
python scripts/prepare-rft-closeout-suite.py --plan "$PRIVATE_PLAN" --detach
python scripts/run-rft-closeout.py --plan "$PRIVATE_PLAN" --detach
python scripts/summarize-rft-project.py --plan "$PRIVATE_REPORT_PLAN" --detach
```

私有评测计划字段：`root, python, gpus, dependencies, model_role,
model_checkpoint, sft_receipt, processor, vggt_source, assets_root,
reuse_prepared, prepared_root`；RFT 另给 `rft_receipt`。
依赖应包含原主实验、奖励消融和 DSR 补测完成且 GPU 已释放。
只读共享资产；输出、代码快照、锁独立。已启动计划不热改，换新版本迁移。

## 评测契约与边界

DSR 补测格式诊断（2026-09-26）：混合题型结构化提示下，三份 SFT
未通过解析门，其中两份经常对选择题输出数值。使用固定 13 条诊断题、
不变权重/视频/解码预算/评分器，改为选择题专用 `mcq-tagged-v2` 提示后，
四份 SFT 均达到解析 13/13、截断 0/13。此证据支持提示敏感性，不证明
完整 benchmark 分数提升或唯一根因。四份模型全部按同协议重测，结果与
历史 structured512 分开报告；不可直接与旧 RFT 配对成绩相减。
`spatial_prompt_inputs(..., instruction=...)` 的默认值仍为旧训练提示，
不会改变运行中的 RFT 或奖励消融。不得通过放宽解析规则把数值映射成字母。

- 共同样本/帧序、最大边448、greedy、512新token、同一结构化提示。
- 视频均匀32帧，真实FPS/原帧号，不使用字幕/音频。STI保留问题指定时间，
  输入全视频；这是固定资源的配对适配，不保证原论文采帧配置一致。
- MMMU 保留题目及选项中图片的交错位置；不能把图片标记当纯文本。
- MMBench 为英文 dev，MMMU 为公开 validation；不冒充隐藏 test 成绩。
- MindCube 使用公开 Tiny 测试子集1050题，不使用约10K训练集。
- CV-Bench 沿用源加权指标；其他选择题保存微平均及类别分数。
- SPAR使用其数值/视角变化算术与任务宏平均，但最终答案提取为金标盲的
  适配版。当前固定数据revision实际7211行，不能照抄README的7207当分母。
- MMMU open 暂用保守最终答案精确匹配，不声称官方宽松解析等价；
  MMBench 暂报循环题目行级准确率，不冒充官方循环一致性分数。
  这些差异必须随表保留，官方重评分仍是收尾审计项。
- SPAR是本模型SFT来源，分数属于训练分布内迁移，不是零样本泛化证明。
- 不使用付费 judge；任何缺媒体/不支持题型都明确失败，不能静默缩小分母。
- 全量前进行固定每类别首题+最长输入 smoke，按解析/截断判定，不按准确率选配置。

源：[SpatialLadder](https://github.com/ZJU-REAL/SpatialLadder)、
[SPAR](https://github.com/LogosRoboticsGroup/SPAR)、
[DSR Suite](https://github.com/TencentARC/DSR_Suite)、
[VLM4D](https://huggingface.co/datasets/shijiezhou/VLM4D)、
[STI-Bench](https://huggingface.co/datasets/MINT-SJTU/STI-Bench)。
数据固定revision和文件范围见 `configs/rft-closeout-assets.json`。

## 权重归档：必须保存依赖闭包

1. 每种正式SFT的完整 final：语言、RGB、几何接口、训练后的VGGT，以及配置。
2. 每种正式RFT final policy：`full_trainable.pt`、`policy_scope.json`、tokenizer等。
3. 与RFT绑定的SFT基础包，不能用 released Qwen/VGGT代替。
4. processor、锁定代码快照、科学/部署配置、训练完成与重载凭据、逐题评测。
5. 需要续训时额外保存完整末步 optimizer/scheduler/RNG；推理包不是续训包。

`policy_scope.json` 的历史绝对路径是溯源约束。异地恢复必须显式重映射
并重新验收，不通过删除 lineage 检查伪装独立权重。清单只统计文件/字节，
不计算SHA256。云端目标须用户确认且默认私有；代码GitHub不上传权重、
私有路径、连接配置、原始基础设施日志或凭据。

## 结项门

训练完成不等于结项。须汇总正式主实验、消融、DSR补测、扩展评测；
失败项清单可保留但不可写成0分。代码测试、真实多模态输入验收、
checkpoint异地恢复、云端上传验证、官方评分差异审计分别记录状态。
尚未完成的项目保持 pending；没有证据时不宣布“完全复现”。
# Closeout repair protocol (2026-09-26)

The paired expanded benchmark suite uses
`rft-closeout-structured512-max32-even-equivalent-options-v2`.
Video sampling selects at most 32 distinct real frames, uniformly across the
full clip; short clips select the largest even count to match native temporal
patching (31 frames -> 30). No duplicated frames or removed questions are used.
Decode integrity, source FPS and exact selected indices are checked/recorded.

The pinned VLM4D real split contains one item (`validation_633`) whose C and D
options have identical text and whose released answer is that text. Preserve
both in `acceptable_answers`; either explicit letter earns credit. This is an
explicit adapted scoring rule, not a claim of official leaderboard parity.
Gold and equivalent-answer metadata are excluded by the inference input whitelist.
All compared models consume the same manifest and scoring snapshot. Existing
completed historical results are unchanged.

Supplemental SFT-only DSR evaluation needs completed SFT checkpoints and its
own idle GPU allocation, not completion of an unrelated RFT on another node.
Version downstream queue dependencies together when replacing a waiting owner;
do not rewrite armed plans or mark superseded queues as successful evaluations.

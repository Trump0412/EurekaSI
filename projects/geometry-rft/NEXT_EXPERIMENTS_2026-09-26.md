# 几何 SFT→RFT：下一轮实验契约

更新：2026-09-27。用户已授权补充训练，部署按真实运行状态单独记录。
最高主张为受控离线比较，不预先宣称 SOTA。

## 最新决策：优先于 GeoRoute

以下安排覆盖本文旧提案的优先级；不扩大数据量，不启动接口 RFT 矩阵。

| 优先级 | 实验 | 固定预算和比较目标 |
|---|---|---|
| 1 | 原始 Qwen3-VL-2B-Instruct → 4D-only full-language RFT；无本地 SFT、无 VGGT | 与已有几何 RFT 固定同清单、采样顺序、G8、prompt batch16、5885更新、奖励和提示 |
| 2（可并行） | Qwen3.5-2B → 下采样 VGGT alignment → 解冻 VGGT SFT → mixed RFT（4DRL 70%＋SpatialLadder 30%） | 对照现有Qwen3-VL mixed组；同 SPAR/Hound 清单、两阶段各1 epoch；SFT batch64；RFT预算同上 |
| 后续 | GeoRoute → GeoFits | 新对照训练和对应测评验收完成后释放 |

原始模型指公开 Instruct/post-trained 权重，不是随机初始化或未指令微调的
基础权重。无SFT指不执行本项目空间SFT。RFT仍冻结原生RGB；语言全参数、
不用LoRA。历史作用域字符串 `language_full_geometry` 保留兼容，原生分支
以 `model_kind=native_qwen3vl` 显式标记，参数清单中没有任何几何模块。
不伪造SFT验收文件。此项同时去掉SFT和VGGT，只支持二者组合收益，不能
分别证明两者的独立贡献；总计算和几何token预算也不同，单列GPU小时。

Qwen3.5沿用当前两阶段参考：alignment batch448、只训练接口；SFT batch64、
语言/RGB/接口/VGGT联合训练。alignment batch448为历史参考预算，不称所有
阶段都是64。RFT prompt batch16×G8=128条响应，不是SFT batch64。
2026-09-27纠正：Qwen3.5原先误排的4D-only等待队列已停用，以新版本mixed
队列替代；与现有mixed组保持同奖励、提示、seed、数据清单和5885次更新。
rollout micro上限同为4，逻辑G8不变。原始Qwen3-VL无SFT/无VGGT对照仍为
4D-only；它对应已有4D-only结果，不与mixed结果当作同数据单因素对照。
更换backbone也更换原始预训练语料和模型架构，因此属于跨模型稳健性，
不是证明某个注意力机制更优的单因素实验。每种backbone都记录原始及训练
后指标，不能拿旧异协议得分相减。

必测DSR/ReVSI/VSI；拓展CV-Bench/MMMU等同ID测评需单独验收，不能把规划
当作已经运行。现有RFT structured提示与新的typed-MCQ指标分表。
新原生RFT走同一structured协议用于直接对照；Qwen3.5阶段间也需相同协议。

阶段门：原生模型加载→真实图像/视频rollout→初始actor/reference一致→
有限非零更新→reference不漂移→FP32保存重载→正式训练→固定终点评测。
Qwen3.5另验混合注意力cache、多帧位置及几何梯度；CPU微型模型测试不能
代替完整GPU门。失败停在该项，不中断已在运行的旧实验，也不跳过验收。
两阶段训练器按micro1/2/4真实profile及梯度等价选择吞吐更好的配置。

下面的加数据2×2和其他SFT权重接RFT方案保留为历史候选，**本轮不执行**。

### 原始模型直接RFT的格式修复记录（2026-09-27）

原始Qwen3-VL的首个真实GPU诊断生成256条回答：236条缺少/重复答案标签，
20条截断，全部奖励为0，没有组内差异和有效更新。初始reference一致性、
reference不漂移及checkpoint重载均通过。直接检查原始token解码也没有
完整答案标签，排除了`skip_special_tokens`丢失标签的解释。

首轮修复新增显式可选提示版本`geopsro-original-template-native-literal-tags-v3`：
保留原提示，附加字面XML标签及简短答案序列化约束。不变更词表、
样本、G8和正式更新预算，不插入SFT，不用测试答案选提示。旧默认提示
保持逐字不变；该修复版本仅允许原生、无SFT/VGGT的对照使用。

仅修改提示仍未通过：256条中84条已有唯一终止答案标签，但由于推理文本
没有think外壳，旧评分器将答案奖励一起拒绝。补充独立版本
`geopsro-final-answer-v4`：对未截断、唯一完整终止answer标签单独评分，
不从推理数字/选项中搜索正确答案；未完成think、重复标签、尾随内容、
显式冲突答案拒绝。格式/词表奖励仍要求原有完整结构，权重和词表不变。
该版本仅配置给新的原生对照，旧训练及Qwen3.5 mixed继续其锁定版本。
固定已存训练诊断离线重评分得到84/256可解析、26条正奖励、12/32组有
差异；这是奖励链路诊断，不是benchmark分数。57项定向回归通过。

失败诊断保留，新运行从原始权重开始，不从零更新诊断checkpoint续训。
这引入了**训练提示和奖励解析的显式适配差异**，所以不能将它与旧几何RFT称为只改变
架构的严格单因素实验。当前新运行自身训练前后测评使用同一新提示；
若要与旧主表严格配对，还需追加相同推理提示的测评，不能直接混表。
真实GPU更新验收与正式训练状态以运行receipt为准，静态/CPU通过不代替它。

## 1. 当前融合实现

当前 backbone 为 Qwen3-VL-2B-Instruct，语言隐藏维 2048。RGB 原生视觉
路径保留。VGGT aggregator 最后一层移除非 patch tokens 后得到
`[B,T,1024,2048]`；输入为每帧 448×448，patch 网格 32×32。

下采样不是平均池化或随机丢 patch：32×32 右/下补零成33×33，将每个
3×3 邻域按通道拼接，得到 `[B,T,121,18432]`。随后投影为：

`LN(18432) → Linear(18432,6144) → GELU → LN(6144)`
`→ Linear(6144,2048) → GELU → Linear(2048,2048)`。

每帧121个几何 token 插在对应 RGB 图像或原生视频时间组之后；其监督
label 为-100，仍参与语言模型注意力。不是先解码深度图，也不是送回
Qwen视觉 encoder。原生 RGB 路径和位置处理保留，VGGT不是额外DeepStack。
几何骨干只在生成 prefill 时执行，通过KV缓存用于后续生成。

Query64 路线将2048维特征投影至256维，加入帧/patch位置编码，64个学习
query经8头cross-attention汇聚全部帧，再用MLP投影至2048维。整段视频
固定64个token；下采样32帧则为3872个token。这是压缩/布局/成本同时
变化的接口比较，不能称为相同token预算下比较。

Alignment只训练接口；正式SFT训练语言、RGB、接口，VGGT按冻结/解冻组
设置。当前正式RFT训练语言全参数及接口，冻结各自SFT的RGB/VGGT。

## 2. 历史候选（已被上述最新决策替代）

| 优先级 | 问题 | 最小实验 | 决策规则 |
|---|---|---|---|
| P0 | 基线差距有多少是提示/格式导致？ | 同一Qwen3-VL原始模型、SFT、两种RFT在同manifest同提示下比较 | 不跨backbone/协议相减；报告解析率 |
| P1 | SFT接口优劣是否在RFT后保留？ | 下采样解冻、下采样冻结、Query64解冻，各接同一4D-only RFT | 固定更新/采样预算；比较最终与增量，并计GPU小时 |
| P2 | VSI/VLM3R应加入SFT还是RL？ | 下述2×2阶段因子实验 | 看DSR/ReVSI/VSI与通用能力的共同变化 |
| P3 | 方法能否迁移backbone？ | Qwen3.5-2B的原模型→alignment→SFT→RFT | 完整接口验收后再训练；更小模型后续再扩 |

P1先做下采样冻结最容易复用；Query64已有matrix loader支持，但其RFT
训练、reference、生成插槽、保存重载仍须真实验收。旧direct模型的冻结
范围和训练阶段不同，只列工程对照，不当成干净单因素实验。

## 3. SFT / RL 加数据的2×2设计

S0=SPAR/Hound；S1=SPAR/Hound+训练版VSI590K/VLM3R。
R0=固定4D-only训练来源；R1=相同基础来源+新增可验证空间QA子集。

| 组 | SFT | RL | 可回答的问题 |
|---|---|---|---|
| A | S0 | R0 | 现有锚点 |
| B | S1 | R0 | 只增强SFT空间知识的作用 |
| C | S0 | R1 | 只改变RL任务分布的作用 |
| D | S1 | R1 | 阶段交互，不假设效果可相加 |

每个SFT终点也先测再做RL，不能只测最终D。预算控制版从相同alignment
开始，匹配更新数、样本曝光预算、LR日程并记录实际视觉/文本token和
GPU小时。额外跑全数据epoch属于扩规模实验，单列，不能把更多计算
冒充数据源收益。先做固定小预算pilot，按source-heldout选择扩展；
禁止用DSR/ReVSI测试得分调比例。

新增RL只先纳入可验证的MCQ及单位明确的数值QA，保持G8与同一奖励定义。
开放描述不能硬套exact match。按任务/场景均衡采样，不直接把整个新增
数据集淹没4D来源。数值容差及归一化固定；检查组内奖励方差、无效输出、
KL与非零更新。当前完整RFT预算为5885次更新、prompt batch16、G8，
不是SFT的global batch64，不能不加区分直接照搬。

## 4. 数据与测量契约

现有六源准备包中VSI590K为590578条、VLM3R为335098条（当前清洗清单，
不是新实验承诺使用的最终条数）。可复用媒体，不需要盲目重下。
新研究仅选择所需训练来源，重新锁定清单、清洗版本和任务比例。

- train/validation/test按场景与原视频组隔离；VLM3R、VSI、SPAR跨来源
  重叠需检查，VSI590K与VSI-Bench不能混淆。
- 遵守VST已有标注纠错，不恢复旧答案。标注/视频缺失显式记录。
- 不沿用OpenSpatial来源映射缺失的豁免来声称本研究无泄漏。
- 同一原始backbone比较，保持帧数、原始时间戳、分辨率、解码预算、
  提示及金标盲解析一致；MCQ用专用指令，不把数值输出映射为字母。
- 现有DSR覆盖1450题，官方说明1484题；适配提示、采帧和覆盖不同，
  67.59%目前不能单独证明官方SOTA。

## 5. 指标、阶段门与停止条件

主指标：DSR准确率及任务宏平均、ReVSI/VSI官方算术的适配分数。
辅助：CV-Bench/MMMU等通用能力、解析/截断率、视频长度分层、成本。
同样本差异优先按场景/视频聚类bootstrap；最终有预算再做多seed确认。

先数据/输入审计→真实GPU冒烟→固定预算pilot→锁定正式运行。必要门：
解析率≥90%、截断≤5%；非零有限梯度；actor/reference正确；保存重载；
模型输入不得含答案。失败不放宽解析、不删除难题来抬分。

Qwen3.5不能只替换checkpoint路径：需重新适配模型类、视频位置、几何
插槽、cache、混合注意力状态、固定reference、精度和保存恢复。小模型
加VGGT-1B不必然更便宜，VGGT前向及几何token的成本可能占主导。

## 6. 运行顺序与复现边界

独立测评节点继续12条扩展benchmark lane，随后原生Qwen3-VL RFT；
另一训练节点完成既有奖励消融和测评后，执行Qwen3.5两阶段及RFT。
GeoRoute等待上述新增工作；其他现有任务保持不变。
扩展三模型都使用typed-MCQ同协议，非MCQ仍保留既定结构化说明。
单benchmark失败留档并继续，GeoRoute语义/数值门失败仍会阻止受影响训练。

配置、样本ID、文件大小/mtime、checkpoint lineage、逐题输出和独立代码
快照保存；无需SHA256。不启动未获授权的全矩阵或数据scaling。

依据：[DSR Suite](https://github.com/TencentARC/DSR_Suite)、
[GeoThinker数据方案](https://github.com/Li-Hao-yuan/GeoThinker)。

# 几何模型的多域评测与遗忘检查

顺序：现有论文 → GeoRoute → GeoFits → 本扩展评测。环境/源码/许可审计可以提前准备，GPU推理和模拟器不能抢占前三项。以下是预登记设计，不是已经完成的测评。

## 1. 不用一个空间分数代表所有能力

每个benchmark比较相同的固定checkpoint：released Qwen3-VL-2B、同数据全参数RGB SFT、GeoRoute、GeoFits。上一篇SFT/RFT作为独立纵向比较，其不同数据和LoRA更新范围不能与全参数组混成架构因果排名。只选预先指定的最终checkpoint，不在测试上选最好结果。

| 维度 | 首批候选 | 能支持的结论 | 不能声称 |
|---|---|---|---|
| 空间主能力 | VSI/ReVSI、MMSI、MindCube-Tiny、ViewSpatial、SITE | 几何/多视图QA变化 | 闭环控制能力 |
| 通用视觉保留 | CV-Bench、BLINK完整任务；AI2D、OCRBench、ChartQA | 感知、图表/图示、文字读取是否退化 | 单靠CV-Bench就覆盖全部通用CV |
| 自动驾驶迁移 | DriveLM公开验证；DriveBench clean/corruption/text-only | 驾驶场景理解及输入依赖/鲁棒性 | 自动驾驶闭环安全、轨迹质量或碰撞率 |
| 物理预测 | Physion V1.5 object-contact prediction | 只看规定前缀预测未来接触，或标准线性readout能力 | 文本VLM已经成为视频生成world model |
| 具身离线 | EmbSpatial-Bench；适配公开具身空间QA |  egocentric空间关系与任务条件理解 | LIBERO/RoboTwin动作成功率 |
| 具身闭环（后置） | EmbodiedBench的单个环境先smoke，再官方完整任务 | 有行动接口、反馈和交互预算时的任务成功率 | 未接控制接口就“直接评测VLA” |

[GeoThinker](https://github.com/Li-Hao-yuan/GeoThinker)已列出CV、BLINK、ChartQA、AI2D、OCR、物理与具身相关资源，并通过LMMs-Eval评测；目录中出现名字不等于其完整task、媒体及当前模型适配已经可用。可以复用任务定义/评分器，不能把其模型backend直接换名字当作GeoRoute/GeoFits。

## 2. 遗忘如何计算

每项报告训练后−训练前的分数差（百分点）、有效回答率、长度截断率、逐题配对变化，另报告“原来对→现在错”和“原来错→现在对”的数量。RGB SFT对照用于区分普通空间数据SFT带来的遗忘与新模块的额外影响。

不要跨不同单位直接平均：OCR分数、QA准确率、物理OCP readout、闭环成功率分别展示。任何总体综合分必须预先定义归一化与权重，不能结果出来后调权重。对同一视频/场景的关联QA采用分组bootstrap；没有可用group则说明统计独立性限制。

CV-Bench仍然偏空间感知，因此增加OCR/图表/图示类任务才能更有力检查域外保留。若希望评估原生视觉encoder而非VLM的输出格式，可追加固定训练集linear probe；probe使用的训练数据/优化过程必须独立标注，不与zero-shot VQA混报。

## 3. 同输入与解码规则

- 每个task保存官方版本、split、样本ID、原图/视频顺序、标记、摄像机顺序、帧数、prompt、输出schema、metric代码版本。
- 模型间同样的可见证据、时间跨度、分辨率与总上下文；多摄像机空间顺序不能当时间顺序。历史帧、未来帧和标注不能混入可见输入。
- OCR/ChartQA/自由回答不能套用“只允许字母/数字”的空间模板。按各task使用相同提示和预声明输出预算，保留金标不可见解析。
- 同时记录官方算术与明确适配后的输入/输出差别。未闭合思维或无最终答案的截断必须记弃权，不能让评分器看标准答案后挑数字。
- 自定义几何模型必须由注册的完整loader加载，并在prefill运行其图/bank分支。原生vLLM或普通AutoModel未支持时不能静默丢分支。

## 4. 执行与许可边界

1. 通用离线QA优先复用固定版本LMMs-Eval的task/metric，使用独立Conda环境。与训练环境分离，不升级当前Transformers。先验收真实输入、batch1与批量一致、输出格式、每任务一条固定smoke。
2. [DriveLM](https://github.com/OpenDriveLab/DriveLM)依赖nuScenes/CARLA数据与自己的任务/评分格式；[DriveBench](https://github.com/worldbench/DriveBench)额外比较干扰和text-only。先核对原始媒体许可、公开验证标签及metric依赖。需外部LLM judge的分项先停用并标not_run，不自行调用付费API或把替代judge叫官方分数。
3. [Physion官方说明](https://physion-benchmark.github.io/)和[V1.5 evaluator](https://github.com/neuroailab/physion_evaluator)的原任务包含readout训练，不是直接拿通用VLM做文字问答。原始线性readout与我们可选zero-shot QA必须分列。仅给规定观察前缀，接触发生后的帧不能泄漏进输入；同一sample的指定对象身份必须可见且一致。
4. [EmbodiedBench](https://github.com/EmbodiedBench/EmbodiedBench)包含不同模拟环境，各自独立Conda、数据和headless渲染验收。先检查当前服务器图形驱动与最小实际render，再下载完整大型simulator资源。读取环境反馈/动作解析预算要对齐；不自动下载受协议约束的资产、不替用户接受条款。
5. 当前模型输出文本，不具备G0.5动作头。不能直接套旧LIBERO评测脚本输出“机器人成功率”。具身闭环要另做agent动作接口；视频生成world-model指标则需要额外生成器/预测头，不属于本轮现成模型能力。

## 5. 初步优先级与停止条件

P0：CV＋BLINK＋AI2D/OCR/ChartQA遗忘表，成本小、最直接回应全参数更新风险。

P1：DriveLM/DriveBench＋EmbSpatial离线迁移；许可证、来源与metric核对通过后完整评测。

P2：Physion标准readout及前缀预测对照；明确训练probe的成本与信息条件。

P3：EmbodiedBench闭环，仅在渲染/动作/环境成功终止判据全通过后进入正式任务。

这些扩展目前是待准备、后置执行项，**不是已安装或已启动的benchmark**。任何缺标签、许可证、原始媒体、模型backend或评分器的项目写`blocked`并给具体原因，不用相似数据集顶替，不把失败计0。ETA在各类真实样本完成预热测速后分别给出。

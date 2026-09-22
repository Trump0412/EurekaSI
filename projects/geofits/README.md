# GeoFits：全参数几何bank融合执行契约

用户项目名GeoFits，关联私有稿件正文称SpatialFit；本次统一工程目录为GeoFits。不把旧稿件数字作为已验证结果，不上传稿件。

2026-09-21修订：与GeoRoute共享六源协议（增加MindCube train10K，VLM3R使用GeoThinker32帧VSI205456＋VST132053，取消100K上限；VST官方errata排除2350条）；下文五源为旧版本记录，以`configs/geometry-followup-studies.json`和新数据验收receipt为准。OpenSpatial 100K获用户授权在缺少场景映射时全部加入train-only，明确记录场景重叠未知，不声称全数据无污染。不得将MindCube完整集中的训练题计入独立测试。

## 顺序与核心比较

等前序论文及GeoRoute全部必选训练/评测完成后执行。本方法从相同released Qwen3-VL-2B初始化，不从GeoRoute权重继续微调，因此阶段顺序不混入初始化优势。五源数据、split、输入、全参数更新范围及通用评测与GeoRoute一致。

主模型使用VGGT层11/17/23、Pi3层17/26/35组成每视觉区域6条候选；在decoder第1/2/3层做TopK2检索与独立sigmoid门控。使用SFT答案监督，无TIP或单独连续性预训练。

计划重新训练的对照：3D-only、4D-only、dense融合、无gate、单层融合；当前优先准备full，其余组不能仅因配置列出便视为已部署。用户已取消新增matched RGB SFT，历史RGB只能作非匹配参考。不同bank大小/宽度/层数会改变参数量与算力，逐组记录，不仅比准确率。新增RFT奖励消融的顺序和待部署边界见[奖励消融计划](../../docs/RFT_REWARD_ABLATION_PLAN.md)。

## 必须补足的条件

| 缺项/风险 | 处理 |
|---|---|
| 原文冻结nativeRGB，而用户要全参数 | 更新language＋nativeRGB＋新增模块全部权重；VGGT/Pi3作为外部教师仍冻结；无LoRA |
| Pi3版本/真实多层宽度未落实 | 固定**原Pi3**而非Pi3X；从实际decoder各层提取，最终两层concat不能冒充三层特征 |
| patch16/patch14网格不同 | 显式共享视场几何变换，再建立每区域一一对应；不能认为相同分辨率2×2pool就足够 |
| video两帧合并与逐帧bank矛盾 | 本轮统一每帧独立image，避免隐式时间2→1；明确这属于补足的输入协议 |
| 终端question hidden写回更早visual | question位置只由无答案prompt边界确定，禁止取teacher-forcing最后token；测试改答案不改prefix |
| fusion与原生DeepStack顺序 | 记录实际TF代码执行顺序；验证全teacher-forcing与prefill/KV增量decode一致，不能只做tensor形状测试 |
| time adapter瓶颈/时间公式未给出 | 显式配置bottleneck256；真实时间normalized sin/cos；只有列表序号则标order-only，不伪造FPS/物理秒数 |
| 门控初始化/检索宽度未给出 | 显式写入配置，gate hidden=LLM宽度/4；主topK2；不在test上选 |
| 全参数与教师缓存 | 不缓存nativeQwen状态；Pi3固定backbone输出可以缓存，但可训练时间adapter必须在no_grad之外 |
| 数据和评测不完整 | 与GeoRoute共享冻结五源manifest及benchmark版本，不复制旧297899数据冒充新混合 |

SFT一epoch、AdamW LR1e-5、global64、warmup3%、cosine、bf16、weight_decay0、seed3407。ZeRO-3及真实micro1/2/4门。这里“全参数”不指把离散/固定外部教师也改造成端到端训练；更新范围必须随参数清单保存。

## 分数解释

主VSI及MMSI/MindCube-Tiny/ViewSpatial/CV，额外ReVSI/SITE单列扩展。沿用各任务官方评分算术，并明确输入/输出格式适配。不能把VGGT/Pi3 bank称为有真实运动/物理时间保证的4D监督；是否实际利用时序需后续顺序/倒序/单帧对照支持。

报告TopK选择频率、门值、各层/题型分布及预测正确性关联；路由统计只说明模型选择偏好，不能单独证明物理解释。已选择的固定最终模型不根据测试路由图再调层位。

## 当前实现边界

2026-09-22 更新：六个变体现在都有独立执行架构；训练、重载、真实教师推理、六项评测入口及逐题融合统计已接通。单层消融只用 decoder 第3层；dense 的完整bank softmax是显式适配。CPU测试通过不等于完整模型GPU验收，部署与剩余门见[实现验收说明](../../docs/GEOFITS_IMPLEMENTATION_ACCEPTANCE.md)。下文保留此前边界背景。

`spatial_intelligence/geofits.py`已实现六entry bank、时间adapter、TopK2、三层门控、显式fusion context。CPU回归含真实tiny Qwen decoder、DeepStack、gradient checkpointing、prefill/KV一致性。

这些不等于完整模型可训：真实双教师多层提取/网格、完整HF注册保存重载、分布式全参数更新、所有benchmark适配仍需逐项验收。GeoFits阶段在这些证据不足时必须停在准备门，不能套用前一篇LoRA RFT worker冒充实现。

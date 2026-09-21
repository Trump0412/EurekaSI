# 两项后续研究的复现实验条件审计

结论：两份方法描述足以指导实现核心算子，**不足以直接唯一确定完整训练与评测流水线**。本轮已把缺项转成配置、测试与准备门，但不能把“补充设定”写成作者原始参数，也不能把旧稿件成绩当作本次复现产物。

本次采用实验设计skill的“语义契约→接口/数值验证→真实模型门→锁定全量”顺序。私有PDF保持原样；需要修改的训练/评测条款集中在以下说明，不直接改写或发布PDF。

## 覆盖清单

| 条件 | GeoRoute | GeoFits | 补足结果/下一门 |
|---|---|---|---|
| 精确checkpoint/revision | 缺revision | 缺revision，Pi3名称有歧义 | 固定Qwen/VGGT/原Pi3，绝不替换Pi3X |
| 模块层位与主公式 | 四visual出口、两STB明确 | 双教师六层、三decoder层、TopK2明确 | 核心算子和真实tiny Qwen路径已实现 |
| graph与tensor布局 | threshold/topK/dedup缺省 | grid一一对应未成立 | 显式像素映射与teacher/native网格适配；不复用不等价GeoWire |
| 更新范围 | 原描述包含LoRA | 冻结nativeRGB | 用户覆盖：两项SFT语言/nativeRGB/新增模块全参数；教师冻结另记 |
| 训练epoch/完整预算 | TIP与SFT遍历数缺失 | SFT一epoch明确 | GeoRoute预声明TIP一遍＋SFT一遍/replay，不能倒推为原文预算 |
| TIP精确定义 | 各项损失/系数/mask/负例缺失 | 不适用 | MSE/hinge/preservation显式定义与回归，不声称官方参数 |
| 五源选集 | VLM3R/OpenSpatial未给清单 | 同样未给清单 | 数据配置已补选集规则；真实分类/scene映射仍阻塞 |
| 去重与泄漏 | 仅排除eval文件不足 | 没给映射/排除记录 | scene/video源级审计、近重复边界、排除ID与实际分母 |
| baseline | 原基座/外部已发表成绩不足以隔离SFT收益 | 同样缺匹配新数据的RGB SFT | 增加相同五源、全参数RGB SFT重新训练对照 |
| 主benchmark版本/聚合 | 大致数量，不含精确revision | 同上 | VSI5130完整集/既有ReVSI，固定源版本与逐任务官方算术 |
| 互补benchmark | MindCube-Tiny与全文总体数量混杂 | MindCube具体split不明 | 主用官方Tiny精确QA，不能用全量21K规模替代Tiny分母 |
| 解码/提示 | temperature/token cap/解析器缺失 | 声称official但缺执行配置 | 固定greedy/nonthinking512与盲金标提取，标为适配协议 |
| correspondence诊断 | 未给scene/point清单和几何转换细节 | 不适用 | 需要独立GT depth/pose/intrinsics，不用VGGT预测自评 |
| 随机性/不确定性 | seed/repeats未给 | seed/repeats未给 | 首轮3407、逐题留存、scene分组区间；单seed不夸大统计结论 |
| 硬件/优化器恢复 | 与当前40GB卡不同 | 与当前40GB卡不同 | 独立ZeRO-3/长样本门、完整optimizer/RNG重载，实测ETA |

## 实现过程中已识别的实质风险

1. **旧方法并不等价。** 旧GeoWire的post-merger/final-only、自环和TIP目标不能当作GeoRoute四路pre-merger实现。新代码独立，不改旧结果。
2. **patch网格差异。** Qwen patch16，VGGT/Pi3 patch14。GeoRoute把像素track投到Qwen网格；GeoFits采用同视场448→392得到相同28×28网格，然后各自2×2pool。该补足必须出现在后续稿件中。
3. **Pi3特征宽度。** 选定的17/26/35层各为1024维；最终输出拼接产生2048维。两者不能互相替代。VGGT选定层是frame/global拼接2048维，特殊token另移除。
4. **question不是answer。** GeoFits条件向量必须取无答案prompt内的终端question位置，不能取teacher-forcing序列末尾；缓存推理与全序列forward需数值一致。
5. **时间描述要降级到有证据的层级。** 多图列表顺序不等于物理时间。无时间戳时只允许明确order-only编码，不能宣称秒级速度/运动建模已成立。
6. **全参数训练改变缓存可用性。** nativeQwen特征不能固定缓存；只复用冻结教师图/特征。TIP几何缓存与跨阶段模型checkpoint必须绑定输入/教师/预处理身份。
7. **自环recovery对照有局限。** 对已mask目的地只给self-loop，自身证据也被mask，劣化不自动证明学到了物理拓扑。保留同目的地/同权重/同源帧的错误端点对照，并报告mask与有效邻居覆盖。

## 真正的数据阻塞

当前官方OpenSpatial发布版与“原3M中的选集”不是同一份可直接复原的清单。官方生成器保留类型标签，但HF发布首批行出现UUID、未知type和数据集级来源，暂未找到对应的scene/type导出映射。不能凭问题关键词伪造五类配额，也不能把排除了eval文件写成scene无重叠。

两个真实Parquet探针已下载；各首3行的内嵌图像路径也为null，仍未恢复scene/type，详情见[数据审计](GEOMETRY_FOLLOWUP_DATA_AUDIT.md)。需要原始选集清单/映射，或明确另立一个可审计的公开数据协议；不能静默少用一源却仍称五源复现。VLM3R的camera-position勘误与所有benchmark转换也需逐项验收。

## 运行边界

下载正在执行；后续训练按前序依赖挂起。当前实现包括数学核心、Qwen hooks、teacher提取接口和GeoRoute阶段worker，**尚不具备已验收的完整无人值守五源训练—全部benchmark流程**。缺项有明确准备门，不会拿诊断权重或过期checkpoint替代。操作与状态见[后续队列指南](GEOMETRY_FOLLOWUPS_RUNBOOK.md)。

# 吞吐、六源数据与奖励迁移

2026-09-21。本文描述实现及验收要求，不表示全部 GPU 候选已经验收。

## 保持科学定义的提速

- 最新用户授权：三组新SFT统一全局batch64，micro1/2/4分别反向调整GA，不减少帧、样本或有效训练轮数。历史384诊断保留但不能作为64验收；用新运行目录从已完成alignment初始化，不是优化器续训。LR2e-5、1epoch、warmup比例0.03、cosine保持，调度总步数按新batch重算。RFT prompt batch16/G8不变。
- 同长度序列沿VGGT的B维合批，不把独立视频拼入时间维；分布式各rank必须采用同样的调用分块，避免ZeRO3参数collective错序。
- trainable VGGT不允许冻结缓存代替反向；frozen VGGT才可读缓存，且媒体顺序、预处理和权重版本一致。
- 显存门默认保留长样本余量；用户本轮明确接受约37.8GiB峰值，因此使用显式`memory_limit_fraction: 1.0`，不再仅因超过92%容量淘汰。实际OOM、数值、长样本及重载检查仍保留；此授权不保证后续绝不OOM。候选按热态有效样本/秒选择，不以reserved显存或瞬时GPU利用率最大化为目标。
- 新版保存间隔可配置为20步。续训需要完整model/optimizer/scheduler/RNG/Trainer状态、相同world和global batch、相同manifest与样本消费位置。只有权重不叫完整断点续训。
- 没有SFT断点的刚启动任务不伪造恢复记录；从alignment初始化重新启动须标记restart，而非resume。
- null-geometry评测的新代码避免先提VGGT再丢弃，显式force-null与正常输入布局不变。tiny模型逐logit回归不替代完整GPU模型验收。

每次迁移使用新root与代码快照，旧checkpoint、日志和失败证据保留。不能热改运行中的训练或评测。

## 数据修订

先前VLM3R100K上限不符合本次目标，现取消。GeoThinker对应32帧清单实际VSI205456、VST132053，总337509（132568是8帧变体数量，不能混用）；官方VST errata排除2350后335159条候选。加入MindCube train10000。与SPAR234K、Hound64K、VSI590K、OpenSpatial ARKitScenes来源100K合成六源，名义约133万。它不是最终训练行数；按媒体可读性、官方errata及已知源场景协议后出清单。

用户随后明确授权OpenSpatial缺场景映射也加入训练。该100K只进train，不构造伪scene验证集；receipt保留`leakage_checked=false`、OpenSpatial重叠unknown和有界授权。其余来源原有已知场景检查继续保留。此例外解除数据准备阻塞，不等于证明评测无污染；报告不能据此声称严格无污染泛化。

MindCube完整集包含训练内容，不可整体作测试。Tiny子集仍需固定ID/媒体与scene审计。VST问题引用帧索引时不得使用少帧变体丢掉所指帧；官方修订答案与删除项需要ledger。

## 恢复旧词表与prompt

迁移前简化词表及短prompt是新参考实现，不是旧GeoPSRO完整配方。新配置选`geopsro-lexicon-strict-v2`：恢复原六类词表、五类加权覆盖、正确答案门控、重复/堆词防刷逻辑；恢复原三段prompt，增加纯数值回答的接口说明。

有意不回退的安全修复：严格金标不可见答案解析、截断/冲突拒绝、SpatialLadder数值评分。format仍采用严格完整结构，不恢复旧分项格式分；短语补边界检查。风险词仅记录命中，不宣称能验证幻觉或自动扣分。

因此新版本是**旧任务资源＋明确安全适配**，不是旧奖励程序原封不动。词表更全面不代表指标一定更好；词奖励是弱文本约束，不验证中间推理是否真实符合图像。

`run-rft-reward-ablation.py`等待主RFT全部完成，仅用相同4DRL验证集题型宏平均选择，分数相同按较短生成再按名称。两个节点可各跑answer-only与answer+format；与full共享SFT初始化、G8、数据抽样和预算。不使用最终测试成绩选组合，不从full-RFT权重续训。源full必须已使用相同v2协议，否则拒绝比较。

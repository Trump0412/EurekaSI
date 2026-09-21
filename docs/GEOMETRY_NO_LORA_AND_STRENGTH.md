# 无 LoRA RFT 与 GeoRoute 强度诊断

本次用户决策：RFT 更新全部语言参数及融合模块，冻结原生 RGB 和 VGGT；不采用 LoRA。这不是整网全部解冻。新版本使用独立、不可更新的 SFT reference，不能使用 LoRA 的 disable-adapter 路径代替 reference。正式训练必须重新通过真实 G=8 采样、非恒定奖励、参数更新、显存和重载验收。

本地 reference 后端使用同步 CPU FP32 master/moment offload 降低优化器显存，仍非 ZeRO/VERL。每次更新增加 CPU/GPU 传输和 CPU 优化器开销；不能假设全参数后速度更快。真实吞吐和主机内存验收未完成，正式 ETA 必须重测。

## 训练后的 DSRBench / ReVSI

两条已部署的无 LoRA RFT 队列均执行 `gate -> train -> evaluate`。数据准备器已有 DSRBench 清单及 ReVSI 外部清单绑定；下载与数据验收结束后自动固化到各自输入快照，不新增重复 GPU 队列。核查时 DSRBench 1450 条、ReVSI 6158 条。清单存在不代表整套 RFT 数据就绪。

每组分别评估固定 SFT 起点和该组最终 RFT checkpoint，保存逐题 ID、生成 token、答案、截断情况、分任务和总体分数。混合与 4D-only 使用相同清单。DSR 当前按选择题答案精确正确率统计，不将格式/词汇训练奖励作为 benchmark 分数；ReVSI 调用已有任务评分器。两者采用同样的结构化推理提示、greedy、512 新 token 和锁定图像/视频输入，因此属于本地配对协议，不等同官方 leaderboard 提示协议。现有 VSI 和来源隔离验证也保留。

队列依赖评测完成才释放后续研究；失败或缺结果必须保持失败/未完成状态，不能记为 0 分。尚未完成真实 full-scope GPU gate、完整训练与评测，当前没有新 RFT 指标。

GeoRoute 保留 full、one_stb、final_only、post_merger、no_tip。取消 matched_rgb_sft 的新增训练，不删除旧结果。没有数据匹配的 RGB baseline 时，不能将历史 RGB 与新模型的分差完全归因于几何模块。

## TIP 是什么

在当前实现中，它是跨帧特征传输的专门训练：遮掉有跨帧支持的部分视觉特征，通过正确的几何对应恢复它；把对应替换为同源帧中的错误位置后，恢复应该变差；未遮挡特征应尽量保持。它不要求人工增加 CoT 标注，也不是 RFT。

初始化阶段只更新 STB；随后全参数 SFT 每 15 次指令 optimizer update 插入一次 TIP 更新。no_tip 同时去掉初始化和回放，因此指令样本量相同、总计算量不同。mask15%、MSE、hinge margin0.1、两项附加权重0.1是已声明的本地实现选择，不冒充论文完整官方配方。

## 强度与可视化

STB 使用 h'=h+g*alpha*F(h,graph)。alpha 是每块可学习门，正式配置保持 alpha_init=0、g=1；alpha=0只指初始化，不代表训练期间恒为零。首步主要更新门，门离开零后分支权重获得梯度；需通过真实轨迹确认。

诊断使用同一已训练 checkpoint、同一组预先锁定的验证样本，依次 g=0/0.5/1/2。g=0是关闭路由的推理干预，不等于训练过的 RGB baseline。此接口已经实现，但尚未产生真实模型诊断结果，不自动改变正式推理参数。

可调用 `route.routing_diagnostics(strength=2.0, telemetry=records.append)`；记录每个出口/块的 alpha、实际输出增量与输入范数之比、有效连接数和节点统计。回调仅接收脱离计算图的标量，开启统计会增加同步开销，不默认用于热态测速。

正式图使用一致色标、相同帧与位置，展示原特征、增量和更新后特征。弱作用本身也是结果；不得为了视觉差异放大正式输出或逐图自动缩放后隐藏色标。是否增强训练由验证准确率、TIP正确/错误对应差距、实际增量比及稳定性共同决定，不由测试集分数或图像鲜艳程度决定。五个结构组保持相同强度策略。

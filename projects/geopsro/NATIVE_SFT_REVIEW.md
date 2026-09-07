# 原生 Qwen3-VL SFT 候选审查

来源：本地交接包的 `upstream-candidates/GeoPSRO/scripts/train_qwen3vl_native_sft.py`。候选没有合入可运行核心，也没有用于更新公共 source commit。

值得后续迁移的能力是 padding 后的真实多样本 VLM batch、非重入 gradient checkpointing、学习率调度、ZeRO 与吞吐/显存记录。这些是公共训练器的缺口，但需要接入公共 schema、样本协议、heldout 审计和结果溯源后再实现。

静态发现的直接接入障碍：

- 读取 `frame_paths` 并按 truthiness 过滤 question/answer；数值零答案会被删除，非法样本也被静默跳过。公共代码保留零值并拒绝无效数据。
- `[:max_frames]` 静默截断原始帧，会破坏带 marker 或指定视图的 QA。公共 manifest 的帧预算检查不能移除。
- 使用字符串答案的最后一次 token 子序列匹配确定 labels 起点。这不能证明找到的是 assistant 边界；应核对 chat template 前缀与完整多模态序列，并测试重复答案/token 边界。
- 没有 train/heldout 场景审计，默认允许 remote code，没有权重/数据/输入协议哈希；输出目录也允许继续写入。
- 当前固定 ZeRO-2；保存 DeepSpeed 状态不等于已实现恢复训练，没有 load_checkpoint、采样顺序和 RNG 恢复路径。候选中的 ZeRO-3 consolidate 分支只在 rank0 调用，未来支持 stage3 时还要检查其 collective 参与要求。
- 日志只保留每个更新末尾 microbatch 的 loss 和样本，不能代表该次全局 batch 的完整预算。

因此保留现有参考训练器。本次不复制 launcher 的服务器环境/GPU 默认值，不将此候选声明为 DeepSpeed 支持。未来迁移先建立 HF labels 对齐、帧预算、全局 batch、保存恢复及多卡数值回归，再开放配置入口。

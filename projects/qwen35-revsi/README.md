# Qwen3.5-2B 空间 SFT 的 ReVSI 受控评测

问题：同一模型和 ReVSI 协议下，GeoThinker 选择的 SPAR/Hound 监督能否改善空间 QA？对照为原始权重与 SFT 权重。只支持离线 QA，不支持机器人控制或几何架构优势；使用 vanilla RGB Qwen，不称为 GeoThinker 架构复现。

## 固定协议

- 模型：Qwen/Qwen3.5-2B 公共 post-trained 权重，revision 在 `configs/qwen35-assets.json`。
- 训练候选：VG-LLM 发布 SPAR 234,277 / Hound 63,750；Hound 用 GeoThinker 同 ID 清单的准确帧与顺序，不重新抽3%/25%。
- SPAR 用固定 renderer，保留规定1/2/3/32帧，彩点/框问题不能配未渲染图。
- 冻结 ReVSI 32-frame test，排除 scene-ID 训练重叠并报告最终数量；近重复和基础模型预训练暴露仍未排除。测试不用来挑 checkpoint/LR。
- 训练：1 epoch、语言侧全参数/visual 冻结、AdamW LR 1e-5、warmup3%、cosine、global batch64、seed3407、bf16。按保留样本遍历，DDP 尾部补齐另记，不是 reference sampler 的有放回抽样。
- 评测：同规定32帧，有序 RGB 多图、最大边448、关闭 thinking、greedy、16新token、官方直接答案提示。**不是声称与官方 video-token 输入实现等价。**
- 分数：上游答案解析/数值阈值，细类型均值→合并组→宏平均，附分母/缺失/逐类/逐题记录。
- VLM3R、VSI590K 本轮仅下载，不产生新的训练结论。

## 阶段门

1. 新环境 pip check、CPU 回归、官方 scorer parity。
2. 真实视觉 batch/帧顺序、答案/padding mask、选择 logits 的 loss/梯度等价。
3. 16题 smoke 完整输出，再做全量 baseline。
4. 全量媒体/marker/scene gate，四卡两步有限 loss/梯度与保存，再长训。
5. 一轮导出→相同协议复测→逐题比较。

操作见 [SERVER_RUNBOOK](../../docs/SERVER_RUNBOOK.md)。当前结果待测，不能用预期提升代替结果。

来源：[GeoThinker](https://github.com/Li-Hao-yuan/GeoThinker)、[ReVSI](https://github.com/3dlg-hcvc/revsi)、[Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B)。

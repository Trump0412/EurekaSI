# geopsro

交接包中的原生 DeepSpeed SFT 候选尚未接入；接口缺口和迁移要求见 [原生 SFT 审查](NATIVE_SFT_REVIEW.md)。

动态空间状态、策略与奖励设计；验证动态推理与后训练。

公共部署、数据、SFT/RL/OPD、评分和结果溯源全部使用根目录的 `spatial` 命令，不复制训练器。

原始论文方法：`spatial source geopsro` 拉取锁定版本并应用已有补丁；使用 `configs/legacy-geopsro-sft.yaml` 指定路径后启动原入口。三个原方法尚未全部移植到原生训练 backend，不能把公共 FusionBackend 结果标为原论文复现。

实验配置及方法扩展保留在本目录，进入共用训练器的模型适配放在 `spatial_intelligence/backends/`。方法差异和未修问题见 `docs/AUDIT.md`。

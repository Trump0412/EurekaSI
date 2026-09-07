# geowire

交接包中的可视化与评测建议已整理为 [证据契约](EVIDENCE_PROTOCOL.md)；候选脚本和未经复核的实验分数未合入。

跨帧几何图与视觉 token 传输；验证对应与跨帧持续性。

公共部署、数据、SFT/RL/OPD、评分和结果溯源全部使用根目录的 `spatial` 命令，不复制训练器。

原始论文方法：`spatial source geowire` 拉取锁定版本并应用已有补丁；使用 `configs/legacy-geowire-sft.yaml` 指定路径后启动原入口。三个原方法尚未全部移植到原生训练 backend，不能把公共 FusionBackend 结果标为原论文复现。

实验配置及方法扩展保留在本目录，进入共用训练器的模型适配放在 `spatial_intelligence/backends/`。方法差异和未修问题见 `docs/AUDIT.md`。

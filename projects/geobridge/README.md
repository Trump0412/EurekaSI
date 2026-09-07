# geobridge

分层几何 bank 与按需桥接；验证参照系、几何证据利用。

公共部署、数据、SFT/RL/OPD、评分和结果溯源全部使用根目录的 `spatial` 命令，不复制训练器。

原始论文方法：`spatial source geobridge` 拉取锁定版本并应用已有补丁；使用 `configs/legacy-geobridge-sft.yaml` 指定路径后启动原入口。三个原方法尚未全部移植到原生训练 backend，不能把公共 FusionBackend 结果标为原论文复现。

实验配置及方法扩展保留在本目录，进入共用训练器的模型适配放在 `spatial_intelligence/backends/`。方法差异和未修问题见 `docs/AUDIT.md`。

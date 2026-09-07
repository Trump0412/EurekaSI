# Changelog

## 0.2.1 — 未发布 / Unreleased

- 项目命名为 EurekaSI，确定公共仓库地址；发行包名为 eurekasi，新增同名 CLI，保留 spatial 命令与 Python 模块兼容性。

- 合并两次静态审查：保留数据清洗、配置覆盖、缓存身份、蒸馏溯源和 ReVSI 协议修复。
- 增加 wheel 的 catalog/config/patch 资源声明与定位，保留源码安装和自定义本地配置行为。
- 统一包版本，Transformers 固定到 4.57.6，保留 torch/torchvision/NumPy 约束及 Python 3.10–3.12 范围。
- 加固环境隔离、未完成安装标记和测试解释器选择；修复分布式测试的 train/heldout 重叠。
- 发布检查与归档共用文件选择，静态解析版本与来源锁，支持 SOURCE_DATE_EPOCH。
- 增加回归用例、维护/安全说明和协作模板。本版本尚未进行运行时验收、构建/安装或正式 release。

## 0.2.0 — 原交付 / Original delivery

公共训练、推理、数据、几何与迁移基础代码。原包及其验证日志保留为历史证据。

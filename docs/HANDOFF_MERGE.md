# 交接增量的适配决策

日期：2026-09-07。输入为本地 `trans/HANDOFF_2026-09-07.md` 与 `Spatial_Intelligence_Handoff_v0.2.1_2026-09-07.zip`。本次以当前工作树为主，按共同 v0.2 基线逐模块比较，没有应用整份补丁或覆盖现有目录。

交接包的 61 个 MANIFEST 项 SHA256 校验通过；只验证归档完整性，不证明其中代码正确。合并前的公共工作树备份在本地 `_archives/pre-handoff-merge-2026-09-07.zip`。`trans/` 已加入忽略规则，原输入、Git bundle 和候选脚本保留原样，不进入发布包。

## 已适配合入

| 增量 | 当前仓库的真实需要 | 适配方式 |
|---|---|---|
| 版本同步与发布元数据 | 原 pyproject 为 0.2.0，包常量仍为 0.1.0 | 统一到未发布的 0.2.1，增加 CHANGELOG/CITATION；不填写虚构作者、DOI、远端地址或发布日期 |
| Transformers 4.57.6 / dtype | 交接选择同一 4.57 系列修订，原生 HF 加载参数同步 | 统一 train/test/native 到 4.57.6，采用 dtype；保留本库精确 torch/torchvision/NumPy 约束和 Python 3.10–3.12 范围，未复制宽泛依赖区间 |
| wheel 运行资源 | 原包只能在源码目录找到 catalog/configs/patches | 添加 data-files 声明与资源定位；处理源码、venv/user/target 布局；补上交接遗漏的 SPATIAL_CONFIG 初始化、UTF-8 及 configs 相对路径回退 |
| Python 3.10 TOML | 声称支持 3.10，但静态 TOML 解析只接受 3.11+ | 添加条件 tomli 依赖；继续不执行应用源码来读取元数据 |
| 环境隔离与失败标记 | ROS/集群路径注入、损坏 venv 和错误解释器会让检查结果失真 | 原生/几何 bootstrap 共用目录验证与失败标记；清理宿主注入，pip check 后移除标记；check 明确选择解释器，发现失败环境就停止 |
| 发布检查 | 原检查主要是语法/链接，缺少元数据、wheel 资源和锁文件的一致性核对 | 增加 release_check；共用实际发布文件选择器，AST 读取版本，校验资源覆盖、依赖、source lock、VSI 哈希、归档内链接及凭据模式 |
| 确定性归档参数 | 归档时间应能复现并与发布流程衔接 | 在现有白名单/原子写入上增加 SOURCE_DATE_EPOCH；保留拒绝覆盖、固定脚本权限及现有目录/文件排除规则 |
| 分布式测试夹具 | 训练题与 heldout 文本完全重叠，真实优化前即触发正确的泄漏保护 | 修改训练题；固定子进程源码路径和工作目录；不强制 Linux `lo` 网卡名；保留原数值容差 |
| 协作与 CI | 缺少变更说明、漏洞报告说明、独立安装包资源校验 | 增加 bug/PR 模板、安全和维护文档；CI 分开静态、wheel/源码归档、CPU/多进程任务；未在本轮触发执行 |
| 无用 import | 有独立、可确认的清理价值 | 只移除当前代码确实不使用的 import；例如本库 assets.py 仍使用 os.walk，不能照搬交接中的删除 |

依赖依据：[Transformers 4.57.6 官方发布页](https://pypi.org/project/transformers/4.57.6/)、[固定版本的 HF 模型加载实现](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/modeling_utils.py)。运行资源使用 setuptools 的 data-files 声明，安装布局限制见 [维护说明](MAINTENANCE.md)；这仍需 CI 的实际 wheel 安装来验收。

## 保留现有实现或不直接合入

| 候选 | 决定与依据 |
|---|---|
| 整份 overlay/patch | 拒绝整体覆盖；否则会退回本库已修的 SPAR 去重、配置覆盖、几何缓存和 KD 溯源逻辑 |
| 交接 release_check | 吸收检查目标，重写执行边界。原实现用 exec 读取包版本、独立遍历全目录，与发布选择器不一致，且无法正确隔离本库 trans/_archives |
| 交接 package_release | 保留本库实现并扩展。原候选按文件系统执行位生成权限，会让 Windows 与 Linux 产生不同结果；它还按任意层级目录名排除 models，不适合未来的源码包布局 |
| 分布式 allclose 容差放宽 | 未合入；夹具先修正，没有实测误差分布就不能把测试放宽视为验证通过 |
| ruff 全量 F 规则与新的强制依赖 | 暂不引入。未在当前树运行过，与构建/运行无关的全量清理不应阻塞核心适配；AST、配置/来源检查及现有 pytest 持续保留 |
| 修改 LICENSE、替换历史 freeze/测试日志 | 未合入。保留原限定许可范围与历史证据；不将另一环境的结果改成合并树通过记录 |
| 关闭全部空白 Issue | 未合入，保留现有反馈入口；增加结构化 bug 模板即可 |
| 本地新 HEAD 与 Git bundles | 只保存/校验交接包，不迁移为公开 source lock，不逐提交重演历史。公共固定 SHA 和已有补丁保持不变 |
| 三份 worktree diff | 已查看变更：主要是服务器路径/conda 环境迁移和 watcher。GeoPSRO 配置还把 train/val 指向同一文件，不符合公共 heldout 契约；全部不合入 |
| GeoBridge 自动占用全部 GPU 的 watcher | 不合入；公共 CLI 已有明确 --gpus/--devices 与 torchrun 路径，自动枚举不等于用户声明的资源预算 |
| GeoPSRO 原生 DeepSpeed SFT/launcher | 暂不合入可运行核心。存在零值过滤、静默截帧、labels 边界、数据泄漏审计、恢复/日志等接口缺口；详见 [原生 SFT 审查](../projects/geopsro/NATIVE_SFT_REVIEW.md) |
| GeoWire 两份可视化脚本 | 暂不合入 runnable tools。依赖旧内部接口，含服务器默认路径；默认放大 gate 128 倍，不能作为训练模型原样推理证据 |
| GeoWire 两份论文/评测报告 | 只提炼可复用协议，不迁入私有分数或运行路径。报告中的 self-loop 推理不等于完整几何方法；原日志/权重不在本轮可复核范围。详见 [证据契约](../projects/geowire/EVIDENCE_PROTOCOL.md) |

## 本次验证范围

仅执行本地文件对照、归档 SHA256 校验、Python AST/结构化配置/链接解析和源码发布静态检查。没有执行 wheel 构建/安装、bootstrap、下载、pytest、训练、推理、仿真或多进程测试，也没有提交/推送。

交接说明中的 56/61 passed 属于另一工作树的历史阶段，并且不能覆盖该 agent 后来的安装包改动。本次不重复引用为当前结果。机器可读来源与检查记录放在 `docs/validation/handoff-provenance.json` 和 `docs/validation/handoff-static-review.json`。

本次补足了发布、安装资源与维护环节；训练器的大规模 batching/ZeRO/精确恢复和真实 VLM/GPU 验收仍是明确的后续工作，见 [READINESS_REVIEW](READINESS_REVIEW.md)。

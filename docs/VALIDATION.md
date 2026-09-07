# 验证记录：当前静态审查与 v0.2 历史证据

## EurekaSI 首次公开提交准备（2026-09-07）

统一项目名称、发行包名和仓库链接，新增兼容 CLI 别名与研究工作流，并增加项目身份一致性检查。当前仅运行 `python scripts/static_check.py` 与 `python scripts/release_check.py`，均通过；没有安装、构建、运行 pytest、训练、推理或仿真。发布准备的记录见 `docs/validation/eurekasi-publication-static.json`。GitHub push/PR 只触发静态任务；打包和 CPU 运行任务须手动触发。

历史 `core-pytest.txt` 的两处机器绝对路径在公开副本中改为 `<historical-workspace>`；测试结果与警告保留。原始字节仍在本地原交付归档和合并前备份中。下述旧报告与哈希描述其产生时的快照，不代表本次树的哈希或运行结果。

## 当前工作树：交接合并（2026-09-07）

本轮选择性适配交接包，版本统一为 0.2.1，Transformers 固定到 4.57.6，并补充 wheel 资源、环境隔离、发布静态检查与回归代码。仅执行文件/归档/源码静态检查，未构建安装 wheel、安装环境、运行 pytest 或任何实验。新增 CI 尚未执行。详见 [HANDOFF_MERGE](HANDOFF_MERGE.md)；当前检查记录为 `docs/validation/handoff-static-review.json`。

## 上轮静态整理（2026-09-07；下述 4.57.1 选择已被本轮替代）

本次按维护者要求只查看、修改代码和文档。没有配置环境、下载数据/权重、执行训练/推理/仿真，也没有运行 pytest（其中包含真实优化步骤）。新增的 `tests/test_maintenance.py` 等回归用例是待执行代码，不是通过证据。

执行范围为 Python AST、JSON/JSONL、YAML/TOML 解析、文档本地链接和原始 ZIP 的 SHA256 清单校验。结果和完整流程限制见 [READINESS_REVIEW](READINESS_REVIEW.md)。这些检查不证明依赖可安装、GPU 显存足够、模型前向正确或 benchmark 已复现。

原生依赖从已撤回的 Transformers 4.57.0 调整为 4.57.1；历史 CPU 环境记录不能验证此变更。`requirements/resolved-cpu-py312.txt` 只描述原始交付环境，不能作为当前环境锁文件。

## 以下为原始交付附带的历史记录（未在当前机器重跑）

日期：2026-09-07。当前环境没有 CUDA GPU、用户真实训练数据或完整模型权重。本次没有产生新的 benchmark 性能、论文复现分数或机器人成功率。

### 历史交付记载的执行

| 验证 | 结果 | 证据和边界 |
|---|---|---|
| 全新 venv bootstrap cpu | 完成 | Python3.12、torch2.5.1+cpu、torchvision0.20.1+cpu、Transformers4.57.0、PEFT0.17.1；安装顺序已修正 |
| pip check | No broken requirements found | 实际解析版本见 requirements/resolved-cpu-py312.txt |
| 单进程/模块/回归测试 | 55 passed | 包含下面的小模型与模块验证；docs/validation/core-pytest.txt、core-junit.xml |
| SFT/RL/OPD/OPSD/offline KD | 完成真实优化、保存、重载生成 | 随机小 GPT2 模型，每模式两步；不代表真实 Qwen 多模态训练 |
| LoRA、response-only loss | 通过 | LoRA 保存/重载；loss 与 HF 原生 labels 对齐 |
| 通用融合 | 通过 | 零门控等价、几何梯度、真实小模型训练/重载、异常后的 hook 清理；不代表 VGGT 实模效果 |
| 数据构建与清洗 | 通过 | marker 必需性、原图/渲染图分离、选项解析、冲突标注、分场景划分、archive 越界拒绝 |
| 几何数学/缓存合约 | 通过 | DA3 反投影、点 token、finite 校验；没有运行真实编码器权重 |
| LIBERO 接口 | 合约测试通过 | 假环境 rollout、动作边界、成功记录；未运行 MuJoCo/LIBERO |
| 资源元数据 | 36/36 仓库确认存在 | 官方 HF API 获取 SHA，写入 catalog；不代表各模型已加载 |
| 实际下载 | 通过 | 只下载 Qwen3-VL-2B config.json，经历超时后成功重试；没有下载完整权重 |
| 原代码获取和自动补丁 | 通过 | 实际 fetch GeoPSRO 固定 SHA、自动 apply 3 文件补丁；未 push 上游 |
| CLI 路径与资源计划 | 通过 | init + download + source + run inference --gpus2 --dry-run 生成可核对配置 |
| 两进程测试 | **环境阻塞，未通过验收** | Gloo 初始化被 Operation not permitted 拒绝，loopback 亦受限；三项测试保留在 CI 必跑集合 |

分布式测试涵盖预期的 unused-gradient 同步、两进程 SFT 对相同 global batch 单进程的数值一致性、分片推理覆盖率；当前没有执行到这些断言，不能写成通过。`pytest -m "not distributed"` 的 55 passed / 3 deselected 明确排除了它们。

轻量发布包解压后再次运行：50 passed、2 个历史源码模块 skipped、3 个分布式用例 deselected。历史回归需先 `spatial source geowire` 和 `spatial source geopsro`，CI 已配置自动获取。两份 ZIP 的全部 MANIFEST.sha256 校验通过。

## v0.1 历史验证

GeoWire 原 tests 29 通过；GeoPSRO 原 tests 10 通过；GeoBridge HGB 与 Stage1 评分选定 tests 11 通过。原环境为 CPU torch2.14.0，GeoBridge FCP 因缺 torchvision 未收集；没有将环境缺依赖认定为方法失败。历史报告保留在 docs/ 中，新环境的记录放在 docs/validation/，不要混用两个环境的结果。

原问题已复现并保留修复回归：GeoWire generate 递归；GeoPSRO 最终答案误读、零值丢失、缺失金标计为正确。补丁在完整复现档案中已应用，轻量开源包通过 source 命令获取并应用。

## 下一道真实验收门槛

1. 在允许进程通信的服务器或 GitHub CI 运行完整 tests；在 GPU 运行 NCCL 单卡/双卡等效检查。
2. 每个真实 Qwen2/2.5/3-VL、InternVL-HF/LLaVA-HF 权重至少做图像输入、梯度、保存/重载验证。
3. VGGT/DA3/Pi3 真实提取 + FusionBackend 端到端，确认帧、尺度和坐标约定。
4. 官方六 benchmark 的数据格式、媒体完整性、任务 prompt/scorer、全覆盖率与实际结果。
5. GeoBridge/GeoPSRO 原方法完整原生适配、模型与奖励和论文的逐项对照；没有用公共融合替换名义复现。
6. 实际 action policy 的 LIBERO 执行；驾驶、世界模型仍为研究计划。

这是已搭建、可继续实验和开源协作的研究代码库，不是上述所有模型/benchmark/硬件已经验收的生产系统。

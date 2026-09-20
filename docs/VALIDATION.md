# 验证记录：当前运行验收与 v0.2 历史证据

## 后训练与三个原项目专项审查（2026-09-19）

重新执行公共后训练/目标/适配测试 **26 passed**（含新增 OPD/OPSD 非零学习信号检查）；GeoWire 原生 **29 passed**、GeoPSRO 原生 **10 passed**。GeoBridge 全套收集阶段 **3 errors**，另选核心六个测试文件 **22 passed**，不可合并写成全套通过。当前未部署 VERL，Qwen3.5 公共 OPD/OPSD 后端也尚未接通。环境、源码依据、日志位置与复现门槛见 [专项审查](POSTTRAINING_READINESS_2026-09-19.md)。本次没有更改生产训练逻辑或现有 Conda 依赖。

## Qwen3.5 空间 SFT / ReVSI 部署（2026-09-19）

新建独立 Python 3.12 Conda 环境，PyTorch 2.7.1+cu126、Transformers 5.3.0、FLA 0.5.2，4×A100 40GB；没有覆盖旧 G0.5 环境。`pip check` 通过。原 native profile 仍为独立旧依赖，不能与本 profile 混装。

FLA 的 chunk gated-delta 核已可用；`causal-conv1d` 未安装，卷积分支仍有 PyTorch fallback，不能宣称整条 fast path 全开。正式前后对照保持同一运行环境，后续内核优化需单独测等价与吞吐。

已执行证据：

- 服务器完整 CPU 回归：**86 passed、2 skipped**（缺少两个可选历史源码）；包含两进程训练/推理测试和固定上游 ReVSI scorer 等价检查。日志位于运行根目录 `logs/cpu-tests.log`，旧失败记录保留，末尾为通过结果。
- 修复 JSON 配置误走 YAML 解析导致 `1e-05` 成为字符串的真实错误，并补测试。分布式 Adam 数值一致性测试改用 FP64 小模型，保留原严格容差，避免零梯度附近 FP32 误差放大；不代表修改了生产训练精度。
- 原始 Qwen3.5-2B 与 ReVSI 已完整下载；实际准备 **6,158题、380场景、每场景32帧**。
- 实际模型真实图像 forward/backward gate：最近重验答案位置投影与全 logits loss 均为 **0.0040816772**，梯度有限，峰值已分配显存 **9.55 GiB**。这是构造短回答的接口诊断，不是 SFT loss 或 benchmark 分数；optimizer 更新次数为0。证据：`receipts/model-gate.json`、`logs/model-gate.log`。
- 四卡16题推理 smoke 完成并保存预测；这16题全部是计数题，不能视为完整 benchmark。随后全量6,158题评测已完成，锁定协议下宏平均 **0.0503971944（5.04%）**。抽查发现有回答在16个生成 token处仍为解释前缀、未给最终答案；这是需要单独量化的格式/截断混杂因素，不能将该分数直接解释为空间能力，也不能用该分数挑训练 checkpoint。原预测保留，后续预算诊断须新建 run，并同步前后对照协议。
- 后续补充字节分卷流读取、下载进度去重测试，专用测试文件服务器 **7 passed**；此处不把它冒充为再次执行了完整测试集。
- 已下载 Hound 小分片并准备32条真实样本，记录 sample IDs；四卡 DDP 两步、从 checkpoint-2 恢复到第三步、权重重载推理均已完成。证据为 `runs/diagnostic-ddp/completion.json` 与对应 state/logs；不属于正式 SFT 结果。

仍待验收：ReVSI 输出预算/格式诊断、训练媒体完整性/场景排除数量、1/2/4 micro-batch 实测、一轮正式 SFT 及配对复测。自动排队不等于这些阶段已通过。以 [SERVER_RUNBOOK](SERVER_RUNBOOK.md) 和运行目录中的 receipts/state 为交接入口；不把静态检查或历史小模型测试当成真实全量训练完成。

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

## 2026-09-19 服务器增量验收

详细配置、产物路径、限制见 [后训练运行指南](POSTTRAINING_RUNBOOK.md)。

- Qwen3.5 OPD/OPSD：真实八图采样、同词表教师答案位置对齐、非零 KL/梯度/参数更新、保存重载通过；只是一条样本的接口诊断，教师并未证明优于学生。
- GeoBridge 恢复被上游删除但仍被引用的 helper 后，原生测试 40 passed；GeoWire 29 passed，GeoPSRO 10 passed。GeoBridge 后续增加本地 VGGT `model.pt` 加载补丁，实际缓存和 initializer/FCP 两步训练已通过；不将此前 40 项测试自动算作补丁后的重测。
- 三个方法均已生成独立真实几何缓存与小规模阶段权重；完整 Stage2、正式样本预算和 benchmark 仍未完成。
- 新增后训练、扩展队列与按需标记缓存回归：`pytest -q tests/test_qwen35_posttraining.py tests/test_extension_queue.py tests/test_marker_cache.py`，8 passed。另对 34 种真实 SPAR 题型进行 eager/lazy 标记逐像素比较，全部相同。
- 四卡正式 SFT 尚在数据准备阶段；分层单卡实测给出一轮约 7–11 小时的启动前估算，不是已完成成绩。VERL v6 正在重试，尚未验收完整 rollout/reward/update/checkpoint 链路。

## 2026-09-20 恢复正式 SFT

- 准备阶段凌晨因把 SPAR 原始 ID 当唯一键而退出，正式训练当时没有启动。现采用过滤前源文件行号构造唯一键，保留原始 ID，不删除同 ID 的不同记录。
- `tests/test_qwen35_study.py` 在服务器运行 8 passed（包含新 ID 回归与官方 scorer 对照）。全部 297899 条 manifest 的唯一键、源 ID 对应与源顺序另行审计通过；SPAR 234149、Hound 63750，仍仅排除 128 条 ReVSI 场景重叠记录。
- 清单构造改为保序线程池并缓存重复路径检查，保留缺图失败规则。复用全部原渲染缓存，不更改图像或答案。
- 长样本 batch 1/2/4 的单卡实测吞吐分别约 1.373/1.218/1.272 samples/s，选择每卡 1、GA16、四卡 global batch64。四卡两步 smoke loss 1.64743 / 2.35719，梯度有限，保存并退出成功。正式 SFT 已由监督器启动，最终成绩仍待测。
- VERL 原图路径修复已通过原生 RLHFDataset → qwen-vl-utils 的真实八图 CPU 解码/顺序检查；尚未重跑 GSPO，不将输入检查算作训练闭环成功。
- 基线 ReVSI 宏平均 0.05039719436，6158 条完整输出；抽查有解释性输出在 16-token 上限截断。该分数仅描述锁定协议，不等同模型完整能力。若修改输出预算，必须独立命名并成对重跑 baseline/SFT，不能混用旧分数。

## 下一道真实验收门槛

### 2026-09-20 batch 与评测修复增量

- 四卡真实混合样本统一 global batch64：micro1/2/4 分别 10.927/10.080/6.844 samples/s；micro8 OOM，micro16 未测。选择 micro1 + GA16，并从完整 checkpoint200 恢复，已再次产出有限 loss/梯度。证据与边界见 [batch/评测协议](BATCH_AND_EVAL_PROTOCOL.md)。
- 服务器答案提取、VSI 官方算术/聚合与训练回归合计 36 passed、1 skipped；另视频位置编码兼容回归 1 passed。Windows 无 torch：相关梯度测试不能在本机验收，不把本机缺依赖算作服务器通过。
- 原生视频修复后 ReVSI 13条 smoke：解析100%、截断0%、strict30.00、extracted30.71；非完整基准成绩。VSI-Bench 5130条/288视频已准备，debiased v1为2362条，按场景ID检查与当前SFT交叉0条。
- 完整新协议 ReVSI/VSI 的 baseline/SFT 结果待测；旧5.04保留作历史协议诊断，不作可信能力基线。

1. 在允许进程通信的服务器或 GitHub CI 运行完整 tests；在 GPU 运行 NCCL 单卡/双卡等效检查。
2. 每个真实 Qwen2/2.5/3-VL、InternVL-HF/LLaVA-HF 权重至少做图像输入、梯度、保存/重载验证。
3. VGGT/DA3/Pi3 真实提取 + FusionBackend 端到端，确认帧、尺度和坐标约定。
4. 官方六 benchmark 的数据格式、媒体完整性、任务 prompt/scorer、全覆盖率与实际结果。
5. GeoBridge/GeoPSRO 原方法完整原生适配、模型与奖励和论文的逐项对照；没有用公共融合替换名义复现。
6. 实际 action policy 的 LIBERO 执行；驾驶、世界模型仍为研究计划。

这是已搭建、可继续实验和开源协作的研究代码库，不是上述所有模型/benchmark/硬件已经验收的生产系统。

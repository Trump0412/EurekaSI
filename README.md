# EurekaSI：空间智能公共基础设施

**EurekaSI · v0.2.1（开发版本，尚无正式 release）**

主仓库：[Trump0412/EurekaSI](https://github.com/Trump0412/EurekaSI)。面向多篇论文的受控实验与下游应用，共享实现、配置契约和可追溯的验证记录。后续项目组织见 [研究与应用工作流](docs/RESEARCH_WORKFLOW.md)。

发行包名为 `eurekasi`，安装后可用 `eurekasi` 或兼容命令 `spatial`；Python 模块仍为 `spatial_intelligence`，现有配置和缓存路径保持兼容。

面向 GeoWire、GeoBridge、GeoPSRO 和新的空间智能研究：共用环境配置、数据清单、几何缓存、SFT / RL / OPD / OPSD、推理评测与实验溯源。三个项目的差异留在 `projects/`，公共实现只有一套。

当前目录是公开维护的研究代码库。早期维护仅做静态审查；后续 Qwen3.5 服务器训练、原生视频 smoke 与针对性回归的实际证据见 [VALIDATION](docs/VALIDATION.md)，完整配对 benchmark 仍待完成。交接取舍见 [合并审查](docs/HANDOFF_MERGE.md)，其他流程的未验收能力见 [就绪度审查](docs/READINESS_REVIEW.md)。历史 CPU/小模型日志不代替新 GPU 验收。

## 四步开始

可变长多模态训练的有效 batch 内分桶、rank 均衡、FLA 数值门和自动吞吐对照见 [吞吐优化](docs/THROUGHPUT_OPTIMIZATION.md)。当前为可选功能；四卡加速倍数待实测。

Qwen3.5 当前四卡测速、断点恢复与 ReVSI/VSI-Bench 最终答案提取采用 [batch 与评测协议 v2](docs/BATCH_AND_EVAL_PROTOCOL.md)。原始输出、解析结果和逐题指标都保留；不要将历史 image-only/16-token 分数当作已验证基线。

Qwen3.5-2B 新服务器部署使用独立 Conda 配方，见 [服务器操作手册](docs/SERVER_RUNBOOK.md)：`bash scripts/start-qwen35-study.sh /persistent/your-root`。包含下载、ReVSI baseline、234K+64K SFT 与配对复测阶段门；新 GPU 验收不能用旧随机模型日志代替。

未来执行时，在项目根目录使用 Linux / WSL2 + Python 3.12。Windows 原生终端可维护源码，但 Bash、CUDA/NCCL 和模拟器路径尚未提供原生 Windows 验收。下列步骤会安装环境并运行测试：

```bash
bash scripts/bootstrap.sh native              # CUDA 12.4 wheels；无 GPU 可改 cpu
source .venv/bin/activate
spatial init --root /data/spatial             # 只在这里配置机器路径
bash scripts/check.sh
```

可设置 `PIP_INDEX_URL` 使用 Python 包镜像，`TORCH_FLAVOR=cu121` 切换 PyTorch wheel，`spatial init --hf-endpoint https://hf-mirror.com --root /data/spatial` 使用 HF 镜像。镜像可用性取决于网络；版本 SHA 不变。不要把 token 写入 YAML。完整命令见 [快速操作手册](docs/QUICKSTART.md)。

```bash
spatial download qwen3-vl-2b
spatial download vg-llm-data                  # 默认只下载两份训练标注
spatial doctor
```

下载标注不等于下载视频。按 [数据构建](docs/DATA.md) 获取对应媒体、渲染 SPAR 标记并生成 `spar.train.jsonl` 等清单后：

```bash
spatial run sft --model qwen3-vl-2b --name sft-001 \
  --train spar.train:0.8 llava-hound.train:0.2 --heldout spatial.val vsibench.test
spatial run rl --model qwen3-vl-2b --adapter /data/spatial/runs/sft-001/final \
  --name rl-001 --train spar.train --heldout spatial.val vsibench.test
spatial run inference --model qwen3-vl-2b --name base-vsi --eval vsibench.test
```

`--set train.learning_rate=0.000005` 等参数覆盖复用原配置校验；`--dry-run` 保存实际配置供核对。训练支持同步数据并行、LoRA、可配置每卡 batch 与梯度累积；`--gpus 4 --devices 0,1,2,3` 启动多进程。详见 [资源配置](docs/DISTRIBUTED.md)，这不是 verl/DeepSpeed 训练器。中间 checkpoint 可作下一阶段初始化，没有精确断点续训。

## 能力与入口

| 工作 | 入口 | 实现与边界 |
|---|---|---|
| 权重/数据下载、镜像、锁定版本 | `download`、`source` | HF 下载断点恢复；source SHA + 补丁；36 个资源元数据已核查 |
| 原始标注与视频转换 | `build-data`、`frames` | JSON/JSONL/Parquet，单轮 QA，视频固定采帧，SPAR 标记渲染 |
| 数据质量与划分 | `clean-data`、`split-data`、`audit` | 无效图片、冲突标注、精确重复、按场景划分与跨集合重叠；不是自动语义认证 |
| SFT / GRPO / GSPO / OPD / OPSD | `run` 或 `train` | 实际优化循环、冻结 teacher/reference；小模型端到端验证 |
| 多模型、多 benchmark | `suite`、`benchmark` | 共享协议矩阵 + EASI/lmms/VLMEvalKit 外部官方任务启动 |
| VGGT / DA3 / Pi3 | `cache-geometry` | 冻结几何提取器，精确帧指纹、缓存校验；真实权重待 GPU 验收 |
| 通用几何融合 | `FusionBackend` | 可训练 cross-attention + 零初始化门控；与原论文三种结构分别命名 |
| 原始三项目复现 | `legacy`、`projects/` | 原环境入口与快照；GeoWire 原生后端；其余原方法尚未完全移植 |
| LIBERO | `libero-eval` | 真实环境闭环接口、动作校验、初始状态与成功率记录；需另训动作策略 |

## 目录

- `spatial_intelligence/`：共用数据、模型、目标函数、训练、评测、几何、迁移代码。
- `projects/{geowire,geobridge,geopsro}/`：三个项目的任务规划与原方法入口。
- `configs/`：模式配置、模型配置、六 benchmark 矩阵、几何配置、LIBERO 配置。
- `catalog/`：资源 URL、权重/数据 SHA、源码锁定版本、官方任务映射。
- `_archives/`：本地保留的两份原始 ZIP、独立研究文档和原 MANIFEST；已忽略，不进入源码发布包。
- 原方法源码通过 `spatial source` 获取；`legacy_sources/` 仅用于显式导入的历史快照，不是公共代码的维护副本。
- `patches/`：GeoWire 生成修复和 GeoPSRO 解析/空值/零值修复。
- `tests/`、`.github/workflows/`：CPU 合约、回归和真实随机小模型优化测试。

## 文档与社区

[快速操作](docs/QUICKSTART.md) · [完整教程](docs/TUTORIAL.md) · [数据构建](docs/DATA.md) · [GPU 与 batch](docs/DISTRIBUTED.md) · [兼容性](docs/COMPATIBILITY.md) · [几何与融合](docs/GEOMETRY.md) · [具身迁移](docs/TRANSFER.md) · [代码审查](docs/AUDIT.md) · [AAAI 方法审查](docs/AAAI_REVIEW.md) · [CVPR 研究设计](docs/CVPR_RESEARCH.md) · [模型与数据资源](docs/RESOURCES.md) · [贡献指南](CONTRIBUTING.md) · [验证记录](docs/VALIDATION.md)

新公共代码采用限定范围的 MIT 许可；原工程与第三方资产保留各自条款，见 [NOTICE](NOTICE.md)。仓库没有上传到 GitHub，也没有修改三个上游仓库；补丁已经在交付快照中应用，`spatial source` 会将它们应用到指定版本的本地 checkout。

仅检查源码和配置时可执行 `python scripts/static_check.py`（Python 3.10–3.12、PyYAML，3.10 另需 tomli），它不导入模型、不执行测试或实验。发布源码时使用 `python scripts/package_release.py --output dist/spatial-intelligence-source.zip`；该脚本按公共源码目录收集文件，生成新的 `MANIFEST.sha256`。实际 Git 提交和发布由维护者完成。

已增加 wheel 运行资源声明与定位，源码维护继续推荐 editable 安装。安装目录、用户配置规则与待验收边界见 [MAINTENANCE](docs/MAINTENANCE.md)。`python scripts/release_check.py` 静态核对归档文件、版本、依赖、来源锁和 wheel 资源覆盖。

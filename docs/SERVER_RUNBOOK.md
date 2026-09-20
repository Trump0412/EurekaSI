# 新服务器：Qwen3.5 → ReVSI → 空间 SFT → ReVSI

**2026-09-20 更新：** 当前 batch 与评测入口以 [四卡测速与评测协议 v2](BATCH_AND_EVAL_PROTOCOL.md) 为准。下文单卡长样本选择、多图/16-token 协议描述的是旧流程，保留用于解释历史记录，不再作为新实验默认配置。

Qwen3.5 OPD/OPSD、隔离 VERL/GSPO、三个几何项目的真实缓存验收见 [后训练操作指南](POSTTRAINING_RUNBOOK.md)。

这是部署入口，不是已完成实验的成绩单。研究协议见 [实验契约](../projects/qwen35-revsi/README.md)。保留原 CLI/环境；本配方使用独立 Conda profile，不能混装 `native.txt`。

本轮下载结束后的旧项目/VERL 隔离部署及扩展数据配置见 [自动后续部署](POST_DOWNLOAD_DEPLOY.md)。该队列独立于当前 Qwen3.5 训练，不会修改当前实验的数据混合或评测协议。

## 给下一位操作员／Codex

只需提供本仓库、SSH 地址、持久盘根目录和可用 GPU，要求“按 SERVER_RUNBOOK 部署/恢复并确认第一条真实输出”。先读 `AGENTS.md`、本文、`docs/VALIDATION.md`；检查进程与 receipts，不重复启动、不覆盖旧结果、不把下载中说成训练中。不提供私钥或 token，不跨服务器复制 `.git`。

前提：Linux、4 × A100 40GB、兼容驱动、Conda、curl/git。压缩资源约 577GB，含分段缓存、解压和 checkpoint 建议预留至少 2TB，另行确认真实配额。模型/数据在服务器经 `hf-mirror.com` 下载；Git 不通时可用本地 `git bundle` 传源码。

```bash
git clone https://github.com/Trump0412/EurekaSI.git
cd EurekaSI
export CONDA_BIN=/path/to/miniconda3/bin/conda
bash scripts/start-qwen35-study.sh /persistent/your-root
python3 scripts/run-qwen35-study.py --root /persistent/your-root --status
python3 scripts/study-status.py --root /persistent/your-root
tail -f /persistent/your-root/logs/supervisor.log
```

替换示例中的 Conda 和持久盘路径。默认使用可见的四张 GPU；共享服务器先设置 `CUDA_VISIBLE_DEVICES`，不得占用他人任务。启动不是验收：检查环境、CPU 回归、16 样本 smoke 的真实预测与正式 baseline 输出增长。

## 流程与故障边界

1. 独立 `envs/qwen35`：PyTorch 2.7.1、Transformers 5.3.0，记录 pip freeze/check，不使用 PyTorch 2.9。
2. 资源版本固定于 `configs/qwen35-assets.json`，独立失败隔离。普通下载续传；并行 Range 校验 Content-Range/字节数，不新建 SHA256 清单。保留原仓库已有完整性检查。
3. 获取固定 renderer/scorer 源码，检查评分算术等价。
4. ReVSI 保留规定 32 帧，以有序多图 RGB 输入，最大边 448、关闭 thinking、greedy、最多16新token。它不是未经验证的 video-token 后端等价复现。
5. 16 题 smoke → 四卡分片 baseline → 覆盖率与逐类宏平均。预测逐样本落盘，重启补缺失 ID；配置变化必须换 run 名。
   baseline 后利用最先下载的 Hound 小分片做独立 `diagnostic-ddp` 两步优化、恢复到第三步、导出权重重载推理。真实数据/梯度但只有接口诊断意义；诊断为演示恢复而延长一步 scheduler，不用于比较收敛，正式 SFT 总预算不变并从原权重重新开始。
6. CPU 准备与 baseline 重叠。SPAR RGBD 是 18 个字节分卷，不能分别解压；当前提取所选 RGB，depth/pose 留在原包。Hound 包里是帧目录而非 mp4，使用 GeoThinker 准确路径与顺序。
7. 原始清单 234,277 / 63,750；先排除 ReVSI scene 重叠，记录排除 ID 和最终分母。缺图、未知 marker、上下文超限均失败，不静默丢样本。
8. 真实长样本 micro-batch 1/2/4 吞吐与显存探测 → 四卡两步 SFT gate → 一轮 SFT → 同协议 ReVSI。语言侧全参数、visual 冻结，非 LoRA；有效 batch64，GA=64/(4×micro)。选择预留至少8%显存且吞吐最高的候选；所有候选失败则停止，不静默降低图像/数据预算。Trainer 保存 optimizer/scheduler/RNG checkpoint，恢复不等于跨硬件逐位复现。
9. 两项评测齐全才生成 `runs/comparison.json`；失败不能记为0分。VLM3R/VSI590K 仅准备资源，不混进本轮训练。

VLM-3R 的 GeoThinker VSI/VST 转换标注复用 SPAR 媒体。同时保存 VLM-3R 原始标注；其 2026-07-13 camera-position 勘误可能尚未反映在旧版 GeoThinker 转换文件中。后续启用 VST 训练前必须逐 ID 核对修订，不可直接把旧转换标注当成已纠错。当前 SPAR/Hound SFT 不使用 VLM-3R 标注。来源：[原始数据卡](https://huggingface.co/datasets/Journey9ni/VLM-3R-DATA)、[GeoThinker 数据说明](https://github.com/Li-Hao-yuan/GeoThinker)。

## 状态与恢复

| 位置 | 用途 |
|---|---|
| `receipts/*.json` | 资源版本、字节与下载状态 |
| `state/*.json` | 阶段、命令、日志、返回码 |
| `logs/gpu.jsonl` | 利用率、显存、功率快照；不是 MFU |
| `manifests/*.jsonl` | 训练/评测/排除清单 |
| `runs/*/predictions.rank*.jsonl` | 可恢复的逐卡预测 |
| `runs/*/metrics.json` | 覆盖率与分数 |
| `runs/*/checkpoint-*` | 可恢复训练状态 |

修复后先测再重启 supervisor，已完成阶段跳过。输入/评分/样本语义改变必须新建 run 并重跑受影响对照，不能沿用完成标记。下载进程独立，已有下载用 `--attach-downloads`；先查锁与进程，不能盲目启动第二套。

```bash
export EUREKASI_ROOT=/persistent/your-root
export SPATIAL_PYTHON="$EUREKASI_ROOT/envs/qwen35/bin/python"
bash scripts/check.sh
python3 scripts/run-qwen35-study.py --root "$EUREKASI_ROOT" --attach-downloads --detach
```

环境失败修复后重跑 `bash scripts/bootstrap-qwen35.sh`。训练检查真实 loss/梯度和 checkpoint，不能仅看 PID；镜像/权限失败须报告具体资源，不擅自替换数据集。

## 效率与验收

- 评测四卡副本/样本分片，不无故做四卡 tensor parallel；记录 batch 耗时和 token 数。
- SFT 真正 tensor batch、DataLoader workers/pinned memory、DDP、bf16、gradient checkpointing、fused AdamW；仅答案位置投影词表 logits，需通过全 logits 的 loss/梯度等价检查。
- 原 reference trainer 的 batch 是逐样本累积，新入口不能混淆。单卡候选并行测量，使用独立丢弃模型、不改研究权重；DDP 冒烟再验显存与梯度。长样本代理不保证覆盖所有极端组合，正式训练仍可能 OOM，失败应保留诊断而非自动改输入协议。
- 看稳态 samples/s、有效 tokens/s、功率、显存与数据等待；不拿显存占满代替高吞吐，不用 dummy 计算填卡。
- 安装/下载期无法承诺每秒满载，应如实记录。正式实验期间不热改模型或数据语义。

当前完整 SFT / 前后 ReVSI 结果待测；自动入口不是已完成验收。实际以 `state` 和当前验证记录为准。

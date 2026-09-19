# 后训练与几何方法：安装、真实验收和恢复

本页补充 [服务器指南](SERVER_RUNBOOK.md)。不是全量方法复现的完成声明。
所有命令在 EurekaSI 仓库根目录执行；`ROOT` 指向用户分配的持久目录。
运行前检查 GPU 占用和现有进程，同一阶段不可重复启动。失败换新 run 名，保留旧证据。

## 隔离环境，不再被无关下载阻塞

```bash
export ROOT=/persistent/your-root
export CONDA_BIN=/path/to/miniconda3/bin/conda
python3 scripts/post-download-deploy.py --root "$ROOT" --conda "$CONDA_BIN" --environments-now --detach
python3 scripts/post-download-deploy.py --root "$ROOT" --conda "$CONDA_BIN" --native-now --detach
python3 scripts/fetch-study-assets.py --root "$ROOT" --catalog configs/geometry-assets.json --workers 2 --detach
```

- `qwen35`：Transformers 5.3，Qwen3.5 的正式 SFT/评测和蒸馏。
- `legacy-geo`：Python 3.10、Torch 2.5.1、Transformers 4.57.6，三个原生几何项目。
- `verl-legacy`：Python 3.12、Torch 2.9、Transformers 4.57.6、VERL 0.7、vLLM 0.11.2。
  **这个 rollout 版本不支持 Qwen3.5。** 用三种旧方法共同依赖的 Qwen3-VL-2B 验收框架，不替换 Qwen3.5 主实验。

安装阶段检查 `pip check`，在 `logs/extension-*-freeze.txt` 留存依赖快照。
初始环境导入成功不代表 GPU 内核、训练器、奖励或 checkpoint 通过。
硬件/驱动改变仍需重新做小规模验收；不能承诺任意服务器免验证一次成功。

## Qwen3.5 OPD / OPSD

可复用后端：`spatial_intelligence.backends.qwen35:Qwen35Backend`，可填入公共训练配置的 `model.backend` 和 `teacher.backend`。
它保留原 HF 后端接口，但明确限制模型族，关闭 thinking，扩展答案的 `mm_token_type_ids`，按各自提示长度选择答案预测位置。
学生与教师必须具有相同 tokenizer；不允许跨词表直接 KL。

以下诊断依赖已准备的 `manifests/sft-probe.train.jsonl`、基础模型和 `runs/diagnostic-ddp/final`：

```bash
CUDA_VISIBLE_DEVICES=0,1 "$ROOT/envs/qwen35/bin/python" scripts/verify-qwen35-distillation.py \
  --root "$ROOT" --mode opd --output "$ROOT/runs/diagnostic-opd-new" --detach
CUDA_VISIBLE_DEVICES=2,3 "$ROOT/envs/qwen35/bin/python" scripts/verify-qwen35-distillation.py \
  --root "$ROOT" --mode opsd --output "$ROOT/runs/diagnostic-opsd-new" --detach
```

诊断为 Hound 训练样本的真实八图、16 个采样 token、LoRA rank 8、一次反向 KL 更新。
OPD 的教师是前期三步诊断 checkpoint，**没有“更强教师”的性能证据**。
OPSD 教师为冻结基础模型，仅教师看到该训练样本的答案；学生不含答案。
没有使用 ReVSI 测试答案。LoRA 和这次单步预算不是正式 SFT 配置。

通过条件：图像栅格/像素一致、教师冻结、同一采样 token、非零有限 KL/梯度/参数变化、全 logits 与选择位置的数值对照、图像消融影响 logits、保存后重载一致。
结果在 `sample.json/config.json/metrics.json/completion.json`，权重在 `checkpoint/`，优化器在 `optimizer.pt`。
这不代表公共训练器已具有完整断点恢复功能。

## GeoBridge 修复与独立缓存

GeoBridge 发布 commit `74cfe79500e5` 删除公共可视化 helper，但仍有脚本和测试导入它。
`catalog/sources.yaml` 记录从 `68a94e26fcf57b6abdfb407ac1eb4830ce822ed5` 恢复原文件；安装器保留恢复来源，拒绝覆盖不同的已有文件。
旧模型的 `SlidingWindowCache` 接口通过独立 Transformers 4.57 环境解决，不修改当前 Qwen3.5 环境。

三个方法使用**不同缓存格式**，不可互相改名冒充：

| 方法 | 真实缓存 | 后续阶段 |
|---|---|---|
| GeoWire | Qwen token layout/语义特征 + VGGT 几何与 tracking + 稀疏图 | 两步 TIP 诊断权重，再单独验收 Stage2 |
| GeoPSRO | 原生 compact depth/point/confidence/camera 缓存 | 两步 geometry alignment 权重，再单独验收 Stage2 |
| GeoBridge | 原生 g11/g17/g23 与 patch/grid/frame 元数据 | FCP 还需 correspondence graph 和前置 initializer，不能用其他目标的权重替代 |

```bash
CUDA_VISIBLE_DEVICES=2 python3 scripts/prepare-geometry-diagnostics.py --root "$ROOT" \
  --method geopsro --output "$ROOT/runs/diagnostic-geometry-geopsro-new" --detach
CUDA_VISIBLE_DEVICES=3 python3 scripts/prepare-geometry-diagnostics.py --root "$ROOT" \
  --method geowire --output "$ROOT/runs/diagnostic-geometry-geowire-new" --detach
CUDA_VISIBLE_DEVICES=2 python3 scripts/prepare-geometry-diagnostics.py --root "$ROOT" \
  --method geobridge --output "$ROOT/runs/diagnostic-geometry-geobridge-new" --detach
```

同 GPU 的几何诊断由锁串行，其他 GPU 可并行。使用两个固定 Hound 训练样本，保留原帧顺序；未知物理时间不伪造秒数。GeoWire 缺失时间戳为空列表，避免在模型提示中写入虚构时间。
GeoPSRO 原生提取器异常时会写 zero fallback；本包装器明确检查 `geometry_valid` 和 NaN，拒绝把 fallback 作为成功缓存训练。
产物是接口诊断，不是论文阶段权重或正式方法成绩。完整复现仍需正式样本预算、所有阶段、checkpoint 重载和独立 benchmark。

## VERL 原生 GSPO 验收

在 OPD/OPSD 结束释放 GPU 后运行：

```bash
CUDA_VISIBLE_DEVICES=0,1 python3 scripts/verify-verl-gspo.py --root "$ROOT" \
  --output "$ROOT/runs/diagnostic-verl-gspo-new" --detach
```

脚本等独立环境与 Qwen3-VL 权重，检查分配卡空闲，准备四条真实 Hound 训练样本 parquet，调用 `verl.trainer.main_ppo`。
配置：Ray/FSDP/vLLM、group=4、prompt batch=2、64 response tokens、GSPO clipping 0.0003/0.0004、一次训练步、保存 checkpoint。
actor 使用 SDPA，不要求本机编译 FlashAttention2；vLLM 独立使用自己的兼容内核。
奖励为训练答案词级 F1，**只验收框架，不是空间 benchmark 的评分器**；关闭验证，不将同源数据包装成 heldout 成绩。

`status.json` 区分依赖等待、运行、失败、checkpoint 已产生但待审计。
只有同时验证 rollout、非恒定组内奖励、非零有限 actor 梯度、checkpoint 重载，才可宣称完整闭环通过。
不能以 `pip check`、进程存在、配置检查或 checkpoint 文件存在单独替代验收。
上游依据：[GSPO 实现](https://github.com/verl-project/verl/blob/v0.7.0/verl/trainer/ppo/core_algos.py)。

## 本轮已验证证据（2026-09-19）

- Qwen3.5：相关回归 13 passed；真实八图 OPD loss=0.892627、grad norm=3.178967；OPSD loss=0.296158、grad norm=1.219249。均非零参数更新，保存/重载最大 logit 差异 0。只支持单样本单步接口结论。
- GeoBridge：恢复模块 + 兼容环境后完整测试 40 passed；GeoWire 29 passed；GeoPSRO 10 passed。
- GeoPSRO：真实 VGGT 缓存通过有效性门，两步 alignment loss 2.001698 / 2.501607，已输出 `geo_adapter.pt`；尚未完成重载/正式效果验收。
- GeoBridge：两个真实样本多层缓存、原生 initializer、correspondence graph、FCP 均已产出。两步 FCP loss 2.123312 → 2.112537，梯度 3.830155 → 1.847192；权重在 `runs/diagnostic-geobridge-stages-v1/{initializer,fcp}/latest.pt`。压缩阶段日程仅验收接口，尚待重载审计，不代表完整复现。
- VERL：独立环境安装通过；已修复 SDPA、Qwen3-VL 嵌套上下文长度配置、八图上限和缺少 `qwen_vl_utils`。v5 到达真实图像处理但缺依赖退出；v6 补依赖重跑，完整闭环仍待验收。兼容补丁 `patches/verl-qwen3vl-context.patch` 已接入安装脚本，依赖已写入独立 requirements。
- GeoWire：真实缓存及两步 TIP 完成，loss 1.25 → 1.237803，`runs/diagnostic-geometry-geowire-v2/stage1/geowire_adapter.pt` 与优化器状态已保存；尚待重载和完整 Stage2 验收。原生环境与 VERL 均固定 `qwen-vl-utils==0.0.14`、`av==16.0.1`，避免旧 PyAV 源码编译失败。

## 自动 SFT 与 ETA

```bash
cd "$ROOT/EurekaSI"
python3 scripts/run-qwen35-study.py --root "$ROOT" --attach-downloads --detach
python3 scripts/training-eta.py --root "$ROOT"
```

监督器顺序执行：数据完整性校验与标记渲染 → 等待四卡空闲 → micro-batch 1/2/4 独立测速 → 两步 DDP smoke → 正式一轮 SFT → 同协议 ReVSI 复测 → comparison.json。已完成阶段跳过，训练保留优化器 checkpoint 以续训。辅助诊断失败不直接中断主实验；没有通过数据/显存门禁则不强行训练。

去除 ReVSI 场景交叉后的计划训练量为 297899 条，global batch 64，GA=64/(4×micro-batch)，约 4655 optimizer steps。冻结视觉部分，语言部分全参数训练，非 LoRA 正式训练。

`runs/diagnostic-sft-throughput-v1/eta.json` 按 1/2/3/8/32 帧各一个真实样本实测，预计四卡纯训练 **7–11 小时**；这是启动前粗估，未覆盖完整长度分布、数据加载和 DDP 开销。正式运行后 `live-eta.json` 用实际 step 速度更新，前 20 步标记 warming_up。相同协议 ReVSI 复测基于 baseline 耗时约 28 分钟计算时间，可预留 30–45 分钟。数据准备另算，不将“脚本已启动”等同于训练已启动。

共享存储的小文件元数据与重复图片编码曾严重拖慢准备。现对 Hound 完整性检查使用 32 线程，对 SPAR 标记使用 16 线程与按需解码；未被标记访问的帧直接引用原图，仍保留规定的全部帧及顺序。`scripts/verify-marker-parity.py` 已对 34 种真实题型逐像素比较通过；不可为了吞吐取消标记或减帧。准备进度与预计剩余时间写入 `receipts/train-render-progress.json`。

这些代码已同步本地和服务器工作树；Git commit/push 另行执行，不宣称已发布到 GitHub。

# 独立节点的原生 VERL GSPO

这是 Qwen3-VL-2B 的 legacy VERL 0.7 验收与小规模空间探索，不是 Qwen3.5 的 VERL 支持声明。Qwen3.5 的 reference GSPO/OPSD 另用隔离环境。

## 隔离与恢复

从 GitHub clone 每节点自己的 EurekaSI；`ROOT` 是节点私有目录，`SOURCE_ROOT` 仅提供只读模型、数据和环境复制源。禁止将运行中的源环境 editable 安装指向新仓库。

```bash
conda create --copy --yes --override-channels \
  -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main \
  --prefix "$ROOT/envs/verl-legacy" --clone "$SOURCE_ROOT/envs/verl-legacy"
python3 scripts/run-node-rl.py --root "$ROOT" --source-root "$SOURCE_ROOT" \
  --devices 0,1 --name diagnostic-verl-node-v1 --detach
```

`--copy` 避免环境文件硬链接后互相污染；不要假定不同节点具有相同 Conda 包缓存。新节点 `--offline` 可能在复制 pip 文件后因缺少 Python/系统库的 Conda 包失败；保留文件、使用镜像补齐即可，不需删除整个环境。显式 `--override-channels` 避免意外使用默认付费/需确认条款的渠道。

监督器等待独立 clone 完成，安装本仓库 editable、检查依赖、应用版本限定的上下文补丁，再运行真实 Hound 八图 GSPO 诊断。所有 state、logs、receipts、runs 都写在节点私有目录。数据和模型通过只读使用约定复用；manifests 仅逐文件链接，允许另外存储本节点清单。

Conda clone 的最后事务可能把 pip 降级过的 setuptools 又装回 Conda 版本。节点入口通过 `requirements/verl-node-compat.txt` 重申 vLLM 的 setuptools<81 约束和 EurekaSI 的 Pillow11.3要求，再做 `pip check`；不修改复制源。

`verify-node-verl-gspo.py` 调用共享验收器，只增加原生 rollout/validation JSONL 留存，不维护另一个训练器副本。

本轮另定位一个真实 rollout 阻塞：vLLM0.11.2 已要求 `execute_model(non_block=True)` 返回 Future，但 VERL0.7 的外部 ZeroMQ executor 将兼容方法错误地限制到 vLLM>=0.12，导致第一次真实图像生成时 `NoneType.result`。`patches/verl-vllm0112-executor.patch` 只调整该版本门槛；`scripts/patch-verl-executor.py` 锁定两个包版本并验证 execute/sample Future 返回契约。它属于引擎 API 兼容问题，不可笼统归因 CUDA 驱动；补丁通过仍需重新跑真实 GPU 闭环。

2026-09-20 的失败 gate-v1 已完成真实八图解码：一个请求的实际 prompt 为1564 tokens、每帧 `image_grid_thw=[1,38,20]`，未超过4096；因此这次失败不是上下文溢出或图片缺失。修复后的 executor 运行测试已验证两种 non-blocking 调用均返回 Future，且 `.result()` 保持原 RPC 的 None/采样结果；新的 gate-v2 留存独立证据，不覆盖 v1。

gate-v2 实测通过：两组真实 rollout 均有组内奖励变化，actor grad norm=56.4151、PG loss=4.75e-7；488个参数 tensor 改变，最大绝对变化1.90735e-6；导出模型重载后真实八图预测成功。单步391.99秒（gen93.34、old-logprob145.39、update153.22），另保存14.88秒；峰值allocated25.48GiB/reserved30.56GiB。长时间无日志最终证明是单步较慢，不是已证实的hang，不应据此盲改通信协议。

导出额外的 `lm_head.weight` 已确认是合法 tied embedding 别名：base/export配置均启用tie，导出head与embedding逐位相等；没有缺失原参数。审核仅允许这个已验证别名，不一般性放宽键集合。tokenizer原始JSON的BPE merges字符串/列表和默认ignore_merges序列化不同，但同一tokenizers运行时规范化后完全相同。CPU全参数审计显式限制8线程，避免本机128+默认线程拖慢中型tensor遍历。

## 进入正式探索的门槛

`audit-verl-gate.py` 自动要求以下证据并写 `acceptance.json`：

- 真实生成的文本和每条奖励已落盘；至少一组采样奖励不是常数。
- 原生 actor 的梯度有限且非零。
- FSDP checkpoint 可导出为 HF 模型；参数键一致、全参数有限，至少一个参数数值相对基础模型改变。
- 导出模型重新加载后能对真实八图产生预测。

只有 `status=accepted` 可解锁下一阶段。诊断奖励是训练样本词级 F1，仅验收算法通路，绝不是正式空间成绩。失败保留原 run，修复后换新 run 名，不把失败当作 0 分或“已完成”。

## 四卡空间探索

用 `prepare-spatial-rl-probe.py` 生成独立场景切分，复用已审计的空间关系 Yes/No 清单；不以 ReVSI 测试题训练，不从已见过该 heldout 的 SFT checkpoint 初始化。

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 scripts/run-verl-spatial.py \
  --root "$ROOT" --gate "$ROOT/runs/diagnostic-verl-node-v1" \
  --train "$ROOT/manifests-local/spatial-boolean-v1/train.jsonl" \
  --val "$ROOT/manifests-local/spatial-boolean-v1/val.jsonl" \
  --output "$ROOT/runs/qwen3vl-spatial-gspo-v1" --steps 100 --detach
```

探索预算：512 train / 128 val，16 prompts/step，4 responses/prompt，100 optimizer steps；每卡 actor microbatch 1、GSPO sequence-mean、clipping 0.0003/0.0004、学习率 1e-6，生成上限16 tokens，保留全部1/3张标记图。每图最大像素面积200704，prompt上限4096，无静默文本截断。这个像素协议不是跨后端等价声明。

VERL0.7 默认 dataset 不消费任意 `data.image_kwargs`。本实验显式使用 `BoundedSpatialDataset` 把面积限制写到真正送入 qwen-vl-utils 的每张图上；准备阶段用真实1/3帧样本验证帧数、尺寸和面积上限，保存 `image_protocol_audit.json`。正式四卡启动前等待每张分配卡显存低于500MiB，避免和另一条已获分配的 SFT 对照冲突；等待超过4小时明确停止，不挤占其他任务。

训练前、训练末用同128条 heldout、greedy、相同输入协议评估。奖励是严格 Yes/No 正确率，歧义或非规范文本得0，不使用词汇重叠。输出 `rollouts/*.jsonl` 与 `validation/*.jsonl`，保留样本、配置和来源 manifest。每25步保存原生 checkpoint。

**正式100步的最终checkpoint单独重载全128题，仍是额外验收项**；当前自动比较是原生训练器内的before/after validation，不能把诊断gate已重载当作正式最终权重已重载。

对已启动的监督PID配置有限后台守护：

```bash
python3 scripts/watch-native-rl.py --output "$ROOT/runs/qwen3vl-spatial-gspo-v1" \
  --supervisor-pid RUN_PID --steps 100 --max-hours 24 --stale-minutes 90 --detach
```

`health.json` 记录实际optimizer step、最近步耗时和粗略剩余训练时间；前10步标记预热。监督器意外退出或超时均持久化失败状态；超时只终止已核实的本任务进程树，不调用全局Ray stop或杀其他GPU任务。GPU利用率不代替optimizer进度。

这只探索小规模二元空间关系学习：不能称为完整 SPAR、ReVSI、论文复现，不能把不同 backbone 的分数差异解释成算法改进。总 ETA 须以实际启动后10–20步速度估计；环境复制或只有 PID 时不报虚构训练吞吐。

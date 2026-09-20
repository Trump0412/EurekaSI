# Qwen3.5 多模态 SFT 吞吐优化验收

日期：2026-09-20。对象：Qwen3.5-2B、SPAR/Hound 混合训练数据、4×A100 40GB。验收类型：数值检查、受控吞吐、保存/恢复；不评估下游空间 QA 正确率。

运行环境：Python3.12、torch2.7.1、transformers5.3.0、accelerate1.12.0、flash-linear-attention/fla-core0.5.2、triton3.3.1。验收复用既有环境，没有在线替换活动环境的内核依赖。

## 1. 范围与当前结论

按用户要求停止原 SFT，保留完整 checkpoint-600（模型、优化器、scheduler、四个 rank 的 RNG）。它不是完成一轮训练的权重。后续原 SFT/配对评测队列也已停止，不自动重启。

全部性能候选以同一 checkpoint-600 为起点，不把诊断产生的参数用于正式实验。验收发现 **现有 auto 后端已经在使用 FLA**；日志中的 fast-path warning 不能解释为所有 DeltaNet 算子都使用 PyTorch 回退。

当前证据不支持把“直接启用 FLA”或“增大 batch”自动认定为等价优化。以下失败结果保留，未放宽原梯度阈值。

最终结论：micro1 有效 batch 内均衡的短程吞吐收益为7.42%，通过预设吞吐门。真实 checkpoint 恢复流程完成；同 checkpoint 分叉的最终模型参数误差通过1e-5门，但 optimizer 并非逐位一致，不宣称严格确定性复现。优化保持显式启用，不自动恢复原研究训练。

## 2. 数值门与排错

共同门槛：loss 相对误差<1%、全部非空参数梯度的整体 relative L2<5%、无 NaN/Inf。relative L2 定义为 `||g_candidate-g_reference||₂ / ||g_reference||₂`。这些是本项目的预设工程阈值，不是官方精度保证。

| 诊断 | 参考路径 | 样本范围 | loss 相对差 | 梯度 relative L2 | 判断 |
|---|---|---|---:|---:|---|
| 原生 FLA 替换 | PyTorch reference、micro1 | 1/2/32图各一条，共3条 | 0.374% | 9.03% | 未通过 |
| 保留外部 Q/K 归一化的 FLA 尝试 | 同上 | 同3条 | 0.526% | 9.60% | 未通过，未保留为推荐实现 |
| auto micro2 | auto micro1 | 1/2/3/32图各一条，共4条 | 0.208% | 5.98% | 未通过 |
| auto micro4 | auto micro1 | 同4条 | 0.313% | 5.87% | 未通过 |

外部归一化尝试最初还触发 Triton 的 fp32/bf16 操作数不一致；补齐显式转换后能够执行，但梯度仍不达门槛。因此“只由归一化位置造成差异”的解释没有得到支持。不能将这些 bf16 数值差异直接解释为模型数学实现错误，也不能据 loss 接近就宣布梯度等价。

后续主验收收窄为 **保持 auto 后端和 micro1，仅改变有效 batch 内样本排布**。用64条真实样本比较原顺序/反序的累积梯度，以检查顺序敏感性；再用实际四卡 sampler 验证吞吐和恢复。batch2/4 只作为诊断，不能进入推荐集合。

## 3. 四卡比较协议

- 数据：训练清单中 seed3407 抽取768条固定真实混合样本，另加64条长样本压力测试，共832条诊断输入；不使用 ReVSI/VSI test 选择配置。
- 每候选：12个 optimizer steps，前3步预热、后9步计时；第13步是长样本压力测试，不计吞吐但计显存。
- 相同 global batch64、bf16、冻结视觉部分、相同帧/分辨率、sample-mean loss、fused AdamW、gradient checkpointing。
- 正序、反序各跑一次；这是两次固定样本测量，不是多 seed 收敛实验或置信区间。
- 主对照：auto legacy micro1 / auto balanced micro1。诊断：auto legacy micro2 / balanced micro2 / balanced micro4。
- 只有数值门通过、两次完成、reserved 显存峰值<容量92%的候选可推荐；要求中位吞吐提升≥5%，且较慢一次也快于基线较快一次。

指标：samples/s 为计时期间实际有效样本数除以耗时；有效 tokens/s 排除 padding；padding 比例为 `1 - 有效输入token数/补齐后输入token数`；显存为 PyTorch reserved 峰值，不是 nvidia-smi 全进程占用，更不是 GPU 计算利用率。

## 4. 主结果

64条样本的原序/反序累积 loss 均为0.46853257，loss相对差为0，全梯度 relative L2 为2.179%，数值有限，通过预设5%门槛。这是单卡顺序敏感性检查，不是实际四卡梯度逐位相等的证明。

| auto 配置 | 每卡 micro / GA | 两次 samples/s | 中位 samples/s | padding | 峰值 reserved GiB/卡 | 数值资格 |
|---|---|---|---:|---:|---:|---|
| legacy | 1 / 16 | 10.831 / 10.797 | 10.814 | 0% | 23.45 | 基线 |
| balanced | 1 / 16 | 11.685 / 11.549 | 11.617 | 0% | 23.48 | 通过 |
| legacy | 2 / 8 | 10.386 / 10.337 | 10.362 | 35.92% | 24.95 | 不通过，仅诊断 |
| balanced | 2 / 8 | 14.505 / 14.561 | 14.533 | 4.46% | 24.65 | 不通过，仅诊断 |
| balanced | 4 / 4 | 11.581 / 11.585 | 11.583 | 10.70% | 27.63 | 不通过，仅诊断 |

**接受配置：micro1、GA16、四卡 global batch64、auto、balanced、sample_mean。** 中位吞吐相对基线提升7.424%，同等计时样本的计算时间约减少6.91%；较慢的优化重复11.549仍快于较快的基线10.831。不是对整轮 wall-clock 或下游分数的保证。

micro1 本来就没有 padding，收益不能解释成“消除 padding”；负载排布变化是主对照因素。micro2 的均衡方案相对 micro2 legacy 快约40.3%，同时 padding 从35.92%降到4.46%，支持减少补齐与改善排布的工程价值，但它未通过整模型梯度门，不能用这一更大的数字作为已验收配置的加速宣传。

## 5. 保存/恢复验收

独立诊断使用同一192条真实训练样本，固定总预算3步；比较连续三步与第二步保存/停止后恢复到第三步。两条路径总预算和 scheduler 不变，不通过延长 max_steps 演示恢复。

检查真实 `resumed_from=checkpoint-2`、最终 global_step/max_steps、完整模型参数差异和 scheduler 相等。参数 relative L2 阈值1e-5。所有输出使用独立 diagnostic 名称，不覆盖 checkpoint-600。

首轮两条独立运行：连续3步，对照独立运行到2步后恢复至3步。恢复路径确实加载 checkpoint-2，scheduler、global_step/max_steps相同，但模型参数relative L2为2.4284e-5，未通过1e-5门；optimizer relative L2为7.682%，非逐位相同。两条路径第1步loss已经分别为0.534028/0.534344，第2步为0.679711/0.679059，因此不能把最终差异全部归因于恢复。

补充同源分叉：完整复制连续路径的同一个checkpoint-2（含模型、optimizer、RNG、scheduler）与训练契约到独立诊断目录，只恢复执行第3步，再与原连续路径的checkpoint-3比较，排除中断前独立运行差异。

| 指标 | 独立运行后恢复 | 同 checkpoint-2 分叉恢复 |
|---|---:|---:|
| 模型参数 relative L2 | 2.4284e-5，未通过 | 7.2219e-6，通过 |
| 参数最大绝对差 | 3.0518e-5 | 1.1444e-5 |
| optimizer relative L2 | 7.682% | 5.024% |
| optimizer 逐位相同 | 否 | 否 |
| scheduler 相同 | 是 | 是 |
| 最终 global_step / max_steps | 3 / 3 | 3 / 3 |

比较覆盖全部2,213,241,664个模型参数；optimizer状态字段/参数组覆盖一致、内部step一致。**通过的仅是预设模型参数误差、scheduler与步数门，以及实际保存/恢复流程；optimizer差异是单独记录的诊断指标，不在该1e-5模型参数门内。** 不将结果解释为optimizer等价、逐位恢复或长期收敛一致。若后续研究要求严格确定性，需要另行定位内核/数值差异；本轮未确定唯一根因，未修改门槛。

三步训练及保存、恢复均已结束，四卡已释放；没有重启原研究SFT。

## 6. 回归、限制与复现

本次初轮完整服务器测试：143 passed、1 skipped、1 failed；唯一失败为服务器既有 GeoBridge catalog/lock 不一致。依赖检查通过；本地发布元数据检查通过。该非本次吞吐改动的问题未被静默改写。随后针对吞吐、训练、答案提取、空间评分、视频兼容和公开摘要的回归为48 passed、1 skipped。

本次工程修复包括：模型构造前设置当前 rank 对应的 CUDA device，避免四个进程在 GPU0 建立初始上下文；在 rank0 写入新运行元数据前同步各 rank，防止新目录检查竞态；诊断脚本使用有界非阻塞文件锁重试，修复共享盘阻塞式等待出现的 `ETIMEDOUT`。这些是具体启动/排队修复，不单独声称贡献了某个加速百分比。

限制：短程吞吐不保证整轮训练速度；bf16 数值一致性门不证明收敛相同；更快不等于 ReVSI/VSI 分数更高；样本长度成本使用代理，实际 token/padding 从 processor 输出计算。原 SFT 已停止，本报告不声称取得新的全量空间评测成绩。

复现与参数见 [吞吐优化指南](THROUGHPUT_OPTIMIZATION.md)，逐配置数字、832条样本ID和同源恢复指标见 [公开验收摘要](evidence/throughput-acceptance-2026-09-20.json)。`scripts/summarize-throughput.py` 按白名单导出数值摘要，不复制连接信息、私有机器路径或原始基础设施日志。公开报告与仓库不包含实际服务器映射。实验设计与报告 skill 使本次按预先数值门拆分主结果、失败对照及结论边界，而不是只呈现最快配置。

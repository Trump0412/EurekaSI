# 多模态 SFT 吞吐优化：实现、对照与验收

日期：2026-09-20。验收与发现见 [验收报告](THROUGHPUT_ACCEPTANCE_2026-09-20.md)。原 SFT 已按用户要求停止；checkpoint-600 保留。当前主验收固定既有 auto 后端和 micro1，只改变有效 batch 内排布。

## 决策与边界

研究问题：当前可变长、多图 SPAR/Hound SFT 是否因 padding、rank 工作量不均或 DeltaNet 内核而低效？目标是在不减少样本、帧数、分辨率或训练预算的情况下提高有效 samples/s 和 tokens/s。最高主张是受控工程吞吐比较，不是空间能力提升。

竞争解释分别通过三组对照检查：同 kernel 下 legacy/balanced；同排布下 reference/FLA；同有效 batch 下 micro1/2/4。显存更满不作为成功标准。

## 已实现的公共接口

| 文件 | 功能 |
|---|---|
| `spatial_intelligence/throughput.py` | 成本代理、有效 batch 内分桶和 rank 均衡、显式 kernel 选择及路径记录 |
| `spatial_intelligence/throughput_sampler.py` | 对接 Trainer/Accelerate 的确定性 sampler |
| `spatial_intelligence/qwen35.py` | 可选 sample-mean completion loss；旧默认 token-mean 不变 |
| `spatial_intelligence/study.py` | 可选训练参数、恢复契约与 kernel inventory |
| `scripts/verify-delta-kernels.py` | bf16 算子输出/梯度检查 |
| `scripts/verify-throughput-model.py` | 真实1/2/3/32图样本的整模型检查；schedule-only 使用64条样本的顺序敏感性检查 |
| `scripts/benchmark-ddp-batches.py` | 四卡吞吐、有效 token、padding、显存、数据等待与真实算子路径 |
| `scripts/tune-throughput.py` | 等待现有 study、数值门、正反序两轮消融、推荐配置 |
| `scripts/verify-throughput-resume.py` | 固定总预算的连续/中断恢复对照，不重启研究 SFT |
| `scripts/summarize-throughput.py` | 白名单导出公开数字，排除机器路径和原始日志 |

分桶成本代理为 `256 × 图像数量 + ceil((问题字符数+答案字符数)/4)`，**不是精确 token 数**。它只指导排布，不过滤、裁剪或修改任何样本。真实 padding/token 指标从实际 processor 输出计算，不用代理值冒充。

在一个固定有效 batch 内将接近长度的样本组成 micro-batch，再将各组分配到预计工作量最低且仍有位置的 rank。每个 optimizer step 的样本集合保持不变，不跨 step 排序。首个有效 batch 原样保留，防止 Accelerate 从开头补齐尾部时改变重复样本；不完整尾部也不重排。改变 micro-batch 本身可能改变 DDP 尾部补齐数量，须单列；同 micro 的 legacy/balanced 尾部已纳入测试。

训练 sampler 使用相同 SeedableRandomSampler 与 seed/epoch；测试覆盖 Accelerate 真实 BatchSamplerShard 的各 rank 输出、完整 step 和尾部重复样本。改变 rank/运算顺序不保证 bf16 逐位相同。

## 为什么加入 sample-mean loss

原每卡 micro1 + GA16 近似于每条样本的答案 token 平均 loss，再对样本取平均。直接改成 micro2 的整批 token 平均，会让答案较长的样本权重更大。

新选项固定为：`mean_i(sum_t(valid_token_loss[i,t]) / valid_answer_tokens[i])`。这让不同 micro 分组保持相同样本权重。玩具模型 loss/梯度等价测试已加入；真实模型 bf16 容差验证是下一道门。旧实验默认不热改；balanced 策略要求显式 sample_mean。

## 内核控制与验收

`reference` 强制 PyTorch DeltaNet；`fla` 只替换 chunk/recurrent DeltaNet，保留相同 reference norm/conv；`auto` 记录环境实际路径，作为当前部署的工程基线。不会根据一句 fast-path warning 推断所有算子均回退。

本次服务器已经存在 flash-linear-attention/fla-core 0.5.2；没有为本任务修改活动环境。没有安装 causal-conv1d，也没有将普通 FlashAttention、FLA 和卷积融合混称为一个功能。参考：[FLA 官方实现](https://github.com/fla-org/flash-linear-attention)。

算子检查长度127/256/513，真实 GPU 上输出 relative L2 为0.00527–0.00544，五种输入梯度的 relative L2 最大0.00668，无非有限数。预先阈值为输出2%、梯度5%；已通过。第一次 Triton 编译约97秒，**这些含编译的单次耗时不是训练加速倍数**。

整模型门固定 loss 相对误差<1%、全梯度 relative L2<5% 且有限。实际验收中，FLA/reference 切换及 auto micro2/4 未达到梯度门槛，因此未作为通过验收的优化。现有 auto 本身已使用 FLA；不能把旧 warning 当成全回退证据。

`--schedule-only` 专门验收固定后端/micro1 的排布变化：64条真实样本正序/反序累积，loss相同、梯度relative L2约2.18%，通过预设门槛。更大 batch 即使测速更快，也不能绕过它自己的数值门。

## 自动四卡实验

```bash
export ROOT=/persistent/your-root
cd "$ROOT/EurekaSI"
python3 scripts/tune-throughput.py --root "$ROOT" \
  --checkpoint "$ROOT/runs/sft-spar234k-hound64k/checkpoint-600" \
  --output "$ROOT/runs/throughput-acceptance-new" --schedule-only --detach
```

等待 `state/study.lock` 释放，最多48小时，不中断训练或其正式评测。拿到锁后仍核对四卡空闲，其他任务占卡则报错，不抢卡。默认用当前 SFT 的 final 权重；可用 `--checkpoint` 明确指定完整现有权重。诊断权重不保存为研究模型。

替换为实际完整 checkpoint，不保证其他服务器存在 step600。固定 seed3407、768条真实混合训练样本、global batch64、12步（3步预热+9步计时），外加64条长样本压力测试。保存逐条 ID 与原 manifest 内容。schedule-only 模式下5组配置正序、反序各跑一遍：

- auto / legacy / micro1：工程基线；
- auto / balanced / micro1：保持单样本计算，均衡 rank 工作量；
- auto / legacy / micro2：诊断，不通过数值门则不参与推荐；
- auto / balanced / micro2、micro4：诊断，不通过数值门则不参与推荐。

不带 schedule-only 的完整模式保留后端对照；当前其严格数值门未通过，不推荐用它跳过已知失败。

所有测速采用 sample_mean。每候选15分钟、整套运行预算60分钟（终止宽限除外）；OOM/超时终止候选进程组并保留日志。只有完成两次、长样本也通过、峰值 reserved<92%显存的候选才参与选择。

接受条件：中位吞吐提升≥5%，且候选两次中的较慢一次也快于基线较快一次。这是小规模稳健性门，不是置信区间。报告 samples/s、有效 tokens/s、padding 比例、数据等待、峰值显存、实际内核函数和两次离散程度。没有达到门槛就保留基线。

产物：`status.json`、`samples.jsonl`、`model-parity.json`、各候选日志/指标/kernel inventory、`recommendation.json`。推荐不自动改写正在进行的研究训练，更不根据 benchmark 正确率挑吞吐配置。

## 使用通过验收的配置

### 基础设施定位与多 batch 使用

本功能保留为 EurekaSI 的可复用、显式启用的训练基础设施，不是一次性实验补丁。支持长度分组、有效 batch 内 rank 均衡、样本等权 loss、吞吐测量与恢复契约。当前接入并实测的是 `spatial_intelligence.study` 的 Qwen3.5 路径，不代表所有模型或其他训练入口已自动启用。

原训练器本来就支持 micro-batch>1；本优化的价值是减少可变长输入带来的无效计算和负载不均，而不是首次提供多样本 batch。四卡保持 global batch64 时，可选配对为 micro1/GA16、micro2/GA8、micro4/GA4。增加 micro 时须同步调整 GA，不能无意改变有效 batch。

例如，以下是**显式选择的 micro2 实验配方**，不是通过全部数值检查的默认推荐。先按上文测量真实数据吞吐、检查显存与数值，再决定是否启动；使用新运行名，不向旧 checkpoint 静默更换训练契约：

```bash
"$ROOT/envs/qwen35/bin/python" -m torch.distributed.run --standalone --nproc_per_node=4 \
  -m spatial_intelligence.study train --root "$ROOT" --name sft-balanced-micro2-new \
  --batch-size 2 --ga 8 --throughput-policy balanced \
  --delta-backend auto --loss-reduction sample_mean
```

本次 micro2 balanced 的中位吞吐为14.533 samples/s，高于 micro1 balanced 的11.617，但 micro2 的梯度检查未通过本次预设工程门。该门不是业界统一的质量标准，未通过也不是准确率下降的证据；保留功能供显式实验使用，不修改自动推荐门槛，不声称多 batch 必然更快或下游效果等价。默认训练参数不变，提交/更新仓库本身不会启动训练。

### 已通过本次数值与吞吐门的 micro1 配方

本次两轮四卡受控测试接受以下配置：中位10.814→11.617 samples/s，提升7.42%；峰值 reserved 约23.48 GiB/卡。其他机器、数据组成或软件版本须重新验收，不外推为整轮训练加速保证。

```bash
"$ROOT/envs/qwen35/bin/python" -m torch.distributed.run --standalone --nproc_per_node=4 \
  -m spatial_intelligence.study train --root "$ROOT" --name sft-throughput-validated \
  --batch-size 1 --ga 16 --throughput-policy balanced \
  --delta-backend auto --loss-reduction sample_mean
```

写入 `throughput-contract.json` 和 `kernel-inventory.json`。恢复时必须匹配契约，拒绝把新 sampler/loss/kernel 静默塞入旧无版本 checkpoint；原研究训练已按用户要求停止，不称为完成。`--processor` 可以明确指定仅含权重的源 checkpoint 所需的 processor。

保存/恢复验收使用已安装 torch 的训练环境：

```bash
"$ROOT/envs/qwen35/bin/python" scripts/verify-throughput-resume.py \
  --root "$ROOT" --suite "$ROOT/runs/throughput-acceptance-new" \
  --checkpoint "$ROOT/runs/sft-spar234k-hound64k/checkpoint-600" \
  --output "$ROOT/runs/throughput-resume-new" --detach
```

它在数值门通过、吞吐队列释放锁后运行192条真实样本、固定三步预算的连续/中断恢复对照。共享盘锁采用有界非阻塞重试，避免阻塞式 NFS 锁等待超时。底层 `--stop-after-steps` 和 `--diagnostic-manifest` 仅允许用于最多五步的 diagnostic 运行，不能意外缩短正式训练。比较最终模型、scheduler、步数并记录 optimizer 状态差异；不是只检查 checkpoint 文件存在。

若两条独立运行在中断前就出现数值差异，使用新输出目录，并加 `--branch-source "$ROOT/runs/diagnostic-<已有连续运行>"`：复制同一个 checkpoint-2 的模型/optimizer/RNG/scheduler 与训练契约，仅重跑最后一步，对比源连续运行的 checkpoint-3。原证据保留；不能把独立运行差异直接归咎于恢复，也不能因为存在 bf16 差异就放宽门槛。

本次使用实验设计 skill 将“可能更快”拆成数值、样本集合、真实四卡吞吐三道门。batch2/4 的速度诊断保留，但未通过整模型数值门，不推荐自动切换；原 SFT 已停止，不给它编造新的完成 ETA。

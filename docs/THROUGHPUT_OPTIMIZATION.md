# 多模态 SFT 吞吐优化：实现、对照与验收

日期：2026-09-20。状态：实现与初步验证完成；四卡端到端对照已部署排队，尚无正式加速倍数。

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
| `scripts/verify-throughput-model.py` | 真实1/2/32图样本的整模型 loss/全部有效参数梯度检查 |
| `scripts/benchmark-ddp-batches.py` | 四卡吞吐、有效 token、padding、显存、数据等待与真实算子路径 |
| `scripts/tune-throughput.py` | 等待现有 study、数值门、正反序两轮消融、推荐配置 |

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

整模型门使用相同 checkpoint、真实图像、相同目标，比较 reference micro1、reference micro2、FLA micro1；loss 相对误差<1%、全梯度 relative L2<5% 且有限。任意门失败停止扩大，不自动换配置。

## 自动四卡实验

```bash
export ROOT=/persistent/your-root
cd "$ROOT/EurekaSI"
python3 scripts/tune-throughput.py --root "$ROOT" \
  --output "$ROOT/runs/throughput-ablation-v1" --detach
```

等待 `state/study.lock` 释放，最多48小时，不中断训练或其正式评测。拿到锁后仍核对四卡空闲，其他任务占卡则报错，不抢卡。默认用当前 SFT 的 final 权重；可用 `--checkpoint` 明确指定完整现有权重。诊断权重不保存为研究模型。

固定 seed3407、768条真实混合训练样本、global batch64、12步（3步预热+9步计时），外加64条长样本压力测试。保存逐条 ID 与原 manifest 内容。7组配置正序、反序各跑一遍：

- auto / legacy / micro1：工程基线；
- reference / legacy / micro1、micro2；
- reference / balanced / micro2；
- FLA / legacy / micro1；
- FLA / balanced / micro2、micro4。

所有测速采用 sample_mean。每候选15分钟、整套运行预算60分钟（终止宽限除外）；OOM/超时终止候选进程组并保留日志。只有完成两次、长样本也通过、峰值 reserved<92%显存的候选才参与选择。

接受条件：中位吞吐提升≥5%，且候选两次中的较慢一次也快于基线较快一次。这是小规模稳健性门，不是置信区间。报告 samples/s、有效 tokens/s、padding 比例、数据等待、峰值显存、实际内核函数和两次离散程度。没有达到门槛就保留基线。

产物：`status.json`、`samples.jsonl`、`model-parity.json`、各候选日志/指标/kernel inventory、`recommendation.json`。推荐不自动改写正在进行的研究训练，更不根据 benchmark 正确率挑吞吐配置。

## 使用通过验收的配置

以下仅示范参数，必须以实际 recommendation 为准，不代表当前推荐 micro2：

```bash
"$ROOT/envs/qwen35/bin/python" -m torch.distributed.run --standalone --nproc_per_node=4 \
  -m spatial_intelligence.study train --root "$ROOT" --name sft-throughput-validated \
  --batch-size 2 --ga 8 --throughput-policy balanced \
  --delta-backend fla --loss-reduction sample_mean
```

写入 `throughput-contract.json` 和 `kernel-inventory.json`。恢复时必须匹配契约，拒绝把新 sampler/loss/kernel 静默塞入旧无版本 checkpoint；本轮已在训练的基线保持原配置完成。正式采用前还应做一小段新配置训练与保存/恢复验证，不能只凭算子测试宣布全流程已验收。

本次使用实验设计 skill 将“可能更快”拆成数值、样本集合、真实四卡吞吐三道门。尚未测出整模型/四卡结果时，不承诺加速倍数或新的训练 ETA。

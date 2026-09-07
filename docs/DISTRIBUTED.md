# GPU 数量、batch 与多进程执行

## 一个入口调整资源

```bash
spatial run sft --model qwen3-vl-2b --name sft-4gpu \
  --gpus 4 --devices 0,1,2,3 \
  --train spar.train:0.8 llava-hound.train:0.2 --heldout spatial.val vsibench.test \
  --set train.batch_size=2 --set train.gradient_accumulation=8
```

此例的全局 prompt batch = 4 张卡 × 每卡 batch 2 × 梯度累积 8 = **64**。减为 2 张卡但想保持 batch64，可设置 batch2、gradient_accumulation16。固定全局 batch 不能自动固定训练 token 数，因为样本长度与 RL 生成长度不同，仍需核查日志。

RL/OPD/OPSD 使用相同资源参数。RL 另有 group_size，例如 group8 则每个 optimizer step 产生 64×8 条 rollout。group_size 不属于 prompt batch；不能为了省显存改 group 后仍称奖励估计条件完全相同。

```bash
spatial run inference --model qwen3-vl-2b --name vsi-4gpu \
  --gpus 4 --devices 0,1,2,3 --eval vsibench.test
```

推理按 rank 切分样本，每 rank 写独立文件，最后按原清单顺序合并；重复/缺失和评分分母仍受校验。每题随机种子由 sample ID 得到，不随 rank 改变。

## 直接 torchrun

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  -m spatial_intelligence train --config /data/spatial/runs/plans/my-run.json
```

用 `spatial run ... --dry-run` 生成该配置。`--gpus` 快捷方式用于单机；跨节点直接使用 torchrun 自身的 rendezvous 配置，数据与输出目录必须共享。跨节点和 NCCL 尚未实机验收。

## 实际 batch 语义

本版的 `train.batch_size` 是每 rank 每次累积的样本数，内部**逐样本前向/反向**，然后同步参数梯度。它与小样本平均损失的同步数据并行在数学上对应，但不是 padding 后一次 VLM forward 的大张量 batch。增大 batch 不会把所有样本同时放进显存，因此容易调参；吞吐效率也低于有动态 batching 的工业训练引擎。

不同长度的每个 completion 在本实现中各自先取 token 平均，再取样本平均。保持 GPU 数量变化时全局 prompt batch 相同，保持同一抽样全局位置、数据顺序和优化配置，SFT 更新应数值接近。RL 还受到采样数值和生成种子的影响。

## 同步细节与边界

- 每 rank 都从同一确定性数据抽样流中取自己的全局 draw 位置。抽样是有放回的，同一个样本可被多次抽中；不会错误地先各自随机 shuffle 再按 rank 分片。
- 参数初始化由 rank0 广播；每 step 在裁剪前平均梯度。未使用参数也参与一致顺序的 collective；所有 rank 都未使用时保留 grad=None。
- 非有限 loss/梯度经全局标志检查，所有 rank 一致终止；不会各自跳过不同 step。
- rank0 保存模型与全局日志，其他 rank 保留本地日志。异常交由 torchrun 终止整个 worker group，不尝试隐蔽跳过故障样本。
- 分布式模式每 rank 的 student 与 teacher 在同一 LOCAL_RANK GPU 上，RL reference 也占该卡显存；它是数据并行，不是模型切分或教师服务。
- 融合模块以同样方式同步；原 legacy 脚本仍使用自己的训练器，这项改动不自动修复原 GeoPSRO 分布式代码。
- 没有 FSDP/ZeRO、优化器分片、精确断点恢复、异步 rollout 或自动 batch 寻优。

已编写两进程 CPU/Gloo 的未使用梯度同步、相同全局 batch 的 SFT 数值对照、并行推理覆盖率测试。当前运行环境拒绝 Gloo 建立进程通信（Operation not permitted），因此这三项没有完成验收；它们保留在 GitHub CI 的必跑测试中。多 GPU/NCCL 也尚未实测。单进程与模块测试可单独执行 `pytest tests -m "not distributed"`，不能据此报告分布式通过。详见 VALIDATION。

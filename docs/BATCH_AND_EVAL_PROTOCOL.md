# 四卡 batch 选择与空间评测协议 v2

本页优先于旧版单卡长样本测速和 image-only/16-token 评测配方。旧产物保留为诊断证据，不能与新协议的成绩混表。

## 1. 停训、测速、恢复

先确认完整 checkpoint 存在，包含 model、optimizer、scheduler、trainer_state 和每个 rank 的 RNG。停止当前监督器，再停止其 torchrun；不能按宽泛进程名杀任务。停在两个 checkpoint 之间会重算上次保存之后的步骤，不删除任何已保存 checkpoint。

```bash
export ROOT=/persistent/your-root
cd "$ROOT/EurekaSI"
python3 scripts/benchmark-ddp-batches.py --root "$ROOT" \
  --checkpoint "$ROOT/runs/sft-spar234k-hound64k/checkpoint-200" \
  --output "$ROOT/runs/diagnostic-ddp-batches-new" --detach
python3 scripts/resume-tuned-study.py --root "$ROOT" \
  --selection "$ROOT/runs/diagnostic-ddp-batches-new/selection.json" --detach
```

将 checkpoint 路径替换为本次真正完整的保存点；不保证示例 step200 在其他服务器存在。不得与旧监督器同时运行。

测速使用同一 checkpoint 的一次性模型副本，测试 micro-batch 1/2/4/8/16，GA 16/8/4/2/1，四卡有效 batch64。固定随机种子抽取同样的 1280 条混合训练样本，保存全部样本清单；按 rank stride 分配，所有候选每个 optimizer step 接触同一组64条。20步中前4步预热，后16步计时，再加一组最长真实样本做显存压力测试，不计入速度。

记录包括真实 DataLoader、前向/反向、梯度累积与同步、AdamW、每卡最大显存、数据等待。优化器为一次性新实例，预热后状态已分配；不冒充完整恢复训练。候选必须训练完成且峰值 reserved 显存低于容量92%，再按 samples/s 选最快者；OOM 不触碰正式 checkpoint。这里只是**当前实现和测量样本上的最优候选**，不是所有内核/packing/sampler 下的全局最优。

恢复器先运行两组预定义 ReVSI 格式 smoke，然后从原训练目录最新 checkpoint 恢复 SFT。保留全局 batch64 和一轮训练预算。Transformers 5.3 在关闭 auto_find_batch_size 时尊重新 micro-batch；按 `global_step × 新GA` 跳过已处理 microsteps。调整 micro-batch 后不声称逐位相同训练轨迹。

## 2. 为什么旧 5.04 分不能直接作能力结论

旧 ReVSI 全部6158条已评分，但2529道选择题中只有17条以有效选项开头；大量解释在16-token上限前没有最终答案。官方 lmms-eval 的默认16-token和首词评分本身没有被改错，旧文本提示也已对全量清单检查一致。问题是**未完成多图适配和回答格式验收，却把分数汇报为可信基线**。这是流程缺陷；不能从中推出模型只具备5分的空间能力。

同样，不能因为预计应该超过30分，就倒推正确答案、放宽数值容差、从推理中挑任意匹配标签，或在 test 上选择最高分配置。

## 3. 新的显式输入和输出契约

- 原生 Qwen video 输入：ReVSI 使用已发布32帧视频中的全部帧；VSI-Bench使用原视频均匀32帧。
- 保存源视频、原帧索引和原视频 FPS，交给 processor 生成时间戳及 video token。禁止默认猜 FPS，禁止隐式再次抽帧；断言 temporal grid 对应规定帧数。
- 最大图像边448，greedy，non-thinking，batch1，输出上限512。
- 主适配协议明确要求 `<answer>字母或数值</answer>`；原官方文字提示保持，额外答案格式要求公开记录。这是**适配协议**，不是未经验证的官方排行榜复现。
- 每条记录保存 raw response、decoded response、生成 token ID、是否长度截断、解析结果及方法、题型和输入 grid。

先做每题型首条的固定 smoke：native-video + 官方提示/16-token 用于诊断；native-video + answer-tag/512-token 为预先声明的主适配协议。格式门只看解析率≥90%、截断率≤5%，不看正确率。未通过不得全量跑出误导成绩，但不会取消已授权的 SFT。

## 4. 提取最终答案，而不是选择标准答案

`spatial_intelligence/answer_extraction.py` 的输入只有生成回答、候选选项和截断标志，**没有 ground_truth**。支持明确 answer tag、boxed、JSON answer 字段、明确最终答案语句、纯字母/数值或唯一的完整选项文本。未闭合 thinking、冲突候选、没有最终答案的截断文本明确弃权；不让 LLM judge 看到标准答案，不从任意推理数字中取最接近答案的值。

同时保存：

1. **strict**：官方首词提取和评分算术作用于当前 raw response。它是格式敏感诊断，不等于官方推理配置的 leaderboard 分数。
2. **extracted**：固定、盲于金标的最终答案提取之后，仍使用官方 MCQ / MRA 容差与题型聚合规则。

数值保留回答给出的数，不进行猜测性的单位换算。两个分数均标明0–100尺度。新旧协议不能混用；SFT前后必须精确匹配样本 ID、帧/分辨率、提示、输出预算及解析器。

## 5. VSI-Bench 不是 VSI-590K

正式测试资源来自固定 revision 的 `nyu-visionx/VSI-Bench`，由 `configs/spatial-eval-assets.json` 下载；VSI-590K 是另一个训练集，不可替代测试。

新测试集的两个 parquet 合计5130条、唯一 ID；报告 full 和 debiased v1（2362条）两份结果，不把 v1 说成完整集。准备时审计与 SFT 的场景交叉并另报无交叉子集，近重复和基座预训练暴露依然未知。当前服务器该场景 ID 交叉数为0。

```bash
python3 scripts/fetch-study-assets.py --root "$ROOT" \
  --catalog configs/spatial-eval-assets.json --detach
"$ROOT/envs/qwen35/bin/python" scripts/prepare-vsibench.py --root "$ROOT" --detach
# 上面的恢复器会在 SFT 完成后依次评测两个数据集的 baseline / SFT。
```

从全新服务器部署，原入口 `scripts/start-qwen35-study.sh` 也已切换到四卡混合测速与新评测协议。旧 `study eval` 保留作历史复算，不是推荐入口。

## 6. 当前验收与限制

2026-09-20 四卡固定混合样本实测（峰值为 PyTorch reserved，不等于 nvidia-smi 进程占用）：

| 每卡 micro-batch | GA | 全局 samples/s | 四卡最大 reserved GiB | 结论 |
|---|---:|---:|---:|---|
| 1 | 16 | 10.927 | 23.45 | 当前最快安全候选，恢复正式训练 |
| 2 | 8 | 10.080 | 24.63 | 比 1 慢 |
| 4 | 4 | 6.844 | 27.57 | 比 1 慢 |
| 8 | 2 | 未完成 | — | CUDA OOM，保留失败日志 |
| 16 | 1 | 未测试 | — | 因 8 已 OOM，跳过 |

不是 batch 越大越快；可变长多图输入的 padding、计算内核与通信都影响吞吐。未进行 packing/新内核的受控优化，不声称已达硬件极限。OOM 时其他 rank 曾卡在同步等待；脚本现检测 OOM 并终止该候选进程组，同时设 15 分钟上限，避免无效占卡。

原生视频 GPU smoke 暴露 Transformers 5.3.0 的 `get_rope_index` 缺少按时间戳拆分 grid，导致 StopIteration。`qwen35_video_compat.py` 仅在该版本回补上游相同操作；视觉 encoder 的原始 grid 和像素不变，不升级运行中的训练环境。单独回归检查修复前报错、修复后位置索引及原 grid 不变。[上游实现](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py)。

修复后 ReVSI 固定每题型首条共 13 条的真实 GPU smoke：解析 13/13、截断 0/13、strict 30.00、extracted 30.71。VSI-Bench 对应 10 条 smoke：解析 10/10、截断 0/10、strict 48.33、extracted 49.58。**这是格式验收，不是完整 benchmark，也不能与旧全量 5.04 作提升比较。** 两组 smoke 均单 GPU、allocator 上限为该卡15%，已退出。SFT 从 checkpoint200 恢复，未使用测速产生的临时权重。完整新协议 baseline/SFT 仍待自动队列完成。

若需替换监督器而不重启健康训练，可先核对并停止旧监督器（不停止其训练子进程），再使用 `resume-tuned-study.py ... --adopt-training-pid <verified-torchrun-pid> --detach`；它核对命令行、等待训练完成凭据，再执行格式验收及配对评测。禁止同时运行两个监督器。

- 答案提取27项回归、VSI官方算术/聚合对照及金标不可见测试合计29项通过。
- 原生32帧 CPU processor 检查已通过，实际 video grid 的 temporal维为16（每两个原帧一组），不是丢了一半视频；带有真实源视频时间戳。
- 全量训练后结果、原生生成 smoke 和新的完整 baseline 以实际运行产物为准；没有预先承诺分数。
- 旧多图/16-token 的约30分钟评测 ETA 不再适用于512-token新协议，要按新输出长度实测。

依据：[ReVSI 官方任务配置](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/1cd474f858a3055407d44a4b823e3d6d3299bbe7/lmms_eval/tasks/revsi/_default_template_yaml)、[VSI 官方评分](https://github.com/vision-x-nyu/thinking-in-space/blob/51e089c3ae69b9435e9489058610f5b3964c56a8/lmms_eval/tasks/vsibench/utils.py)、[VSI-Bench 数据版本](https://huggingface.co/datasets/nyu-visionx/VSI-Bench/tree/bdcadb3fea447621a828a24911801faba3587c12)。

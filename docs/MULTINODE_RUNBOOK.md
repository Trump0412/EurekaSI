# 多服务器并行实验：独立写入，共享只读输入

后续纠正：读取旧八卡SFT的 `training_args.bin` 确认 seed/data_seed 实际均为 **3407**，旧目录/CLI的3408未生效；不能称作独立seed重复。以下3408为历史意图而非实际配置。新Qwen3-VL＋VGGT对照使用已验证的3407，见 [几何SFT指南](GEOMETRY_SFT_RUNBOOK.md)。

## 最新实测补充（2026-09-20 13:21，中国时间）

13:26更新：原生 VERL gate 已 **accepted**。两个真实prompt组均有组内奖励差异，actor梯度有限非零，488个参数张量发生数值变化；导出后重载8帧图像推理成功。导出的额外 `lm_head.weight` 经配置 `tie_word_embeddings=true` 和逐位相等检查，确认为词嵌入别名，而非缺失/错配参数。CPU核验限制为8线程，避免大量小张量比较时过度并行。此验收证明链路更新与可重载，不证明空间任务指标提高。

13:29更新：正式原生队列 `runs/qwen3vl-spatial-gspo-v4` 已进入 `running_native_gspo`（GPU0–3、100步、prompt batch16、group4）；此状态仅表示训练器已启动，首个正式loss仍待产出。旧v1/v2/v3失败/阻塞产物保留，不覆盖。训练前后128条native验证及每25步checkpoint自动进行；最终checkpoint独立重载128条尚不是此正式脚本的验收项，不得与gate的重载验收混淆。正式ETA需热态步骤计时，不能把双卡8帧冷启动gate直接外推。

SFT 节点与 RL 节点分别使用独立的 `$NODE_ROOT`，链接到共享持久盘的不同节点目录；仓库、环境和输出隔离。原有基线节点任务未由本次部署改动。真实主机映射、连接地址和用户目录只保存在本地私有配置中，不进入公开仓库。

- SFT 节点：正式 SFT 已到 464/4655，loss 0.484659、grad 0.910156，实测剩余约 4.43 小时；checkpoint 已按每100步保存。训练后自动执行锁定协议 ReVSI，评测额外预留20–35分钟，仍待实测。
- RL 节点：Qwen3.5 reference GSPO 与配套 SFT 对照均完成100步及权重重载评测；它们不是原生 VERL 训练。
- RL 节点：原生 VERL gate 已真实完成一步并保存 model/optimizer checkpoint；grad 56.4151。该步407秒（含保存），其中生成93秒、old log-prob145秒、actor更新153秒。此前静默不能据此认定卡死。验收脚本尚待修复/完成重载验收，正式队列不能提前宣称运行。

| 同一128条场景隔离验证集 | 正确数 | 准确率 | 有效 Yes/No 输出率 |
|---|---:|---:|---:|
| 原始 Qwen3.5-2B | 49 | 38.28% | 60.94% |
| reference GSPO，100步 | 78 | 60.94% | 100% |
| SFT 对照，100步 | 86 | 67.19% | 100% |

Always-No 基线为65/128（50.78%）。两种训练均为400次prompt抽样，RL额外生成1600次rollout，计算预算不相等。单seed、小验证集，不是ReVSI正式成绩，不能据此判定算法普遍优劣。

配对分析显示：原模型输出合法的78题，正确数49→51；原模型输出非法的50题，正确数0→27。因此RL净增29题中27题来自原先非法输出子集。该事后分层提示格式遵循是重要因素，但不构成因果消融。SFT更高的点估计也不支持此小实验中“RL优于直接监督”的结论。

证据分别位于 `runs/qwen35-spatial-gspo-v1/{comparison,paired-format-analysis,completion}.json` 与 `runs/qwen35-spatial-sft-control-v1/{comparison,completion}.json`。逐题预测和固定manifest保留；不得用验证集挑选最佳checkpoint。

本页记录 2026-09-20 的扩展方案。初始化与算法说明见 [服务器指南](SERVER_RUNBOOK.md)、[后训练指南](POSTTRAINING_RUNBOOK.md)。安装完成、GPU 分配、首步训练、完整结果是不同状态；以实际进程和产物为准。

## 本轮实验分工

| 节点角色 | GPU | 任务 | 科学意义与限制 |
|---|---|---|---|
| 原四卡节点 | 4 × A100 40GB | seed 3407 完整 SPAR/Hound SFT 主线 | 保留其 checkpoint，不在其他节点覆盖或擅自续启 |
| 新八卡 SFT 节点 | 8 × A100 40GB | seed 3408，同数据、1 epoch、global batch64、冻结视觉 | 独立 seed 重复，辅助判断稳定性；4/8卡和DDP尾部补齐有差异，不宣称逐位可复现 |
| 新八卡 RL 节点 GPU0–3 | 4 × A100 40GB | Qwen3-VL 原生 VERL GSPO | 先两卡真实图像闭环门，再四卡空间 Yes/No 小实验 |
| 新八卡 RL 节点 GPU4–7 | 4 × A100 40GB | Qwen3.5 原生 reference GSPO | 使用 EurekaSI 公共训练器，不冒充 VERL 对 Qwen3.5 的支持 |

两个 RL 分支分别比较自身原始权重与训练后权重，不能把不同 backbone/引擎的结果直接归因于算法。独立 root、进程锁、端口与输出目录；不执行 dummy 计算占卡。

在双卡 VERL gate 期间，GPU2–3 暂时运行 `scripts/run-spatial-sft-control.py`：与 Qwen3.5 reference RL 相同数据、初始模型、LoRA、LR、100步和global prompt batch4的监督训练对照（2卡×GA2）。它也重载权重评估同128条，复用已完成的原模型基线。两者训练目标和rollout计算预算不同；这不是“纯格式”消融，而是比较直接监督是否同样足够。分析结合准确率、balanced accuracy和valid_yes_no_rate，不将输出格式改善全部解释成空间推理改善。原生四卡RL有显存空闲等待门，必须等GPU2–3释放后再启动。

## 新机器接入

1. 先读 SSH 配置与 GPU UUID，确认别名没有指向同一 GPU；查现有进程，不能把显存占用当作训练成功。
2. 用 `findmnt -T <目录>` 确认哪些目录是持久共享盘。用户指定的工作入口可以是指向该节点持久 root 的符号链接，但先检查原入口是否存在，禁止覆盖已有目录。
3. 每个节点在自己的 root 从 GitHub clone：

```bash
git clone https://github.com/Trump0412/EurekaSI.git "$NODE_ROOT/EurekaSI"
```

4. 可只读复用已验证的公共权重、媒体、标记图片。训练/评测 manifests 复制快照并审计；诊断只读清单可逐文件链接，避免整个 manifests 目录链接后误写公共文件。不要复用共享 state/runs/锁/日志作为新节点状态。
5. 独立 Conda prefix；不能在共享原环境中 `pip install`。若已有兼容环境，可复制后重做 editable 安装：

```bash
"$CONDA_BIN" create -y --copy --prefix "$NODE_ROOT/envs/qwen35" \
  --clone "$SOURCE_ROOT/envs/qwen35"
"$NODE_ROOT/envs/qwen35/bin/python" -m pip install --no-deps -e "$NODE_ROOT/EurekaSI"
"$NODE_ROOT/envs/qwen35/bin/python" -m pip check
```

`--copy` 避免通过硬链接编辑源环境文件。仅当本机 Conda 包缓存齐全时才使用 `--offline`；共享 prefix 可见不等于本机包缓存齐全。本轮 RL 节点即遇到 offline 缺包，保留已有文件后改为从原镜像联网补齐。新建环境可使用现有 bootstrap 脚本和镜像源。

## 八卡 SFT 队列

入口 `scripts/run-node-sft.py`：克隆环境结束 → pip/check与CPU回归 → manifest副本和场景隔离审计 → micro-batch 1/2/4 实测 → 八卡两步烟测 → seed3408一轮训练 → 同协议ReVSI → 对比记录。

计划样本 297899（SPAR234149、Hound63750），global batch64，`GA=64/(8×micro)`，约4655 optimizer steps。保持模型、标记图、帧顺序、LR与原主线相同。八卡 ETA 必须用正式步骤实测；不能把四卡7–11小时机械承诺为八卡一半。

SPAR 原始 source ID 并不唯一；使用固定源文件的过滤前行号标识，同时保存 source_id/source_index。重复 ID 修复不能变成删掉有意义训练样本。正式 gate 审计 manifest 唯一性、源索引与无 ReVSI 场景交叉。

## 空间 RL 小实验

`scripts/prepare-spatial-rl-probe.py` 默认从已验证 SFT 来源中筛选 **Yes/No 空间关系**，保留原始1/3帧及图像标记，固定 seed3407 按 scene 分开后取512训练、128验证。候选1184条、207场景；不因样本不足自动放松约束。保存所有ID、场景、来源与过滤原因。

```bash
python3 scripts/prepare-spatial-rl-probe.py --source-root "$SOURCE_ROOT" \
  --output "$NODE_ROOT/manifests-local/spatial-boolean-v1" --train-count 512 --val-count 128
python3 scripts/run-qwen35-spatial-rl.py --root "$NODE_ROOT" --source-root "$SOURCE_ROOT" \
  --gpus 4,5,6,7 --steps 100 --detach
```

使用**原始 VLM**初始化，不能从完整 SPAR SFT 权重开始后再称这128条未见：它们本来就在原SFT来源中。基础模型预训练暴露及近重复图像仍未排除。

Qwen3.5 reference 分支：语言 LoRA rank8（q/v）、视觉冻结，GSPO group4、每卡一个prompt、global prompt batch4、LR1e-5、clip0.0004、KL系数0.02，64新token。奖励只看训练答案正确性，不奖励答案格式，也不把金标注入学生输入；记录每组真实response/reward。8步 gate 必须有组内奖励差异、非零有限梯度，之后从同一原始模型重新启动100步pilot。100步是400次有放回prompt抽样、1600次rollout，**不是完整1 epoch或正式benchmark**。

训练前后使用同128条验证、相同64token greedy解码；保存逐题记录、总体准确率、Yes/No分层、always-No基线及balanced accuracy，以排除仅学习标签先验。验证结果不得用于事后挑最优checkpoint。最后一次权重重载推理也是阶段门。

公共 reference trainer 目前保存adapter、不保存可精确续训的优化器状态。短pilot失败保留日志，以新run名重跑；不能把adapter重新加载称为断点续训。正式SFT的Trainer checkpoint则包含optimizer/scheduler/RNG。

## 只读跨节点状态

```bash
python scripts/fleet-status.py \
  --node train-host /persistent/node-sft \
  --node rl-host /persistent/node-rl
```

脚本通过本机现有 SSH 别名并行读 GPU、进程与状态，关闭agent/X11转发，不持久化凭据。输出同时报告状态文件时间与PID是否存在：`running` 文件不能单独证明进程还活着。

已有 ReVSI 主线的16-token协议存在解释性输出被截断的风险；只能支持该锁定协议下比较，不能直接当模型空间能力上限。后续若做更长输出预算实验，必须同时重测原模型和训练模型，使用新run名，不能修改已完成主表。

## 部署验收快照（2026-09-20 12:54，中国时间）

- 八卡 SFT：独立环境 pip check通过；真实 batch1/2/4 测速选择每卡1、GA8。两步DDP烟测有限loss并保存checkpoint/final；正式训练31/4655，loss1.549558、grad norm2.140625，显存约27.2–30.8GB/卡。近期3.5–3.8秒/step，保守训练预留5–6小时，后续复测预留20–35分钟（八卡评测时长尚未实测）。
- 全量 manifest 独立逐行审计：问答、source_index/source_id、原帧顺序和计数一致；297899保留+128排除完整覆盖来源；未因重复source ID删除样本。
- RL 子集已生成512/128，scene交叉为0。两种Yes/No题型再次逐像素核对原renderer与实际manifest一致：多帧题的首JPEG可能本来不需要标记，不能按扩展名认定缺框。
- Qwen3.5 reference GSPO四卡gate于12:56完成8步：32个prompt组中16组奖励非恒定，8步均非零有限梯度，末步grad0.66218。已转入同128条验证的原模型基线，随后自动100步pilot与权重重载复测。监督器在环境重装后保留了旧editable导入路径，已显式优先本checkout并恢复；已完成gate未重训。
- 原生VERL独立环境复制后还修复setuptools/Pillow依赖约束，具体进度见当前state；不把后台等待器称为训练运行。
- RL objective、reward审计、split、类别平衡指标、fleet命令测试在新节点隔离Qwen环境 **21 passed**。Windows未装torch，不用于证明GPU/目标环境运行通过。

以上是时间点状态，不是最终结果。`scripts/fleet-status.py` 与各节点当前日志优先于本文静态快照。

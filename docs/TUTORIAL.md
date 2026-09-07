# 汇编教程（部分内容来自原交付）

当前操作顺序、安装版本和修订边界以 [QUICKSTART](QUICKSTART.md)、[COMPATIBILITY](COMPATIBILITY.md) 和 [READINESS_REVIEW](READINESS_REVIEW.md) 为准。原交付的历史实验记录不代表本轮运行结果。

> v0.2 首次部署请优先按 [QUICKSTART](QUICKSTART.md) 的固定版本脚本安装；本文保留详细原理与配置说明。

# 从数据到空间推理实验

所有命令在项目根目录执行。配置中的相对路径也以当前工作目录为基准。以下 `/path/to/...` 是需要替换的本地路径，不是随包提供的数据。

## 1. 选择运行路径

**复现已有论文**：使用 `legacy`，在原方法环境中调用原训练脚本。不要同时更换预处理、prompt、优化器和训练环境后再把变化归因给方法。

**研究 SFT/RL/蒸馏的因果贡献**：使用公共 `train`。它的处理逻辑更透明，适合先验证单卡小实验。v0.2 已加入同步数据并行；GPU 与 batch 配置及测试边界见 [DISTRIBUTED](DISTRIBUTED.md)。

**比较模型推理能力**：统一清单、图像预算、prompt、解码策略和 scorer，调用 `infer`。只改变模型路径、checkpoint 或明确的模型后端。

本版每个 rank 逐样本执行，`train.batch_size × gradient_accumulation × world_size` 决定全局 prompt batch；torchrun 下同步梯度、参数与日志。不是动态张量 batching，也没有优化器分片。

## 2. 环境与安装

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[train]'
python -m pip install -e '.[video,vsi]'
python -m pip install pytest
python -m pytest -q tests
```

真实 GPU 运行应先安装与服务器驱动匹配的 PyTorch。新包不依赖 FlashAttention 或 DeepSpeed；原 GeoBridge 路径则保留自己的版本要求，包括 Python 3.10。CPU 验证环境与原 GPU 环境不是同一个依赖组合。

先检查模型类是否被 Transformers 支持，再选 checkpoint。当前 HF 图像后端允许 Qwen2-VL、Qwen2.5-VL、Qwen3-VL 的 dense 类型，以及 Transformers 原生 InternVL、LLaVA、LLaVA-NeXT 类型。后几类使用标准 AutoProcessor 接口，尚未用真实权重验收；不包含历史 remote-code 格式。列表之外的家族会报错，不会当文本模型加载后静默丢图。

## 3. 统一数据清单

一行一个样本，使用 JSONL：

```json
{"dataset":"DSR-Bench","id":"scene12_q03","split":"test","scene_id":"sourceA/scene12","media":["/data/frame0.png","/data/frame1.png"],"frame_indices":[0,30],"timestamps_s":[0.0,1.0],"question":"Where is the red cup relative to the box?","choices":{"A":"left","B":"right"},"answer":"A","task":"relative_position","metric":"choice"}
```

| 字段 | 约定 |
|---|---|
| `dataset` + `id` | 共同构成唯一键，保留数值零 ID |
| `split` | train/val/test，训练入口只接收 train |
| `scene_id` | 尽可能保留上游场景标识；同一真实场景的别名应归一 |
| `media` | 有序的已解码帧或多视角图像；不接受隐式视频解码 |
| `frame_indices`, `timestamps_s` | 如提供，其长度必须等于帧数 |
| `choices` | 显式标签映射；多选题 gold 必须是标签，而非猜测的索引 |
| `answer` | 训练/评分需要；纯推理可为 null；数值 0 不会丢失 |
| `task` | 子任务分类，用于按任务聚合 |
| `metric` | 通用 choice / numeric / exact；benchmark 正式指标由官方插件补充 |
| `metadata` | 数据来源、教师输出等；默认不传给推理模型 |

模型推理只收到问题、选项、媒体和帧信息，不收到 answer 或任意 metadata。训练器在确实需要监督、reward 或 privileged teacher 时单独读取标签。

### 3.1 转换已有清单

支持常见 `sample_id/clip_id/qid`、`frame_paths/images/media_paths`、`target/label` 别名：

```bash
python -m spatial_intelligence convert \
  --input /data/spar-source.jsonl --output /data/spar.train.jsonl \
  --dataset SPAR-7M --split train --media-root /data/media
```

对单轮 LLaVA 风格 conversations：

```bash
python -m spatial_intelligence convert \
  --input /data/llava-frames.json --output /data/llava.train.jsonl \
  --dataset LLaVA-Hound --split train --media-root /data/media \
  --adapter spatial_intelligence.benchmarks:conversation_qa
```

这里要求提前将视频对应到 `frame_paths`；多轮对话不能被悄悄截成一轮，会报错。不同来源数据的字段不能仅靠名称猜测；可新增 `module:function` 转换函数。

### 3.2 视频采帧

```bash
python -m spatial_intelligence frames \
  --video /data/scene12.mp4 --output /data/frames/scene12-f8 --count 8
```

输出 PNG 和 `frames.json`，包含原视频 SHA256、采样索引、fps 和时间记录。少于 8 帧时不会人为复制帧。将这份帧列表提供给所有模型，防止每家模型各自解码、各自采到不同画面。

默认采用首尾均匀采样。对于可变帧率视频，`index/fps` 不等于精确 PTS；正式 benchmark 若给出时间戳，应采用官方解码/采帧结果，再写入统一清单。不要称默认 8 帧协议为所有 benchmark 的官方协议。

多视角任务还需要注意：独立相机顺序与时间顺序不是一回事。不要对本来无时间语义的图像列表自动解释为运动过程。

### 3.3 泄漏检查

```bash
python -m spatial_intelligence audit \
  --train /data/spar.train.jsonl /data/llava.train.jsonl \
  --test /data/dsr.test.jsonl /data/mmsi.test.jsonl \
  --output /data/audit.json
```

检查 dataset::id、场景 ID、媒体字节、问题+媒体重叠。存在重叠时返回失败。新训练器还会自动做同一检查，并写入实际训练/测试清单的内容指纹。

这是一道必要检查，不是无泄漏证明。相似视频、同一场景不同压缩版本、其他模型预训练数据污染还需要来源清查。若某 benchmark 官方允许共享场景，先单独设计并记录该协议；当前严格训练入口默认不放行重叠。

## 4. SFT

复制 `configs/sft.yaml`，替换模型、数据与输出路径。核心参数：

```yaml
data:
  train:
    - {path: /data/spar.train.jsonl, weight: 0.8}
    - {path: /data/llava.train.jsonl, weight: 0.2}
  heldout: [/data/dsr.test.jsonl, /data/mmsi.test.jsonl]
  eval: /data/dsr.test.jsonl
train:
  mode: sft
  steps: 1000
  gradient_accumulation: 8
  learning_rate: 0.00001
```

这里只展示修改字段，实际配置请保留完整 YAML。`weight` 是抽取数据源的概率权重，不是把 80% 文件永久截出来；选中数据源后均匀抽样，允许有放回采样。运行日志记录实际抽样次数。

```bash
python -m spatial_intelligence train --config configs/sft.yaml
```

监督目标为 `<answer>标准答案</answer>`，仅 completion token 和 EOS 计算交叉熵。视觉 token、问题和 prompt 前缀不参与监督。超过上下文长度直接报错，不静默截断图片或答案。

默认 LoRA。`model.lora.enabled=false` 可切换原生 HF 全参训练；这会显著增加显存，需在实验表明确全参/LoRA 和可训练参数量。LoRA dropout 固定为 0，避免同一参考实现中在线 rollout 与更新概率不一致。

有效全局 prompt batch 是 `batch_size × gradient_accumulation × world_size`；若要与原论文全局 batch 对齐，不能只对齐 YAML 的 batch 名字，还要检查每步样本数、累计训练 token、图像数量和更新次数。

### 4.1 恢复已有结果用于推理或新训练

LoRA 训练结果：保持 `model.path` 为原始基础模型，将 `model.adapter` 改为 `runs/.../final`。

全参训练结果：将 `model.path` 改为 `runs/.../final`，`model.adapter=null`。

这是加载模型权重继续做新的实验。**本版没有精确中断续训**：checkpoint 未保存 optimizer、调度器和全部采样器状态；不能把加载权重说成完全恢复原轨迹。

### 4.2 复用已有 GeoBridge/GeoWire SFT

```bash
python -m spatial_intelligence legacy --config configs/legacy-geowire-sft.yaml
python -m spatial_intelligence legacy --config configs/legacy-geowire-sft.yaml --execute
python -m spatial_intelligence legacy --config configs/legacy-geobridge-sft.yaml --execute
```

`legacy` 展示的 cwd 是对应方法源码根目录，传给原脚本的数据、模型和缓存建议用绝对路径。GeoBridge 示例通过 env 选择原虚拟环境、DATASETS、GPU 和 Stage 1 checkpoint；dataset registry 与原数据路径约定仍由该仓库负责。

GeoWire 多卡时应保留原 DeepSpeed 参数、parity/readiness 检查和 QA/TIP 配比；包装器不会擅自从公共 YAML 猜测这些方法特有参数。

GeoPSRO 兼容 SFT 保留历史“文本+几何前缀”的行为。要研究 RGB 增强效果，应先明确要比较原始方法还是新的 RGB 版本。

## 5. RL：GRPO 与 GSPO

```bash
python -m spatial_intelligence train --config configs/rl.yaml
python -m spatial_intelligence train --config configs/rl.yaml \
  --set train.algorithm=gspo --set output=runs/gspo-qwen3
```

先在 YAML 中将 `model.path/model.adapter` 指向 SFT 结果。流程是：冻结初始 reference → 学生对同一问题采样 K 条 completion → 计算 reward → 分组标准化 → 优化裁剪策略目标与 reference KL。

对组内 reward：

\[
A_i=\frac{r_i-\bar r}{\max(\operatorname{std}(r),10^{-6})}.
\]

本版 GRPO 使用 token 级概率比，GSPO 使用 response 平均 log-ratio 的指数作为序列比率。每组只做一次更新，因此旧策略来自更新前的同一学生。没有 PPO 多 epoch、异步 rollout 或重要性修正队列。

| 参数 | 作用 |
|---|---|
| `group_size` | 每个问题生成几条回答；至少 2 |
| `algorithm` | grpo / gspo |
| `clip` | 概率比裁剪范围 |
| `beta` | 冻结 reference 的 KL 约束系数 |
| `generation.temperature` | rollout 采样温度；logprob 使用同一温度 |
| `gradient_accumulation` | 每次更新累积的问题组数 |
| `reward_plugin` | 自定义 reward 的 `module:function` |

默认 reward 是严格答案正确性。若全组都正确或全组都错误，优势为零；此时没有来自组内排名的学习信号。这是需要记录的数据/采样难度问题，不应误判为优化器失败。

参考实现要求 `top_p=1.0, top_k=0`，不支持截断采样、重复惩罚和 beam search 的训练概率修正。直接接受这些参数但仍计算原始 softmax logprob 会引入错误的 on-policy 假设，所以这里明确拒绝。

### 5.1 复用 PSRO reward

把 `legacy_sources/GeoPSRO` 加入运行环境的 `PYTHONPATH`，实现以下小适配函数：

```python
def psro_reward(response, sample, config):
    from geopsro4d.reward.reward_fn import compute_score
    details = compute_score(
        sample['dataset'], response, sample['answer'],
        {'choices': sample['choices'], 'answer_type': 'multiple_choice'}
    )
    return details['score']
```

将它放入可导入模块，并把 `reward_plugin` 设置成 `模块名:psro_reward`。根据数值题/方向题补齐 answer_type，不能把所有题强行设为 multiple_choice。PSRO 格式需要相应 instruction；换 prompt 就形成了新的协议分组。

训练用 reward 可以与正式评测 scorer 不同，但必须分别记录。关键词奖励上升不等于空间正确性上升。

## 6. OPD：蒸馏外部强模型

```bash
python -m spatial_intelligence train --config configs/opd.yaml
```

设置 `teacher.path` 为真实教师 checkpoint，`teacher.device` 可用另一块 GPU；设置 student 的模型/adapter 为当前学生。两个模型都需要可访问的本地权重或授权访问权限。

本版操作定义：学生在线生成 \(y\sim\pi_\theta(\cdot|x)\)，教师在同一回答前缀上输出完整词表分布，学生只在 response 位置优化 KL。默认：

\[
L=\mathbb E_{y\sim\pi_\theta}\left[\frac1{|y|}\sum_t D_{KL}\big(\pi_\theta(\cdot|x,y_{<t})\Vert\pi_T(\cdot|x,y_{<t})\big)\right].
\]

采样 token 本身按 stop-gradient 处理；这是可检查的 on-policy distillation 参考目标，不声称与所有论文中名为 OPD 的实现细节完全相同。

`kl_direction=forward` 改为教师到学生 KL；`distill_temperature` 控制蒸馏分布温度，loss 乘以温度平方；`sft_weight` 可添加真实答案 CE。教师不接收梯度。

必须满足：

- tokenizer 语义完全一致，而不只是 vocab_size 相同；代码核对 vocabulary、special tokens 和 tokenizer 实现配置。
- teacher 与 student 对同一个 completion token ID 评分；不同 prompt 长度可以，但回答前缀必须对齐。
- 教师冻结；记录 checkpoint 来源，避免将教师看到的测试数据误称为学生泛化。

完整 logits KL 的内存规模约为 response 长度 × vocabulary，外加模型与反向激活。2B 学生+大教师不等于单卡一定可放下；先短序列小步测试。当前未实现 top-k 蒸馏近似、教师 RPC 或分片全词表 KL。

## 7. OPSD：自蒸馏实验

```bash
python -m spatial_intelligence train --config configs/opsd.yaml
```

本版 OPSD 明确定义为 **on-policy self-distillation 的答案条件变体**：教师从学生初始化状态复制并冻结；学生在线采样；`privileged_answer=true` 时，教师 prompt 额外获得训练集标准答案，学生 prompt 不获得该信息；随后对同一回答前缀进行 KL 对齐。

这是有意使用训练标签构造 privileged teacher，不是测试时把答案提供给模型。推理入口永远不传标准答案。

`teacher_refresh_steps=0` 表示始终冻结初始教师；正数表示每 N 步硬更新为学生快照。没有 EMA 更新。关闭 privileged answer 且教师、学生完全相同时，初始 KL 应为零，不会凭空产生新能力。

OPSD 缩写可能在不同论文中指不同方法。如果要复现某篇指定论文，必须进一步对齐它的教师条件、刷新策略、KL 方向、采样来源和目标函数，不能只因为都叫 OPSD 就认为一致。

## 8. 异构教师：离线蒸馏

不同 tokenizer、仅提供文本的 API 教师，不能直接按 token ID 计算 KL。使用教师输出作为文本监督：

```bash
python -m spatial_intelligence teacher-export \
  --config configs/inference.yaml --output /data/teacher.train.jsonl
python -m spatial_intelligence train --config configs/offline_kd.yaml
```

第一步需把 inference 配置的 `data.eval` 指向 **train split** 清单，model 指向教师。它将教师输出写入 `metadata.teacher_response`，保留原 answer，不伪装为人工标签。第二步将训练数据路径改为该输出。

本版导出命令通过已实现后端访问教师。只提供 API 的其他模型仍需自行接入该服务的后端，当前不包含商业 API 调用代码。

离线教师数据也需要质量控制：至少区分正确答案、可解析答案和未经筛选输出；如果筛选仅保留教师答对的样本，应记录保留率和任务分布变化。本版不默认过滤，避免不透明地改变数据量。

## 9. 推理、benchmark 与公平对比

```bash
python -m spatial_intelligence infer --config configs/eval-dsr.yaml
python -m spatial_intelligence infer --config configs/eval-mmsi.yaml
python -m spatial_intelligence infer --config configs/eval-vsi.yaml
```

每次输出 `run.json`、流式 `predictions.jsonl`、`metrics.json`、逐题 details。run 中记录完整配置、环境、数据内容指纹、协议指纹、预测文件指纹；预测记录还包括输入/输出 token 数、耗时和帧索引。

HF tokenizer/processor 会根据模型原生结构进一步 patchify 图像。统一图像尺寸上限不意味着各家得到相同视觉 token 数：需要同时报告输入像素预算和实际 token 数。当前记录的是完整输入 token 数，尚未统一提取各家单独的视觉 token 数。

### 9.1 VSI 正式指标

```bash
python -m pip install -e '.[vsi]'
python -m spatial_intelligence convert \
  --input /data/vsi-with-frames.jsonl --output /data/vsi.test.jsonl \
  --dataset VSI-Bench --split test \
  --adapter spatial_intelligence.benchmarks:vsi_row
```

原数据需要保留 `id,dataset,scene_name,question_type,ground_truth,options`，并添加已解码的 `frame_paths`。`eval-vsi.yaml` 使用直接答案 prompt，并调用包内固定 commit 的官方 scorer；不要给这个 scorer 使用 XML 格式答案。

官方分数在 `metrics.json` 的 `official.metrics`，其中 `overall` 以百分数表示。顶层 `accuracy` 仍是内部通用指标，不能拿它当 VSI 正式总分。官方 scorer 要求完整子任务集合；只抽取部分任务时用内部指标，不声称完整 VSI overall。

固定评分源代码并不固定采帧预算。论文汇报应写“官方 scorer + 本文统一 8 帧协议”，或者完整复现官方模型配置后再声称官方协议一致。

DSR/MMSI/ViewSpatial/ReVSI 当前提供统一清单推理与研究指标。原始清单转换、成对/一致性指标及正式聚合仍需按所用版本的官方代码接入 `score.official_scorer`；本版没有冒充全部已完成。

### 9.2 同一 benchmark 比较多模型

```bash
python -m spatial_intelligence infer --config configs/inference.yaml \
  --set model.path=/models/Qwen3-VL-2B-Instruct --set output=runs/qwen3-dsr
python -m spatial_intelligence infer --config configs/inference.yaml \
  --set model.path=/models/Qwen2.5-VL-7B-Instruct --set output=runs/qwen25-dsr
python -m spatial_intelligence compare \
  runs/qwen3-dsr/metrics.json runs/qwen25-dsr/metrics.json \
  --output runs/dsr-comparison.json
```

合并前核对协议、数据、scorer 的指纹。改变帧数、prompt、输出预算或评分容差后，需要分组报告，不能直接并入同一公平对比表。Mock 输出禁止进入模型对比。

`compare` 的基础表展示通用 accuracy，并保留 `official` 字段；benchmark 正式得分应从该字段单独汇报。不能跨 benchmark 直接比较 overall，必须先定义任务归一和聚合方式。

公平比较至少有两条轨道：

| 轨道 | 固定项 | 允许变化 |
|---|---|---|
| 原生能力 | 数据、帧、prompt、输出预算、scorer | 原厂模型架构与权重 |
| 后训练效果 | 同一初始模型、数据来源、预算、评测协议 | SFT/RL/OPD/OPSD 的算法与目标 |

不同模型应报告参数量、是否经过额外空间预训练、几何教师成本、输入 token 数、延迟、训练预算、蒸馏教师成本。不能仅按“训练 1000 step”宣称计算公平。

### 9.3 参数组合

```bash
python -m spatial_intelligence sweep --config configs/rl.yaml \
  --grid configs/sweep-rl.yaml --output runs/rl-grid
python -m spatial_intelligence sweep --config configs/rl.yaml \
  --grid configs/sweep-rl.yaml --output runs/rl-grid-executed --execute train
```

网格默认仅写配置，方便先审阅数量和预算。执行模式按顺序运行，出现失败即停止。未知配置键和拼写错误会报错。给每个种子保留独立输出目录，不覆盖之前实验。

## 10. GeoWire 图缓存与原生后端

使用 `configs/inference-geowire.yaml`。LoRA rank/alpha/target_modules 必须和原训练一致；`options.phase=2` 加载组合的 geowire/qwen_trainable checkpoint，phase=1 则只加载图传输权重。

缓存路径采用 `cache_root/dataset/id/`。除原 `graph_coo.npz` 外，需要构图时记录的 `graph_contract.json`：

```json
{"media_sha256":["每张输入帧的SHA256"],"frame_indices":[0],"image_max_side":448,"image_grid_thw":[[1,32,32]],"graph_sha256":"graph_coo.npz的SHA256"}
```

这里的数字仅用于展示结构，不能照抄。契约必须来自实际构图输入和 processor 输出；对旧缓存，先核实帧顺序、resize/crop、grid 和坐标约定后再迁移。事后随意补一份相同尺寸记录，不能证明图节点真的对齐。

统一适配器会拒绝缺失或不一致的契约，避免错用其他视频/其他 resize 的缓存。当前只支持 normal 条件；几何 zero/shuffle 干预尚需在构图层显式实现。

## 11. 一次新服务器验收的最小顺序

1. 运行 CPU tests，检查入口和配置。
2. 取真实模型及 2–8 个真实样本，跑 baseline 推理，核对图像确实被读取、输出可解析、标签不进 prompt。
3. 用同样样本核对 SFT loss 与 HF labels 的位置对应、checkpoint 重载结果。
4. RL 检查组内 reward 方差和梯度；OPD 检查 tokenizer 对齐、教师无梯度；OPSD 检查 privileged answer 只在教师分支出现。
5. 对几何方法做 zero-adapter parity、frame/grid/cache 一致性检查，再使用完整数据。
6. 固定环境和数据清单，运行多种子实验。硬件上完成这些验收前，不把 CPU 单元测试描述为真实模型性能复现。

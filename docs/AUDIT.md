> 本文件保留原交付的审查背景；当前静态修订与未完成项见 [READINESS_REVIEW](READINESS_REVIEW.md)。

# 三个实验仓库审查

审查依据为实际拉取的源码，不依据仓库标题推断功能。你的服务器实验已经能运行；以下结论限定于本次读取的 GitHub commit，服务器工作目录可能包含未提交代码或外部训练入口。

## 1. 冻结的源码与方法对应

| 仓库 | commit | README 中的方法名 | 当前核心能力 |
|---|---|---|---|
| GeoWire | `d427cc7dd849e96682a8153fa6e2d5a30fd348e2` | Georoute | 稀疏跨帧图、TIP、图控制语义传输、QA/TIP 交替 SFT |
| GeoPSRO | `4d7db6e3e3dc02e5e9197c731a92bf8bacfc8c36` | gap-4D | VGGT 几何 token 对齐、几何前缀训练、PSRO reward、RL 数据准备 |
| GeoBridge | `74cfe79500e59d92af21ab4e7b1a746031a94711` | SpatialFit | FCP/continuity bank、异构几何桥接、Qwen2.5/Qwen3 的 Stage 2 |

名称曾发生变化。统一代码库用稳定的方法 ID 对接，保留 checkpoint 内部符号，不进行大范围类名替换。

## 2. 已复现并处理的问题

### A. GeoWire 生成入口递归：高优先级

位置：`geowire/geowire/models/qwen3vl_bridge.py` 的 `generate()`。

原实现将 `base_model.forward` 替换成调用桥接器 `forward()` 的函数；桥接器又调用 `base_model(...)`，于是再次进入被替换的 `forward()`。本次使用最小 `nn.Module` 复现 `RecursionError`。

修复：生成期间只临时包装 `get_image_features()`，复用与训练相同的图传输逻辑，并在 `finally` 恢复；不替换 `base_model.forward`。补充零门控生成一致性和异常后恢复测试。

补丁：`patches/geowire-generation.patch`。新代码仍通过临时方法替换实现，单实例不适合多个线程并发请求；服务化时应转为实例级正式模块插入。

源码依据：[GeoWire bridge](https://github.com/Trump0412/GeoWire/blob/d427cc7dd849e96682a8153fa6e2d5a30fd348e2/geowire/geowire/models/qwen3vl_bridge.py)。

### B. GeoPSRO 最终答案解析可能读到推理中的字母：高优先级

位置：`geopsro4d/reward/answer_parser.py`。原评分解析优先识别 `Answer:` 文本，没有优先提取 `<answer>` 标签，并在失败后取首字符。

复现输入：`<think>A is wrong</think><answer>B</answer>`，两个选项时原解析器返回 `A`。这是评分路径的问题；`reward_fn.py` 另有解析器，不应笼统称所有奖励都存在同一错误。

修复：优先唯一的 `<answer>`；移除 `<think>`；多个答案或无法明确解析的文本返回失败，不按首字母猜。保留直接字母和完整选项文本匹配。

### C. 缺失答案可被计为正确：高优先级

位置：`geopsro4d/eval/eval_dsr.py`。当标准答案缺失且预测也解析为 `None` 时，`None == None` 成立。最小复现一行 `{"id":"x","response":""}` 得到 accuracy=1.0。

修复：标准答案缺失直接报错；预测解析失败不能计为正确。统一评分同时检查重复 ID、缺失预测、额外预测及真实分母。

### D. 数值 0 的 ID/答案丢失：中优先级

位置：`geopsro4d/data/schema.py`。`row.get(...) or ...` 会把合法数值 `0` 当作不存在。已复现 ID=0 报缺失 ID；答案=0 也受相同逻辑影响。

修复：明确按 `None`/空串判断字段，保留零值。B/C/D 的补丁为 `patches/geopsro-scoring-schema.patch`。

源码依据：[parser](https://github.com/Trump0412/GeoPSRO/blob/4d7db6e3e3dc02e5e9197c731a92bf8bacfc8c36/geopsro4d/reward/answer_parser.py)、[DSR score](https://github.com/Trump0412/GeoPSRO/blob/4d7db6e3e3dc02e5e9197c731a92bf8bacfc8c36/geopsro4d/eval/eval_dsr.py)、[schema](https://github.com/Trump0412/GeoPSRO/blob/4d7db6e3e3dc02e5e9197c731a92bf8bacfc8c36/geopsro4d/data/schema.py)。

## 3. 能运行，但需要明确实验含义的入口

### GeoPSRO Stage 3 没有实际调用 verl

`train_stage3_rft.py` 的主路径读取数据，写 `psro_rft_train.jsonl` 和 `verl_launch_plan.json`，随后结束。`--backend verl` 只进入 plan 字段，没有创建 trainer、rollout worker 或优化器。

因此本次将该兼容入口命名为 `geopsro_prepare_rl`。新框架的 `train.mode=rl` 则包含实际生成、reward、group advantage、反向传播和 optimizer step。它是单进程参考训练器，不替代服务器已有的大规模 verl 运行脚本。

原 `configs/stage3_psro_rft.yaml` 中的多数参数也未由该脚本读取，脚本写出的 plan 使用硬编码默认值。修改 YAML 并不自动修改那条实际运行路径。

此外 plan 指向 `compute_reward`，而仓库提供的 verl 风格签名是 `compute_score(data_source, solution_str, ground_truth, extra_info)`。接入具体 verl 版本时应核对 reward manager 的调用签名，不能直接假设通用 wrapper 可用。

源码依据：[Stage 3](https://github.com/Trump0412/GeoPSRO/blob/4d7db6e3e3dc02e5e9197c731a92bf8bacfc8c36/geopsro4d/train/train_stage3_rft.py)。

### GeoPSRO 当前 SFT 主路径没有输入 RGB

`run_qwen_smoke()` 是真实入口调用的函数，名字中的 smoke 不等于完全模拟；它会加载真实 Qwen 和优化器。但 `_make_training_example()` 只编码 question/answer，再拼接几何 embeddings，没有读取 `sample.media_paths`、没有 `pixel_values`，也没有常规多图 chat template。

这条路径实际训练的是“文本 + 几何前缀”。若论文实验还包含 RGB，需回查服务器上真正使用的入口。不能把该路径直接当成“RGB VLM + 几何增强”的等价复现。

同一路径还没有在 question 后自动拼 choices；如果 manifest 的 question 尚未包含选项，会与统一多选题协议不同。新 HF 后端明确加载图像并显式拼接 choices。

源码依据：[Stage 2](https://github.com/Trump0412/GeoPSRO/blob/4d7db6e3e3dc02e5e9197c731a92bf8bacfc8c36/geopsro4d/train/train_stage2_sft.py)。

### “eval” 有些只负责评分

GeoPSRO `run_eval_dsr.sh` 接收 checkpoint 后只将其打印出来，没有加载该 checkpoint；`geometry_mode` 是评分结果中的标签，不会对几何输入施加干预。单改该参数不能形成几何消融实验。

GeoWire `scripts/evaluate.py` 同样给预测文件评分。`geowire/evaluation/eval_vsi.py` 则明确写着 scaffold 并退出。统一入口因此拆成 `infer` 和 `score`，且评分会核对生成记录。

## 4. 尚需在真实多卡环境核实的问题

这些是源码可见的风险，本次没有将它们写成“多卡已复现的故障”。

| 位置 | 风险与影响 | 建议 |
|---|---|---|
| GeoPSRO Stage 2 的 `seed_everything(3418+rank)` → `random.shuffle(usable)` → 按 rank 切片 | 各 rank 先得到不同排列，再各自切片，无法保证数据 shard 不重叠 | 用共同种子生成全局索引，再切片；epoch/样本种子独立 |
| `_average_gradients()` 对 `grad is None` 直接跳过 | 不同 rank 随机选择 full/zero 分支时，参与 collective 的参数集合可能不同 | 每个 rank 使用一致 collective 顺序；明确处理 unused 参数 |
| rank 本地发现非有限 loss 后直接 `continue` | 其他 rank 可能进入 backward/all-reduce 而等待 | 使用全局 finite 标志，所有 rank 同步跳过 |
| `_cached_samples()` 只取清单前缀中已有缓存的样本 | 得到的实际训练分布不一定等于完整清单或设定比例 | 输出实际样本 ID、抽样次数、缓存覆盖率 |
| `QwenVGGTWrapper.load_geometry()` 缓存缺失返回零几何 | 缺缓存和主动关闭几何可能混在一起 | 正式实验默认 fail-fast；显式记录缺失率 |
| `load_phase2_adapters(... strict=False)` | 错模型、错 LoRA 配置时可能有权重未加载 | 新 GeoWire 后端检查 LoRA key 集合并严格加载图模块 |

## 5. 公平评测与奖励的研究风险

GeoWire 的通用 `spatial_choice` 正则从全文搜索 A–D，不能覆盖 VSI 的数值题指标，也可能匹配到推理文本。官方 VSI 使用数值相对误差评分和按任务聚合，不是所有题统一 exact accuracy。本版保留官方评分源文件，避免自行“修正”其阈值细节而造成新的协议偏差。

PSRO 关键词奖励衡量的是文本中出现了哪些词，不直接验证这些观察是否符合图像。即使已经针对词表堆砌设置惩罚，也不能据此证明模型获得了可验证的空间推理过程。保留原 reward 用于复现实验，增加 answer-only、answer+format、完整 PSRO 三组对照。

训练数据中出现 `VSI_590K` 之类名称，**不能仅凭名字认定测试集泄漏**。应检查具体 split、scene/video 来源和帧内容。新框架检查精确内容、ID、场景重叠，并明确此检查不能排除近重复视频或教师预训练污染。

## 6. 冗余整理策略

| 可统一的部分 | 保留独立的部分 |
|---|---|
| 样本字段归一化、清单校验、抽帧记录 | 三种几何表示及其缓存内容 |
| 生成参数、最终答案提取、结果记录 | GeoWire 的图约束和 TIP 损失 |
| 数据采样、实验配置、训练目标、参数网格 | GeoBridge 的 FCP/HGB/router/gate |
| 通用 I/O、随机种子、指标与来源记录 | GeoPSRO 原始过程奖励和历史 prompt |

GeoBridge 的依赖安装同时包含训练、视频、几何、开发工具、评测及可选 encoder；还出现 `hf_transfer` 重复列项。将整个依赖列表强制合并进一个新环境会扩大版本冲突面。本版公共核心保持轻量，训练、VSI、视频作为 extras；原方法各自保留环境。

没有凭文件名相似就删除 VGGT、旧 continuity 实现或 historical launcher：这些可能影响 checkpoint 兼容。源码扫描没有发现 GeoBridge 中大于 500 字节的 Python 文件存在逐字相同副本；这里的冗余主要是职责和接口重复，不能据此声称大量文件完全相同。

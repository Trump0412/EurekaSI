# 开源整理与流程就绪度审查

日期：2026-09-07。范围：当前公共 codebase 的静态阅读、修复和文档整理。

**结论：可以作为继续开发和实验的公共基础设施，但还不能承诺从新机器安装到所有数据集训练、推理、官方评测一次走通。** 通用训练和推理有实质实现；原数据到 manifest 的映射、实际模型兼容性、多卡、几何编码器和仿真仍有待验收项。下文区分“代码已修”与“运行已证实”。

本轮交接适配新增 wheel 资源定位、发布检查和环境隔离；详细决策见 [HANDOFF_MERGE](HANDOFF_MERGE.md)。未执行运行测试，旧静态报告属于上轮快照。

## 本次整理的工作树

- 以 `Spatial_Intelligence_Codebase_v0.2.zip` 展开公共源码至当前根目录。两份 ZIP 的公共文件完全一致，只有各自的 MANIFEST 不同；完整版另有 367 个历史源码文件。
- 两份原 ZIP、原独立研究文档和原始 MANIFEST 保存在本地 `_archives/`，没有修改其内容。校验原包：轻量包 123 个清单项、完整版 490 个清单项，哈希均吻合。
- 保留公共代码、配置、资源目录、第三方声明、补丁和研究文档；没有把旧三项目的整份源码并入日常维护目录。
- 补充 `.editorconfig`、`.gitattributes`、忽略规则、维护约定、静态检查入口和针对实质缺陷的回归用例；初始化本地 Git `main` 分支，未暂存、提交、配置远端或推送。

## 沿使用流程的判断

| 流程 | 静态判断 | 实际运行前还需要什么 |
|---|---|---|
| 环境配置 | 原生依赖、源码 editable 安装和 CLI 声明已对齐；默认 Transformers 经交接合并改为 4.57.6；安装脚本检查 Python 范围并执行 pip check | Linux/WSL2，推荐 Python 3.12；新依赖组合尚未安装验收；CUDA wheel/驱动/硬件兼容性需实机确认 |
| 资源下载 | HF 固定 revision、receipt 和断点重试有实现；源码固定 SHA 并应用补丁；只下载元数据不等于模型完整 | 网络/权限、足够磁盘、完整权重和各数据集媒体；本次没有逐项重新核查整个 catalog 的远端可用性 |
| 数据构建 | 支持 JSON/JSONL/Parquet、图像/视频、单轮 QA、SPAR renderer；缺媒体或超预算会报错 | 标注实际 schema、媒体根路径、规定帧序与标记类型逐例核对；MindCube/DSR 等仍需上游专用配方 |
| 清洗、划分、泄漏 | 按渲染后的视觉证据判断重复和标签冲突；按真实 scene_id 划分；训练前进行跨 split 审计 | scene_id 缺失时须补充真实身份；同场景不同别名、近重复帧、教师预训练污染不能由精确哈希排除 |
| SFT / offline KD | 有 response-only loss、LoRA、优化器更新、保存与重载入口；KD 完整清单和导出来源现在有哈希记录 | 真实多模态 processor、标签/EOS 对齐、显存、梯度和 checkpoint 重载均待验收 |
| RL / OPD / OPSD | 有在线采样、组优势/策略损失、冻结 reference/teacher 与 KL；增加参数预检 | 仅参考训练器；教师 tokenizer 必须相容；单次采样后单次更新不等于生产级 PPO/GRPO 引擎；需实机检查奖励、采样与梯度 |
| 单卡/多卡推理 | 逐样本生成、分片合并、哈希和覆盖率校验有实现 | 真实模型未加载；需要完整权重/adapter 和有效 manifest；不支持断点接着写同一输出目录 |
| 多 benchmark | suite 可按任务覆盖输入协议；ReVSI 单独使用 32 帧；正式任务走外部引擎 | 当前模型矩阵与各自 schema/引擎都未端到端运行；诊断 XML 提示与官方直接答案协议不能混用 |
| 几何融合 | 通用七维点缓存、FusionBackend 和独立权重保存有实现，缓存验证加强 | VGGT/DA3/Pi3 尚未加载；DA3 有重依赖；上游源码 commit 的声明还须与实际安装核对；不能把本库融合称为原论文结构 |
| 原方法复现 | GeoWire 有原生桥接后端；GeoBridge/GeoPSRO 保留外部原环境入口 | GeoWire 仍需要匹配图与 graph_contract，当前没有一键从通用点缓存生成其原生图契约的工具；其余原方法尚未全部移植 |
| LIBERO / 应用迁移 | 提供动作契约和闭环评测接口 | 需要另行训练的动作策略及原模拟器环境；QA 训练不会自动得到可控制机器人的策略；没有 VLA、驾驶或世界模型训练闭环 |

## 已修的明确问题

以下均为代码修复，未执行新增运行时回归。

| 优先级 | 原问题与影响 | 当前处理 |
|---|---|---|
| 高 | 发布脚本递归扫描整个目录，可能收录 models/datasets/frames 中的非权重文件、旧 ZIP 和本地材料 | 改为公共源码根目录白名单，排除 symlink、本地配置和环境文件；输出不可覆盖，归档内生成新哈希清单 |
| 高 | 只按 `geometry_media` 去重，会把同原图、同题但不同彩点对象的正确 QA 全部标成冲突 | 清洗改用实际 `media`；跨集合泄漏检查继续使用原始视觉/场景身份 |
| 高 | `run` 在应用 `--set` 后覆盖部分字段，导致用户以为指定了 adapter/配置而实际未生效 | 先解析专用参数，再严格应用覆盖；路径/模式冲突显式拒绝；无效 GPU 参数不再占用计划名；检查 run/suite 名称和重复项 |
| 高 | 包元数据遗漏下载/数值/视觉依赖，安装 extras 与脚本版本范围不同；Transformers 4.57.0 已撤回 | 对齐直接依赖与 extras，补充 torchvision、HF Hub、NumPy 和 data extra；默认现为 4.57.6；不伪造新的 resolved lock |
| 高 | 六任务示例统一最多 8 帧，但 ReVSI builder 要求完整规定视图 | 增加 suite 的逐任务协议覆盖，ReVSI 使用 32 帧预算并检查 num_frames；64 帧任务需要另设配方 |
| 高 | 几何缓存改动后再次提取可能把新哈希登记为可信；不同提取器的冲突到实际计算后才发现 | 提前校验 extractor/schema；拒绝与已有索引不符的缓存；适配器源码哈希进入缓存描述 |
| 中 | 不同媒体根目录/渲染器的构建共用样本输出路径；视频缓存复用未校验帧图 | 构建配方隔离路径；记录 builder 哈希；视频缓存包含并核对帧图哈希；数字帧名按数值排序 |
| 中 | KD 教师响应存在 metadata 中，原训练数据指纹忽略 metadata | 额外记录完整 train/heldout manifest 文件哈希；teacher-export 记录模型身份、输入/输出哈希和环境，保护已有导出 |
| 中 | `SPATIAL_CONFIG` 只影响读取、不影响 init；Windows 路径冒号被当混合权重分隔符；多处文本读取依赖系统编码 | init 与读取统一路径配置；修复路径解析；公共代码使用显式 UTF-8；Git 来源信息从源码目录读取，缺 Git 时保留空值 |
| 中 | NaN/Inf/零步数等参数可能通过计划阶段；空 scene_id、非法 split 不够严格 | 增加训练数值/组大小/采样配置预检；统一 scene_id 字符串；限制 split 为 train/val/test |
| 中 | 几何环境后续安装可因 xformers 等依赖静默升级 torch/torchvision | 增加全程约束文件与 pip check；Pi3 补装上游 requirements；不能满足约束时显式失败 |
| 中 | GeoWire 推理未给 checkpoint 时可能使用随机几何模块 | 推理明确要求训练后的 options.checkpoint |
| 中 | paired 分析重新读了可能已变动的清单/场景，却继续使用旧评分 | 当前清单/媒体指纹必须与历史评测数据指纹相符 |

依赖修订依据：[Transformers 4.57.0 的官方撤回说明](https://pypi.org/project/transformers/4.57.0/)、[4.57.6 发布页](https://pypi.org/project/transformers/4.57.6/)。几何安装限制依据：[DA3 锁定版本依赖](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/3d835ec1a5802d64a8b8b15f817a1ab54809bfe4/pyproject.toml)与 [Pi3 锁定版本 requirements](https://github.com/yyfz/Pi3/blob/9fa3ddb3f8d53041f8b2738df404f62223bbaa7b/requirements.txt)。

## 尚未完善的关键部分

1. **真实环境验收缺口。** 新版依赖、真实 Qwen/InternVL/LLaVA 输入输出、GPU 梯度、多卡通信和几何编码器没有本轮运行证据。旧随机 GPT2 测试不能验证视觉 token 或 RoPE 行为。
2. **数据不是下载即训练。** SPAR marker 依赖规定帧；LLaVA-Hound 标注与媒体要匹配；MindCube 的图像标签不能充当 QA；DSR 可能需要按原始 URL 取视频。仍需按具体 revision/subset 编写并验收小样本配方。
3. **规模限制。** JSON/manifest 与混合数据整体载入内存，多个 rank 各有副本；视频/图像哈希扫描开销明显。batch 参数控制逐样本累积量，没有真正的动态张量 batch。没有流式数据、optimizer/RNG 精确续训、FSDP/ZeRO、自动模型并行和训练中验证选优。
4. **数据语义仍需人核对。** 划分要求可靠的 scene_id；SPAR 与 VSI 同源场景必须排除重叠。官方视图、坐标尺度、confidence 校准、答案单位和媒体帧顺序不能仅凭字段名认定正确。
5. **复现条件尚未全部固定。** 直接依赖已固定，但完整 GPU/外部引擎传递依赖仍需安装后记录；catalog 中历史 metadata_check 不是本轮在线可用性认证。几何原生图契约和独立环境不是统一接口自动解决的问题。
6. **发布仍需维护者的提交。** 工作树保留本地原始材料并设置忽略规则，公共源码发布脚本已整理。本轮不替维护者提交、推送、登记论文作者或发布成绩。运行时验证完成后，再把对应证据补入验证记录。

## 本次检查与下一步

本次只执行文件读取、解析和原始归档校验：`python scripts/static_check.py` 检查 Python AST、YAML/TOML、JSON/JSONL 和本地文档链接，不导入应用，不调用模型、不执行 pytest。测试日志中的旧通过数继续放在 [VALIDATION](VALIDATION.md) 的历史区，不计为本次结果。

机器可读的本次检查记录保存在 `docs/validation/static-review.json`；另按 AST 中的配置字段定义核对了 15 份模型配置，并核对 core 依赖与 pyproject 声明一致。语法通过不等于回归测试通过。

未来按 [QUICKSTART](QUICKSTART.md) 准备环境与数据后，先跑完整 CPU/分布式回归，再做一个真实多模态样本的加载、响应监督、保存/重载与推理；随后进行小规模 scene-disjoint 训练和官方评测。每个阶段记录失败和覆盖范围，通过后再扩大数据/模型规模。

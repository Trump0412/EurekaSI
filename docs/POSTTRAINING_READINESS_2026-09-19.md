# OPD / OPSD / VERL-GSPO 与三个项目运行审查

本次是在部署服务器实际执行的检查，不沿用旧报告中的通过状态。没有安装或升级正在供 Qwen3.5 流水线使用的依赖，没有启动长时间 RL，也没有改动研究模型权重。

## 当前证据

| 范围 | 实际检查 | 结论边界 |
|---|---|---|
| 公共后训练、目标函数、几何适配 | 26 passed，`logs/posttraining-audit.xml` | 小型随机 GPT2 的优化/保存/重载，非真实 Qwen 多模态后训练 |
| OPD / OPSD 非零训练信号 | 新增两项测试通过：正 KL、正梯度范数、参数确实更新；OPSD 包含 teacher refresh | 避免相同 teacher/student 导致零 KL 仍被误认为有效蒸馏；不是性能提升证据 |
| GeoWire 原生测试 | 29 passed，`logs/geowire-native-audit.xml` | 图构建、TIP 小缓存、传输和桥接基础测试，不含真实 VLM/几何权重完整复现 |
| GeoPSRO 原生测试 | 10 passed，`logs/geopsro-native-audit.xml` | schema、采帧、reward、几何适配基础测试，不含完整 RL |
| 已提供的原生适配补丁 | 5 passed，`logs/legacy-adapter-audit.xml`；亦包含在上面的26项中，不重复加总 | GeoWire 递归/恢复、GeoPSRO 最终答案/零值/缺失金标检查 |
| GeoBridge 核心子集 | FCP/HGB、评分、划分、compact cache 六个文件 **22 passed**，`logs/geobridge-core-audit.xml` | 核心张量与缓存逻辑通过，不含真实 backbone 完整训练 |
| GeoBridge 全套 | **3 个收集错误**，`logs/geobridge-native-audit.xml` | 一个缺源码模块、两个 Transformers 接口不兼容；不能把上述子集通过写成全套通过 |

源码按 `catalog/sources.yaml` 固定的 commit 获取，服务器放在忽略的 `legacy_sources/`；GeoWire/GeoPSRO 使用仓库已有补丁。源码未提交、未作为新公共依赖 vendoring。

## OPD / OPSD：已实现什么，还缺什么

公共入口在 `spatial_intelligence/training.py`。OPD 在学生当前策略生成的回答位置，计算冻结教师分布与学生分布的 KL；要求 tokenizer 语义相同。OPSD 克隆学生为教师，只给教师追加训练期金标答案，可按步刷新教师；学生输入不能带该答案。

当前 `backends/hf.py` 的多模态家族白名单**没有 qwen3_5**。新的 `study.py` / `qwen35.py` 是 SFT/评测专用实现，没有因此自动获得 OPD/OPSD 支持。不得仅修改白名单就宣布接入：还须测试图像 token/`mm_token_type_ids` 扩展、teacher/student completion 对齐、在线采样、冻结教师、KL 内存，以及保存重载。

参考训练器的 batch 为逐样本累积，不是新的 Qwen3.5 Trainer 真正 tensor batch；没有完整 optimizer/scheduler/RNG 断点续训。OPD 正式实验还需要指定兼容 tokenizer 的实际教师权重，不能将随机小模型测试或同权重零 KL 当成有效教师实验。

## VERL / GSPO：尚未部署成可运行框架

对本次检查的四个环境做了模块/包清点：`qwen35`、Conda base、`flagscale-train`、`flagscale-inference` 中均未发现 `verl`。Qwen3.5 环境同时没有 Ray/vLLM/DeepSpeed；旧 flagscale-inference 有 Ray 2.43.0 和 vLLM 0.8.6 开发版，但这不代表 VERL 或 Qwen3.5 rollout 已可用。可重跑 `scripts/audit-posttraining-env.py`；模块可发现也不等于 import/CUDA/多卡验收。

`objectives.py` 中实现了 GSPO 的序列级重要性比率，目标函数测试通过。但当前公共训练器是一次 rollout 后一次更新，不能代替 VERL 的 rollout workers、分布式 actor、权重同步和多次 PPO 更新。当前流程的重要性比率在更新前起点为1，不能据此验证生产 GSPO 的离策略裁剪行为或收益。

固定 commit 的 GeoPSRO `train_stage3_rft.py` 里，`--backend verl` 仅写 `psro_rft_train.jsonl` 和 `verl_launch_plan.json` 后结束；没有 VERL trainer 或 optimizer。plan 的 `compute_reward` 也不能直接当作已验证的 VERL reward-manager 签名。`--smoke` 是对预写字符串打分，不是模型 rollout。

## 三项目完整复现门槛

- **GeoWire**：需要 Qwen3-VL 权重、正确的跨帧图/graph contract、TIP 缓存和 phase1 checkpoint，再验证真实 RGB 生成和阶段2训练。现有桥接器明确是 Qwen3-VL，不能用 Qwen3.5 替代后仍称原方法复现。
- **GeoBridge**：需要与其 fork 匹配的独立环境、FCP/continuity bank、Stage1 权重、HGB 接入和几何缓存；`legacy-geobridge-sft.yaml` 仍有 `/path/to/...` 占位。此次检出的具体阻塞：固定 commit 缺少可视化测试导入的 `scripts/analysis/stage1_vggt_similarity_common.py`；旧 `modeling_qwen2_5_vl.py` 导入 `SlidingWindowCache`，而当前 Transformers 5.3.0 不提供此接口。上游 `setup.py` 要求 Python 3.10、torch 2.5.1、Transformers 4.57.0，与当前 Python 3.12/Qwen3.5 环境不同；这里没有安装或验证这套旧锁定环境，不应混装降级当前环境。
- **GeoPSRO**：需要几何对齐 checkpoint、实际数据/cache 和多卡审计；已核对原 Stage3 不是训练器。既有 Stage2 的 RGB/choices 输入及多卡同步风险见 `docs/AUDIT.md`，不能由10项基础测试推导为已解决。

后续实施应单独隔离 VERL 环境，锁定兼容版本并建立真实图像 reward/rollout/optimizer/checkpoint 小闭环，再扩大实验。三个原方法保留自己的配置与缓存，公共 FusionBackend 只能标为新基线，不能冒充原方法。

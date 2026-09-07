# 兼容性与环境隔离

“统一代码库”统一的是样本、配置、任务接口、结果格式与实验管理；不能把不兼容的三代 Transformers 塞进一个环境。源代码和资源版本锁在 catalog；外部环境安装完成应保存 pip freeze 和 GPU/驱动信息。

| 路径 | 依赖策略 | 验证状态 |
|---|---|---|
| 本库 native | Python 3.12，torch 2.5.1，torchvision 0.20.1，Transformers 4.57.6，PEFT 0.17.1；支持的 Python 范围为 3.10–3.12 | 本轮仅静态审查；原 4.57.0 环境附有历史测试日志，新组合与 CUDA 均待运行验收 |
| Qwen2/2.5/3-VL dense 原生 HF | 同一 native 环境；Qwen2.5-VL 原生类不等于 GeoThinker 修改过的类 | AutoProcessor/AutoModel 接口，真实权重待验收 |
| InternVL HF / LLaVA HF | 只接受已注册原生 model_type，拒绝旧 remote-code 和未接入 OneVision | 尚无真实模型验收；不能仅凭名字认为所有变体通用 |
| GeoThinker | 原仓库说明 Qwen2.5-VL 训练需 4.50.0，Qwen3-VL 需 4.57.0；分别建 venv | 源码核查，尚未重跑论文 |
| GeoSR | 3D/4D 分支的环境和脚本分别使用，保留自己的 fusion 结构 | 源码核查，目录注册；尚未统一后端移植 |
| SpatialStack | 已读取分支为 Qwen3.5，torch 2.10.0/cu129、Transformers 5.3.0 等；跟随锁定 README 独立安装 | 链接及源码核查；不进 native4.57 |
| EASI | 使用仓库自己的 scripts/setup.sh，Python3.11/uv 与子模块 | 任务参数映射已核查；安装及实模评测未在本机完成 |
| VGGT / DA3 / Pi3 | bootstrap-geometry.sh 每个编码器单独 venv，仅共享缓存文件 | 接口源码核查及缓存数学/合约测试，真实几何权重未运行 |
| LIBERO | 使用原项目支持的仿真环境，保留 torch/robosuite/MuJoCo 组合，不装 native train extras | 闭环接口测试使用假环境；未在真实仿真器运行 |

## 安装外部评测环境

```bash
spatial source easi
cd /data/spatial/external/easi
# 需已安装 uv，且有 Python3.11；脚本自身创建其 .venv
bash scripts/setup.sh
```

本库 `spatial benchmark` 接收这个环境的 Python 绝对路径，以 subprocess 启动外部命令，不污染 native 包环境。MindCube-Tiny、ViewSpatial、MMSI、VSI 的 EASI 名称映射直接取自所锁定源码，不根据相似名称猜测。ReVSI/DSR 若该版本不包含任务，预检报错；改用对应官方支持版本并记录实际 commit。

EASI 脚本中某些依赖为范围约束；catalog 的源码 SHA 不能固定其所有 transitive dependencies。安装后保存 `pip freeze`，不声称外部 GPU 环境已完全锁定。原交付 CPU 环境另附 `requirements/resolved-cpu-py312.txt`，仅作历史记录。

4.57.0 被维护者以安装问题为由撤回，本轮交接合并采用同一小版本系列的 4.57.6，避免引入 5.x 迁移。此选择是静态修订，不代表安装/训练已验证。[官方撤回说明](https://pypi.org/project/transformers/4.57.0/)、[4.57.6 发布页](https://pypi.org/project/transformers/4.57.6/)。

## 几何编码器依赖

VGGT 的核心 package 可以 editable 安装；DA3 用其 pyproject 核心依赖，不装 Gaussian/app extras，但核心列表本身包含 xformers、Open3D、pycolmap 等重依赖，不能视为轻量安装。几何安装脚本现在始终约束 torch/torchvision，防止后续依赖解析静默替换已经选择的版本；若没有兼容包会显式失败。Pi3 同时安装锁定源码的 requirements，RoPE 扩展与可选加速按该版本 README 处理。bootstrap-geometry 安装完成不等于模型能在当前驱动上运行，首个 cache-geometry 调用会执行真实 forward 并校验输出。

依赖依据：[DA3 固定版本 pyproject](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/3d835ec1a5802d64a8b8b15f817a1ab54809bfe4/pyproject.toml)、[Pi3 固定版本 requirements](https://github.com/yyfz/Pi3/blob/9fa3ddb3f8d53041f8b2738df404f62223bbaa7b/requirements.txt)。

## 尚未实现的规模化能力

本库已有 torchrun 同步数据并行参考实现（见 DISTRIBUTED.md），仍没有 FSDP/ZeRO/verl 集成、动态 batch、多机调度、精确恢复 optimizer/RNG、自动模型并行或者 VLA 训练 pipeline。原项目训练入口仍可在原环境使用。把研究参考训练器升级到大规模生产需单独做数值等效、并行采样、同步失败处理、预算与 checkpoint 测试，不应只加一个 torchrun 命令就宣称支持多卡。

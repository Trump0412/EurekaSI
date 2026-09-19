# 下载结束后的自动部署

启动入口：

```bash
python3 scripts/post-download-deploy.py --root /persistent/root --conda /path/to/miniconda/bin/conda --detach
```

`configs/post-download.json` 固定本轮9项资源屏障。全部 receipt 为 complete 后才安装；失败/缺失不会被当成完成，也不会偷偷替换数据源。进程独立于 SSH，单实例锁，30秒检查一次。状态为 `state/post-download.json`，逐阶段 `state/extension-*.json`，日志 `logs/extension-*.log`。重新运行可跳过已成功阶段；原进程仍在时不能重复启动。

## 自动执行范围

1. 生成 `extension-configs/resources.json` 和独立 `downloads.json`。
2. 并行安装两套独立 Conda：`legacy-geo`（Python3.10、Torch2.5.1、Transformers4.57.6）与 `verl-legacy`（Python3.12、VERL0.7.0、vLLM0.11.2、Torch2.9.0、Transformers4.57.6）。后者是旧模型框架验证栈，**不是 Qwen3.5 rollout 支持声明**。安装前提与版本依据：[VERL0.7.0](https://github.com/verl-project/verl/blob/v0.7.0/setup.py)、[vLLM0.11.2依赖](https://pypi.org/pypi/vllm/0.11.2/json)。不安装到现有 qwen35 环境，不修改驱动/CUDA系统目录。
3. pip check、包导入、OPD/OPSD 小模型测试；拉取固定版本三个原方法并应用仓库已有补丁，在隔离旧环境运行各自测试。任一安装/项目失败独立记录，其它项继续。

当前所有部署子进程隐藏 GPU，避免与原 SFT 抢卡；没有自动启动额外长训、额外 benchmark，或把 import 通过写成 GPU RL 验收。VERL GPU rollout/GSPO 更新、Qwen3.5 后训练适配、GeoBridge 缺失可视化源码仍需后续实现或验证。宿主机驱动和新 engine CUDA 的兼容性必须实测；不会为安装自行升级驱动。

## 其它数据配置

评测配置：VSI-Bench、ReVSI、MMSI、ViewSpatial、MindCube、DSR，以及 CVBench/BLINK/EmbSpatial/3DSRBench/SITE 图像与视频扩展入口。全部保留原 engine/task 名和验证门槛；MindCube-Tiny 不冒充全量，DSR train/test 必须隔离，部分媒体需额外获取。

训练配置：VLM-3R、VSI590K、Cambrian-S、SpatialStack。既有数据复用，不重复下载；新增集合混合权重为0，启用前审计 train split、场景重叠、媒体、标记/坐标、VLM-3R 勘误。当前 SPAR/Hound 正式实验不改变。

这里只准备扩展数据下载/实验配置，不默认下载所有额外TB级数据。确认某项预算后，可以从独立 catalog 选取具体资源：

```bash
python3 scripts/fetch-study-assets.py --root /persistent/root \
  --catalog /persistent/root/extension-configs/downloads.json --assets extra-vsibench
```

没有 asset 的评测项由其外部 engine 提供下载协议，不虚构 Hugging Face 地址。资源配置 `enabled:false` 和未通过门槛是有意的：配置就绪不等于媒体、适配和官方分数已就绪。

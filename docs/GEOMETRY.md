# VGGT、DA3、Pi3 与统一融合

几何编码器放在独立环境离线提取，训练环境只加载 NumPy 缓存。这样不会把三个编码器的 CUDA 扩展与主模型版本强行混装。

```bash
spatial download vggt
bash scripts/bootstrap-geometry.sh vggt
# 修改 configs/geometry/vggt.yaml 的 weights 路径（保留 source_commit）
.venv-vggt/bin/spatial cache-geometry \
  --manifest /data/spatial/manifests/spar.train.jsonl \
  --config configs/geometry/vggt.yaml --output /data/spatial/geometry/spar-vggt
```

DA3/Pi3 把命令中的 vggt 替换成 da3/pi3。source SHA 已写入各自配置。安装脚本和适配器已按上游接口实现，**三个真实几何权重尚未在本环境加载或 GPU 验收**；特殊扩展的安装以锁定版本 README 为准。失败会直接报错，不生成随机/零几何充当成功输出。

## 缓存定义

- VGGT：world_points、world_points_conf。
- DA3：depth + intrinsics + world-to-camera extrinsics 反投影到世界点。
- Pi3：points + sigmoid(conf)。Pi3 默认使用显式 518×518 resize，属于本库共同实验协议，不是其官方 demo 的长宽比处理；对照实验须保持一致。
- 每帧固定网格抽取 `xyz, confidence, frame_fraction, u, v` 七维 token。点坐标保留原输出尺度，不能把相对尺度模型的数字直接视为米。
- 缓存 key 包含有序原图哈希、模型文件哈希、源码 commit、提取参数和 schema。索引记录 npz 文件哈希，训练前核对；缺缓存不回退。
- 另记录本库几何适配器源码哈希；换配方会在提取前拒绝复用不一致索引。已有索引中的 npz 被修改会报错，不重新登记为有效缓存。上游 `source_commit` 仍是声明值，真实运行还需核对实际安装源码及其修改状态。

这是轻量、可检查的几何输出基线；不是取 VGGT 多层 hidden bank，也没有声称复现 GeoThinker 的 SGF、GeoBridge HGB、GeoSR GGF 或 SpatialStack 的层间融合。

## 共用融合训练

```bash
spatial run sft --model qwen3-vl-2b --name fusion-sft \
  --geometry-index /data/spatial/geometry/spar-vggt/index.json \
  --train spar.train --heldout spatial.val vsibench.test
```

`FusionBackend` 用 MLP 将七维几何投影到语言维度，由输入 embedding query 做 cross-attention，再以 tanh(gate) 残差相加。gate 初始为零，初始输出与基模一致；优化器同时更新 LoRA 与融合模块。这是新的共同研究基线，需要按该名字报告。编码器冻结，几何缓存不参与反向传播。

```bash
spatial run inference --model qwen3-vl-2b \
  --adapter /data/spatial/runs/fusion-sft/final \
  --fusion-checkpoint /data/spatial/runs/fusion-sft/final/spatial_fusion.pt \
  --geometry-index /data/spatial/geometry/vsi-vggt/index.json \
  --eval vsibench.test --name fusion-vsi
```

评测集也要先提取自己的缓存。full-model 模式则 `--model` 指向完整 final，省略 adapter。融合结构的 heads 参数必须与训练一致。加载融合 state_dict 使用 strict=True；普通 HF checkpoint 单独保存，避免未知自定义键。

消融支持 geometry_condition=normal/zero/reverse_frames。zero 完全关闭融合残差；reverse_frames 只反转 frame_fraction 时间标签，保留 XYZ 与 confidence，用于时间标签错配实验。不能把它描述为重新运行反序几何编码器。更强的错配场景、相机扰动、随机图等对照应实现独立实验配置并匹配预算。

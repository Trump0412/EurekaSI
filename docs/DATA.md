# 数据获取、构建、清洗与能力标签

## 两种评测路径

正式复现优先通过 `spatial benchmark` 调用官方引擎，让其保留数据集自己的 prompt、视频布局、指标与 split。`build-data` 构建的是供统一训练和受控评测使用的 manifest，不能自动等同于论文官方协议。

| 数据 | 获取 | 构建要点 |
|---|---|---|
| SPAR-7M | `download spar7m` 或 `download vg-llm-data` | 单图/多图 QA 需 `spar_info` 的彩点、框或地图渲染；训练别名和比例必须记录 |
| LLaVA-Hound | VG-LLM-Data 中 `llava_hound_64k.json`；`llava-hound-media` 获取对应媒体 | “64K 标注子集”“原始视频库”“LLaVA-Video-178K”不是一个数据集，不能互换 |
| VSI-Bench | `download vsibench --extract` | 原数据的 dataset/scene_name 对应 `<dataset>/<scene_name>.mp4`；数值题与选项题分别处理 |
| ReVSI | `download revsi --extract` | 16/32/64_frame 各自的 QA 和固定视图视频；builder 用 num_frames 对应 `<N>_frame/<scene_id>.mp4`，不能把 32 帧题任意变 8 帧 |
| MMSI-Bench | `download mmsi` | Parquet 内嵌图像 bytes 可自动落盘；嵌入题干的 Options 会分离 |
| ViewSpatial | `download viewspatial --extract` | image_path 可能是字符串化列表；answer 如 `A. right`；图片另有归档，media-root 指向相对路径起点 |
| MindCube | `download mindcube`，另取项目提供的 QA | HF default/train 是图像和 label，不能用 label 伪造空间问答。QA 与图像映射未提供时 builder 会明确报错；外部 EASI 默认 MindCube-Tiny |
| DSR-Bench / DSR train | `download dsr` | benchmark.parquet 与 train_qa_pairs 分开；Koala/Panda 视频可能只有原始 URL。遵循原项目媒体获取流程，缺视频就报错 |

源仓库、论文与权重链接全部见 [RESOURCES](RESOURCES.md)。每次构建保存原始标注哈希、行号、渲染器哈希、清单哈希。框架不会在运行时静默删除缺图样本。

## SPAR 单图／多视图构建实例

```bash
spatial source geothinker
spatial download vg-llm-data
spatial download vg-llm-data --include 'train/spar_7m.tar.gz' --extract
spatial build-data \
  --input /data/spatial/datasets/vg-llm-data/train/spar_234k.json \
  --dataset spar --split train \
  --media-root /data/spatial/datasets/vg-llm-data/media \
  --marker-source /data/spatial/external/geothinker/src/qwen_vl/data/draw_marker.py \
  --max-frames 8 --output /data/spatial/manifests/spar.raw.train.jsonl
```

媒体归档真实顶层目录需先与标注中的路径核对；若多一层目录，只改 `--media-root`，不要逐个改标注。没有匹配媒体时此命令会失败，而不是产出“成功”的残缺训练集。

`media` 是渲染后的 VLM 输入；`geometry_media` 是顺序一致的原图，交给 VGGT/DA3/Pi3。这避免把人工彩点当成真实三维几何，同时保留语言题目的指代证据。渲染直接调用锁定的 GeoThinker 实现，不重写其颜色或坐标规则。

带 `spar_info` 的原始视频不能经过任意均匀采样后绘制：marker 的帧索引可能错位。先按上游配方形成规定的有序帧，再交给 builder；遇到超预算帧目录会拒绝重采样。地图/导航等复杂类型同样需要上游渲染函数和完整证据，不能随便用图像截断替代。

## 通用视频和 Parquet

```bash
spatial build-data --input /data/annotations/llava_hound_64k.json \
  --dataset llava-hound --split train --media-root /data/videos/train_300k \
  --max-frames 8 --output /data/spatial/manifests/llava-hound.train.jsonl

spatial build-data --input /data/mmsi/test.parquet --dataset mmsi --split test \
  --media-root /data/mmsi --max-frames 8 \
  --output /data/spatial/manifests/mmsi.test.jsonl
```

输入支持 JSON 数组、JSONL、Parquet。媒体字段支持 media/frame_paths/images/image_path/image，以及 video/videoID。仅接受单轮 QA；未知多轮结构需要 `convert --adapter module:function` 明确转换。金标选项接受显式字母或唯一精确匹配的选项文本，不猜测数字标签是 0-based 还是 1-based。

视频帧选择缓存按原视频 bytes 和帧预算命名；记录帧索引、时间、原文件哈希。同一视频可跨多个问题复用解码结果。可变帧率视频的 index/fps 只是近似时间，时序评测应保留上游实际时间戳。

## 清洗不是“删掉模型不会的题”

`clean-data` 自动剔除：无法解码的图片、无视觉证据、完全重复的问题+选项+证据、同证据同题但冲突的所有标签。去重使用实际渲染后的 `media`：同一原图上的不同彩点/框可指向不同对象，不能仅因 `geometry_media` 相同就判冲突。跨 split 的泄漏审计继续使用原图/场景，避免标记变化掩盖场景复用。全部删除原因进入 excluded.jsonl。缺场景或能力标签进入 review.jsonl，保留人工审查，不偷偷删除难例。

构建目录按原标注、媒体根目录、帧预算、渲染器和 builder 哈希隔离；视频缓存记录每张帧图的哈希。旧版本没有帧哈希的缓存需要在新的 frames 目录重建。标注清单和训练混合当前会整体载入内存，7M 级数据尚无流式训练实现；多卡每个进程各持一份，运行前需要独立评估内存预算。

新增人工标注建议放在 metadata：

```json
{"spatial_capability":"reference_frame","evidence_required":"multi_view","reference_frame":"camera","temporal_order_required":false,"answer_grounded":true,"annotation_status":"double_reviewed","source_scene_family":"scannet/scene0011_00","split_group":"scannet/scene0011_00"}
```

这些字段不自动送入模型，也不会因关键词就认证能力。划分前将真实 `scene_id` 归一到 canonical scene family；工具目前按 scene_id 分组，不自动读 split_group。室内场景别名、近重复视频、教师预训练泄漏需要额外人工/感知哈希检查。不得跨来源给同场景编不同别名来绕过审核。

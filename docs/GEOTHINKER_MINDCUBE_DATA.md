# GeoThinker VLM3R 与 MindCube 的六源扩展

旧版 VLM3R 100K 上限不是 GeoThinker 对应子集，不能继续用于此实验的新数据定义。
官方 32 帧转换发布 VSI 205456 条、VST 132053 条，共337509条；8帧VST为132568条，不能从文件名或其他变体推断32帧条数。只选32帧变体，不能把不同帧数版本叠加计算。VST问题直接引用32帧序号，8帧版本可能遗漏问题要求的帧。

来源：[GeoThinker 数据准备说明](https://github.com/Li-Hao-yuan/GeoThinker)、[转换清单](https://huggingface.co/lihy285/GeoThinker/tree/main/train)、[MindCube 官方数据卡](https://huggingface.co/datasets/MLL-Lab/MindCube)。下载应在服务器经镜像，优先复用已下载资产；已存在文件不覆盖。

GeoThinker旧VST转换与后来官方纠错不完全一致。按相同source ID连接原始VST修订版，校验source/scene/task后同时使用修订问题与答案，并记录全部变化：本次68542条答案变化、3108条问题变化，2350条官方erratum排除，VSI/VST共保留335159条候选。只改答案会把新答案套到旧帧序号问题，是错误的。

MindCube 仅使用官方 `MindCube_train.jsonl` 10000 条训练，`MindCube_tinybench.jsonl` 1050 条评测。对已下载官方包的审查：train/tiny ID、媒体、媒体目录组均无交集；完整 `MindCube.jsonl` 21154 条包含全部10000训练 ID，**不作为 heldout**。目录组隔离只是可观测代理，并非已证明所有真实物理房间隔离。

```bash
python scripts/audit-geothinker-mindcube-data.py \
  --source-root "$SHARED_INPUTS" \
  --mindcube-zip "$ASSET_ROOT/datasets/mindcube/data.zip" \
  --output "$NODE_ROOT/studies/geometry-followup-data-v3"
```

脚本要求新输出目录，保留旧版本。输出 VLM3R 与 MindCube 组件清单、纠错/排除 ledger、媒体读取失败表及 `data-readiness.json`。检查 ReVSI/VSI-Bench scene，逐张实际读取所用媒体。失败不静默删样本。

该组件完成并不意味着六源已可训练。SPAR、Hound、VSI590K、OpenSpatial ARKitScenes 100K 还需统一合并、去重、全局 source-scene train/val 隔离；相同 ScanNet 场景不能因跨 SPAR/VLM3R/VSI 来源而绕过 split。大于100万是原始规模预期，不是已验收训练量。receipt 保持 `ready_for_training=false` 直到六源联合验收。

## 当前可证实的缺口

`prepare-six-source-geometry.py` 对实际输入生成库存/阻塞报告，不伪装为已完成merge。VSI590K实际590667条，包含5963个独立视频与44857张独立静态图像，不能统一假设有video字段。媒体解码尚需独立验收。

OpenSpatial已取得100925条ARKitScenes候选，全部无原始媒体路径、无scene字段；meta_info仅分辨率，UUID唯一但不代表场景。原始图片无精确重复不能证明场景无泄漏。后续用户明确授权仅对OpenSpatial接受这个限制：seed3407选100000条全部train-only，不构造伪scene或加入validation。协议版本为`shared-six-source-geothinker-vlm3r-mindcube-v4-openspatial-waiver`，`leakage_checked=false`，`known_sources_leakage_checked=true`仅在其他五源验收后成立。必须随报告保留`contamination_waiver`；不是完全无污染的泛化结果。

`prepare-six-source-waiver.py` 实际执行自动链：VSI九归档定向安全解包、最多32真实帧采样、静态图解码、OpenSpatial导图、其余五源规范场景split/精确样本去重、六源合并。缺媒体或未实现的多轮对话格式报错，不静默丢数据。`merge-five-source-geometry.py` 保留规范样本ID与source_index；只有scene+有序媒体+问题+答案完全相同的候选才记入重复ledger，不能仅凭重复source ID删除SPAR行。

`watch-six-source-readiness.py` 为正式队列提供稳定endpoint，追踪不可变producer版本，使用共享`followup_data_policy.validate_leakage_policy`验收clean或显式OpenSpatial-only例外，不将ready自动等同leakage_checked。其他媒体/六源/清单gate仍需通过。启动必须显式指定`--authorize-openspatial-scene-waiver`，不是默认全局跳过审查。

```bash
python scripts/prepare-six-source-waiver.py \
  --source-root "$SHARED_INPUTS" --component-root "$COMPONENT_ROOT" \
  --openspatial-acquisition "$ARKIT_ROOT/acquisition.json" \
  --output "$NODE_ROOT/studies/six-source-waiver-v1" \
  --authorize-openspatial-scene-waiver
```

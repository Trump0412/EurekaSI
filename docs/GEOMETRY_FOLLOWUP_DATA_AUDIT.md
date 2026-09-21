# 五源几何实验资产与数据门

这是实现协议，不是既有实验结果，也不包含私有稿件正文。两种待比较架构共享同一冻结 manifest；下载成功不等于可训练。

## 最新选集覆盖：ARKitScenes来源100K

用户已指定OpenSpatial选用ARKitScenes来源的100K条记录，替代旧版五类各20K配额；不再以未知type阻塞类别配额。筛选`data_source=arkitscenes`，不代表100K个独立场景。两方法及匹配RGB基线共用相同100K清单。若用户已有清单则保留；否则seed3407确定性选集。100K按总选集解释，随后按独立scene划出1%验证，记录实际训练数；不足不得混入其它来源补齐。

这确认了来源范围，但尚未证明100K媒体和scene映射已齐。仍须验证具体清单、媒体、场景级分割和benchmark排除；旧探针不等于完整100K。下文五类方案仅为历史背景，以本节和v2配置为准。

## 已固定的选集原则

`configs/geometry-followup-assets.json` 固定 seed3407、source-scene 1% validation、最多32个真实帧。短序列保留原帧，不复制、不编造 FPS；题目标记必须保留。

- SPAR/Hound复用既有234149/63750精确来源，新增全benchmark来源排除后报告实际数量。
- VSI使用released train；VLM3R使用空间与时序指令，先核对camera-position勘误，再按canonical顺序最多100K。
- OpenSpatial测量、关系、相机、多视一致性、场景五类，各最多20K，不足不跨类补齐。必须先核对原始type到五类的映射，不通过关键词猜测。此选择是新实现协议，不声称原论文精确选集。

## 官方来源与许可

下载revision在配置中完整固定，旧训练来源revision沿用`qwen35-assets.json`；禁止用相似名称第三方仓库替换。

| 资产 | 官方发布 | 数据卡许可/边界 |
|---|---|---|
| OpenSpatial | [官方代码](https://github.com/VINHYU/OpenSpatial)、[官方链接的数据](https://huggingface.co/datasets/jdopensource/JoyAI-Image-OpenSpatial) | Apache2.0；当前default2335335条约2.36TB、web3632071条约778GB，并非可直接声称原始3M。先只下载两个default shard作为schema探针 |
| MMSI | [RunsenXu/MMSI-Bench](https://huggingface.co/datasets/RunsenXu/MMSI-Bench) | CC-BY4.0；使用Parquet内嵌图像，不再重复下载TSV |
| ViewSpatial | [lidingm/ViewSpatial-Bench](https://huggingface.co/datasets/lidingm/ViewSpatial-Bench) | Apache2.0；保留原始场景来源与底层图像使用条件 |
| MindCube | [MLL-Lab/MindCube](https://huggingface.co/datasets/MLL-Lab/MindCube)、[官方代码](https://github.com/mll-lab-nu/MindCube) | MIT；必须使用官方Tiny QA及图像映射，图像label不能替代QA，Tiny不是全量 |
| SITE | [官方项目](https://wenqi-wang20.github.io/SITE-Bench.github.io/)、[franky-veteran/SITE-Bench](https://huggingface.co/datasets/franky-veteran/SITE-Bench) | CC-BY4.0；image/video两部分均需原协议 |
| CV-Bench | [nyu-visionx/CV-Bench](https://huggingface.co/datasets/nyu-visionx/CV-Bench) | Apache2.0；2D/3D分别保留，Parquet含图像 |
| Pi3 | [yyfz/Pi3](https://github.com/yyfz/Pi3)、[yyfz233/Pi3](https://huggingface.co/yyfz233/Pi3) | BSD2Clause；原Pi3，不是Pi3X；代码归档及权重仅下载，不执行安装 |

上述数据卡许可不抹去底层ScanNet、Matterport等来源限制。脚本不接受访问协议、不绕过gated访问、不发布原始数据。

## 执行与状态

```bash
python scripts/download-geometry-followup-assets.py \
  --config configs/geometry-followup-assets.json \
  --root "$ASSET_ROOT" --reuse-root "$SOURCE_ROOT" --workers 2 --detach
```

最多两路下载、每路20MiB/s、CPU-only、进程锁；已有训练数据仅检查只读引用，不重下载。HF在服务器通过镜像源下载。逐资产`state/*.json`记录版本、字节与失败；中断`.part`保留可续。`downloaded_not_prepared`不是ready。

若计算节点不能直连HF API，已审查的`geometry-followup-inventory.json`提供固定revision对应的精确文件与字节数，下载URL仍包含不可变revision；不把`main`或第三方同名镜像当成来源。GitHub归档不执行安装；中断归档保留为`.interrupted-*`再重新获取。

真实下载检查：`data/000001.parquet`为76行、6,331,306字节，`data/000002.parquet`为342行、24,941,863字节。两文件各检查的首3行均为UUID、`type=unknow`、数据集级`data_source=arkitscenes`；`meta_info`仅尺寸，内嵌`images.path=null`，没有question_types/question_tags。不是只依据网页viewer推测，但也不能由6行检查断言全部发布数据均缺字段。当前证据无法恢复五类配额和独立scene身份，仍需原始映射；不得按关键词猜类或跳过benchmark泄漏门。

原Pi3权重（3,834,909,248字节）和源码归档（43,535,584字节）已下载，状态为`downloaded_not_prepared`；尚不代表真实教师运行与全部层位验收完成。

`data-readiness.json`必须继续保持非ready，直到五源转换、实际媒体解码、全部benchmark来源排除、split审计、标记与几何输入语义通过。下载器不会伪造训练manifest。当前必须补足VLM3R勘误映射、OpenSpatial五类选择/媒体导出、benchmark专用转换与官方评分器。几何缓存还需区分真实深度/相机GT与模型预测，不能将RGB存在等同TIP监督就绪。

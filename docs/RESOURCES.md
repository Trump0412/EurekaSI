# 空间智能资源目录

核查日期：2026-09-07。以下36个 Hugging Face 资源已通过官方 API 核查仓库存在并取得 SHA；这不等于已下载权重或复现性能。下载器默认使用 catalog 中固定 SHA。参考源码也已读取并锁定 commit。

## 源码与评测基础设施

| 名称 | 上游源码 | commit |
|---|---|---|
| geothinker | [geothinker](https://github.com/Li-Hao-yuan/GeoThinker) | `e9b3786ab584` |
| geosr | [geosr](https://github.com/SuhZhang/GeoSR) | `cac5017c867f` |
| spatialstack | [spatialstack](https://github.com/jzh15/SpatialStack) | `b84ebaa5451d` |
| easi | [easi](https://github.com/EvolvingLMMs-Lab/EASI) | `0c1a41e55d49` |
| vggt | [vggt](https://github.com/facebookresearch/vggt) | `a288dd0f1478` |
| da3 | [da3](https://github.com/ByteDance-Seed/Depth-Anything-3) | `3d835ec1a580` |
| pi3 | [pi3](https://github.com/yyfz/Pi3) | `9fa3ddb3f8d5` |
| libero | [libero](https://github.com/Lifelong-Robot-Learning/LIBERO) | `8f1084e3132a` |
| dsr | [dsr](https://github.com/TencentARC/DSR_Suite) | `ae81203a6726` |
| revsi | [revsi](https://github.com/3dlg-hcvc/revsi) | `fd36516db5b1` |
| geowire | [geowire](https://github.com/Trump0412/GeoWire) | `d427cc7dd849` |
| geobridge | [geobridge](https://github.com/Trump0412/GeoBridge) | `74cfe79500e5` |
| geopsro | [geopsro](https://github.com/Trump0412/GeoPSRO) | `4d7db6e3e3dc` |

## 模型与几何辅助编码器

| CLI 名称 | 官方资源 | 接入状态 | SHA |
|---|---|---|---|
| `qwen3-vl-2b` | [Qwen/Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) | hf_native | `89644892e4d8` |
| `qwen3-vl-4b` | [Qwen/Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) | hf_native | `ebb281ec70b0` |
| `qwen3-vl-8b` | [Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) | hf_native | `0c351dd01ed8` |
| `qwen25-vl-3b` | [Qwen/Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) | hf_native | `66285546d2b8` |
| `qwen25-vl-7b` | [Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) | hf_native | `cc594898137f` |
| `qwen2-vl-2b` | [Qwen/Qwen2-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) | hf_native | `895c3a49bc3f` |
| `internvl3-8b-hf` | [OpenGVLab/InternVL3-8B-hf](https://huggingface.co/OpenGVLab/InternVL3-8B-hf) | hf_native | `259a3b64a146` |
| `llava-onevision-7b-hf` | [llava-hf/llava-onevision-qwen2-7b-ov-hf](https://huggingface.co/llava-hf/llava-onevision-qwen2-7b-ov-hf) | external_environment | `0d5068052768` |
| `vggt` | [facebook/VGGT-1B](https://huggingface.co/facebook/VGGT-1B) | geometry_extractor | `860abec7937d` |
| `da3` | [depth-anything/DA3NESTED-GIANT-LARGE](https://huggingface.co/depth-anything/DA3NESTED-GIANT-LARGE) | geometry_extractor | `8615eefb62f2` |
| `pi3` | [yyfz233/Pi3](https://huggingface.co/yyfz233/Pi3) | geometry_extractor | `ae722e703928` |
| `geothinker` | [lihy285/GeoThinker](https://huggingface.co/lihy285/GeoThinker) | external_environment | `7ba854d3d989` |
| `geosr` | [SuhZhang/GeoSR-Model](https://huggingface.co/SuhZhang/GeoSR-Model) | external_environment | `67e6636bbd81` |
| `spatialstack` | [Journey9ni/SpatialStack-Qwen3.5-4B](https://huggingface.co/Journey9ni/SpatialStack-Qwen3.5-4B) | external_environment | `777b2289252f` |
| `sensenova-si` | [sensenova/SenseNova-SI-1.3-Qwen3-VL-8B](https://huggingface.co/sensenova/SenseNova-SI-1.3-Qwen3-VL-8B) | external_environment | `a6328154b47f` |
| `spatialladder` | [hongxingli/SpatialLadder-3B](https://huggingface.co/hongxingli/SpatialLadder-3B) | external_environment | `0819c3adf882` |
| `spatial-mllm` | [Diankun/Spatial-MLLM-subset-sft](https://huggingface.co/Diankun/Spatial-MLLM-subset-sft) | external_environment | `7c07e6ab0981` |
| `spacer` | [RUBBISHLIKE/SpaceR](https://huggingface.co/RUBBISHLIKE/SpaceR) | external_environment | `669cff6b1cf7` |
| `vst-3b` | [rayruiyang/VST-3B-SFT](https://huggingface.co/rayruiyang/VST-3B-SFT) | external_environment | `b9f90a7b723f` |
| `cambrian-s-3b` | [nyu-visionx/Cambrian-S-3B](https://huggingface.co/nyu-visionx/Cambrian-S-3B) | external_environment | `9d8cbd75ab3e` |
| `mindcube-sft` | [MLL-Lab/MindCube-Qwen2.5VL-RawQA-SFT](https://huggingface.co/MLL-Lab/MindCube-Qwen2.5VL-RawQA-SFT) | external_environment | `d5ecb32d91d9` |
| `dsr-model` | [TencentARC/DSR_Suite-Model](https://huggingface.co/TencentARC/DSR_Suite-Model) | external_environment | `d0df6d2adced` |

## 训练与评测数据

| CLI 名称 | 官方资源 | 接入状态 | SHA |
|---|---|---|---|
| `spar7m` | [jasonzhango/SPAR-7M](https://huggingface.co/datasets/jasonzhango/SPAR-7M) | download_catalog; recipe required | `0fe664cbada1` |
| `vg-llm-data` | [zd11024/VG-LLM-Data](https://huggingface.co/datasets/zd11024/VG-LLM-Data) | download_catalog; recipe required | `4dc7eb0e9a5b` |
| `llava-video` | [lmms-lab/LLaVA-Video-178K](https://huggingface.co/datasets/lmms-lab/LLaVA-Video-178K) | download_catalog; recipe required | `6d8c562dc26d` |
| `llava-hound-media` | [ShareGPTVideo/train_video_and_instruction](https://huggingface.co/datasets/ShareGPTVideo/train_video_and_instruction) | download_catalog; recipe required | `7ed0b5422bfd` |
| `vsibench` | [nyu-visionx/VSI-Bench](https://huggingface.co/datasets/nyu-visionx/VSI-Bench) | download_catalog; recipe required | `bdcadb3fea44` |
| `revsi` | [3dlg-hcvc/ReVSI](https://huggingface.co/datasets/3dlg-hcvc/ReVSI) | download_catalog; recipe required | `e80b1cde454a` |
| `mmsi` | [RunsenXu/MMSI-Bench](https://huggingface.co/datasets/RunsenXu/MMSI-Bench) | download_catalog; recipe required | `ec7c92bfaf77` |
| `viewspatial` | [lidingm/ViewSpatial-Bench](https://huggingface.co/datasets/lidingm/ViewSpatial-Bench) | download_catalog; recipe required | `95d9ea3a94da` |
| `mindcube` | [MLL-Lab/MindCube](https://huggingface.co/datasets/MLL-Lab/MindCube) | download_catalog; recipe required | `9c941b46a6bd` |
| `dsr` | [TencentARC/DSR_Suite-Data](https://huggingface.co/datasets/TencentARC/DSR_Suite-Data) | download_catalog; recipe required | `414132f02d03` |
| `vlm3r-data` | [Journey9ni/VLM-3R-DATA](https://huggingface.co/datasets/Journey9ni/VLM-3R-DATA) | download_catalog; recipe required | `aae3535e513d` |
| `vsi590k` | [nyu-visionx/VSI-590K](https://huggingface.co/datasets/nyu-visionx/VSI-590K) | download_catalog; recipe required | `346fbd4e41de` |
| `cambrian-s-data` | [nyu-visionx/Cambrian-S-3M](https://huggingface.co/datasets/nyu-visionx/Cambrian-S-3M) | download_catalog; recipe required | `4b3baf755096` |
| `spatialstack-data` | [Journey9ni/SpatialStackData](https://huggingface.co/datasets/Journey9ni/SpatialStackData) | download_catalog; recipe required | `4f078beb949f` |

## 阅读路径

- 基础模型：[Qwen3-VL](https://github.com/QwenLM/Qwen3-VL)、[InternVL](https://github.com/OpenGVLab/InternVL)、[LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT)。
- 几何增强：[VG-LLM](https://github.com/LaVi-Lab/VG-LLM)、[VLM-3R](https://github.com/VITA-Group/VLM-3R)、GeoThinker、GeoSR、SpatialStack。
- 训练数据：[SPAR](https://github.com/fudan-zvg/spar)、DSR Suite、VSI-590K、Cambrian-S 数据。
- 能力评测：[VSI](https://vision-x-nyu.github.io/thinking-in-space.github.io/)、[ReVSI](https://3dlg-hcvc.github.io/revsi/)、[MMSI](https://github.com/InternRobotics/MMSI-Bench)、[ViewSpatial](https://zju-real.github.io/ViewSpatial-Page/)、[MindCube](https://www.mll.lab.northwestern.edu/mind-cube/)。
- 具身：[LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)、[EmbSpatial-Bench](https://huggingface.co/datasets/FlagEval/EmbSpatial-Bench)、[RefSpatial-Bench](https://huggingface.co/datasets/BAAI/RefSpatial-Bench)。

非 catalog 的扩展阅读链接未全部进行权重元数据核查。模型卡的评测数字只代表原作者报告，不能合并成本库公平对比表。不同模型的代码、权重、训练数据可能分别适用不同许可。

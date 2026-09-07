# 部署和启动操作手册

以下是未来执行用的操作说明，本轮维护没有执行安装、下载或实验。命令在 Linux / WSL2 的仓库根目录运行，推荐 Python 3.12（支持 3.10–3.12）。源码维护推荐 editable 安装；已增加 wheel 运行资源声明与定位，但实际构建/安装尚待验收，见 [维护说明](MAINTENANCE.md)。先查看 [当前就绪度](READINESS_REVIEW.md)。

## 1. 安装与路径

```bash
bash scripts/bootstrap.sh native
source .venv/bin/activate
spatial init --root /data/spatial
bash scripts/check.sh
spatial doctor
```

CPU 主流程用 `bash scripts/bootstrap.sh cpu`。`core` 只装清单、下载和配置所需依赖，不能训练。默认不安装 FlashAttention/DeepSpeed，优先保证少依赖可复现。

只需编辑被 git 忽略的 `spatial.local.yaml`。自定义位置用 `SPATIAL_CONFIG=/path/spatial.local.yaml`。数据盘包含 models/datasets/manifests/frames/geometry/runs/external/receipts。不自动搬动旧实验文件。

国内网络：先设置 `PIP_INDEX_URL`，PyTorch wheel 用 `TORCH_FLAVOR=cu121` 或默认 cu124；HF 镜像通过 init 的 `--hf-endpoint` 设置。若镜像缺少固定 SHA，换回官方 endpoint，不能悄悄改成最新权重。私有资源通过标准 Hugging Face token 环境配置访问，不记录密钥。

## 2. 下载

```bash
spatial catalog assets
spatial download qwen3-vl-2b --dry-run
spatial download qwen3-vl-2b
spatial download vg-llm-data
spatial download vg-llm-data --include 'train/spar_7m.tar.gz' --extract
spatial source geothinker
```

引号确保通配符传给下载器而非 shell 展开。大型视频和模型可能占用数百 GB；`--dry-run` 显示实际目标与过滤条件。下载 receipt 固定 revision，重复命令恢复同一快照。`--extract` 拒绝路径越界和 archive symlink。下载不自动改造上游目录；转换时按标注中的相对路径指定 `--media-root`。

## 3. 构建和检查数据

见 [DATA.md](DATA.md)。创建训练、验证和测试清单后：

```bash
spatial clean-data --input /data/spatial/manifests/spar.raw.train.jsonl \
  --output /data/spatial/manifests/spar-clean
spatial split-data --input /data/spatial/manifests/spar-clean/clean.train.jsonl \
  --output /data/spatial/manifests/spar-splits
spatial audit --train /data/spatial/manifests/spar-splits/train.jsonl \
  --test /data/spatial/manifests/vsibench.test.jsonl --output /data/spatial/runs/leakage.json
```

同一场景不得分到多个 split。默认 80/10/10 只划分 train pool，不重新划分公开 benchmark。论文 benchmark 保持原分割。小数据池可能为空，程序不会复制测试样本凑数。

`split-data` 要求每条样本都有可靠的 `scene_id`；builder 的字段映射/路径识别不能保证所有来源都有场景标签。出现缺失时先补全真实场景身份。若 SPAR 与 VSI 来自相同 ScanNet 场景，需要在训练池中排除测试场景；审计发现重叠后会停止训练。

最小闭环可以直接使用上一步生成的三个清单，无需再命名为后文的多数据集别名：

```bash
# 前提：三个 split 非空，图像齐全，基模已完整下载
spatial run sft --model qwen3-vl-2b --name sft-spar \
  --train spar-splits/train --heldout spar-splits/val spar-splits/test --dry-run
# 核对该计划后直接执行，避免重复使用 run 命令占用同一个名称
spatial train --config /data/spatial/runs/plans/sft-spar.json
spatial run inference --model qwen3-vl-2b --adapter /data/spatial/runs/sft-spar/final \
  --name sft-spar-test --eval spar-splits/test
```

这里得到的是训练池独立划分上的诊断结果，公开 benchmark 另按各自官方 test 构建。

## 4. 训练、蒸馏、推理

示例假定 `spar.train.jsonl`、`llava-hound.train.jsonl`、`spatial.val.jsonl`、`vsibench.test.jsonl` 已放入 manifests 根目录；可直接传绝对路径代替简称。

```bash
spatial run sft --model qwen3-vl-2b --name sft-a \
  --train spar.train:0.8 llava-hound.train:0.2 --heldout spatial.val vsibench.test \
  --set train.steps=1000 --set train.learning_rate=0.00001

spatial run rl --model qwen3-vl-2b --adapter /data/spatial/runs/sft-a/final \
  --name grpo-a --train spar.train --heldout spatial.val vsibench.test \
  --set train.algorithm=grpo --set train.group_size=8

spatial run opsd --model qwen3-vl-2b --adapter /data/spatial/runs/sft-a/final \
  --name opsd-a --train spar.train --heldout spatial.val vsibench.test

spatial run opd --model qwen3-vl-2b --teacher /data/my-compatible-spatial-teacher \
  --name opd-a --train spar.train --heldout spatial.val vsibench.test

spatial run inference --model qwen3-vl-2b --adapter /data/spatial/runs/sft-a/final \
  --name sft-vsi --eval vsibench.test
```

OPD 默认学生 cuda:0、教师 cuda:1，可用 `--set teacher.device=cuda:0` 共卡，但需要足够显存。教师必须使用适配后端；不能仅因为来自 Qwen 就假定几何模型可由 HF 普通类加载。原生 token KL 要求 tokenizer 语义完全一致。不同 tokenizer 的先进教师用 `teacher-export` 收集响应，再用 `offline_kd`；这是异构序列蒸馏，不标成同词表 OPD。教师 rollout 仅在训练 pool 生成，不把 benchmark gold 放入教师训练。

LoRA 输出重新加载时，`--model` 仍是原基模，`--adapter` 指向 final；完整权重训练的输出可直接作为 `--model`。常见参数已有 `sweep-sft.yaml` 和 `sweep-rl.yaml`，用 `spatial sweep --help` 生成实验矩阵。路径与超参通过最终 `runs/plans/<name>.json` 核对。

`run` 先解析快捷参数，再应用 `--set`。`output`、`train.mode`、model/teacher 的 path 和 data 字段由专用参数负责，冲突覆盖会报错；需要完全自定义时使用 `spatial train/infer --config`。`--dry-run` 只保存计划，不做图片、权重、GPU 或显存验收；之后可直接运行这份计划。同名计划不会被覆盖。

显卡和 batch 可用 `--gpus 4 --devices 0,1,2,3 --set train.batch_size=2 --set train.gradient_accumulation=8`；此时全局 prompt batch 为64。内部仍逐样本执行，具体语义与分布式边界见 [DISTRIBUTED](DISTRIBUTED.md)。

## 5. 多数据集／多模型

```bash
spatial suite --config configs/suites/common-six.yaml
# 修改 suite.name 为新的名字后执行，已有目录不会覆盖
spatial suite --config configs/suites/common-six.yaml --execute
spatial run inference --model qwen3-vl-2b --name base-vsi --eval vsibench.test
spatial compare /data/spatial/runs/base-vsi/metrics.json \
  /data/spatial/runs/sft-vsi/metrics.json --output /data/spatial/runs/vsi-comparison.json
```

suite 会实际加载配置中的每个模型并在每份 manifest 上推理，要求模型先下载、清单先构建。其余任务默认最多 8 帧，ReVSI 单独使用最多 32 帧的配置，必须提供对应的 32 帧标注和完整规定视图。任务间不共享同一个总分协议；同一任务内各模型共享输入配置。所有输出仍是诊断结果，不能冒充排行榜分数。

## 6. 官方任务

```bash
spatial source easi
# EASI 的隔离环境按 docs/COMPATIBILITY.md 建好后：
spatial benchmark vsibench --engine-root /data/spatial/external/easi \
  --python /data/spatial/external/easi/.venv/bin/python \
  --model qwen3_vl --model-args pretrained=/data/spatial/models/qwen3-vl-2b,max_num_frames=32 \
  --output /data/spatial/runs/official-vsi --execute
```

同入口支持 mmsi/viewspatial/mindcube（默认 Tiny 子集），以及 ReVSI 的 lmms 与 DSR 的 VLMEvalKit 路径。后两者需要对应 engine checkout，任务存在性会预检；详见 catalog。外部引擎有自己的数据下载位置和模型参数，保留原始结果，默认关闭外部 judge 和提交功能。启动成功仅表示进程完成，仍需检查官方结果与完整覆盖率。

## 7. 把修复提交到原仓库

当前尚未提交上游。`spatial source geowire` 与 `spatial source geopsro` 已自动应用交付的补丁，用户可在该 checkout 中检查 `git diff`，建自己的修复分支再提交。不要重复手工 apply。若直接在自己的原仓库应用，先 `git apply --check /path/to/patches/xxx.patch`，检查通过再 `git apply`；在不同版本上需人工解决上下文差异。

# RGB 与冻结几何分支的受控 SFT

## 本轮问题与边界

判断同一 Qwen3-VL-2B-Instruct、同一训练清单和一轮预算下，额外的场景几何特征是否改善空间 QA。队列顺序不等于权重继承：RGB 基线和几何模型都从相同发布权重开始，几何组不加载 RGB SFT 的 final。

本轮是匹配既有 full-language SFT 的架构探索：不是设计稿中的完整 alignment/LoRA/70:30/RFT 复现。保留原始清单的所有帧、原比例一遍遍历，不暗中改成8帧或重采样。两组的输入token数、模型参数量与计算量不同，不能声称等算力比较。几何模型额外做相同token布局的 learned-null 推理消融。

资源依据：[Qwen3-VL](https://github.com/QwenLM/Qwen3-VL)、[VGGT聚合器](https://github.com/facebookresearch/vggt/blob/a288dd0f14786c93483e45524328726ab7b1b4ce/vggt/models/aggregator.py)、[RoboRefer](https://github.com/Zhoues/RoboRefer)。RoboRefer的独立深度分支/对齐训练是参考思路；这里接入VGGT聚合特征，不把两者视为同一架构。用户提供的私有手稿不得进入公开仓库，也不能把其表中数字当作本轮实测结果。

## 锁定契约

| 项目 | RGB基线 | 几何组 |
|---|---|---|
| 起点 | Qwen3-VL-2B-Instruct 原始权重 | 同左，新初始化几何接口 |
| 数据 | SPAR234149 + Hound63750，297899条 | 完全相同，继承基线manifest快照 |
| 主训练 | 1 epoch，global batch64，seed3407 | 同左，micro1×8卡×GA8 |
| 可训练模块 | 全语言侧，冻结原视觉侧 | 全语言侧＋几何接口；两种视觉编码器冻结 |
| 优化 | AdamW，LR1e-5，cosine，warmup3% | 相同LR，包括几何接口 |
| 几何 | 无 | VGGT final aggregate patch features，2048维 |
| 接口 | 无 | 256维瓶颈、64 learned-query slots、两层projector |
| Dropout | 无 | 20%整样本learned-null几何替换 |
| 主评测 | 相同ReVSI/VSI清单、native32video、tagged512、greedy | 同左，额外normal/null两种条件 |

额外使用frame/patch正弦位置编码，明确属于本实现选择；flatten本身不能使cross-attention对顺序敏感。VGGT使用同一有序、已标记RGB图像，经上游square padding缩放到448，保留所有帧；不读取真实depth/pose/答案作为几何输入。64个几何token放在最后一个native vision-end之后、问题之前，labels=-100、mm type=0。保留Qwen原生DeepStack视觉mask和mRoPE生成逻辑。

旧节点目录名含seed3408，但checkpoint实际seed/data_seed均为3407（已读保存参数确认）。本轮明确使用3407；不能把旧任务写成3408独立种子复现。TF LazyModule曾覆盖class导出替换，新入口改为同一class初始化方法包装，保留可pickle的原TrainingArguments，并测试实际值。

## 为什么在线提取

完整清单有276782组唯一有序帧，共1791565帧。按每帧1024×2048×2字节保存dense bf16特征，仅payload约7.51TB。不会默认创建这种缓存。冻结VGGT在线no-grad提取，只在诊断中缓存少量样本；不换成XYZ统计、随机特征或静默降采样。

VGGT权重版本 `860abec7937da0a4c03c41d3c269c366e82abdf9`，源码版本 `a288dd0f14786c93483e45524328726ab7b1b4ce`。原权重严格加载，去掉不执行的预测heads；聚合器仍跨全部帧计算。

## 自动顺序与阶段门

1. RGB真实多图loss/mask/梯度、更新、保存重载、native视频检查。
2. 混合数据batch测速与长样本压力测试；八卡DDP smoke；RGB正式SFT和评测。
3. 等基线明确释放GPU，确认融合真实GPU验收accepted，再八卡融合DDP smoke。
4. 几何组从原始权重开始一轮SFT，checkpoint保存model/optimizer/scheduler/RNG，可从一致合同恢复。
5. final重载，固定各题型首题格式门（parse≥90%、truncation≤5%，不按正确率挑配置）；通过后normal/null全量评测，失败标blocked而非0分。

入口例子（真实路径只放本地配置）：

```bash
"$NODE_ROOT/envs/qwen35/bin/python" scripts/run-qwen3vl-geometry.py \
  --root "$NODE_ROOT/studies/qwen3vl-geometry-v1" \
  --model "$SOURCE_ROOT/models/Qwen3-VL-2B-Instruct" \
  --source "$VGGT_SOURCE" --weights "$SOURCE_ROOT/models/VGGT-1B" \
  --baseline-state "$BASELINE_ROOT/state/pipeline.json" \
  --gate "$GEOMETRY_GATE/acceptance.json" --detach
```

等待器最多等基线7天、等GPU空闲2小时，阶段命令有超时；异常记录在独立state/logs，不启动后续昂贵步骤。启动时冻结本次代码副本，避免等待期间修改公共代码影响排队实验。训练输出和本机权重路径不提交Git。

## 已验证与尚未完成

- 独立几何模块12项CPU测试：含all-null微批1的全部参数保留计算图，避免DDP unused-parameter故障。
- tiny真实Qwen集成7项CPU测试：原DeepStack/mRoPE、subsetlogits loss/grad、save/reload、generation只在prefill消费几何、新增模块from_pretrained初始化。
- 真实GPU：3帧loss0.7498；32帧loss2.8428，接口非零梯度及参数更新；峰值reserved23.04GiB；保存重载logit最大差0；normal/null输出不同。这些只证明接口，不是指标提升。
- 额外真实32帧native-video生成已通过：重载模型生成4个token，geometry adapter仅prefill调用一次，保留原video grid；无需丢帧或转成文本几何。
- 后台融合等待器已启动；队列11项CPU/mock测试通过，实际tiny Qwen集成含online分支8项通过。等待期间不占GPU，须等RGB基线训练及评测释放全部卡。
- 初次真实gate发现新增模块缺失权重初始化不完整，已为queries/null/MHA补显式初始化，重新通过；初次tinygenerate发现forward签名隐藏原参数，已修复并重测。旧失败产物保留。
- 八卡融合训练、全量成绩与稳态ETA仍以实际队列产物为准，不能从单样本耗时承诺完成时间。

网盘工具与手动认证见 [下载指南](BAIDU_DATA_DOWNLOAD.md)。未登录不阻塞已有SPAR/Hound数据上的SFT。

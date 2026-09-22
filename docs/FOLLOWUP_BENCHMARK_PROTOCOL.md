# 互补 benchmark 适配契约

`spatial_intelligence.followup_benchmarks` 提供 `validate_rows(rows, benchmark)`、`score_prediction(row, response, benchmark, truncated=False)`、`summarize(scores, benchmark)`。ReVSI/VSI-Bench 委托既有 `spatial_eval`，不改原算术。

## 已准备的四项

| 名称 | 全量问题数 | 汇总 |
|---|---:|---|
| MMSI | 1000 | 题目微平均及类别分层 |
| MindCube Tiny | 1050 | 题目微平均；among600/around250/rotation200 |
| ViewSpatial | 5712 | 微平均及类别宏平均，后者不冒充已核实官方overall |
| CV-Bench | 2638 | 0.25×ADE20K + 0.25×COCO + 0.5×Omni3D |

评分使用0–1单位；SITE的CAA允许负值。保留无效输出、超长截断输出在分母中，均计错。输出`total/expected_total/complete_benchmark`；少量gate样本不能称全量正式结果。归一化清单保留`original_question`、规范`choices`、一致的`answer/ground_truth`、媒体顺序与source ID；question不重复拼接选项，instruction仅要求输出字母。

当前选择器仅接受清楚的最终字母或`<answer>/<final>`标签，不使用金标引导解析、不随机猜、不把数字问题自动交给VSI算术。这是预先声明的**严格抽取适配**，不等同于各项目宽松正则或LLM judge。所有方法共享同一清单和解析器。

MindCube仅tiny作为本项评测，已逐一验证与官方10K训练集ID、媒体和媒体目录组无交集；完整21,154条不能当独立测试集。官方实现排除translation，其tiny实际没有translation，故本清单分母不变。跨数据集或checkpoint预训练污染未因此得到排除。

## SITE：评分已实现，输入仍阻塞

官方SITE是4449图像题+3619视频题，共8068，指标为：

`CAA = sum(correct_i - 1/K_i) / sum(1 - 1/K_i)`

不是普通accuracy，也不是每题标准化分数的简单平均。模块同时报告micro accuracy、CAA分子/分母、类别CAA。官方抽取器解析失败时随机选项；本适配明确不采用随机猜测。

SITE有1187道题的选项包含图像占位，1477道图像题的问题含图像占位，必须保存交错位置及image-option对应关系。现有全部图片前置的collator不能直接视为等价。故准备器产生`blocked_input_adapter` receipt，不生成冒充可评测的全量manifest；视频也未擅自采帧。GeoRoute论文要求SITE时，该阻塞应保留；GeoFits自己的四项互补评测不应被当成已含SITE。

## 证据与复现

- [MMSI官方接入评分](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/lmms_eval/tasks/mmsi_bench/utils.py)：题目微平均。
- [MindCube源码](https://github.com/mll-lab-nu/MindCube/tree/b8b7062adf6d3e49d588a7d014a0a787553d09ec/src/evaluation/core)：base_metrics与extractors。
- [CV-Bench数据卡](https://huggingface.co/datasets/nyu-visionx/CV-Bench)：明确分来源加权式。
- [ViewSpatial数据卡](https://huggingface.co/datasets/lidingm/ViewSpatial-Bench)：字段及五类任务；尚未把公开表格overall与本适配宣称完全等价。
- [SITE官方汇总](https://github.com/wenqi-wang20/SITE-Bench/blob/main/eval_scripts/aggregate.py)与[输入/抽取实现](https://github.com/wenqi-wang20/SITE-Bench/blob/main/eval_scripts/sitebench/utils.py)。

```bash
python scripts/prepare-followup-benchmarks.py \
  --assets-root "$ASSET_ROOT/datasets" --output "$NODE_ROOT/benchmarks-v1"
python -m pytest tests/test_followup_benchmarks.py -q
```

工具仅复用已下载资产，逐图实际解码，独立输出且不覆盖旧清单；每项失败独立记入receipt。`full_model_verified=false`表示这里只验数据，不能替代GPU模型输入/生成gate。

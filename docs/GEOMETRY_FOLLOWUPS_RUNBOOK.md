# 后续几何实验：顺序、资产与验收门

最新用户指令已重新授权部署：当前论文训练/评测及RFT → GeoRoute及其对照 → GeoFits。此前手动暂停记录保留，采用新v3计划，不重启旧v1空命令计划。不能抢占现有训练；失败不写成0分。当前数据和真实runtime验收仍不足，授权不等于立即GPU开跑。

实现契约：[GeoRoute](../projects/georoute/README.md)、[GeoFits](../projects/geofits/README.md)。
共用配置：[study contract](../configs/geometry-followup-studies.json)。
资源来源与尚未解决的转换：[data audit](GEOMETRY_FOLLOWUP_DATA_AUDIT.md)。

## 状态含义

| 状态 | 实际含义 |
|---|---|
| downloading | 数据/模型字节在传输，不代表标签/媒体可用于训练 |
| waiting_previous_paper | 现有论文尚未全部成功释放GPU |
| waiting_preparation | 五源数据或完整实现验收不足；不能启动正式GPU训练 |
| blocked_implementation | 尚无经过验收的完整可执行入口；并非训练已经排好并能无人值守完成 |
| running_stage | 子进程运行中；是否实际更新仍需独立证据 |
| complete + accepted | 配置定义的全部阶段验收成功 |

## 队列格式

`scripts/build-geometry-followup-plans.py`从私有节点映射生成三个角色的实际GeoRoute训练、重载和ReVSI/VSI命令：primary负责full/noTIP，control_a负责匹配RGB/oneSTB，control_b负责finalOnly/postMerger。所有角色等待前篇两组RFT和原四卡对照完成。各角色需同时完成次级benchmark证据，GeoFits主组才可跨节点放行。GeoFits非full消融与完整评测串联仍未实现，不用空命令或同一full结果冒充。

```bash
python scripts/build-geometry-followup-plans.py --bindings private-bindings.json \
  --scientific configs/geometry-followup-studies.json --output private-generated-v3
```

生成不启动；运行需显式授权flag与独立root。每个worker计划写进queue的`stage_plans`并随代码冻结。新的数据receipt必须匹配`shared-five-source-arkitscenes100k-v2`；旧五类配额receipt不会误放行。

`scripts/run-geometry-followups.py`读取ignored私有plan，要求明确root/python/gpus、前序receipts、逐阶段requirements及argv命令。计划和代码立即快照；变更需要新版本。阶段入口尚未准备好时，`commands: []`必须保持阻塞，不能填一个不存在的文件或把CPU单元测试当作正式训练完成。

```bash
"$NODE_ROOT/envs/geometry-matrix/bin/python" scripts/run-geometry-followups.py \
  --plan "$NODE_ROOT/.private/geometry-followups-plan.json" --detach --authorize-run
```

所有前序条件使用精确字段匹配，旧RFT要求`status=complete`且`gpu_work_finished=true`。多节点需要在私有plan中列出所有对应前序状态；仅本机空闲不足以跳过其它节点的论文实验。

新阶段应分别检查数据receipt的`media_verified`/`leakage_checked`和完整runtime验收，不把下载的`assets-receipt.json`冒充数据receipt。完整acceptance由真实程序输出；不得手工把`trained_pending_acceptance`改成accepted来放行。

队列在依赖通过后还检查所分配GPU的显存空闲，并持有独立进程锁。只管理自己的进程组；既有训练、checkpoint、环境均保留。超时或失败保留日志并阻断相应下游；独立无依赖的对照可继续。

## 当前已做与仍欠缺

- 新资产下载器有固定revision及文件字节清单，服务器使用镜像URL续传，避免计算节点直连HF API失败；限制并发，不在GPU上跑下载工作。
- 两个模型的数学核心、Qwen hooks和若干真实tiny模型CPU回归已有实现；独立stage worker支持全参数GeoRoute，不更改旧LoRA RFT。
- GeoRoute新进程重载/更新检查入口和专用ReVSI/VSI推理入口已写好，尚未完成真实GPU端到端验收；GeoFits仍缺完整训练、重载与评测串联。不能用vanilla模型加载器悄悄丢掉新增模块。
- 最新检查：静态解析179个Python文件、20个JSON及111个本地链接通过；阶段/队列测试5通过、1项Windows锁测试跳过。另在Linux使用真实Qwen processor完成5项输入回归通过。这些不是正式训练或benchmark结果。
- 真实教师大权重、多源数据转换、完整训练—重载—多benchmark流水线尚未全部验收。**当前挂起的是后续实验准备/验收队列，不宣称无人值守全量训练已经全部就绪。**
- 完整吞吐和ETA尚未知。GeoRoute所有帧作tracking anchor的图生成可能比SFT更昂贵；需分别报告冷缓存构图时间、热缓存训练、TIP额外更新和评测。不能沿用旧融合模型的8.1小时作为本次ETA。

## 后续接管检查

1. 查`state/followups.json`和下载receipt；保留失败记录，避免重复下载/重复controller。
2. OpenSpatial采用用户指定ARKitScenes来源100K，不再要求五类配额；确认实际选集ID、媒体与scene隔离，解决VLM3R勘误及各benchmark实际schema/评分，冻结五源manifest。
3. 完成真实VGGT tracking与Pi3层位/网格验收；不得把模拟教师测试当成真权重通过。
4. 为GeoRoute执行真实分布式诊断、保存恢复、完整评分，再放行正式TIP/SFT与对照；GeoFits必须等GeoRoute所有必选项完成。
5. 将实际receipt路径与已验收的可执行stage commands写入**新版本**plan，按节点空闲资源部署；不能热改已armed快照。

## 前篇RFT的SFT来源与LoRA边界

两组前篇RFT（混合数据、4D专用对照）均指定上游`geometry-matrix-v4/runs/roborefer-downsample-unfrozen-sft/final`，而非诊断两步权重，也不是根据test挑选的checkpoint。最近核查时该正式final尚未产出并验收，RFT仍等待前序。实际节点绝对路径只保留在私有plan。

上游从Qwen3-VL-2B-Instruct和VGGT初始化，先对齐接口，再用SPAR234149＋Hound63750进行正式SFT；该阶段语言、原生视觉、接口和VGGT均可训练。LoRA出现在**后续RFT**：`spatial_intelligence/geometry_rft.py:add_policy_adapter`对语言attention的q/k/v/o及MLP的gate/up/down应用rank64、alpha128、dropout0；`geometry_adapter`通过modules_to_save全量训练，原生RGB与SFT后的VGGT冻结。这不是全参数RFT。

这两项新研究的SFT按用户要求不使用LoRA；冻结外部教师不等于对主模型使用LoRA。前篇已经armed的RFT配置没有被静默改为全参数，若改变必须单独版本化并重新做显存/更新验收。

两篇之后的多领域评测见[多样化评测方案](GEOMETRY_GENERALIZATION_EVAL_PLAN.md)，该方案不代表全部新环境已安装。

# 三节点排程v3与ETA边界

后续决策已替换本页旧队列：GeoRoute v4 移除 matched RGB，RFT v3 改为语言全参数＋融合模块、冻结视觉、无 LoRA。三个 follow-up 等待器与两个 RFT 等待器已重新部署；旧等待器确认没有训练子进程后停用，历史产物保留。见[最新契约](GEOMETRY_NO_LORA_AND_STRENGTH.md)。下文 v3 排程及 LoRA 待确认描述仅为历史快照。

## 部署复核

三个角色的 v3 controller 均已启动并复核存活，状态为 `waiting_previous_paper`，不是新研究已经开训。既有训练未中断。节点地址、PID 和本机路径仅保存在私有部署记录中。

OpenSpatial ARKitScenes 来源补充下载已在服务器通过镜像启动：首个约 5 分钟快照为 13,727 条来源记录、约 1.01 GB，目标 100,000 条；按这一短窗口外推还需约 30–45 分钟，仅为网络下载估计，不包含 scene 映射、泄漏排查、图像导出和训练验收。下载限制候选总量 20 GiB，并保留磁盘余量。采用固定版本的确定性分片前缀，不能称为对完整来源总体的均匀随机抽样。

GeoFits CPU 集成测试 3 项、worker 逻辑测试 2 项通过；发布教师权重、多卡 2B/ZeRO、断点恢复和正式 benchmark 尚未验收，正式运行 gate 保持关闭。该 CPU 结果不代表部署快照已通过真实 GPU 验收。

用户已授权RFT→GeoRoute→GeoFits，不额外空等16小时。使用实验设计skill区分实际运行、数据准备、可执行命令和完整验收。

## 当前实测（2026-09-20）

| 资源角色 | 当前任务 | 本阶段剩余 | 后续链路 |
|---|---|---|---|
| 8卡主组 | downsample alignment 110/665，loss1.2252 | 约7.1h | 解冻VGGT SFT→评测→混合RFT |
| 4卡对照组 | query64 alignment85/665，loss1.3583 | 约13.6h | 解冻VGGT SFT→评测 |
| 8卡第二组 | 原始VGGT融合SFT200/4655，loss1.0152 | 约16.7h | 评测→共享主组alignment后的冻结VGGT SFT→评测→4D-only RFT |

ETA只覆盖已实测的当前阶段，不能把alignment速度套到解冻VGGT SFT。原始融合组本身已覆盖未来约16–17小时；其后还有正式工作。两种局部吞吐优化仍未切入正式快照，ETA不乘虚构加速倍数。

## RFT实际门

现有两个controller存活，无重复启动。共用正式主组SFT final，group8；下载和原始SFT验收之后才开始实际RFT gate。媒体6142个中观察到2780个、已解码验收563个，不能把36710/14752原始标注数当最终train rows。完整RFT ETA需最终清单及热态步速。

现有v2是语言LoRA+接口训练，不是全参数。用户已就此质疑，本次再次明确询问是否将RFT也改为全参数；未静默改动已armed配置，不称RoboRefer原版RFT。两个新研究SFT明确无LoRA。

## 新研究资源映射

- primary：GeoRoute full、noTIP；后接GeoFits full。
- control_a：GeoRoute matched RGB、oneSTB。
- control_b：GeoRoute finalOnly、postMerger。
- GeoFits其它消融的训练/评测尚未完整实现，显式列为pending；不会作为已部署工作虚报占卡时长。

每组GeoRoute有真实TIP/SFT/重载/两主benchmark命令；数据和runtime未通过时保持等待。次级benchmark同样必须真实有验收结果，不能仅因两主benchmark结束就宣称论文完成。GeoFits等待三角色GeoRoute完成。

OpenSpatial采用用户指定ARKitScenes来源100K；现有两个探针不等于100K到齐。筛选/导出脚本已准备，必须有源scene映射和benchmark排除，不借未知类别为借口阻塞来源选择，也不伪造scene隔离。该缺口与VLM3R转换、真实GPU runtime及次级评测适配仍影响启动日期。

结论：当前工作量已经超过16小时；尚不能承诺16小时后RFT和两篇新研究均在训练，更不能提供三篇全完成的精确ETA。状态依据实际receipt，不依据计划任务数量。

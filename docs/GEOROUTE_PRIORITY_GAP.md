# GeoRoute 填补 RFT 等待窗口

用户授权：当前本机 SFT 和全部评测结束后，可提前运行独立的 GeoRoute
`final_only` 正式 TIP/SFT。它从 released VLM 初始化，不依赖 RFT 权重。
RFT 优先：初始化权重及 RFT 数据就绪后，请求 GeoRoute 在完整优化器断点
暂停；RFT 及其测评结束后恢复，再进行既定奖励消融。后续 GeoRoute 队列
复用已验收的 TIP/SFT 产物，继续评测及 post-merger，不重复训练。

## 实际前提，不用填充计算代替实验

- 六源训练清单必须 ready、媒体可读；OpenSpatial-only 授权不豁免损坏媒体。
- `verify-georoute-runtime.py` 必须真实完成多图与32帧的 TIP/SFT 更新、保存、
  重载和生成验证，才写正式训练所需的 runtime receipt。
- 当前 SFT/eval 未释放 GPU 时不能启动。诊断权重不初始化正式训练。
- 如果 RFT 先就绪，跳过提前训练窗口，优先运行 RFT；不为填空而延迟它。
- 若数据或 GPU 验收尚未完成，队列仍是等待，不等于正式训练已启动。
- 已开始的一次短 runtime gate 会执行到命令安全结束；正式训练的抢占发生
  在 optimizer 边界，不承诺收到请求的同一秒立即释放显卡。

## 互斥与可恢复性

`run-georoute-gap.py` 是单一资源所有者。私有部署前必须核实旧 RFT 和奖励
消融控制器仅在等待、没有活动子任务，交接后才启动该所有者；当前 SFT
不受影响。正常后续队列依赖奖励消融完成，因此不会与提前任务并发写入。
不能只用瞬时显存空闲来代替任务所有权。

worker 的 `--pause-request` 文件在所有 rank 一致的 optimizer 边界生效。
它保存模型、优化器、调度器、Trainer 状态及逐 rank RNG，写 `pause.json`，
然后退出；不生成成功的 final/completion。`torchrun` 可能把 rank 的75退出码
转换为1，所以调度器核对本次请求对应的新暂停收据和完整 checkpoint，
不能仅凭父进程退出码认定暂停成功。恢复保持数据顺序、world、micro、GA、
总预算和调度一致；不缩短 epoch，且最终参数审计与最初初始化比较。

异常有可见 failure 状态；这不是已经在完整 GPU 模型上证明任意崩溃精确恢复。
现有 CPU Trainer 轨迹测试与后续真实 GPU 验收应分别记录。

## 数据链修复

原 `Unsafe archive member` 来自复用文件的合法 symlink 指向旧只读目录，
被普通解包的路径归属检查误拒绝。修复只允许显式白名单复用根，核对普通
归档成员及精确文件大小，仍拒绝 `..`、归档链接、目录链接、非授权目标。
视频以视频解码器验证，不再交给 PIL。另两条官方多图片标注混入 MP4，
保留原始记录并显式隔离，不伪造替代帧；未知缺失仍阻止数据验收。

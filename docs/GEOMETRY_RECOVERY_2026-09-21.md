# 几何 SFT / RFT 健康检查与恢复

日期：2026-09-21。本文为排错快照，不是完整实验结果报告。资源用角色描述，私有部署映射不发布。

## 后续授权与进度更新

用户已明确同意排除源端17个空视频对应的98条训练QA。新版准备仅在显式 `--exclude-upstream-empty` 下执行：4DRL训练32656→32558，验证691不变；SpatialLadder训练14517、验证235不变；总训练47075条。保留原split、其余样本顺序和完整排除原因，原媒体不删除。旧失败receipt保留，不改写成成功。最终条数仍需新版解码验收核实。

后续检查中，下采样micro1完成6步混合样本、2步长样本诊断，四组件更新和独立重载均通过；混合样本热态1.4964样本/秒、峰值reserved25.82GiB。仅按该诊断外推完整SFT约55.3小时，不含评测，也不是正式已完成训练。原始融合SFT完成4655步并进入ReVSI评测；两组RFT仍等待前序正式SFT与数据验收。

旧128条实验的数据来源明确为SPAR-234K，不含Hound或4DRL：从1184条合格的1–3帧物体空间关系Yes/No问答中，按scene分组后固定抽取512训练、128验证。两种题型 `obj_spatial_relation_oo` / `obj_spatial_relation_oo_mv` 各256条训练、64条验证。训练/验证场景交集0，显式排除ReVSI场景。初始化是原始Qwen3.5-2B而非完整SPAR SFT权重；历史实验使用LoRA rank8，不能与当前无LoRA的4DRL RFT混淆。

## 结果边界

4DRL 混合 RFT 与 4D-only RFT 均未完成，也没有 DSRBench / ReVSI 配对成绩，不能判断哪种组合最好。两个 RFT 的阻塞来自正式 SFT producer 失败；数据准备还存在独立阻塞。GeoRoute / GeoFits 后续依赖未完成，不将等待队列算作训练。

历史 Qwen3.5 空间 Yes/No 小实验在同一 128 条来源隔离验证集上：原模型 49/128（38.28%），100 步 reference GSPO 78/128（60.94%），100 步 SFT 对照 86/128（67.19%）。两种训练 prompt 抽样数一致，但 RL 额外生成 rollout，计算预算不等；单 seed 小实验的点估计以 SFT 更高，不支持“正式 4DRL 中 SFT 一定优于 RL”。此前原生 Qwen3-VL GSPO pilot 被用户取消，仅完成 4 步，不能称为完整 RL 结果。

## 本轮发现与修复

| 对象 | 证据 / 根因 | 恢复策略与当前验收 |
|---|---|---|
| 下采样＋解冻 VGGT | Alignment 665 步完成并重载；SFT 首次前向遇到 FP32 residual 与 BF16 LayerNorm 参数冲突 | 仅在 adapter 输入边界显式对齐参数 dtype，保留 VGGT FP32 图像归一化和梯度；回归通过，新版本重新诊断，不重做 alignment |
| 64-query＋解冻 VGGT | Alignment 665 步完成、reload delta0；SFT micro1 mixed/pressure 与更新通过，但 mixed reserved37.49/39.39GiB（95.17%）超过92%门；micro2/4 OOM | 新版本使用 expandable_segments，复用完全相同 mixed/pressure 样本和 alignment；92%门不放宽；诊断完成前不宣称修复成功 |
| 原始 VGGT 融合 | 检查时正式 SFT 仍推进、loss/grad有限 | 不打断；结束后按既有队列评测 |
| 冻结下采样对照 | 尚在等待前序工作，可能遇到相同 dtype 边界 | 旧等待器安全停用，新快照包含公共修复；alignment只读复用 |
| 4DRL 媒体 | 6142个路径中17个源端即0字节，精确重试也返回0字节；涉及98条候选QA | 不伪造完整下载或静默排除。等待明确排除授权或补齐原视频，原始证据保留 |

### ETA

原64-query micro1诊断吞吐约0.8031样本/秒，297899条外推约103小时纯SFT。这是旧诊断配置估计，不含评测，也不是已接受的新正式ETA。显存优化后的实际吞吐需重新测量。下采样SFT曾在首个前向失败，因此没有可用正式ETA。不要把已完成alignment的ETA延用于SFT/RFT。

## 验证与发布

提交候选快照通过静态解析和本地链接检查。隔离服务器 CPU 回归为79 passed、1 skipped；另有下采样精度边界及既有接口测试通过。CPU结果不证明8卡全参数RFT的显存、吞吐或奖励有效性。新的多卡诊断尚在运行。

所有复原均使用新版本root，保留失败日志与已完成alignment；依赖RFT和后续研究重新绑定恢复后的producer。没有覆盖正在训练的代码，没有将失败结果填成0分。发布前检查暂存区隐私标识，私有稿件、服务器映射、凭据、模型和数据不提交。

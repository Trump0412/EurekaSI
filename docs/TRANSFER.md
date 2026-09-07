# 从空间问答到具身、驾驶与世界模型

## LIBERO 接口已经做什么

`transfer/libero.py` 调用真实 LIBERO Benchmark/OffScreenRenderEnv API：固定任务顺序、选择官方初始状态、seed/reset、warmup、逐步动作执行、成功谓词、逐 episode 记录、每任务与总体成功率。输入动作必须是七维有限数值、符合 normalized OSC_POSE [-1,1]。不把自然语言答案自动当机器人动作。

先在独立 LIBERO 环境完成官方安装和资产路径配置，然后安装本库的最小包，不装 `[train]`：

```bash
/path/to/libero-env/bin/python -m pip install -e /path/to/spatial-intelligence --no-deps
# 该环境还需 PyYAML、Pillow、NumPy；保持与仿真器兼容版本
MUJOCO_GL=egl /path/to/libero-env/bin/python -m spatial_intelligence libero-eval \
  --config configs/libero-smoke.yaml
```

所附 ZeroPolicy 只测试接口，会明确写 `smoke_only_not_model_result`。本次没有执行真实 MuJoCo/LIBERO；通过的是动作与 rollout 合约测试。无训练好的 action policy 时，无法声称“空间模型已经在 LIBERO 上评测”。

## 接入你训练好的模型

实现一个动作策略 factory，接口为：

```python
class MyActionPolicy:
    def __init__(self, options):
        # Load your spatial encoder checkpoint + action-trained head + normalization.
        ...
    def reset(self, *, instruction, seed):
        ...
    def act(self, observation, instruction):
        # RGB/proprioception -> encoded spatial state -> normalized OSC_POSE action[7].
        ...
    def identity(self):
        # Return encoder/head/normalization file hashes and training-demo split.
        ...
```

配置 `policy: spatial_intelligence.transfer.policies:PluginPolicy`，在 policy_options 中填 factory 与 config，`smoke_only: false`。PluginPolicy 只转发 agentview/wrist RGB 与公开本体状态，避免使用 simulator object state 作为特权视觉输入。相机翻转、图像处理、动作归一化、gripper 符号必须与动作训练完全一致。

当前尚未附训练好的动作 head、LIBERO demonstration 的 BC 训练器或远程多环境策略服务；它们需要数据与训练资源。可先复用你已有 VLA 的 action head，替换/增加本库空间编码分支，再按下面的公平协议研究迁移。

## 最小可发表迁移设计

1. 固定同一 action head 架构、训练 demonstration、优化步数、相机输入、初始状态与 seeds。
2. 比较 base VLM、SFT VLM、清洗数据 SFT、OPD、几何融合五种冻结 encoder。只变 encoder，避免动作训练量同时变化。
3. 报告成功率、每任务结果、无效动作率、推理时延、OOD 方位/颜色/布局变化；按场景/任务重采样置信区间。
4. 分开报告 freeze encoder + BC head 与 end-to-end action finetuning。后者提升不能直接归因于预先获得的空间能力。

GeoThinker 论文摘要报告 embodied referring 与 autonomous driving 泛化，这不等于公开提供了 LIBERO 策略或 action checkpoint。空间指代是更轻量的首个迁移测试，之后才进入闭环操作。

## 自动驾驶和世界模型

这两条线暂列研究协议，不提供伪造的可运行 integration。驾驶先以封闭数据集测试多目标相对位置、运动关系、遮挡后跟踪和规划约束，再研究 simulator planning 指标；空间问答涨分不能代替安全性评估。世界模型用冻结空间表征预测未来位置/关系/可达性，对比 RGB latent 与语义特征，在跨场景/跨相机条件衡量一致性和预测误差。

第一篇研究宜选“空间指代 + 一个 LIBERO suite”作为应用闭环，驾驶和世界模型不要同时变成三个未完成的大项目。

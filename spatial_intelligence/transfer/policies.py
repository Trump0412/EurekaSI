import numpy as np

class ZeroPolicy:
    def __init__(self,options):pass
    def reset(self,*,instruction,seed):pass
    def act(self,observation,instruction):return np.array([0,0,0,0,0,0,-1],dtype=np.float32)
    def identity(self):return {'status':'smoke_only_not_model_result'}

class PluginPolicy:
    """Wrap a separately trained action policy without giving it simulator privileged state."""
    def __init__(self,options):
        from ..io import symbol
        self.policy=symbol(options['factory'])(options['config'])
    def reset(self,**kwargs):self.policy.reset(**kwargs)
    def identity(self):return self.policy.identity()
    def act(self,observation,instruction):
        allowed=['agentview_image','robot0_eye_in_hand_image','robot0_eef_pos','robot0_eef_quat','robot0_gripper_qpos']
        return self.policy.act({k:observation[k] for k in allowed},instruction)

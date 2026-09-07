import numpy as np
import pytest
from spatial_intelligence.transfer.libero import valid_action,rollout
from spatial_intelligence.transfer.policies import ZeroPolicy

class Env:
    def seed(self,seed):pass
    def reset(self):self.steps=0
    def set_init_state(self,s):return {}
    def step(self,a):self.steps+=1;return {},0,self.steps==3,{}
    def check_success(self):return self.steps==3


def test_rollout_records_actual_success():
    result=rollout(Env(),ZeroPolicy({}),np.zeros(2),'move',42,10,warmup=1)
    assert result['success'] and result['steps']==2 and len(result['actions'])==2

@pytest.mark.parametrize('action',[[0]*6,[2]*7,[float('nan')]*7])
def test_invalid_actions_rejected(action):
    with pytest.raises(ValueError):valid_action(action)

import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('spatial_verl_reward', Path(__file__).resolve().parents[1] / 'scripts/verl-spatial-reward.py')
reward = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reward)


def test_binary_reward_requires_unambiguous_answer():
    assert reward.compute_score('spar_spatial_yesno', 'Yes.', 'Yes')['score'] == 1
    assert reward.compute_score('spar_spatial_yesno', '<answer>No</answer>', 'No')['score'] == 1
    assert reward.compute_score('spar_spatial_yesno', 'Yes, but maybe no.', 'Yes')['score'] == 0
    assert reward.compute_score('spar_spatial_yesno', 'No', 'Yes')['score'] == 0


def test_binary_reward_rejects_unsupported_source_and_gold():
    with pytest.raises(ValueError):
        reward.compute_score('revsi_test', 'Yes', 'Yes')
    with pytest.raises(ValueError):
        reward.compute_score('spar_spatial_yesno', 'Yes', 'left')

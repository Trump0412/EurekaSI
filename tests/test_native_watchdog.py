import importlib.util
from pathlib import Path


spec = importlib.util.spec_from_file_location('native_watchdog', Path(__file__).parents[1] / 'scripts/watch-native-rl.py')
watchdog = importlib.util.module_from_spec(spec)
spec.loader.exec_module(watchdog)


def test_duration_is_not_optimizer_progress():
    text = 'step:4 - training/global_step:4 - timing_s/step:449.513927 - actor/loss:0.1\n'
    assert watchdog.optimizer_steps(text) == 4
    assert watchdog.optimizer_steps('timing_s/step:422.123\n') == 0


def test_training_counter_max_and_boundaries():
    assert watchdog.optimizer_steps('training/global_step:1 - x\ntraining/global_step:100\n') == 100
    assert watchdog.optimizer_steps('foo/training/global_step:500 training/global_step:4.5') == 0

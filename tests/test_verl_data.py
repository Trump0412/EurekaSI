import asyncio
import importlib.util
from pathlib import Path
import sys
import types


def test_pixel_limits_are_consumed_without_mutating_images(monkeypatch):
    base = types.ModuleType('verl.utils.dataset.rl_dataset')
    base.RLHFDataset = object
    monkeypatch.setitem(sys.modules, base.__name__, base)
    helper = types.ModuleType('qwen_vl_utils')
    observed = []
    def decode(messages, **kwargs):
        observed.extend(messages[0]['content'])
        return ['image0', 'image1'], None
    helper.process_vision_info = decode
    monkeypatch.setitem(sys.modules, helper.__name__, helper)
    spec = importlib.util.spec_from_file_location('bounded_spatial_test', Path(__file__).resolve().parents[1] / 'spatial_intelligence/verl_data.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    content = [{'type': 'image', 'image': '/first.png'}, {'type': 'image', 'image': '/second.png'}]
    result = asyncio.run(module.BoundedSpatialDataset.process_vision_info(
        [{'role': 'user', 'content': content}], 16, {'image_kwargs': {'min_pixels': 3136, 'max_pixels': 200704}}))
    assert result[0] == ['image0', 'image1']
    assert [item['image'] for item in observed] == ['/first.png', '/second.png']
    assert all(item['max_pixels'] == 200704 for item in observed)
    assert all('max_pixels' not in item for item in content)

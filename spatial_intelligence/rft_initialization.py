"""Explicit released-native initialization, separate from formal SFT evidence."""
import json
from pathlib import Path


def is_native(plan):
    return plan.get('model_kind', 'geometry') == 'native_qwen3vl'


def validate_native_initialization(plan):
    if not is_native(plan) or plan.get('initialization') != 'released_native':
        raise ValueError('Explicit native released initialization required')
    if plan.get('sft_receipt') or plan.get('vggt_source'):
        raise ValueError('Native control must not declare SFT or VGGT inputs')
    if plan.get('use_lora') is not False or plan.get('policy_scope') != 'language_full_geometry':
        raise ValueError('Native control requires non-LoRA full-language training')
    root = Path(plan['model_checkpoint'])
    config = json.loads((root/'config.json').read_text())
    if config.get('model_type') != 'qwen3_vl' or config.get('geometry_matrix'):
        raise ValueError('Not a plain Qwen3-VL checkpoint')
    weights = list(root.glob('*.safetensors'))
    if not weights or any(p.stat().st_size == 0 for p in weights):
        raise ValueError('Released model weight files missing')
    for index in root.glob('*.safetensors.index.json'):
        entries = json.loads(index.read_text())['weight_map'].values()
        if any(Path(name).name != name or not (root/name).is_file() for name in entries):
            raise ValueError('Missing or unsafe released checkpoint shard')
    return dict(checkpoint=str(root.resolve()), origin='released_native',
                sft_performed=False, geometry_present=False)

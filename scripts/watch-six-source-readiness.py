"""Stable blocked/pending receipt across immutable six-source preparation attempts.

This supervisor never promotes a partial inventory into training readiness.
Promotion requires an externally completed merged-manifest receipt and all gates.
"""
import argparse
import json
import os
from pathlib import Path
import time
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spatial_intelligence.followup_data_policy import validate_data_readiness


def read(path):
    return json.loads(Path(path).read_text()) if Path(path).is_file() else {'status': 'missing'}


def aggregate(component, audit, merged):
    complete = True
    try:
        validate_data_readiness(merged)
    except ValueError:
        complete = False
    if component.get('status') in {'failed', 'media_failed'}:
        complete = False
    authorized = (merged.get('contamination_waiver') or {}).get('authorized') is True
    blockers = [] if complete else ['six-source merged manifest and all media acceptance gates required']
    if component.get('status') in {'failed', 'media_failed'}:
        blockers.append('VLM3R/MindCube component media recovery required')
    if merged.get('error'):
        blockers.append(merged['error'])
    return dict(status='ready' if complete else 'pending_data_preparation',
                data_contract_version=1,
                six_sources_verified=merged.get('six_sources_verified', False),
                global_scene_split_verified=merged.get('global_scene_split_verified', False),
                ready_for_training=bool(complete), media_verified=bool(complete), leakage_checked=merged.get('leakage_checked', False),
                known_sources_leakage_checked=merged.get('known_sources_leakage_checked', False),
                contamination_waiver=merged.get('contamination_waiver'),
                train_manifest=merged.get('train_manifest') if complete else None,
                validation_manifest=merged.get('validation_manifest') if complete else None,
                component_status=component.get('status'), component_counts=component.get('counts', {}),
                inventory_status=audit.get('status'), counts=merged.get('counts', {}),
                raw_inventory_counts=audit.get('counts', {}),
                train_rows=merged.get('train_rows'), validation_rows=merged.get('validation_rows'),
                producer_status=merged.get('status'),
                limitations=['OpenSpatial scene overlap unknown; accepted by user, not a blocker'] if authorized else [],
                blockers=blockers)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--component-receipt', required=True)
    p.add_argument('--inventory-receipt', required=True)
    p.add_argument('--merged-receipt', required=True)
    p.add_argument('--interval', type=float, default=60)
    a = p.parse_args()
    root = Path(a.output)
    root.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (root/'supervisor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    while True:
        receipt = aggregate(read(a.component_receipt), read(a.inventory_receipt), read(a.merged_receipt))
        receipt.update(selection_name='shared-six-source-geothinker-vlm3r-mindcube-v4-openspatial-waiver',
                       pid=os.getpid(), updated=time.time(),
                       current_producer=dict(component=a.component_receipt, inventory=a.inventory_receipt, merged=a.merged_receipt),
                       protocol='v4: canonical source-scene split for five known sources; OpenSpatial train-only authorized unknown-scene exception')
        temp = root/'data-readiness.tmp'
        temp.write_text(json.dumps(receipt, indent=2))
        temp.replace(root/'data-readiness.json')
        if receipt['ready_for_training']:
            break
        time.sleep(a.interval)

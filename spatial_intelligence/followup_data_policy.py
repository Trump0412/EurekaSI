"""Explicit acceptance of an authorized, bounded contamination limitation."""
from pathlib import Path


DATA_READY_STATUSES = frozenset({'ready', 'prepared', 'prepared_with_user_waiver'})


def validate_data_readiness(receipt):
    """Accept historical producer labels only with unchanged substantive gates.

    Canonical consumers should receive status=ready from the receipt adapter;
    historical prepared labels alone never confer readiness.
    """
    if receipt.get('status') not in DATA_READY_STATUSES:
        raise ValueError('Data producer has not completed preparation')
    for key in ('ready_for_training', 'media_verified', 'six_sources_verified'):
        if receipt.get(key) is not True:
            raise ValueError('Missing data acceptance gate: '+key)
    policy = validate_leakage_policy(receipt)
    if policy == 'checked' and receipt.get('global_scene_split_verified') is not True:
        raise ValueError('Clean data require verified global scene split')
    if receipt.get('error'):
        raise ValueError('Data producer reports an error')
    for key in ('train_manifest', 'validation_manifest'):
        value = receipt.get(key)
        if not isinstance(value, str) or not value or not Path(value).is_file() or Path(value).stat().st_size == 0:
            raise ValueError('Missing or empty data manifest: '+key)
    return policy


def validate_leakage_policy(receipt):
    if receipt.get('leakage_checked') is True:
        return 'checked'
    waiver = receipt.get('contamination_waiver', {})
    if (receipt.get('known_sources_leakage_checked') is True
            and waiver.get('authorized') is True
            and waiver.get('scope') == 'openspatial_only'
            and waiver.get('scene_overlap') == 'unknown'
            and waiver.get('validation_policy') == 'all OpenSpatial train-only'
            and isinstance(waiver.get('reason'), str) and waiver['reason'].strip()):
        return 'openspatial_scene_overlap_unknown_user_authorized'
    raise ValueError('Require verified isolation or an explicit OpenSpatial-only user authorization')

"""Explicit acceptance of an authorized, bounded contamination limitation."""


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

import importlib.util
from pathlib import Path
import pytest
from spatial_intelligence.followup_data_policy import validate_data_readiness


def producer(tmp_path, status='prepared_with_user_waiver'):
    train=tmp_path/'train.jsonl';train.write_text('{}\n')
    val=tmp_path/'validation.jsonl';val.write_text('{}\n')
    return dict(status=status, ready_for_training=True, media_verified=True, six_sources_verified=True,
                leakage_checked=False, known_sources_leakage_checked=True,
                train_manifest=str(train), validation_manifest=str(val),
                contamination_waiver=dict(authorized=True, scope='openspatial_only', scene_overlap='unknown',
                    validation_policy='all OpenSpatial train-only', reason='Explicit user instruction'))


@pytest.mark.parametrize('status',['ready','prepared','prepared_with_user_waiver'])
def test_compatible_producer_labels(tmp_path,status):
    assert 'user_authorized' in validate_data_readiness(producer(tmp_path,status))


@pytest.mark.parametrize('key',['ready_for_training','media_verified','six_sources_verified','known_sources_leakage_checked'])
def test_prepared_does_not_override_gates(tmp_path,key):
    value=producer(tmp_path);value[key]=False
    with pytest.raises(ValueError):validate_data_readiness(value)


def test_missing_manifest_and_pending_status_rejected(tmp_path):
    value=producer(tmp_path);Path(value['validation_manifest']).write_text('')
    with pytest.raises(ValueError):validate_data_readiness(value)
    value=producer(tmp_path,'extracting_vsi_media')
    with pytest.raises(ValueError):validate_data_readiness(value)


def test_adapter_emits_consumer_ready_without_changing_waiver(tmp_path):
    spec=importlib.util.spec_from_file_location('readiness_adapter',Path(__file__).parents[1]/'scripts/watch-six-source-readiness.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    value=producer(tmp_path);value.update(counts={'openspatial':100000},train_rows=100001,validation_rows=10)
    result=module.aggregate({'status':'component_prepared'},{'counts':{'raw':200000}},value)
    assert result['status']=='ready'
    assert result['ready_for_training'] and not result['leakage_checked']
    assert result['counts']==value['counts'] and result['raw_inventory_counts']=={'raw':200000}
    assert result['producer_status']=='prepared_with_user_waiver'
    assert validate_data_readiness(result)
    result=module.aggregate({'status':'media_failed'},{},value)
    assert result['status']!='ready' and not result['ready_for_training']

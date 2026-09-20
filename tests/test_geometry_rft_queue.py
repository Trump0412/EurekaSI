"""CPU-only RFT scheduling, immutable inputs and paired scientific gates."""
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('geometry_rft_queue',
    Path(__file__).resolve().parents[1] / 'scripts/run-geometry-rft-queue.py')
queue = importlib.util.module_from_spec(spec); spec.loader.exec_module(queue)


@pytest.fixture
def plan(tmp_path):
    checkpoint = tmp_path / 'sft/final'; checkpoint.mkdir(parents=True)
    queue.write(checkpoint / 'config.json', {'geometry_matrix': {'adapter': 'downsample'}})
    (checkpoint / 'model.safetensors').write_bytes(b'test-only-weight-identity')
    queue.write(checkpoint.parent / 'contract.json',
        {'stage': 'sft', 'diagnostic': False, 'epochs': 1, 'max_steps': -1})
    receipt = checkpoint.parent / 'completion.json'
    queue.write(receipt, dict(status='complete', diagnostic_only=False, finite_loss=True,
        nonzero_update=True, reload_verified=True, steps=776, checkpoint=str(checkpoint)))
    train = tmp_path / 'train.jsonl'
    train.write_text(''.join(json.dumps({'id': f'train-{i}'}) + '\n' for i in range(16)))
    spatial = tmp_path / 'spatial.jsonl'; spatial.write_text('{"id":"spatial-0"}\n')
    evaluation = tmp_path / 'evaluation.jsonl'; evaluation.write_text('{"id":"heldout"}\n')
    data = tmp_path / 'data.json'
    queue.write(data, dict(status='ready', ready_for_training=True, accepted_train_rows=17,
        train_test_overlap=0, leakage_checked=True, media_verified=True,
        manifests={'4drl': str(train), 'spatialladder': str(spatial)},
        eval_manifest=str(evaluation), eval_manifests={'dsr': str(evaluation)}))
    recipe = tmp_path / 'recipe.json'; queue.write(recipe, {'group_size': 8, 'global_prompt_batch': 16})
    return dict(root=str(tmp_path / 'rft'), python='python', model_checkpoint=str(checkpoint),
        processor=str(tmp_path / 'processor'), vggt_source=str(tmp_path / 'vggt-source'),
        sft_receipt=str(receipt), data_receipt=str(data), scientific_config=str(recipe),
        pair_id='fixed-checkpoint-data-mixture-v1', gpus=[0, 1, 2, 3],
        dependencies=[{'path': str(tmp_path / 'prior.json'), 'statuses': ['complete']}],
        jobs=[{'name': 'mixed'}], group_size=8, prompts_per_update=16, seed=3407)


def prepare_inputs(plan):
    folder = Path(plan['root']) / 'inputs'; folder.mkdir(parents=True)
    (folder / 'scientific-config.json').write_bytes(Path(plan['scientific_config']).read_bytes())
    return queue.resolve_inputs(plan)


def gate(runtime, arm='mixed', mode='gate'):
    return dict(status='complete', arm=arm, mode=mode,
        initial_checkpoint=runtime['model_checkpoint'], finite_loss=True,
        nonzero_update=True, reload_verified=True, geometry_preserved=True,
        within_group_reward_variation=True, reference_initial_parity=True,
        all_group_responses_verified=True, group_size=8)


def test_single_node_subset_preserves_pair_contract(plan):
    assert queue.validate_plan(plan) is plan
    control = dict(plan, jobs=[{'name': 'four_d_only'}])
    assert queue.validate_plan(control)['model_checkpoint'] == plan['model_checkpoint']


@pytest.mark.parametrize('change', [
    {'group_size': 4}, {'prompts_per_update': 8}, {'gpus': [0, '0']},
    {'jobs': [{'name': 'mixed', 'model_checkpoint': '/other'}]},
    {'jobs': [{'name': 'mixed', 'four_d_rl_fraction': .5}]},
    {'dependencies': []}, {'jobs': [{'name': '../escape'}]},
])
def test_rejects_unmatched_or_unsafe_plan(plan, change):
    with pytest.raises(ValueError): queue.validate_plan(dict(plan, **change))


def test_dependency_requires_all_gpu_work_finished(plan):
    dep = plan['dependencies'][0]
    assert not queue.dependency_ready(dep)
    queue.write(dep['path'], {'status': 'complete'})
    assert not queue.dependency_ready(dep)
    queue.write(dep['path'], {'status': 'complete', 'gpu_work_finished': True})
    assert queue.dependency_ready(dep)


def test_budget_is_identical_prompt_draws_not_control_epochs():
    value = queue.budget(17)
    assert value['updates'] == 3 and value['prompt_budget'] == 48
    assert value['tail_padding_prompt_draws'] == 14 and value['rollout_budget'] == 384


def test_formal_sft_gate_rejects_diagnostic_and_wrong_checkpoint(plan):
    queue.validate_sft(plan)
    value = queue.read(plan['sft_receipt']); value['diagnostic_only'] = True
    queue.write(plan['sft_receipt'], value)
    with pytest.raises(ValueError, match='formal'): queue.validate_sft(plan)


def test_source_sft_must_be_full_epoch_contract(plan):
    path = Path(plan['model_checkpoint']).parent / 'contract.json'
    value = queue.read(path); value['max_steps'] = 2; queue.write(path, value)
    with pytest.raises(ValueError, match='full one-epoch'): queue.validate_sft(plan)


def test_data_must_have_audited_nonempty_count_and_no_overlap(plan):
    data = queue.read(plan['data_receipt']); queue.validate_data(data)
    for changed in ({'accepted_train_rows': 0}, {'train_test_overlap': 1},
                    {'ready_for_training': False}, {'media_verified': False}, {'leakage_checked': False}):
        with pytest.raises(ValueError): queue.validate_data(dict(data, **changed))


def test_resolved_budget_and_manifests_are_immutable(plan):
    runtime = prepare_inputs(plan)
    assert runtime['prompt_budget'] == 48 and runtime['group_size'] == 8
    assert queue.read(runtime['data_receipt'])['train_manifests'] == runtime['train_manifests']
    assert Path(runtime['eval_manifest']).read_text() == '{"id":"heldout"}\n'
    assert queue.resolve_inputs(plan) == runtime
    (Path(plan['model_checkpoint']) / 'model.safetensors').write_bytes(b'changed')
    with pytest.raises(ValueError, match='checkpoint changed'): queue.resolve_inputs(plan)


@pytest.mark.parametrize('field', ['geometry_preserved', 'all_group_responses_verified',
    'reference_initial_parity', 'within_group_reward_variation', 'reload_verified'])
def test_gate_cannot_be_replaced_by_process_success(plan, field):
    runtime = prepare_inputs(plan); value = gate(runtime); value[field] = False
    with pytest.raises(ValueError, match='runtime gate'):
        queue.validate_mode_receipt(value, runtime, 'mixed', 'gate')


def test_training_must_finish_exact_paired_budget(plan):
    runtime = prepare_inputs(plan)
    value = gate(runtime, mode='train')
    value.update(updates=runtime['updates'], prompt_budget=runtime['prompt_budget'], checkpoint=plan['model_checkpoint'])
    queue.validate_mode_receipt(value, runtime, 'mixed', 'train')
    value['updates'] -= 1
    with pytest.raises(ValueError, match='budget mismatch'):
        queue.validate_mode_receipt(value, runtime, 'mixed', 'train')


def test_evaluation_requires_exact_manifest_and_verified_ids(plan):
    runtime = prepare_inputs(plan); value = gate(runtime, mode='evaluate')
    value.update(paired_ids_verified=True, eval_manifest=runtime['eval_manifest'])
    queue.validate_mode_receipt(value, runtime, 'mixed', 'evaluate')
    value['eval_manifest'] = '/different/manifest.jsonl'
    with pytest.raises(ValueError, match='IDs changed'):
        queue.validate_mode_receipt(value, runtime, 'mixed', 'evaluate')


def test_snapshot_refuses_missing_worker_and_hot_recipe_changes(plan, tmp_path):
    source = tmp_path / 'source'; source.mkdir()
    with pytest.raises(ValueError, match='worker must exist'): queue.snapshot(plan, source)
    for name in ('scripts', 'spatial_intelligence', 'configs', 'catalog'): (source / name).mkdir()
    (source / 'scripts/train-geometry-rft.py').write_text('# test worker\n')
    frozen = queue.snapshot(plan, source)
    assert (frozen / 'scripts/train-geometry-rft.py').is_file()
    queue.write(plan['scientific_config'], {'changed': True})
    with pytest.raises(ValueError, match='recipe changed'): queue.snapshot(plan, source)


def test_pair_contract_ignores_node_allocation_but_locks_scientific_recipe(plan):
    data = queue.read(plan['data_receipt']); recipe = queue.read(plan['scientific_config'])
    original = queue.pair_contract(plan, data, recipe)
    other_node = dict(plan, root='/different/node', python='/different/python',
                      gpus=list(range(8)), jobs=[{'name': 'four_d_only'}])
    assert queue.pair_contract(other_node, data, recipe) == original
    changed = queue.pair_contract(other_node, data, dict(recipe, training={'learning_rate': 1e-3}))
    assert changed != original


def test_count_claim_must_match_manifest_rows(plan):
    data = queue.read(plan['data_receipt']); data['accepted_train_rows'] = 18
    with pytest.raises(ValueError, match='row count'):
        queue.pair_contract(plan, data, queue.read(plan['scientific_config']))


def test_validation_manifests_are_snapshotted_and_paired(plan, tmp_path):
    data = queue.read(plan['data_receipt'])
    paths = {}
    for name in ('4drl', 'spatialladder'):
        path = tmp_path / (name + '-validation.jsonl')
        path.write_text(json.dumps({'id': name + '-heldout'}) + '\n')
        paths[name] = str(path)
    data['validation_manifests'] = paths
    queue.write(plan['data_receipt'], data)
    runtime = prepare_inputs(plan)
    prepared = queue.read(runtime['data_receipt'])
    paired = queue.read(Path(plan['root']) / 'state/pair-contract.json')
    for name, original in paths.items():
        key = 'validation_' + name
        frozen = runtime['eval_manifests'][key]
        assert frozen != original
        assert Path(frozen).read_bytes() == Path(original).read_bytes()
        assert prepared['validation_manifests'][name] == frozen
        assert paired['evaluation_ids'][key] == [name + '-heldout']
    assert runtime['accepted_train_rows'] == 17


def test_validation_name_collision_is_rejected():
    with pytest.raises(ValueError, match='Conflicting validation'):
        queue.evaluation_sources({'eval_manifests': {'validation_4drl': 'a'},
                                  'validation_manifests': {'4drl': 'b'}})


def test_shared_pair_locks_content_not_only_ids(plan, tmp_path):
    pytest.importorskip('fcntl')
    data = queue.read(plan['data_receipt'])
    contract = queue.pair_contract(plan, data, queue.read(plan['scientific_config']))
    target = tmp_path / 'shared-pair.json'
    queue.verify_shared_pair(target, contract)
    queue.verify_shared_pair(target, contract)
    changed = dict(contract, seed=123)
    with pytest.raises(ValueError, match='scientific contract mismatch'):
        queue.verify_shared_pair(target, changed)
    source = Path(data['eval_manifest'])
    row = json.loads(source.read_text()); row['question'] = 'Changed text with same sample ID'
    source.write_text(json.dumps(row) + '\n')
    with pytest.raises(ValueError, match='manifest content changed'):
        queue.verify_shared_pair(target, contract)

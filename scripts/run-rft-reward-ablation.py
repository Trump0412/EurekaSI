"""Deferred validation-selected reward ablation. Uses no final-test score.

Run once per allocation with a private plan. Main RFT queues must finish first.
The chosen SFT initialization is reused, never the trained full-RFT checkpoint.
"""
import argparse
import copy
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2), encoding='utf-8'); temp.replace(path)


def validation_evidence(root, arm):
    folder = Path(root) / 'runs' / arm / 'evaluate'
    value = read(folder / 'completion.json')
    if value.get('status') != 'complete' or not value.get('paired_ids_verified') or not value.get('reload_verified'):
        raise ValueError('Candidate evaluation not accepted')
    report = value['paired']['validation_4drl']['rft']
    tasks = report['per_task']
    if not tasks or any(t['count'] < 1 for t in tasks.values()):
        raise ValueError('No source-heldout task coverage')
    macro = sum(t['mean_answer_reward'] for t in tasks.values()) / len(tasks)
    if not math.isfinite(macro) or not 0 <= macro <= 1:
        raise ValueError('Invalid heldout answer score')
    predictions = [json.loads(line) for path in sorted(folder.glob('rft-validation_4drl.rank*.jsonl'))
                   for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    expected = [json.loads(line)['id'] for line in Path(value['eval_manifests']['validation_4drl']).read_text(encoding='utf-8').splitlines() if line.strip()]
    ids = [row['id'] for row in predictions]
    if len(ids) != report['count'] or len(ids) != len(set(ids)) or set(ids) != set(expected):
        raise ValueError('Incomplete or duplicate heldout IDs')
    return dict(macro_answer=macro, mean_tokens=sum(len(r['tokens']) for r in predictions)/len(predictions),
                ids=sorted(ids), initial_checkpoint=value['initial_checkpoint'])


def select(candidates):
    evidence = {}
    for candidate in candidates:
        key = candidate['name']
        if key in evidence: raise ValueError('Duplicate candidate')
        evidence[key] = validation_evidence(candidate['root'], candidate['arm'])
    values = list(evidence.values())
    if not values or any(v['ids'] != values[0]['ids'] for v in values):
        raise ValueError('Selection requires identical independent validation IDs')
    winner = sorted(evidence, key=lambda name: (-evidence[name]['macro_answer'], evidence[name]['mean_tokens'], name))[0]
    return dict(winner=winner, metric='4DRL validation task-macro answer accuracy; then fewer tokens; then name',
                test_used=False, evidence=evidence)


def execute(plan):
    import fcntl
    root = Path(plan['root']); root.mkdir(parents=True, exist_ok=True)
    with (root/'supervisor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        def state(status, **extra):
            write(root/'state.json', dict(status=status, pid=os.getpid(), updated=time.time(), **extra))
        try:
            deadline = time.monotonic() + plan.get('wait_timeout_seconds', 30*86400)
            while True:
                ready = []
                for candidate in plan['candidates']:
                    path = Path(candidate['root'])/'state/rft-queue.json'
                    value = read(path) if path.exists() else {}
                    if value.get('status') in ('blocked','failed','complete_with_failures'):
                        raise ValueError('Main RFT candidate failed; no silent candidate exclusion')
                    ready.append(value.get('status') == 'complete' and value.get('gpu_work_finished') is True)
                if all(ready): break
                if time.monotonic() > deadline: raise TimeoutError('Main RFT wait exceeded')
                state('waiting_for_main_rft'); time.sleep(30)
            selection = select(plan['candidates'])
            saved = root/'selection.json'
            if saved.exists() and read(saved) != selection: raise ValueError('Selection changed')
            write(saved, selection)
            chosen = next(c for c in plan['candidates'] if c['name'] == selection['winner'])
            original = read(Path(chosen['root'])/'plan.json')
            recipe = read(Path(chosen['root'])/'inputs/scientific-config.json')
            if recipe['training']['reward']['version'] != 'geopsro-lexicon-strict-v2':
                raise ValueError('Full experiment did not use the restored lexical protocol')
            variant = plan['variant']
            if variant not in ('answer_only','answer_format'): raise ValueError('Unknown reward ablation')
            recipe = copy.deepcopy(recipe)
            recipe['training']['reward']['words_weight'] = 0.0
            recipe['training']['reward']['structure_weight'] = 0.0 if variant == 'answer_only' else 0.5
            recipe_path = root/'scientific-config.json'; write(recipe_path, recipe)
            child = copy.deepcopy(original)
            child.update(root=str(root/'experiment'), python=plan['python'], gpus=plan['gpus'],
                         scientific_config=str(recipe_path), jobs=[{'name':chosen['arm']}],
                         pair_id='reward-ablation-'+variant, pair_contract_path=str(root/'pair-contract.json'),
                         dependencies=[dict(path=str(Path(c['root'])/'state/rft-queue.json'), statuses=['complete']) for c in plan['candidates']])
            child.pop('training', None)
            path = root/'child-plan.json'
            if path.exists() and read(path) != child: raise ValueError('Ablation plan changed')
            write(path, child)
            spec = importlib.util.spec_from_file_location('rft_queue', REPO/'scripts/run-geometry-rft-queue.py')
            module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
            module.validate_plan(child); code = module.snapshot(child)
            state('running_ablation', variant=variant, selection=selection['winner'])
            with (root/'worker.log').open('ab') as log:
                subprocess.run([plan['python'], str(code/'scripts/run-geometry-rft-queue.py'), '--plan', str(root/'experiment/plan.json')],
                               cwd=code, stdout=log, stderr=subprocess.STDOUT, check=True)
            receipt = read(root/'experiment/state/rft-queue.json')
            if receipt.get('status') != 'complete': raise ValueError('Ablation did not complete successfully')
            state('complete', gpu_work_finished=True, variant=variant, selection=selection['winner'])
        except Exception as exc:
            state('blocked', error=repr(exc)); raise


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--detach', action='store_true')
    args=parser.parse_args(); plan=read(args.plan)
    if args.detach:
        root=Path(plan['root']); root.mkdir(parents=True,exist_ok=True)
        saved=root/'plan.json'
        if saved.exists() and read(saved)!=plan: raise ValueError('Use a new immutable root')
        write(saved,plan)
        with (root/'supervisor.log').open('ab') as log:
            child=subprocess.Popen([plan['python'],str(Path(__file__).resolve()),'--plan',str(saved)],
                cwd=REPO,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps(dict(status='waiting_not_training',pid=child.pid,root=str(root))))
    else:
        execute(plan)

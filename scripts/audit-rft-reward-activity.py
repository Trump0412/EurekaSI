"""Offline audit of saved grouped rollouts; no model, GPU or scoring changes."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spatial_intelligence.rft_reward_activity import group_activity, merge_activity, activity_gate


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train-root', required=True)
    p.add_argument('--output')
    args = p.parse_args()
    root = Path(args.train_root)
    contract = json.loads((root/'contract.json').read_text())
    config = contract['training']['reward']
    activity = {}; seen = set(); examples = []; heading_count = 0
    for path in sorted(root.glob('rollouts.rank*.jsonl')):
        for line in path.open(encoding='utf-8'):
            row = json.loads(line)
            key = (row['step'], row['prompt_index'])
            if key in seen:
                raise ValueError('Duplicate rollout group: interrupted/replayed log needs explicit reconciliation')
            seen.add(key)
            responses = row['responses']
            if len(responses) != contract['group_size']:
                raise ValueError('Incomplete rollout group')
            activity = merge_activity([activity, group_activity([x['reward'] for x in responses], config)])
            for x in responses:
                heading_count += 'Spatial Observation:' in x['text']
            if len(examples) < 3:
                examples.append(dict(step=row['step'], id=row['id'], response=responses[0]))
    result = dict(activity=activity, gate=activity_gate(activity, config),
                  configured_reward=config, heading_occurrences=heading_count, examples=examples,
                  coverage=dict(observed_groups=len(seen), expected_groups=contract['updates']*contract['prompts_per_update'],
                                complete=len(seen)==contract['updates']*contract['prompts_per_update']))
    if args.output:
        dest = Path(args.output)
        if dest.exists():
            raise FileExistsError('Keep prior audit evidence; choose a new output')
        dest.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

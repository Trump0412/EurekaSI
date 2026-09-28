"""Strict manifest-aligned native validation summary with class-balanced accuracy."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(args.output)
    manifest = [json.loads(line) for line in (root / 'val.manifest.jsonl').read_text().splitlines()]
    expected = {row['id']: row['answer'].strip().lower() for row in manifest}
    assert len(expected) == len(manifest)
    reports = {}
    for path in sorted((root / 'validation').glob('*.jsonl'), key=lambda path: int(path.stem)):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        actual = {row['sample_id']: row for row in rows}
        if len(actual) != len(rows) or set(actual) != set(expected):
            raise ValueError(f'Incomplete or duplicated validation coverage: {path}')
        by_answer = {}
        for label in ('yes', 'no'):
            selected = [actual[key] for key, gold in expected.items() if gold == label]
            by_answer[label] = {'n': len(selected), 'accuracy': sum(row['score'] for row in selected) / len(selected)}
        reports[path.stem] = {'n': len(rows), 'accuracy': sum(row['score'] for row in rows) / len(rows),
            'balanced_accuracy': sum(row['accuracy'] for row in by_answer.values()) / 2,
            'parse_rate': sum(row['parse_success'] for row in rows) / len(rows), 'by_answer': by_answer,
            'always_no_accuracy': by_answer['no']['n'] / len(rows), 'always_no_balanced_accuracy': .5}
    report = {'status': 'measured' if len(reports) >= 2 else 'partial', 'validation': reports,
              'scope': 'SPAR scene-heldout Yes/No exploration, not official full benchmark'}
    (root / 'comparison.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

"""Read-only schema and exact filename coverage audit; no video integrity claim."""
import argparse
import json
from pathlib import Path
import pandas as pd


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--node-root', type=Path, required=True)
    a = p.parse_args()
    root = a.node_root
    base = root / 'datasets/dynamic-annotations'
    inventories = {name: set(json.loads((root / f'.private/baidu-downloads/{name}.inventory.json').read_text())['files'])
                   for name in ('4drl_clips', 'dsr-bench')}
    reports = []
    for path in sorted(base.glob('*/*')):
        if path.suffix not in ('.jsonl', '.parquet', '.json'):
            continue
        if path.suffix == '.parquet':
            rows = pd.read_parquet(path).to_dict('records')
        elif path.suffix == '.jsonl':
            rows = [json.loads(line) for line in path.open() if line.strip()]
        else:
            rows = json.loads(path.read_text())
        if not isinstance(rows, list):
            reports.append({'file': str(path.relative_to(base)), 'top_level': type(rows).__name__,
                            'keys': list(rows)[:12]})
            continue
        first = rows[0] if rows else {}
        fields = {k: {'type': type(v).__name__, 'example': str(v)[:180]}
                  for k, v in first.items() if any(x in k.lower() for x in ('video', 'image', 'path', 'id'))}
        refs = set()
        row_refs = []
        for row in rows:
            names_for_row = set()
            for k, v in row.items():
                if isinstance(v, str) and v.lower().endswith('.mp4'):
                    names_for_row.add(Path(v).name)
                elif k == 'videoID' and isinstance(v, str):
                    names_for_row.add(v + '.mp4')
            refs.update(names_for_row)
            row_refs.append(names_for_row)
        coverage = {name: {'matched_video_names': len(refs & names), 'missing_video_names': len(refs - names)}
                    for name, names in inventories.items()} if refs else {}
        for name, counts in coverage.items():
            counts['qa_rows_with_inventory_media'] = sum(bool(rr) and rr <= inventories[name] for rr in row_refs)
        if refs:
            missing = sorted(refs - set.union(*inventories.values()))
            (base / (path.parent.name + '-' + path.stem + '-missing-media.json')).write_text(json.dumps(missing, indent=2))
        reports.append({'file': str(path.relative_to(base)), 'rows': len(rows), 'columns': list(first),
                        'media_fields': fields, 'unique_mp4_refs': len(refs), 'inventory_coverage': coverage})
    report = {'note': 'Exact basename inventory matching only, not decoded-media verification. No training manifests modified.',
              'inventory_name_overlap': len(inventories['4drl_clips'] & inventories['dsr-bench']), 'annotations': reports}
    (base / 'coverage-audit.json').write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()

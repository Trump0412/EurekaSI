"""Fetch official spatial annotations/media and select ordered multi-image rows."""
import argparse
from collections import Counter
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import zipfile

REVISION = '41d36f87a67ff676d438417b5a09079add8de866'
REPO = 'hongxingli/SpatialLadder-26k'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--detach', action='store_true')
    a = p.parse_args()
    root = a.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if a.detach:
        with (root / 'prepare.log').open('ab') as log:
            child = subprocess.Popen([sys.executable, __file__, '--root', str(root)],
                                     stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
        print(json.dumps({'pid': child.pid, 'root': str(root)}))
        return
    import fcntl
    lock = (root / 'prepare.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = {'repo': REPO, 'revision': REVISION, 'status': 'downloading', 'files': []}
    def save():
        tmp = root / 'receipt.json.tmp'
        tmp.write_text(json.dumps(state, indent=2))
        tmp.replace(root / 'receipt.json')
        print(json.dumps(state), flush=True)
    save()
    try:
        for filename in ('README.md', 'spld_spatial_data.jsonl', 'images.zip'):
            dest = root / filename
            if not dest.exists():
                url = f'https://hf-mirror.com/datasets/{REPO}/resolve/{REVISION}/{filename}'
                subprocess.run(['curl', '-fL', '--retry', '8', '--retry-delay', '10',
                                '--connect-timeout', '30', '--max-time', '14400', '-C', '-',
                                '-o', str(dest) + '.part', url], check=True,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                Path(str(dest) + '.part').replace(dest)
            state['files'].append({'file': filename, 'bytes': dest.stat().st_size})
            if filename == 'spld_spatial_data.jsonl':
                rows = [json.loads(line) for line in dest.open() if line.strip()]
                if any(not isinstance(r.get('image'), list) for r in rows):
                    raise ValueError('Unexpected image field schema')
                selected = [r for r in rows if len(r['image']) > 1]
                state.update(spatial_rows=len(rows), selected_rows=len(selected),
                             selection='len(image) > 1; original row and image order preserved',
                             selected_types=dict(Counter(r['data_type'] for r in selected)))
                with (root / 'multi-image.jsonl').open('w') as out:
                    for row in selected:
                        out.write(json.dumps(row, ensure_ascii=False) + '\n')
            save()
        state['status'] = 'extracting_selected_media'
        save()
        wanted = {s for row in selected for s in row['image']}
        with zipfile.ZipFile(root / 'images.zip') as archive:
            names = set(archive.namelist())
            mapping = {}
            for ref in sorted(wanted):
                rel = PurePosixPath(ref)
                if rel.is_absolute() or '..' in rel.parts or '\\' in ref:
                    raise ValueError('Unsafe media reference')
                candidates = [s for s in (str(rel), 'images/' + str(rel)) if s in names]
                if len(candidates) != 1:
                    raise ValueError(f'Missing or ambiguous archive path: {ref}')
                mapping[ref] = candidates[0]
            for ref, member in mapping.items():
                target = root / 'media' / ref
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, target.with_suffix(target.suffix + '.part').open('wb') as out:
                    shutil.copyfileobj(source, out)
                target.with_suffix(target.suffix + '.part').replace(target)
        from PIL import Image
        for ref in sorted(wanted):
            with Image.open(root / 'media' / ref) as im:
                im.verify()
        state.update(status='ready', unique_images=len(wanted), image_validation='ZIP CRC extraction and Pillow verify; all selected images',
                     note='No training queue changed; benchmark source-scene overlap not audited')
        save()
    except Exception as exc:
        state.update(status='failed', error=str(exc))
        save()
        raise


if __name__ == '__main__':
    main()

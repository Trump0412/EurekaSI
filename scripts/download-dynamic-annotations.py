"""Download pinned official annotations only; never alter training queues."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.request

DATASETS = {
    '4dthinker': ('jankin123/4DThinker-Training-Data', '37b53b800f07f2363364dd6c0f9b394d37138bd0',
                  ['README.md', '4drl_data_filtered.jsonl', 'dift_data.jsonl']),
    'dsr-suite': ('TencentARC/DSR_Suite-Data', '414132f02d03cecc583cc6ae77c9cfdc661f87a7',
                  ['README.md', 'LICENSE.txt', 'benchmark.parquet', 'train_qa_pairs.json', 'train_qa_pairs.parquet']),
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--detach', action='store_true')
    a = p.parse_args()
    root = a.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if a.detach:
        with (root / 'download.log').open('ab') as log:
            child = subprocess.Popen([sys.executable, __file__, '--root', str(root)],
                                     stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
        print(json.dumps({'pid': child.pid, 'root': str(root)}))
        return
    import fcntl
    lock = (root / 'download.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    receipts = []
    for name, (repo, revision, files) in DATASETS.items():
        target = root / name
        target.mkdir(exist_ok=True)
        for filename in files:
            dest = target / filename
            url = f'https://hf-mirror.com/datasets/{repo}/resolve/{revision}/{filename}'
            # curl retries and .part prevent an interrupted fetch being accepted.
            result = subprocess.run(['curl', '-fL', '--retry', '5', '--retry-delay', '5',
                                     '--connect-timeout', '30', '--max-time', '1800',
                                     '-o', str(dest) + '.part', url], stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL)
            if result.returncode:
                raise RuntimeError(f'Download failed: {repo}/{filename}')
            Path(str(dest) + '.part').replace(dest)
            receipt = {'repo': repo, 'revision': revision, 'file': filename, 'bytes': dest.stat().st_size}
            receipts.append(receipt)
            (root / 'receipts.json').write_text(json.dumps(receipts, indent=2))
            print(json.dumps(receipt), flush=True)
    print('ANNOTATIONS_DOWNLOADED', flush=True)


if __name__ == '__main__':
    main()

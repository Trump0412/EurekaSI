"""Recover only missing/zero-byte files from an existing private flat inventory.

Uses the already-authorized Baidu client without printing credentials or signed
URLs. Staging and logs stay private; complete originals are never overwritten.
The new receipt is an acquisition gate, not dataset decoding/training readiness.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spatial_intelligence.rft_recovery import display_size_matches


def size_matches(size, display):
    return size > 0 and display_size_matches(size, display)


def upstream_empty_files(files):
    return [name for name, display in files.items()
            if re.fullmatch(r'0(?:\.0+)?B', display)]


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--folder', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--detach', action='store_true')
    args = parser.parse_args()
    os.umask(0o077)
    root = args.root.resolve(strict=True)
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.folder):
        raise ValueError('Explicit flat folder basename required')
    output = args.output.resolve()
    if root/'.private' not in output.parents:
        raise ValueError('Recovery logs must remain under the node private directory')
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    if args.detach:
        with (output/'supervisor.log').open('ab') as log:
            child = subprocess.Popen([sys.executable, __file__, *[a for a in sys.argv[1:] if a != '--detach']],
                stdout=log, stderr=log, stdin=subprocess.DEVNULL, start_new_session=True)
        print(json.dumps({'pid': child.pid, 'receipt': str(output/'receipt.json')}))
        return
    lock = (root/'.private/baidu-downloads/download.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    inventory = json.loads((root/f'.private/baidu-downloads/{args.folder}.inventory.json').read_text())
    files = inventory['files']
    if len(files) != inventory['count'] or any(Path(n).name != n or n in ('.','..') for n in files):
        raise ValueError('Invalid flat inventory')
    destination = root/'datasets/baidu'/args.folder
    missing = [n for n in files if not (destination/n).is_file() or (destination/n).stat().st_size == 0]
    report = {'status': 'recovering', 'pid': os.getpid(), 'expected_files': len(files),
              'missing_or_zero': missing, 'started': time.time(), 'destination': str(destination)}
    receipt = output/'receipt.json'
    write(receipt, report)
    binary = root/'tools/baidupcs/v4.0.2/BaiduPCS-Go'
    env = dict(os.environ, BAIDUPCS_GO_CONFIG_DIR=str(root/'.private/baidu'), BAIDUPCS_GO_VERBOSE='0')
    staging = output/'media'
    staging.mkdir(exist_ok=True)
    try:
        upstream_empty = upstream_empty_files(files)
        if upstream_empty:
            report.update(upstream_empty_files=upstream_empty,
                          blocked_reason='Remote inventory itself contains empty source media; retry cannot reconstruct it')
            raise ValueError('Empty upstream source media requires replacement or explicit revised-data authorization')
        for mode in ('pcs', 'locate', 'stream'):
            pending = [n for n in missing if not (staging/args.folder/n).is_file()
                       or not size_matches((staging/args.folder/n).stat().st_size, files[n])]
            if not pending:
                break
            # Only staging files may be overwritten. Complete dataset files are
            # not passed to the client, and originals remain unchanged on failure.
            command = [str(binary), 'download', '--saveto', str(staging), '--fullpath', '--ow',
                       '--mode', mode, '-p', '2', '-l', '2', '--retry', '3',
                       *['/'+args.folder+'/'+name for name in pending]]
            report.update(mode=mode, pending=len(pending), updated=time.time())
            write(receipt, report)
            with (output/f'{mode}.log').open('ab') as log:
                child = subprocess.Popen(command, env=env, stdout=log, stderr=log, start_new_session=True)
                try:
                    child.wait(timeout=7200)
                except subprocess.TimeoutExpired:
                    import signal
                    os.killpg(child.pid, signal.SIGTERM)
                    child.wait(timeout=30)
                    raise
        failures = [n for n in missing if not (staging/args.folder/n).is_file()
                    or not size_matches((staging/args.folder/n).stat().st_size, files[n])]
        if failures:
            raise ValueError(f'{len(failures)} files remain incomplete after bounded retries')
        for name in missing:
            candidate = staging/args.folder/name
            probe = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                '-show_entries', 'stream=codec_type,width,height', '-of', 'json', str(candidate)],
                capture_output=True, timeout=60, check=True)
            streams = json.loads(probe.stdout)['streams']
            if not streams or streams[0].get('width', 0) < 1 or streams[0].get('height', 0) < 1:
                raise ValueError('Recovered file has no video stream')
        backup = output/'original-zero-files'
        backup.mkdir(exist_ok=True)
        for name in missing:
            target = destination/name
            if target.exists():
                if target.stat().st_size:
                    raise ValueError('Target changed during recovery; refusing overwrite')
                shutil.copy2(target, backup/name)
            (staging/args.folder/name).replace(target)
        mismatch = [n for n in files if not (destination/n).is_file()
                    or not size_matches((destination/n).stat().st_size, files[n])]
        if mismatch:
            raise ValueError(f'{len(mismatch)} full-inventory size mismatches; no ready acquisition receipt')
        report.update(status='download_exited_inventory_present', observed_files=len(files), recovered=len(missing),
            verification='All declared names/nonzero sizes match rounded remote listing; recovered video headers valid. Full decode is a separate preparation gate.',
            ready_for_training=False, completed=time.time())
        write(receipt, report)
    except Exception as error:
        # Error text from a network client may contain signed URLs. Keep only
        # the class in the structured receipt; raw client output stays private.
        report.update(status='failed', error_type=type(error).__name__, updated=time.time())
        write(receipt, report)
        raise


if __name__ == '__main__':
    main()

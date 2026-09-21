"""Recover exact missing released media members from the original split archive."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import time


def run(a):
    previous, out = Path(a.previous), Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    original = json.loads((previous/'data-readiness.json').read_text())
    missing = json.loads((previous/'media-failures.json').read_text())
    wanted = {}
    for entry in missing:
        path = entry['path'];suffix = path.split('/media/spar/', 1)[1]
        wanted['spar/'+suffix] = path
    state = dict(original, status='recovering_exact_archive_members', component_ready=False,
                 ready_for_training=False, recovery_previous=str(previous), missing=len(wanted))
    def save():
        tmp=out/'data-readiness.tmp';tmp.write_text(json.dumps(state,indent=2));tmp.replace(out/'data-readiness.json')
    save()
    parts = sorted(Path(a.archive_root).glob('spar-rgbd-??.tar.gz'))
    if len(parts) != 18:raise ValueError('Expected all 18 original split archive parts')
    pipe = subprocess.Popen(['cat', *map(str, parts)], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    recovered = {}
    try:
        with tarfile.open(fileobj=pipe.stdout, mode='r|gz') as archive:
            for index, member in enumerate(archive):
                name = member.name.removeprefix('./')
                if name in wanted:
                    target = out/'recovered'/name
                    if not member.isfile() or not target.resolve().is_relative_to((out/'recovered').resolve()):
                        raise ValueError('Unsafe member')
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.extractfile(member) as src, target.open('wb') as dst:shutil.copyfileobj(src,dst)
                    from PIL import Image
                    with Image.open(target) as im:im.load()
                    recovered[wanted[name]]=str(target)
                if index % 10000 == 0:
                    state.update(members_scanned=index, recovered=len(recovered), updated=time.time());save()
                if len(recovered)==len(wanted):break
    finally:
        pipe.stdout.close()
        if pipe.poll() is None:pipe.terminate()
        pipe.wait()
    (out/'recovery-ledger.json').write_text(json.dumps(recovered,indent=2))
    if len(recovered)!=len(wanted):
        state.update(status='exact_archive_missing', recovered=len(recovered), remaining=[p for p in wanted.values() if p not in recovered]);save();return
    for filename in ['vlm3r.jsonl','mindcube.train.jsonl','mindcube.tiny.test.jsonl']:
        with (previous/filename).open() as src,(out/filename).open('w') as dst:
            for line in src:
                row=json.loads(line);row['media']=[recovered.get(p,p) for p in row['media']];dst.write(json.dumps(row,ensure_ascii=False)+'\n')
    shutil.copy2(previous/'source-ledger.jsonl',out/'source-ledger.jsonl')
    state.update(status='component_prepared',component_ready=True,recovered=len(recovered),
                 blockers=['six-source merge still required'], media_recovery='Exact source archive bytes; no interpolation/substitute frames')
    state['vlm3r_media']['failed']=0;save()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--previous',required=True);p.add_argument('--output',required=True);p.add_argument('--archive-root',required=True)
    a=p.parse_args()
    try:run(a)
    except Exception as exc:
        path=Path(a.output)/'data-readiness.json'
        if path.exists() and not isinstance(exc,FileExistsError):
            d=json.loads(path.read_text());d.update(status='failed',component_ready=False,error=str(exc));path.write_text(json.dumps(d,indent=2))
        raise

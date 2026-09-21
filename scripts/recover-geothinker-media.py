"""Recover exact missing released media members from the original split archive."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import time

KNOWN_INVALID_IDS={'b4909db6-9528-4eee-ba63-8e1fb01dfa3b','268f44c6-05f4-44b5-a51f-debdb8bb506b'}


def quarantine_reason(row):
    videos=[p for p in row['media'] if Path(p).suffix.lower() in {'.mp4','.mov','.avi','.mkv'}]
    if not videos:return None
    if (row.get('source_id') in KNOWN_INVALID_IDS
            and row.get('source_file')=='vlm3r_vsi_205k_32frames.json'
            and len(videos)==1 and videos[0].endswith('/scene0290_00/video_color/scene0290_00_video.mp4')):
        return 'invalid_media_type: released 32-image annotation includes MP4; explicitly quarantined, no substitute frame'
    raise ValueError('Unexpected video in image annotation: '+row['id'])


def validate_media(path):
    if Path(path).suffix.lower() in {'.mp4', '.mov', '.avi', '.mkv'}:
        import cv2
        cap = cv2.VideoCapture(str(path))
        frames = 0
        try:
            if not cap.isOpened():raise ValueError('Cannot open recovered video: '+str(path))
            expected = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            while True:
                ok, frame = cap.read()
                if not ok:break
                frames += 1
            if frames == 0 or (expected > 0 and frames != expected):
                raise ValueError(f'Incomplete recovered video: {path}; decoded={frames}, expected={expected}')
        finally:cap.release()
        return {'type':'video','decoded_frames':frames}
    from PIL import Image
    with Image.open(path) as im:im.load()
    return {'type':'image'}


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
                    validate_media(target)
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
    non_image_rows=[]
    for filename in ['vlm3r.jsonl','mindcube.train.jsonl','mindcube.tiny.test.jsonl']:
        with (previous/filename).open() as src,(out/filename).open('w') as dst:
            for line in src:
                row=json.loads(line)
                reason=quarantine_reason(row)
                if reason:
                    non_image_rows.append(dict(id=row['id'],reason=reason,original_row=row));continue
                row['media']=[recovered.get(p,p) for p in row['media']]
                dst.write(json.dumps(row,ensure_ascii=False)+'\n')
    shutil.copy2(previous/'source-ledger.jsonl',out/'source-ledger.jsonl')
    state.update(status='component_prepared',component_ready=True,recovered=len(recovered),
                 blockers=['six-source merge still required'], media_recovery='Exact source archive bytes; no interpolation/substitute frames')
    state['vlm3r_media']['failed']=0
    if {r['original_row']['source_id'] for r in non_image_rows} != KNOWN_INVALID_IDS:
        raise ValueError('Expected exactly two known invalid-media annotations')
    (out/'annotation-media-type-errors.json').write_text(json.dumps(non_image_rows,indent=2))
    state['counts']['vsi_accepted']-=len(non_image_rows)
    state['exclusions']['explicit_invalid_media_type']=len(non_image_rows)
    state.update(non_image_annotation_rows=len(non_image_rows),
                 invalid_media_policy='Only two explicitly identified source IDs quarantined with complete original rows; arbitrary missing media still block')
    save()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--previous',required=True);p.add_argument('--output',required=True);p.add_argument('--archive-root',required=True)
    a=p.parse_args()
    try:run(a)
    except Exception as exc:
        path=Path(a.output)/'data-readiness.json'
        if path.exists() and not isinstance(exc,FileExistsError):
            d=json.loads(path.read_text());d.update(status='failed',component_ready=False,error=str(exc));path.write_text(json.dumps(d,indent=2))
        raise

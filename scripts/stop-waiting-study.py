"""Stop only an explicitly identified waiting controller, never a training child.

Use for immutable queue replacement. Preserve all old artifacts. Endpoint and
PID bindings belong in private deployment records, not this script.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import time

def stop(root, state_name, expected_pid):
    root=Path(root).resolve()
    scripts={'rft-queue.json':'run-geometry-rft-queue.py',
             'followups.json':'run-geometry-followups.py','matrix.json':'run-geometry-matrix.py'}
    if state_name not in scripts:
        raise ValueError('Unsupported controller state')
    path=root/'state'/state_name
    state=json.loads(path.read_text())
    if state.get('pid')!=expected_pid or not state.get('status','').startswith('waiting_'):
        raise ValueError('Not the declared waiting controller; inspect without stopping')
    proc=Path('/proc')/str(expected_pid)
    cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode()
    script=scripts[state_name]
    if script not in cmd or str(root) not in cmd:
        raise ValueError('Process identity mismatch')
    children=(proc/'task'/str(expected_pid)/'children').read_text().strip()
    if children: raise ValueError('Controller has children; refuse automated stop')
    os.kill(expected_pid,signal.SIGTERM)
    for _ in range(50):
        if not proc.exists(): break
        time.sleep(.1)
    if proc.exists(): raise RuntimeError('Controller did not exit; no escalation')
    receipt=dict(status='superseded_before_training',old_pid=expected_pid,old_state=state,time=time.time())
    target=root/'state'/'superseded.json'
    with target.open('x') as f: json.dump(receipt,f,indent=2)
    print(json.dumps(dict(status=receipt['status'],old_pid=expected_pid)))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True);p.add_argument('--state',required=True)
    p.add_argument('--pid',type=int,required=True)
    a=p.parse_args();stop(a.root,a.state,a.pid)

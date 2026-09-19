"""One actual SPAR sample per renderer type: eager vs lazy decoded RGB equality."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
from spatial_intelligence.marker_cache import materialize

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);a=p.parse_args()
root=Path(a.root)
spec=importlib.util.spec_from_file_location('renderer',root/'sources/draw_marker.py')
renderer=importlib.util.module_from_spec(spec);spec.loader.exec_module(renderer)
rows=json.loads((root/'datasets/vg-llm/train/spar_234k.json').read_text())
selected={}
for row in rows:
    if not row.get('spar_info'):continue
    info=json.loads(row['spar_info']) if isinstance(row['spar_info'],str) else row['spar_info']
    selected.setdefault(info['type'],(row,info))
result=[]
for kind,(row,info) in selected.items():
    paths=[str(root/'media'/s) for s in row['images']]
    eager=[Image.open(s).convert('RGB') for s in paths]
    render=renderer.DRAW_FUNCTIONS[kind];render(eager[0] if len(eager)==1 else eager,info)
    lazy=materialize(paths,info,render,root/'runs/diagnostic-marker-parity'/kind)
    assert all(np.array_equal(np.asarray(expected),np.asarray(Image.open(actual).convert('RGB')))
               for expected,actual in zip(eager,lazy)),kind
    result.append({'type':kind,'sample_id':row['id'],'frames':len(paths),'pixel_equal':True})
    print(json.dumps(result[-1]),flush=True)
(root/'receipts/marker-parity.json').write_text(json.dumps({'status':'complete','types':result},indent=2))

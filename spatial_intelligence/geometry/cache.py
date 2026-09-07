"""Content-addressed frozen reconstruction cache, independent of backbone environments."""
import json
from pathlib import Path
import numpy as np
from ..io import digest, file_digest, write_json
from ..data import load_samples


def media_hashes(sample):
    return [file_digest(p) for p in sample.get('geometry_media',sample['media'])]


def cache_key(sample, extractor):
    return digest({'media':media_hashes(sample),'extractor':extractor,'schema':'spatial-points-v1'})


def point_tokens(points, confidence, grid=8):
    """Ordered grid samples: xyz, confidence, frame index, uv. No unit/scale normalization."""
    if type(grid) is not int or grid<1:raise ValueError('grid must be a positive integer')
    points=np.asarray(points,dtype=np.float32); confidence=np.asarray(confidence,dtype=np.float32)
    if points.ndim!=4 or points.shape[-1]!=3:raise ValueError('Expected points [frames,H,W,3]')
    n,h,w,_=points.shape
    if min(n,h,w)<1:raise ValueError('Geometry dimensions must be nonempty')
    if confidence.shape==points.shape[:3]+(1,):confidence=confidence[...,0]
    if confidence.shape!=points.shape[:3]:raise ValueError('Confidence shape mismatch')
    yy=np.linspace(0,h-1,min(grid,h)).round().astype(int);xx=np.linspace(0,w-1,min(grid,w)).round().astype(int)
    rows=[]
    for f in range(n):
        for y in yy:
            for x in xx:rows.append([*points[f,y,x],confidence[f,y,x],f/max(n-1,1),x/max(w-1,1),y/max(h-1,1)])
    tokens=np.asarray(rows,dtype=np.float32)
    if not np.isfinite(tokens).all():raise ValueError('Nonfinite geometry: reject sample, never silently replace with zero')
    return tokens


def unproject(depth,intrinsics,extrinsics):
    depth=np.asarray(depth);k=np.asarray(intrinsics);e=np.asarray(extrinsics)
    if depth.ndim!=3 or k.shape!=(len(depth),3,3) or e.shape not in {(len(depth),3,4),(len(depth),4,4)}:
        raise ValueError('DA3 calibration shape mismatch')
    n,h,w=depth.shape;y,x=np.mgrid[:h,:w];pix=np.stack([x,y,np.ones_like(x)],-1)
    camera=np.einsum('nij,hwj->nhwi',np.linalg.inv(k),pix)*depth[...,None]
    # extrinsics are world-to-camera: X_world = R^-1 (X_camera - t).
    return np.einsum('nij,nhwj->nhwi',np.linalg.inv(e[:,:3,:3]),camera-e[:,:3,3][:,None,None,:])


def load_extractor(cfg):
    import torch
    name=cfg['name'];device=cfg.get('device','cuda:0');weights=cfg['weights']
    if name=='vggt':
        from vggt.models.vggt import VGGT
        from vggt.utils.load_fn import load_and_preprocess_images
        model=VGGT.from_pretrained(weights).to(device).eval()
        def run(paths):
            images=load_and_preprocess_images(paths).to(device)
            with torch.no_grad():pred=model(images)
            return pred['world_points'][0].float().cpu().numpy(),pred['world_points_conf'][0].float().cpu().numpy()
    elif name=='pi3':
        from pi3.models.pi3 import Pi3
        from PIL import Image
        model=Pi3.from_pretrained(weights).to(device).eval()
        def run(paths):
            # Fixed square resize is an explicit common protocol, not Pi3's official benchmark preprocessing.
            size=cfg.get('image_size',518)
            if size%14:raise ValueError('Pi3 image_size must be divisible by 14')
            images=[]
            for p in paths:
                with Image.open(p) as im:images.append(np.asarray(im.convert('RGB').resize((size,size)),dtype=np.float32)/255.)
            batch=torch.from_numpy(np.stack(images)).permute(0,3,1,2)[None].to(device)
            with torch.no_grad():pred=model(batch)
            return pred['points'][0].float().cpu().numpy(),pred['conf'][0].sigmoid().float().cpu().numpy()
    elif name=='da3':
        from depth_anything_3.api import DepthAnything3
        model=DepthAnything3.from_pretrained(weights).to(device).eval()
        def run(paths):
            with torch.no_grad():p=model.inference(paths,process_res=cfg.get('image_size',504),process_res_method='upper_bound_resize')
            if p.extrinsics is None or p.intrinsics is None or p.conf is None:raise ValueError('DA3 checkpoint must predict calibrated multiview geometry')
            return unproject(p.depth,p.intrinsics,p.extrinsics),p.conf
    else:raise ValueError('Extractor must be vggt/da3/pi3')
    return run


def extract(manifest,output,cfg):
    # Weight bytes are pinned locally, not merely a mutable HF alias.
    weights=Path(cfg['weights'])
    files=sorted(p for p in weights.rglob('*') if p.is_file() and p.suffix in {'.safetensors','.pt','.pth','.bin','.json'})
    if not weights.is_dir() or not files:raise ValueError('Download weights first; pass a local directory with weight/config files')
    descriptor={k:v for k,v in cfg.items() if k not in {'weights','device'}}
    descriptor['adapter_sha256']=file_digest(__file__)
    descriptor['weights_sha256']={str(p.relative_to(weights)):file_digest(p) for p in files}
    if not descriptor.get('source_commit'):raise ValueError('Record the geometry implementation source_commit in config')
    out=Path(output)
    index=out/'index.json'
    prev=json.loads(index.read_text(encoding="utf-8")) if index.exists() else None
    if prev and (prev.get('schema')!='spatial-points-v1' or prev['extractor']!=descriptor):
        raise ValueError('Different extractor/schema in existing cache directory')
    out.mkdir(parents=True,exist_ok=True)
    run=None;entries={}
    for sample in load_samples(manifest):
        h=digest(media_hashes(sample)); ck=cache_key(sample,descriptor);path=out/(ck+'.npz')
        existing=prev['entries'].get(h) if prev else None
        if existing and (existing.get('cache_key')!=ck or existing['sha256']!=file_digest(path)):
            raise ValueError('Existing geometry cache changed; preserve the old index and rebuild elsewhere')
        if not path.exists():
            if run is None:run=load_extractor(cfg)
            pts,conf=run(sample.get('geometry_media',sample['media']))
            tokens=point_tokens(pts,conf,cfg.get('grid',8))
            temp=path.with_suffix('.tmp.npz');np.savez_compressed(temp,tokens=tokens);temp.replace(path)
        entries[h]={'path':str(path.resolve()),'sha256':file_digest(path),'cache_key':ck}
    if prev:
        entries={**prev['entries'],**entries}
    write_json(index,{'schema':'spatial-points-v1','extractor':descriptor,'entries':entries})
    return {'index':str(index),'entries':len(entries)}


def cached_tokens(index,sample):
    h=digest(media_hashes(sample));entry=index['entries'].get(h)
    if entry is None:raise FileNotFoundError('No geometry for these exact ordered frames; run cache-geometry')
    if file_digest(entry['path'])!=entry['sha256']:raise ValueError('Geometry cache changed')
    with np.load(entry['path'],allow_pickle=False) as blob:tokens=blob['tokens'].copy()
    if tokens.ndim!=2 or tokens.shape[-1]!=7 or not len(tokens) or not np.isfinite(tokens).all():raise ValueError('Invalid geometry cache')
    return tokens

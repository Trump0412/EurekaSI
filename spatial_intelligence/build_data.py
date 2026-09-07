"""Deterministic conversion of released annotations and ordered visual evidence."""
import ast
import importlib.util
import io
import json
import re
from pathlib import Path

from PIL import Image

from .data import normalize, key, load_samples
from .io import digest, file_digest, write_json, write_jsonl
from .video import freeze_frames


def literal(value):
    if isinstance(value,str) and value.strip().startswith(("[","{")):
        try:return json.loads(value)
        except json.JSONDecodeError:return ast.literal_eval(value)
    return value


def choices_from(value):
    value=literal(value)
    if isinstance(value,str):value=value.strip()
    if isinstance(value,dict):return {str(k):str(v) for k,v in value.items()}
    if isinstance(value,list):
        return {chr(65+i):re.sub(r"^[A-Z]\s*[.):]\s*","",str(v)) for i,v in enumerate(value)}
    if isinstance(value,str):
        marks=list(re.finditer(r"(?:^|[,\n]\s*)([A-Z])\s*[.):]\s*",value))
        if not marks:return None
        return {m.group(1):value[m.end():marks[i+1].start() if i+1<len(marks) else len(value)].strip() for i,m in enumerate(marks)}
    return None


def annotation_rows(path):
    path=Path(path)
    if path.suffix==".parquet":
        import pyarrow.parquet as pq
        for batch in pq.ParquetFile(path).iter_batches():yield from batch.to_pylist()
    elif path.suffix==".jsonl":
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():yield json.loads(line)
    elif path.suffix==".json":
        value=json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value,list):raise ValueError("JSON input must be an array; choose the released QA file")
        yield from value
    else:raise ValueError("Annotations must be .json/.jsonl/.parquet")


def materialize_image(value, root, out, number):
    value=literal(value)
    if isinstance(value,str):
        p=(root/value).resolve()
        if not p.is_file():raise FileNotFoundError(p)
        with Image.open(p) as im:im.verify()
        return str(p)
    if isinstance(value,dict):
        if value.get("bytes") is not None:
            with Image.open(io.BytesIO(value["bytes"])) as im:
                out.mkdir(parents=True,exist_ok=True);p=out/f"image-{number}.png";im.convert("RGB").save(p)
            return str(p.resolve())
        if value.get("path"):return materialize_image(value["path"],root,out,number)
    raise ValueError("Unsupported image representation")


def build_annotations(input_path,output,dataset,split,media_root,frames_root,*,max_frames=8,marker_source=None,limit=None):
    if max_frames<1 or (limit is not None and limit<1):raise ValueError("Positive frame budget and sample limit required")
    if split not in {"train","val","test"}:raise ValueError("split must be train/val/test")
    source_hash=file_digest(input_path)
    output=Path(output)
    if output.exists():raise FileExistsError("Manifest already exists; use a new build name")
    root=Path(media_root).resolve(); frame_root=Path(frames_root).resolve()
    # A rebuild with another media root or renderer must not overwrite older evidence.
    builder_hash=file_digest(__file__)
    build_id=digest([source_hash,str(root),max_frames,file_digest(marker_source) if marker_source else None,builder_hash])[:24]
    renderer=None
    if marker_source:
        spec=importlib.util.spec_from_file_location("spatial_marker_renderer",marker_source)
        renderer=importlib.util.module_from_spec(spec);spec.loader.exec_module(renderer)
    rows=[]
    for index,raw in enumerate(annotation_rows(input_path)):
        if limit is not None and index>=limit:break
        question=raw.get("question");answer=raw.get("answer",raw.get("ground_truth",raw.get("label")))
        if "conversations" in raw:
            turns=raw["conversations"]
            q=[x["value"] for x in turns if x.get("from") in {"human","user"}]
            a=[x["value"] for x in turns if x.get("from") in {"gpt","assistant"}]
            if len(q)!=1 or len(a)!=1:raise ValueError("Single-turn QA required; multi-turn data needs an explicit recipe")
            question,answer=q[0],a[0]
        if question is None or answer is None:raise ValueError(f"No QA in row {index}; downloaded images are not QA annotations")
        question=str(question).replace("<image>","").replace("<video>","").strip()
        options=choices_from(raw.get("choices",raw.get("options")))
        if options is None and "Options:" in question:
            question,option_text=question.split("Options:",1);options=choices_from(option_text)
            if not options:raise ValueError("Cannot parse embedded options")
        if options:
            m=re.fullmatch(r"\s*([A-Z])(?:\s*[.):]\s*.*)?",str(answer),re.S)
            if m and m.group(1) in options:answer=m.group(1)
            else:
                exact=[k for k,v in options.items() if v.strip()==str(answer).strip()]
                if len(exact)!=1:raise ValueError("Choice answer must be an explicit label or one exact option text")
                answer=exact[0]
        sid=raw.get("id",raw.get("sample_id",f"{source_hash[:12]}-{index}"))
        sample_dir=frame_root/'samples'/digest([dataset,build_id,str(sid)])[:24]
        media=literal(raw.get("media",raw.get("frame_paths",raw.get("images",raw.get("image_path",raw.get("image",[]))))))
        video=raw.get("video")
        if dataset.lower() in {"vsi-bench","vsibench"} and not media and not video:
            video=str(raw["dataset"])+"/"+str(raw["scene_name"])+".mp4"
        if dataset.lower()=="revsi" and not media and not video:
            frames=int(raw["num_frames"])
            if frames>max_frames:raise ValueError("ReVSI prescribed view count exceeds budget; do not resample official views")
            video=f"{frames}_frame/{raw['scene_id']}.mp4"
        if not media and not video and raw.get("videoID"):video=str(raw["videoID"])+".mp4"
        indices=raw.get("frame_indices",[]);timestamps=raw.get("timestamps_s",[])
        if video and not media:
            vp=(root/video).resolve()
            if vp.is_dir():
                # frame-2 precedes frame-10 even when filenames are not zero-padded.
                media=sorted((str(p) for p in vp.iterdir() if p.suffix.lower() in {".jpg",".png",".jpeg"}),
                             key=lambda p:[int(s) if s.isdigit() else s for s in re.split(r'(\d+)',p)])
                if not media:raise ValueError(f"Empty frame directory {vp}")
                if raw.get("spar_info") and len(media)>max_frames:raise ValueError("SPAR marker frame indices require the original prescribed frames; cannot resample")
                n=min(len(media),max_frames)
                indices=[round(i*(len(media)-1)/(n-1)) for i in range(n)] if n>1 else [len(media)//2]
                media=[media[i] for i in indices]
            else:
                if raw.get("spar_info"):raise ValueError("Marked SPAR video requires prescribed decoded frames; generic video sampling could move target markers")
                cached=frame_root/"videos"/digest([file_digest(vp),max_frames])[:24]
                if (cached/"frames.json").exists():
                    selection=json.loads((cached/"frames.json").read_text(encoding="utf-8"))
                    if selection.get('media_sha256') != [file_digest(p) for p in selection['media']]:
                        raise ValueError('Cached video frames changed or lack hashes; rebuild in a new frames directory')
                else:selection=freeze_frames(vp,cached,max_frames)
                media,indices,timestamps=selection["media"],selection["frame_indices"],selection["timestamps_s"]
        if isinstance(media,(str,dict)):media=[media]
        if not media:raise ValueError(f"No visual evidence in row {index}")
        if len(media)>max_frames:raise ValueError("Multi-image QA exceeds frame budget; do not discard referenced views")
        if dataset.lower()=='revsi' and len(media)!=int(raw['num_frames']):
            raise ValueError('ReVSI evidence does not match the prescribed num_frames')
        media=[materialize_image(v,root,sample_dir,i) for i,v in enumerate(media)]
        clean_media=list(media)
        marker_info=raw.get("spar_info")
        if marker_info:
            if renderer is None:raise ValueError("SPAR markers required: fetch geothinker source and pass its draw_marker.py")
            info=literal(marker_info); fn=renderer.DRAW_FUNCTIONS[info["type"]]
            images=[]
            for p in media:
                with Image.open(p) as im:images.append(im.convert("RGB"))
            fn(images[0] if len(images)==1 else images,info)
            sample_dir.mkdir(parents=True,exist_ok=True)
            media=[]
            for i,im in enumerate(images):
                p=sample_dir/f"marked-{i}.png";im.save(p);media.append(str(p))
        scene=raw.get("scene_id",raw.get("scene_name",raw.get("videoID",video)))
        if scene is not None and raw.get("dataset") and raw["dataset"]!=dataset:
            scene=str(raw["dataset"])+"/"+str(scene)
        if scene is None:
            found=re.search(r"scene\d+_\d+"," ".join(clean_media))
            scene=found.group() if found else None
        marker_type=literal(marker_info).get('type') if marker_info else None
        row=normalize({"id":sid,"dataset":dataset,"question":question,"answer":answer,"choices":options,
              "media":media,"geometry_media":clean_media,"split":split,"scene_id":scene,
              "frame_indices":indices,"timestamps_s":timestamps,"task":raw.get("question_type",raw.get("task_type",raw.get("type",marker_type or "unspecified"))),
              "metric":"choice" if options else raw.get("metric", "numeric" if dataset.lower() in {"vsi-bench","vsibench","revsi"} else "exact"),
              "metadata":{"source_sha256":source_hash,"source_row":index,"markers_applied":bool(marker_info),
                 "marker_renderer_sha256":file_digest(marker_source) if marker_info else None,"raw_dataset":raw.get("dataset")}})
        rows.append(row)
    if not rows:raise ValueError("No rows built")
    if len({key(r) for r in rows})!=len(rows):raise ValueError("Duplicate IDs in source")
    write_jsonl(output,rows)
    write_json(output.with_suffix(".build.json"),{"dataset":dataset,"split":split,"rows":len(rows),"source_sha256":source_hash,
               "manifest_sha256":file_digest(output),"builder_sha256":builder_hash,"max_frames":max_frames,"status":"completed"})
    return {"manifest":str(output),"rows":len(rows)}


def partition(manifest,output,seed=3407):
    rows=load_samples(manifest);groups={}
    for row in rows:
        if row["split"]!="train":raise ValueError("Partition only the training pool; do not repartition released test sets")
        scene=row["scene_id"]
        if scene is None:raise ValueError("Scene IDs required for leakage-resistant splitting")
        group=str(scene)
        ratio=int(digest([seed,group])[:12],16)/16**12
        split="train" if ratio<.8 else "val" if ratio<.9 else "test"
        groups.setdefault(split,[]).append({**row,"split":split})
    out=Path(output)
    if out.exists() and any(out.iterdir()):raise ValueError("Split destination is nonempty")
    for split in ["train","val","test"]:write_jsonl(out/(split+".jsonl"),groups.get(split,[]))
    write_json(out/"split.json",{"seed":seed,"group_by":"scene_id","input_sha256":file_digest(manifest),
        "counts":{s:len(r) for s,r in groups.items()},"note":"Tiny pools can have an empty split; do not reuse test samples to fill it"})
    return {s:len(r) for s,r in groups.items()}

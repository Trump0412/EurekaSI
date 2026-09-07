"""Pinned source checkout and resumable HF downloads with receipts."""
import json
import os
import subprocess
import tarfile
import zipfile
from pathlib import Path

from .io import file_digest, write_json
from .workspace import resource_path, catalog, settings


def safe_extract(archive, destination):
    archive, destination = Path(archive), Path(destination).resolve()
    def check(name):
        p=(destination/name).resolve()
        if p!=destination and destination not in p.parents:
            raise ValueError("Archive path escapes destination")
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            for m in z.infolist():
                check(m.filename)
                if (m.external_attr>>16)&0o170000 == 0o120000:
                    raise ValueError("Symlinks in archive rejected")
            z.extractall(destination)
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as t:
            for m in t.getmembers():
                check(m.name)
                if not m.isfile() and not m.isdir():
                    raise ValueError("Only ordinary files/directories may be extracted")
            t.extractall(destination)
    else:
        raise ValueError(f"Unsupported archive: {archive}")


def download(name, *, include=None, extract=False, dry_run=False, local=None):
    cfg=settings(local); item=catalog("assets")[name]
    if item.get("repo_type") not in {"model","dataset"}:
        raise ValueError("Asset is a link-only resource, not a downloadable HF repository")
    target=Path(cfg["models" if item["repo_type"]=="model" else "datasets"])/name
    patterns=include if include is not None else item.get("allow_patterns")
    receipt=Path(cfg["receipts"])/(name+".json")
    previous=json.loads(receipt.read_text(encoding="utf-8")) if receipt.exists() else None
    if previous and (previous['repo_id']!=item['repo_id'] or previous.get('repo_type')!=item['repo_type']):
        raise ValueError('Asset ID now points to a different repository or repository type')
    revision=previous['resolved_revision'] if previous else item.get('revision','main')
    plan={"asset":name,"repo_id":item["repo_id"],"repo_type":item["repo_type"],
          "requested_revision":item.get("revision","main"),"destination":str(target),
          "endpoint":cfg["hf_endpoint"],"allow_patterns":patterns,
          "resolved_revision":revision if previous else None}
    if dry_run:return plan
    from huggingface_hub import HfApi,snapshot_download
    # First successful resolution freezes the revision across interrupted retries.
    if not previous:
        info=HfApi(endpoint=cfg["hf_endpoint"]).repo_info(item["repo_id"],repo_type=item["repo_type"],revision=item.get("revision","main"))
        revision=info.sha
    plan.update(resolved_revision=revision,status="downloading")
    write_json(receipt,plan)
    snapshot_download(repo_id=item["repo_id"],repo_type=item["repo_type"],revision=revision,
                      endpoint=cfg["hf_endpoint"],local_dir=target,allow_patterns=patterns)
    extracted=[]
    if extract:
        # Extract upstream archives only, never archives created by previous extraction.
        archives=[]
        for directory, dirs, files in os.walk(target):
            if Path(directory)==target:
                dirs[:]=[d for d in dirs if d not in {'media','.cache'}]
            archives.extend(Path(directory)/name for name in files)
        for p in sorted(archives):
            if p.is_file() and p.name.endswith((".zip",".tar",".tar.gz",".tgz")):
                stamp=p.with_name(p.name+".extracted.json")
                h=file_digest(p)
                if not stamp.exists() or json.loads(stamp.read_text(encoding="utf-8")).get("sha256")!=h:
                    safe_extract(p,target/"media")
                    write_json(stamp,{"sha256":h})
                extracted.append(p.name)
    plan.update(status="completed",extracted=extracted)
    write_json(receipt,plan)
    return plan


def fetch_source(name, *, dry_run=False, local=None):
    cfg=settings(local); item=catalog("sources")[name]
    dest=Path(cfg["external"])/name
    plan={"source":name,"url":item["url"],"commit":item["commit"],"destination":str(dest)}
    if dry_run:return plan
    if not (dest/".git").exists():
        if dest.exists() and any(dest.iterdir()):raise ValueError("Source destination is nonempty")
        subprocess.run(["git","clone","--no-checkout",item["url"],str(dest)],check=True)
        subprocess.run(["git","-C",str(dest),"checkout","--detach",item["commit"]],check=True)
    actual=subprocess.check_output(["git","-C",str(dest),"rev-parse","HEAD"],text=True).strip()
    if actual!=item["commit"]:
        raise ValueError("Existing source checkout differs from pinned commit; preserve it and select another workspace")
    if item.get("submodules"):
        subprocess.run(["git","-C",str(dest),"submodule","update","--init","--recursive"],check=True)
    for relative in item.get("patches",[]):
        patch=resource_path(relative)
        applied=subprocess.run(["git","-C",str(dest),"apply","--reverse","--check",str(patch)],capture_output=True).returncode==0
        if not applied:
            subprocess.run(["git","-C",str(dest),"apply","--check",str(patch)],check=True)
            subprocess.run(["git","-C",str(dest),"apply",str(patch)],check=True)
    plan["patches"]={p:file_digest(resource_path(p)) for p in item.get("patches",[])}
    plan["status"]="completed"
    write_json(Path(cfg["receipts"])/("source-"+name+".json"),plan)
    return plan

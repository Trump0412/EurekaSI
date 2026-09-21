"""Prepare immutable, CPU-only RFT manifests after explicit download gates.

The supplied-video subset is explicit: never treat missing released videos as
successful downloads. Benchmark source IDs and internal source groups are
disjoint; unknown scene aliases/perceptual duplicates remain an audit limit.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spatial_intelligence.rft_data import (read_jsonl, normalize_4drl,
    normalize_dsr, normalize_spatialladder, benchmark_groups, filter_leakage,
    group_split, extract_video)


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def write_rows(path, rows):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def verify_image(path):
    from PIL import Image
    try:
        with Image.open(path) as image:
            image.load()
        return None
    except Exception as error:
        return dict(path=path, error_type=type(error).__name__, error=str(error))


def prepare(args):
    import pandas as pd
    # Node-local convenience symlinks are not portable to a different worker.
    for name, value in vars(args).items():
        if isinstance(value, Path): setattr(args, name, value.resolve())
    args.benchmark_manifest = [path.resolve() for path in args.benchmark_manifest]
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (root / "prepare.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (root / "receipt.json").exists() and json.loads((root / "receipt.json").read_text()).get("status") == "ready":
        raise RuntimeError("Ready manifest is immutable: choose a new output directory")
    report = dict(status="preparing", ready_for_training=False, started=time.time(),
                  protocol="rft-source-group-v1", seed=args.seed,
                  validation_fraction=args.validation_fraction, media_verified=False,
                  leakage_checked=False, data_root=str(root),
                  limitations=["Identifier/source-group leakage audit, not perceptual or unknown-alias deduplication",
                               "Released SFT checkpoint pretraining contamination not audited",
                               "SpatialLadder original frame list used as multi-image, no invented FPS or verified temporal order"],
                  arguments={k:str(v) if isinstance(v,Path) else [str(x) for x in v] if isinstance(v,list) else v for k,v in vars(args).items()})
    receipt = root / "receipt.json"
    write(receipt, report)
    try:
        recovery_audit = None
        if getattr(args, 'exclude_upstream_empty', False):
            from spatial_intelligence.rft_recovery import audit_source_inventory
            if args.supplied_inventory is None:
                raise ValueError('Explicit upstream-empty exclusions require the original supplied inventory')
            recovery_audit = audit_source_inventory(args.supplied_inventory, args.four_d_media)
            report['explicit_source_empty_recovery'] = recovery_audit
            report['limitations'].append('Explicitly excludes upstream zero-byte source videos; not complete supplied-media reproduction')
        dsr = [normalize_dsr(x, i, args.dsr_media) for i,x in enumerate(
            pd.read_parquet(args.dsr_annotations).to_dict("records"))]
        four_d, annotation_exclusions = [], []
        for i,x in enumerate(read_jsonl(args.four_d_annotations)):
            try: four_d.append(normalize_4drl(x, i, args.four_d_media))
            except ValueError as error:
                annotation_exclusions.append(dict(id=f"4drl:{i}",source="4drl",raw_annotation=x,
                    reason="invalid_annotation",detail=str(error)))
        spatial = [normalize_spatialladder(x, args.spatial_media) for x in read_jsonl(args.spatial_annotations)]
        auxiliary = {Path(path).name: read_jsonl(path) for path in args.benchmark_manifest}
        if len(auxiliary) != len(args.benchmark_manifest):
            raise ValueError("Benchmark manifest basenames must be unique")
        forbidden = benchmark_groups(dsr)
        for rows in auxiliary.values():
            forbidden |= benchmark_groups(rows)
        exclusions = annotation_exclusions
        if args.supplied_inventory:
            supplied = set(json.loads(args.supplied_inventory.read_text())["files"])
            available = []
            for row in four_d:
                if Path(row["video_path"]).name not in supplied:
                    exclusions.append(dict(id=row["id"], source=row["source"],
                        scene_id=row["scene_id"], reason="not_in_declared_supplied_video_subset"))
                else:
                    available.append(row)
            four_d = available
        report["raw_rows"] = dict(four_d=len(read_jsonl(args.four_d_annotations)), spatialladder=len(spatial), dsr=len(dsr))
        candidates, leaked = filter_leakage(four_d + spatial, forbidden)
        exclusions.extend(leaked)
        seen = set()
        unique = []
        for row in candidates:
            if row["answer_type"] not in {"mcq", "numeric"}:
                exclusions.append(dict(id=row["id"],source=row["source"], raw_answer=row["raw_answer"], reason="unsupported_text_reward"))
                continue
            signature = json.dumps([row["source_group"], row.get("media", row.get("video_path")),
                                    row["question"], row["answer"], row["choices"]], sort_keys=True)
            if signature in seen:
                exclusions.append(dict(id=row["id"],source=row["source"], reason="duplicate_qa_within_source_group"))
            else:
                seen.add(signature); unique.append(row)
        split_rows = group_split(unique, args.seed, args.validation_fraction)
        if recovery_audit is not None:
            from spatial_intelligence.rft_recovery import exclude_source_empty_after_split
            split_rows, empty_exclusions = exclude_source_empty_after_split(
                split_rows, recovery_audit['upstream_empty_files'])
            exclusions.extend(empty_exclusions)
            report['explicit_source_empty_recovery']['excluded_rows'] = len(empty_exclusions)
            report['explicit_source_empty_recovery']['excluded_by_original_split'] = dict(
                Counter(row['original_split'] for row in empty_exclusions))
            report['explicit_source_empty_recovery']['split_policy'] = 'Remove only explicit empty media after original split; all retained assignments unchanged'
        if len({x["id"] for x in split_rows}) != len(split_rows):
            raise ValueError("Duplicate canonical ID")
        report["exclusions"] = dict(Counter(x["reason"] for x in exclusions))
        report["frame_counts"] = dict(Counter(len(x["media"]) for x in spatial))
        report["spatial_nonmonotonic_video_rows"] = sum(x["data_type"] == "video" and x["nonmonotonic_frame_order"] for x in spatial)
        write_rows(root / "exclusions.jsonl", exclusions)
        train_groups = {x["source_group"] for x in split_rows if x["split"] == "train"}
        validation_groups = {x["source_group"] for x in split_rows if x["split"] == "validation"}
        if train_groups & validation_groups or (train_groups | validation_groups) & forbidden:
            raise ValueError("Source group leakage")
        report.update(leakage_checked=True, train_test_overlap=0,
                      leakage_scope="canonical scene IDs, ScanNet building across rescans, 11-character source video IDs across clips",
                      groups=dict(train=len(train_groups), validation=len(validation_groups), benchmark=len(forbidden)))
        # Validate all provided SpatialLadder images, not only post-exclusion ones.
        image_paths = sorted({p for row in spatial for p in row["media"]})
        report.update(status="verifying_images", unique_images=len(image_paths)); write(receipt, report)
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            failures = [x for x in executor.map(verify_image, image_paths) if x]
        if failures:
            write(root / "media-failures.json", failures)
            raise RuntimeError(f"{len(failures)} image decode failures")
        # Prepare benchmark while training videos are still downloading.
        def decode_rows(rows, frames, name):
            paths = sorted({x["video_path"] for x in rows})
            results, failures = {}, []
            def task(path):
                try:
                    destination = root / "frames" / name / Path(path).stem
                    reuse_root = getattr(args, 'reuse_decoded_root', None)
                    if reuse_root is not None:
                        from spatial_intelligence.rft_recovery import reusable_cache
                        previous = reuse_root / "frames" / name / Path(path).stem
                        if reusable_cache(path, previous, frames):
                            destination = previous
                    return path, extract_video(path, destination, frames), None
                except Exception as error:
                    return path, None, dict(path=path,error_type=type(error).__name__,error=str(error))
            report.update(status="decoding_"+name, decode_total=len(paths), decode_completed=0)
            write(receipt, report)
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                for index, (path, result, error) in enumerate(executor.map(task, paths), 1):
                    if error: failures.append(error)
                    else: results[path] = result
                    if index % 10 == 0 or index == len(paths):
                        report.update(decode_completed=index, decode_failed=len(failures),updated=time.time());write(receipt,report)
            if failures:
                write(root / (name+"-media-failures.json"), failures)
                raise RuntimeError(f"{name}: {len(failures)} video decode failures; no ready receipt")
            return [dict(row, **{k:v for k,v in results[row["video_path"]].items() if k != "identity"}) for row in rows]
        dsr = decode_rows(dsr, 32, "dsr32")
        write_rows(root / "dsr.test.jsonl", dsr)
        if args.download_receipt and recovery_audit is None:
            start_wait = time.time()
            while True:
                state = json.loads(args.download_receipt.read_text()) if args.download_receipt.exists() else {}
                if state.get("status") == "download_exited_inventory_present":
                    break
                if state.get("status") in {"failed", "incomplete"}:
                    raise RuntimeError("Existing download supervisor failed/incomplete; no partial training")
                if not args.watch or time.time() - start_wait > args.wait_hours * 3600:
                    raise RuntimeError("Download not ready (use --watch for bounded persistent preparation)")
                report.update(status="waiting_for_download", download_observed=state.get("observed_files"),
                              download_expected=state.get("expected_files"), updated=time.time())
                write(receipt, report); time.sleep(30)
        four_d_rows = decode_rows([x for x in split_rows if x["source"] == "4drl"], 8, "4drl8")
        accepted = four_d_rows + [x for x in split_rows if x["source"] == "spatialladder"]
        manifests, validation = {}, {}
        for source in ("4drl", "spatialladder"):
            for split, mapping in (("train",manifests),("validation",validation)):
                selected=[x for x in accepted if x["source"] == source and x["split"] == split]
                if not selected: raise ValueError(f"Empty {source}/{split}")
                target=root/f"{source}.{split}.jsonl"; write_rows(target,selected);mapping[source]=str(target)
        # Existing benchmark records retain their original video/frame semantics.
        eval_manifests = {"dsr":str(root/"dsr.test.jsonl")}
        for path in args.benchmark_manifest:
            eval_manifests[Path(path).name.split('.')[0]]=str(Path(path).resolve())
        report.update(status="ready", ready_for_training=True, media_verified=True,
                      train_manifests=manifests, manifests=manifests,
                      validation_manifests=validation, eval_manifests=eval_manifests,
                      eval_manifest=eval_manifests["dsr"],
                      accepted_train_rows=sum(x["split"] == "train" for x in accepted),
                      rows=dict(Counter(x["source"]+"."+x["split"] for x in accepted)),
                      completed=time.time())
        write(receipt, report)
    except Exception as error:
        report.update(status="failed", ready_for_training=False, error_type=type(error).__name__,error=str(error),updated=time.time())
        write(receipt, report)
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("four-d-annotations","four-d-media","dsr-annotations","dsr-media","spatial-annotations","spatial-media","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--benchmark-manifest",type=Path,action="append",required=True)
    parser.add_argument("--supplied-inventory",type=Path)
    parser.add_argument("--download-receipt",type=Path)
    parser.add_argument('--exclude-upstream-empty', action='store_true',
                        help='Explicit revised-data authorization: exclude only inventory-declared/local zero-byte videos after original split')
    parser.add_argument('--reuse-decoded-root', type=Path,
                        help='Read-only reuse of old decoded caches with matching source size/mtime/frame-count identity')
    parser.add_argument("--workers",type=int,default=2)
    parser.add_argument("--seed",type=int,default=3407)
    parser.add_argument("--validation-fraction",type=float,default=.02)
    parser.add_argument("--wait-hours",type=float,default=72)
    parser.add_argument("--watch",action="store_true")
    parser.add_argument("--detach",action="store_true")
    args=parser.parse_args()
    if not 1 <= args.workers <= 4: parser.error("Bounded CPU workers must be 1..4")
    if args.detach:
        args.output.mkdir(parents=True,exist_ok=True)
        with (args.output/"prepare.log").open("ab") as log:
            child=subprocess.Popen([sys.executable,__file__,*[x for x in sys.argv[1:] if x!="--detach"]],
                stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True,
                env=dict(os.environ,CUDA_VISIBLE_DEVICES="",OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1"))
        print(json.dumps(dict(pid=child.pid,receipt=str(args.output/"receipt.json"))));return
    prepare(args)


if __name__ == "__main__": main()

"""Persistent, fail-closed geometry ablation queue (private JSON deployment plan).

Each node uses an independent root and immutable code/input snapshots. Existing
pipelines finish first. Profiles are disposable; formal training always starts
from the declared base/alignment checkpoint. A failed job does not cancel peers.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
TERMINAL = {"complete", "failed", "training_complete_evaluation_blocked", "complete_with_failures"}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def validate_plan(plan):
    validate_memory_limit(plan.get('memory_limit_fraction', .92))
    sft_batch=plan.get('sft_global_batch',384)
    if type(sft_batch) is not int or sft_batch < 1:
        raise ValueError('sft_global_batch must be a positive integer')
    encoder_batch = plan.get('encoder_batch_size', 1)
    if type(encoder_batch) is not int or encoder_batch < 1:
        raise ValueError('encoder_batch_size must be a positive integer')
    if type(plan.get('trainable_encoder_batching', False)) is not bool:
        raise ValueError('trainable_encoder_batching must be boolean')
    if type(plan.get('save_steps', 100)) is not int or plan.get('save_steps',100)<1:
        raise ValueError('save_steps must be a positive integer')
    for field in ("root", "python", "model", "processor", "vggt_source", "vggt_weights", "input_root"):
        if not plan.get(field):
            raise ValueError(f"Missing explicit {field}")
    devices = plan.get("gpus", [])
    if not devices or len(devices) != len(set(devices)) or any(not str(x).isdigit() for x in devices):
        raise ValueError("gpus must be distinct device indices")
    if any(batch % len(devices) for batch in (448, sft_batch)):
        raise ValueError("World size must divide both official global batches")
    jobs = plan.get("jobs", [])
    names = [j.get("name") for j in jobs]
    if not jobs or len(set(names)) != len(names):
        raise ValueError("Provide uniquely named jobs")
    for job in jobs:
        if not job.get("name") or Path(job["name"]).name != job["name"] or job["name"] in (".", ".."):
            raise ValueError("Job names must be plain directory names")
        if job.get("adapter") not in ("query64", "downsample") or type(job.get("train_vggt")) is not bool:
            raise ValueError("Explicit adapter and boolean train_vggt are required")
        if bool(job.get("alignment_checkpoint")) != bool(job.get("alignment_receipt")):
            raise ValueError("Shared alignment requires both checkpoint and acceptance receipt")
    if plan.get("profile_steps", 6) < 3:
        raise ValueError("Profiles need warmup and measured optimizer updates")
    return plan


def dependency_ready(dependency, proc_root=Path("/proc")):
    """Terminal receipts, not a transient moment of low GPU utilization."""
    if "pid" in dependency:
        process = proc_root / str(dependency["pid"])
        if not process.exists():
            return True
        try:
            command = (process / "cmdline").read_bytes().replace(b"\0", b" ").decode()
            fields = (process / "stat").read_text().rsplit(") ", 1)[1].split()
        except FileNotFoundError:
            return True
        if fields[0] == "Z":
            return True
        expected = dependency.get("start_time_ticks")
        if expected is not None and int(fields[19]) != int(expected):
            return True  # PID reused: the original supervisor already exited.
        token = dependency.get("command_contains")
        if not token:
            raise ValueError("PID dependencies need a verified command identity")
        return token not in command
    path = Path(dependency["path"])
    if not path.exists():
        return False
    value = read(path)
    statuses = dependency.get("statuses", ["complete"])
    if value.get("status") not in statuses:
        return False
    if dependency.get("require_gpu_work_finished") and not value.get("gpu_work_finished"):
        return False
    return True


def validate_memory_limit(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= 1:
        raise ValueError('memory_limit_fraction must be finite and in (0, 1]')
    return value


def select_profile(profiles, world, global_batch, memory_limit_fraction=.92):
    validate_memory_limit(memory_limit_fraction)
    candidates = []
    for item in profiles:
        micro = item.get("micro", 0)
        if not isinstance(micro, int) or micro < 1 or global_batch % (world * micro):
            continue
        if item.get("status") != "complete" or not item.get("long_sample_passed"):
            continue
        if item.get("warmup_steps", 0) < 1 or item.get("measured_steps", 0) < 2:
            continue
        speed = item.get("samples_per_second", 0)
        peak, capacity = item.get("peak_reserved_gib", math.inf), item.get("gpu_total_gib", 0)
        if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in (speed, peak, capacity)):
            continue
        fits = peak < memory_limit_fraction * capacity or (memory_limit_fraction == 1 and peak == capacity)
        if speed > 0 and capacity > 0 and fits:
            candidates.append(item)
    if not candidates:
        raise RuntimeError("No measured candidate passed warmup, throughput and long-sample memory gates")
    return max(candidates, key=lambda x: x["samples_per_second"])


def validate_gate(receipt):
    if receipt.get("status") != "complete":
        raise ValueError("Stage did not complete")
    for field in ("finite_loss", "nonzero_update", "reload_verified"):
        if receipt.get(field) is not True:
            raise ValueError(f"Missing actual stage acceptance: {field}")


def validate_components(receipt, job, stage):
    validate_gate(receipt)
    if stage == "sft":
        expected = ["model.visual", "model.language_model"]
        if job["train_vggt"]:
            expected.append("geometry_backbone")
        if not all(receipt.get("component_updates", {}).get(key) is True for key in expected):
            raise ValueError("Missing actual updates in intended unfrozen components")


def worker_command(plan, job, stage, model, name, micro, *, profile=False):
    command = [plan["python"], "-m", "torch.distributed.run", "--standalone",
               f"--nproc_per_node={len(plan['gpus'])}", str(REPO / "scripts/train-geometry-stage.py"),
               "--root", plan["root"], "--model", str(model), "--processor", plan["processor"],
               "--vggt-source", plan["vggt_source"], "--vggt-weights", plan["vggt_weights"],
               "--adapter", job["adapter"], "--stage", stage, "--name", name,
               "--micro", str(micro), "--global-batch", str(448 if stage == "align" else plan.get('sft_global_batch',384))]
    if stage == "sft" and job["train_vggt"]:
        command.append("--train-vggt")
    if plan.get('encoder_batch_size', 1) != 1:
        # A micro1 fallback must use the original serial path, not extra
        # collective scheduling for a singleton batch.
        command += ['--encoder-batch-size', str(min(plan['encoder_batch_size'], micro))]
    if stage == 'sft' and job['train_vggt'] and plan.get('trainable_encoder_batching', False):
        command.append('--trainable-encoder-batching')
    if plan.get('save_steps',100)!=100:
        command += ['--save-steps',str(plan['save_steps'])]
    if stage=='sft' and not profile and job.get('resume_checkpoint'):
        command += ['--resume-checkpoint',str(job['resume_checkpoint'])]
    if plan.get("manifest"):
        command += ["--manifest", plan["manifest"]]
    # Frozen-language activation checkpointing is validated with DDP alignment;
    # ZeRO-3 is reserved for the jointly trainable SFT stage.
    if stage == "sft" and plan.get("deepspeed"):
        command += ["--deepspeed", plan["deepspeed"]]
    if profile:
        command += ["--profile", "--max-steps", str(plan.get("profile_steps", 6))]
    return command


def diagnostic_manifests(root, stage, world, steps, sft_global_batch=384):
    """Fixed mixed rows and longest real rows; all candidates see the same IDs."""
    rows = [json.loads(line) for line in (root / "manifests/sft.train.jsonl").read_text().splitlines() if line.strip()]
    batch = 448 if stage == "align" else sft_global_batch
    count = steps * batch
    if len(rows) < count:
        raise ValueError("Formal manifest too short for matched profile")
    random.Random(3407).shuffle(rows)
    mixed = rows[:count]
    longest = sorted(rows, key=lambda row: (len(row.get("media", [])),
                     len(str(row.get("question", ""))) + len(str(row.get("answer", "")))), reverse=True)[:2 * world * 4]
    outputs = []
    for label, values in (("mixed", mixed), ("pressure", longest)):
        path = root / "manifests" / f"diagnostic-{stage}-{label}.jsonl"
        content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in values)
        if path.exists() and path.read_text(encoding="utf-8") != content:
            raise ValueError("Diagnostic sample identity changed")
        if not path.exists():
            path.write_text(content, encoding="utf-8")
        outputs.append(path)
    return outputs


def snapshot(plan, source=REPO):
    root = Path(plan["root"]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    saved = root / "plan.json"
    if saved.exists() and read(saved) != plan:
        raise ValueError("Plan changed: use a fresh queue root")
    write(saved, plan)
    target = root / "code"
    if not target.exists():
        staging = root / "code.pending"
        if staging.exists():
            raise RuntimeError("Incomplete snapshot exists; inspect it before retrying")
        staging.mkdir()
        for folder in ("scripts", "spatial_intelligence", "configs", "catalog"):
            shutil.copytree(source / folder, staging / folder,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        staging.rename(target)
    inputs = Path(plan["input_root"]).resolve()
    if inputs == root:
        raise ValueError("Input root must differ from independent queue output root")
    inputs_done = root / "state/input-snapshot.json"
    if inputs_done.exists():
        return target
    for folder in ("manifests", "receipts"):
        for src in (inputs / folder).glob("*.json*"):
            # Diagnostics belong to the destination's stage/global-batch contract.
            # A previous run can be an input root, but its generated profiles
            # must not become immutable inputs to a different training recipe.
            if src.name.startswith("diagnostic-"):
                continue
            destination = root / folder / src.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() and destination.read_bytes() != src.read_bytes():
                raise ValueError(f"Input snapshot changed: {src.name}")
            if not destination.exists():
                shutil.copy2(src, destination)
    write(inputs_done, dict(status="complete", copied_at=time.time()))
    return target


class Queue:
    def __init__(self, plan):
        self.plan = plan
        self.root = Path(plan["root"])
        self.env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(map(str, plan["gpus"])),
                        PYTHONPATH=str(REPO), PYTHONNOUSERSITE="1", OMP_NUM_THREADS="4",
                        TOKENIZERS_PARALLELISM="false", HF_ENDPOINT="https://hf-mirror.com",
                        TORCHINDUCTOR_COMPILE_THREADS="2",
                        TORCH_EXTENSIONS_DIR=str(self.root / "cache/torch-extensions"),
                        TRITON_CACHE_DIR=str(self.root / "cache/triton"))
        for folder in ("logs", "state", "runs"):
            (self.root / folder).mkdir(parents=True, exist_ok=True)
        for folder in ("torch-extensions", "triton"):
            (self.root / "cache" / folder).mkdir(parents=True, exist_ok=True)

    def status(self, phase, **extra):
        write(self.root / "state/matrix.json", dict(status=phase, pid=os.getpid(), updated=time.time(), **extra))

    def run(self, command, name, timeout):
        path = self.root / "state" / f"process-{name}.json"
        if path.exists():
            old = read(path)
            proc = Path("/proc") / str(old.get("pid", 0))
            if old.get("status") == "running" and proc.exists():
                raise RuntimeError(f"Previous child still exists: {name}; refuse duplicate launch")
        self.status("running", task=name)
        with (self.root / "logs" / f"{name}.log").open("ab") as log:
            child = subprocess.Popen(command, cwd=REPO, env=self.env, stdout=log,
                                     stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
            write(path, dict(status="running", pid=child.pid, command=command, started=time.time()))
            try:
                code = child.wait(timeout=timeout)
            except BaseException:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                write(path, dict(status="failed", pid=child.pid, error="interrupted_or_timed_out", finished=time.time()))
                raise
        write(path, dict(status="complete" if code == 0 else "failed", returncode=code, finished=time.time()))
        if code:
            raise subprocess.CalledProcessError(code, command)

    def idle(self):
        result = subprocess.check_output(["nvidia-smi", "--id=" + self.env["CUDA_VISIBLE_DEVICES"],
            "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True)
        values = [int(x.strip()) for x in result.splitlines()]
        return len(values) == len(self.plan["gpus"]) and all(x < 500 for x in values)

    def wait_dependencies(self):
        started = time.monotonic()
        while not all(dependency_ready(x) for x in self.plan.get("dependencies", [])):
            for dependency in self.plan.get("dependencies", []):
                if "pid" in dependency:
                    continue
                if Path(dependency["path"]).exists():
                    value = read(dependency["path"])
                    if value.get("status") in TERMINAL and value.get("status") not in dependency.get("statuses", ["complete"]):
                        raise RuntimeError("Prior pipeline terminated outside explicitly accepted statuses")
            if time.monotonic() - started > self.plan.get("dependency_timeout_seconds", 14 * 86400):
                raise TimeoutError("Prior pipeline wait exceeded configured deadline")
            self.status("waiting_for_prior_pipeline")
            time.sleep(30)
        while not self.idle():
            self.status("waiting_for_allocated_gpus")
            time.sleep(30)

    def stage(self, job, stage, model):
        name = f"{job['name']}-{stage}"
        completion = self.root / "runs" / name / "completion.json"
        if completion.exists():
            self.verify(job, stage, name)
            validate_components(read(completion), job, stage)
            final = self.root / "runs" / name / "final"
            if not (final / "config.json").exists():
                raise RuntimeError("Completed receipt has no final checkpoint config")
            return final
        profiles = []
        world = len(self.plan["gpus"])
        batch = 448 if stage == "align" else self.plan.get('sft_global_batch',384)
        mixed, pressure = diagnostic_manifests(self.root, stage, world, self.plan.get("profile_steps", 6),
                                             self.plan.get('sft_global_batch',384))
        for micro in (1, 2, 4):
            if batch % (world * micro):
                continue
            trial = f"diagnostic-{name}-b{micro}"
            receipt = self.root / "runs" / trial / "completion.json"
            try:
                if not receipt.exists():
                    command = worker_command(self.plan, job, stage, model, trial, micro, profile=True)
                    command += ["--manifest", str(mixed)]
                    self.run(command,
                             trial, self.plan.get("profile_timeout_seconds", 7200))
                self.verify(job, stage, trial)
                measured = read(receipt)
                validate_components(measured, job, stage)
                stress_name = trial + "-pressure"
                stress_receipt = self.root / "runs" / stress_name / "completion.json"
                if not stress_receipt.exists():
                    command = worker_command(self.plan, job, stage, model, stress_name, micro, profile=True)
                    command[command.index("--max-steps") + 1] = "2"
                    command[command.index("--global-batch") + 1] = str(world * 4)
                    command += ["--manifest", str(pressure)]
                    self.run(command, stress_name, self.plan.get("profile_timeout_seconds", 7200))
                self.verify(job, stage, stress_name)
                stress = read(stress_receipt)
                validate_gate(stress)
                measured["peak_reserved_gib"] = max(measured["peak_reserved_gib"], stress["peak_reserved_gib"])
                measured["long_sample_passed"] = True
                measured["micro"] = micro
                profiles.append(measured)
                if self.plan.get("rows") and measured.get("samples_per_second", 0) > 0:
                    write(self.root / "state" / f"{name}-provisional-eta.json",
                          dict(status="provisional_not_selected", micro=micro,
                               estimated_training_hours=self.plan["rows"] / measured["samples_per_second"] / 3600,
                               samples_per_second=measured["samples_per_second"], updated=time.time(),
                               excludes="remaining diagnostics, checkpoint save/reload and evaluation"))
            except Exception as exc:
                profiles.append(dict(status="failed", micro=micro, error=repr(exc)))
        # Real batch > 1 needs this exact stage's loss/full-gradient comparison,
        # not a historical receipt from another adapter or frozen component set.
        parity = {}
        allowed = [1]
        for micro in (2, 4):
            if not any(item.get("micro") == micro and item.get("status") == "complete" for item in profiles):
                continue
            parity_name = f"diagnostic-{name}-parity-b{micro}"
            parity_path = self.root / "runs" / parity_name / "parity.json"
            try:
                if not parity_path.exists():
                    command = worker_command(self.plan, job, stage, model, parity_name, micro)
                    command = [self.plan["python"], *command[5:], "--parity", "--manifest", str(mixed)]
                    self.run(command, parity_name, self.plan.get("profile_timeout_seconds", 7200))
                parity[str(micro)] = read(parity_path)
                if parity[str(micro)].get("status") == "accepted":
                    allowed.append(micro)
            except Exception as exc:
                parity[str(micro)] = dict(status="failed", error=repr(exc))
        chosen = select_profile([x for x in profiles if x["micro"] in allowed], world, batch,
                                self.plan.get('memory_limit_fraction', .92))
        selection = dict(selected=chosen, profiles=profiles, global_batch=batch,
                         memory_limit_fraction=self.plan.get('memory_limit_fraction', .92),
                         ga=batch // (world * chosen["micro"]), world_size=world,
                         numerical_parity=parity, permitted_micro_candidates=allowed)
        if self.plan.get("rows"):
            selection["estimated_training_hours"] = self.plan["rows"] / chosen["samples_per_second"] / 3600
        write(self.root / "state" / f"{name}-selection.json", selection)
        self.run(worker_command(self.plan, job, stage, model, name, chosen["micro"]), name,
                 self.plan.get("training_timeout_seconds", 7 * 86400))
        self.verify(job, stage, name)
        formal = read(completion)
        validate_components(formal, job, stage)
        final = self.root / "runs" / name / "final"
        if not (final / "config.json").exists():
            raise RuntimeError("Formal checkpoint config missing")
        return final

    def verify(self, job, stage, name):
        receipt = self.root / "runs" / name / "completion.json"
        if read(receipt).get("reload_verified") is True:
            return
        command = worker_command(self.plan, job, stage, self.root / "runs" / name / "final", name, 1)
        # Reload is deliberately fresh-process, single GPU, not another update.
        command = [self.plan["python"], *command[5:], "--verify-reload"]
        self.run(command, name + "-reload", 3600)

    def shared_alignment(self, job):
        receipt = Path(job["alignment_receipt"])
        checkpoint = Path(job["alignment_checkpoint"])
        started = time.monotonic()
        while not receipt.exists() or read(receipt).get("reload_verified") is not True:
            failure = job.get("alignment_failure_state")
            if failure and Path(failure).exists() and read(failure).get("status") == "failed":
                raise RuntimeError("Shared alignment producer failed; no replacement initialization allowed")
            if time.monotonic() - started > self.plan.get("dependency_timeout_seconds", 14 * 86400):
                raise TimeoutError("Shared alignment acceptance wait exceeded deadline")
            self.status("waiting_for_shared_alignment", job=job["name"])
            time.sleep(30)
        value = read(receipt)
        validate_gate(value)
        if Path(value["checkpoint"]).resolve() != checkpoint.resolve() or not (checkpoint / "config.json").exists():
            raise ValueError("Shared alignment checkpoint identity mismatch")
        contract = read(checkpoint.parent / "contract.json")
        if contract.get("stage") != "align" or contract.get("adapter") != job["adapter"] or contract.get("seed") != 3407:
            raise ValueError("Shared alignment recipe mismatch")
        if contract.get("diagnostic") or contract.get("train_vggt") or contract.get("global_batch") != 448:
            raise ValueError("Shared checkpoint is not the formal frozen alignment stage")
        if Path(contract["initial_model"]).resolve() != Path(self.plan["model"]).resolve():
            raise ValueError("Shared alignment used a different initial model")
        for key, expected in {"epochs": 1, "lr": .001, "weight_decay": 0., "warmup": .03}.items():
            if contract.get(key) != expected:
                raise ValueError(f"Shared alignment hyperparameter mismatch: {key}")
        if Path(contract["manifest"]).read_bytes() != (self.root / "manifests/sft.train.jsonl").read_bytes():
            raise ValueError("Shared alignment used a different training manifest")
        write(self.root / "state" / f"{job['name']}-shared-alignment.json",
              dict(status="accepted", checkpoint=str(checkpoint), source_receipt=str(receipt), read_only_reuse=True))
        return checkpoint

    def evaluate(self, job, model):
        errors = {}
        for benchmark in ("revsi", "vsibench"):
            try:
                for smoke in (True, False):
                    name = f"{job['name']}-{benchmark}" + ("-format-smoke" if smoke else "")
                    base = [self.plan["python"], str(REPO / "scripts/train-geometry-stage.py"),
                            "--evaluate", "--root", str(self.root), "--model", str(model),
                            "--processor", self.plan["processor"], "--vggt-source", self.plan["vggt_source"],
                            "--vggt-weights", self.plan["vggt_weights"], "--adapter", job["adapter"],
                            "--stage", "sft", "--name", name, "--benchmark", benchmark]
                    if smoke:
                        base += ["--smoke-per-type", "1"]
                    metric_path = self.root / "runs" / name / "metrics.json"
                    if not metric_path.exists():
                        distributed = [self.plan["python"], "-m", "torch.distributed.run", "--standalone",
                                       f"--nproc_per_node={len(self.plan['gpus'])}", *base[1:]]
                        self.run(distributed, name, self.plan.get("evaluation_timeout_seconds", 86400))
                        self.run(base + ["--merge"], name + "-merge", 1800)
                    metrics = read(metric_path)
                    if smoke and (metrics["parse_rate"] < .9 or metrics["truncation_rate"] > .05):
                        raise RuntimeError("Format gate failed; no full benchmark score claimed")
            except Exception as exc:
                errors[benchmark] = repr(exc)
        if errors:
            raise RuntimeError(json.dumps(errors))

    def execute(self, retry_failed=False):
        import fcntl
        with (self.root / "state/matrix.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                self.wait_dependencies()
            except Exception as exc:
                self.status("failed", error=repr(exc), gpu_work_finished=True)
                raise
            outcomes = {}
            for job in self.plan["jobs"]:
                receipt = self.root / "state" / f"job-{job['name']}.json"
                previous = read(receipt) if receipt.exists() else {}
                if previous.get("status") == "complete" or (previous.get("status") == "failed" and not retry_failed):
                    outcomes[job["name"]] = previous
                    continue
                try:
                    while not self.idle():
                        self.status("waiting_for_allocated_gpus", job=job["name"])
                        time.sleep(30)
                    aligned = self.shared_alignment(job) if job.get("alignment_checkpoint") else self.stage(job, "align", self.plan["model"])
                    final = self.stage(job, "sft", aligned)
                    self.evaluate(job, final)
                    outcome = dict(status="complete", final=str(final), finished=time.time())
                except Exception as exc:
                    outcome = dict(status="failed", error=repr(exc), finished=time.time())
                write(receipt, outcome)
                outcomes[job["name"]] = outcome
            self.status("complete" if all(v["status"] == "complete" for v in outcomes.values()) else "complete_with_failures",
                        jobs=outcomes, gpu_work_finished=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()
    plan = validate_plan(read(args.plan))
    root = Path(plan["root"])
    if args.detach:
        frozen = snapshot(plan)
        (root / "logs").mkdir(exist_ok=True)
        command = [sys.executable, str(frozen / "scripts" / Path(__file__).name), "--plan", str(root / "plan.json")]
        if args.retry_failed:
            command.append("--retry-failed")
        with (root / "logs/matrix-supervisor.log").open("ab") as log:
            process = subprocess.Popen(command, cwd=frozen, stdout=log, stderr=subprocess.STDOUT,
                                       stdin=subprocess.DEVNULL, start_new_session=True)
        print(json.dumps(dict(status="supervisor_launched_not_training_acceptance", pid=process.pid, root=str(root))))
    else:
        Queue(plan).execute(args.retry_failed)


if __name__ == "__main__":
    main()

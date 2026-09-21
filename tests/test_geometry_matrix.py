"""CPU-only safety contracts for the geometry matrix supervisor."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location("geometry_matrix_queue", Path(__file__).resolve().parents[1] / "scripts/run-geometry-matrix.py")
queue = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(queue)


def plan():
    return dict(root="/output", python="/env/bin/python", model="/models/base", processor="/models/base",
                vggt_source="/source/vggt", vggt_weights="/models/vggt", input_root="/inputs",
                gpus=list(range(8)), jobs=[dict(name="frozen-downsample", adapter="downsample", train_vggt=False)])


def profile(micro=1, speed=4., peak=30.):
    return dict(status="complete", micro=micro, samples_per_second=speed, peak_reserved_gib=peak,
                gpu_total_gib=40., warmup_steps=1, measured_steps=5, long_sample_passed=True)


class GeometryMatrixTests(unittest.TestCase):
    def test_global64_changes_sft_commands_and_diagnostics_not_alignment(self):
        p=plan();p['sft_global_batch']=64
        queue.validate_plan(p)
        sft=queue.worker_command(p,p['jobs'][0],'sft','/align','probe',2)
        align=queue.worker_command(p,p['jobs'][0],'align','/base','probe',2)
        self.assertEqual(sft[sft.index('--global-batch')+1],'64')
        self.assertEqual(align[align.index('--global-batch')+1],'448')
        self.assertEqual(64//(len(p['gpus'])*2),4)
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);(root/'manifests').mkdir()
            rows=[dict(id=str(i),media=['x']*(i%3+1),question='q',answer='a') for i in range(2700)]
            (root/'manifests/sft.train.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
            mixed,pressure=queue.diagnostic_manifests(root,'sft',8,6,64)
            self.assertEqual(len(mixed.read_text().splitlines()),384)
            self.assertEqual(len(pressure.read_text().splitlines()),64)
            align_mixed,_=queue.diagnostic_manifests(root,'align',8,6,64)
            self.assertEqual(len(align_mixed.read_text().splitlines()),2688)
        for invalid in (0,-1,True,64.5,65):
            p['sft_global_batch']=invalid
            with self.assertRaises(ValueError):queue.validate_plan(p)

    def test_explicit_physical_memory_cap_does_not_reject_accepted_high_peak(self):
        high=profile(2,8,37.8)
        self.assertEqual(queue.select_profile([profile(1,5),high],8,384)['micro'],1)
        self.assertEqual(queue.select_profile([profile(1,5),high],8,384,1.0)['micro'],2)
        exact=profile(2,8,40)
        self.assertEqual(queue.select_profile([exact],8,384,1.0)['micro'],2)
        for change in ({'peak_reserved_gib':40.1},{'long_sample_passed':False},
                       {'status':'failed'},{'samples_per_second':float('nan')}):
            bad=dict(high,**change)
            with self.assertRaises(RuntimeError):queue.select_profile([bad],8,384,1.0)
        for invalid in (0,-1,1.01,float('nan'),float('inf'),True,'1'):
            with self.assertRaises(ValueError):queue.select_profile([high],8,384,invalid)
            p=plan();p['memory_limit_fraction']=invalid
            with self.assertRaises(ValueError):queue.validate_plan(p)

    def test_encoder_throughput_options_are_explicit(self):
        p = plan(); p['encoder_batch_size'] = 4
        p['trainable_encoder_batching'] = True
        job = dict(name='trainable', adapter='downsample', train_vggt=True)
        command = queue.worker_command(p, job, 'sft', '/align', 'probe', 2)
        self.assertIn('--trainable-encoder-batching', command)
        self.assertEqual(command[command.index('--encoder-batch-size')+1], '2')
        serial = queue.worker_command(p, job, 'sft', '/align', 'probe', 1)
        self.assertEqual(serial[serial.index('--encoder-batch-size')+1], '1')
        frozen = queue.worker_command(p, p['jobs'][0], 'sft', '/align', 'probe', 2)
        self.assertNotIn('--trainable-encoder-batching', frozen)
        p['encoder_batch_size'] = True
        with self.assertRaises(ValueError): queue.validate_plan(p)

    def test_explicit_configuration(self):
        queue.validate_plan(plan())
        bad = plan(); bad["jobs"][0].pop("train_vggt")
        with self.assertRaises(ValueError): queue.validate_plan(bad)

    def test_duplicate_gpus_and_unsafe_name(self):
        bad = plan(); bad["gpus"] = [0, 0]
        with self.assertRaises(ValueError): queue.validate_plan(bad)
        bad = plan(); bad["jobs"][0]["name"] = "../outside"
        with self.assertRaises(ValueError): queue.validate_plan(bad)

    def test_fastest_safe_not_highest_memory(self):
        selected = queue.select_profile([profile(1, 5), profile(2, 8), profile(4, 10, 38)], 8, 384)
        self.assertEqual(selected["micro"], 2)

    def test_requires_pressure_and_warmup(self):
        bad = profile(); bad["long_sample_passed"] = False
        with self.assertRaises(RuntimeError): queue.select_profile([bad], 8, 448)
        bad = profile(); bad["warmup_steps"] = 0
        with self.assertRaises(RuntimeError): queue.select_profile([bad], 8, 448)

    def test_nonfinite_and_invalid_ga_rejected(self):
        with self.assertRaises(RuntimeError): queue.select_profile([profile(speed=float("nan"))], 8, 448)
        with self.assertRaises(RuntimeError): queue.select_profile([profile(micro=3)], 8, 448)

    def test_real_acceptance_not_file_exists(self):
        with self.assertRaises(ValueError): queue.validate_gate(dict(status="complete"))
        queue.validate_gate(dict(status="complete", finite_loss=True, nonzero_update=True, reload_verified=True))

    def test_unfrozen_requires_vggt_update(self):
        receipt = dict(status="complete", finite_loss=True, nonzero_update=True, reload_verified=True,
                       component_updates={"model.visual": True, "model.language_model": True})
        queue.validate_components(receipt, dict(train_vggt=False), "sft")
        with self.assertRaises(ValueError): queue.validate_components(receipt, dict(train_vggt=True), "sft")
        receipt["component_updates"]["geometry_backbone"] = True
        queue.validate_components(receipt, dict(train_vggt=True), "sft")

    def test_shared_alignment_requires_paired_receipt(self):
        value = plan(); value["jobs"][0]["alignment_checkpoint"] = "/shared/final"
        with self.assertRaises(ValueError): queue.validate_plan(value)
        value["jobs"][0]["alignment_receipt"] = "/shared/completion.json"
        queue.validate_plan(value)

    def test_snapshot_does_not_refresh_mutable_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp); source = base / "repo"; inputs = base / "inputs"
            for folder in ("scripts", "spatial_intelligence", "configs", "catalog"):
                (source / folder).mkdir(parents=True)
            (inputs / "manifests").mkdir(parents=True)
            manifest = inputs / "manifests/sft.train.jsonl"; manifest.write_text("original")
            value = plan(); value.update(root=str(base / "study"), input_root=str(inputs))
            queue.snapshot(value, source)
            manifest.write_text("modified upstream")
            queue.snapshot(value, source)
            self.assertEqual((base / "study/manifests/sft.train.jsonl").read_text(), "original")

    def test_snapshot_excludes_source_diagnostics_for_new_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp); source = base / "repo"; inputs = base / "inputs"
            for folder in ("scripts", "spatial_intelligence", "configs", "catalog"):
                (source / folder).mkdir(parents=True)
            for folder in ("manifests", "receipts"):
                (inputs / folder).mkdir(parents=True)
                (inputs / folder / "diagnostic-sft-mixed.jsonl").write_text("old batch384")
            manifest = inputs / "manifests/sft.train.jsonl"
            manifest.write_text("".join(json.dumps(dict(id=i, media=['frame'],question='q',answer='a'))+'\n' for i in range(2304)))
            value = plan(); value.update(root=str(base / "study"),input_root=str(inputs),sft_global_batch=64)
            queue.snapshot(value,source)
            out = base / "study"
            self.assertEqual((out / "manifests/sft.train.jsonl").read_bytes(),manifest.read_bytes())
            self.assertFalse((out / "manifests/diagnostic-sft-mixed.jsonl").exists())
            self.assertFalse((out / "receipts/diagnostic-sft-mixed.jsonl").exists())
            mixed,_ = queue.diagnostic_manifests(out,'sft',4,6,64)
            self.assertEqual(len(mixed.read_text().splitlines()),384)

    def test_dependency_terminal_and_gpu_release(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            dep = dict(path=str(path), statuses=["complete"], require_gpu_work_finished=True)
            self.assertFalse(queue.dependency_ready(dep))
            queue.write(path, dict(status="complete"))
            self.assertFalse(queue.dependency_ready(dep))
            queue.write(path, dict(status="complete", gpu_work_finished=True))
            self.assertTrue(queue.dependency_ready(dep))

    def test_wait_exact_prior_supervisor_not_transient_gpu_idle(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); process = root / "123"; process.mkdir()
            (process / "cmdline").write_bytes(b"python\0scripts/prior.py\0")
            (process / "stat").write_text("123 (python) S " + "0 " * 18 + "456")
            dep = dict(pid=123, command_contains="scripts/prior.py", start_time_ticks=456)
            self.assertFalse(queue.dependency_ready(dep, root))
            dep["start_time_ticks"] = 455
            self.assertTrue(queue.dependency_ready(dep, root))

    def test_align_never_unfreezes_vggt(self):
        p = plan(); job = dict(name="unfrozen", adapter="query64", train_vggt=True)
        align = queue.worker_command(p, job, "align", p["model"], "align", 2)
        self.assertNotIn("--train-vggt", align)
        self.assertEqual(align[align.index("--global-batch") + 1], "448")
        sft = queue.worker_command(p, job, "sft", "/output/align/final", "sft", 4)
        self.assertIn("--train-vggt", sft)
        self.assertEqual(sft[sft.index("--model") + 1], "/output/align/final")
        self.assertEqual(sft[sft.index("--global-batch") + 1], "384")

    def test_profiles_do_not_change_formal_budget(self):
        p = plan(); job = p["jobs"][0]
        formal = queue.worker_command(p, job, "align", p["model"], "formal", 1)
        self.assertNotIn("--max-steps", formal)
        diagnostic = queue.worker_command(p, job, "align", p["model"], "diagnostic", 1, profile=True)
        self.assertIn("--profile", diagnostic)
        self.assertEqual(diagnostic[diagnostic.index("--max-steps") + 1], "6")

    def test_alignment_ddp_sft_zero3(self):
        p = plan(); p["deepspeed"] = "/configs/zero3.json"; job = p["jobs"][0]
        align = queue.worker_command(p, job, "align", p["model"], "align", 1)
        sft = queue.worker_command(p, job, "sft", "/align/final", "sft", 1)
        self.assertNotIn("--deepspeed", align)
        self.assertIn("--deepspeed", sft)

    def test_diagnostic_ids_stable_and_real(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / "manifests").mkdir()
            rows = [dict(id=str(i), media=["image"] * (32 if i < 100 else 1), question="Q", answer="A") for i in range(3000)]
            (root / "manifests/sft.train.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
            first = queue.diagnostic_manifests(root, "align", 8, 6)
            content = [p.read_bytes() for p in first]
            second = queue.diagnostic_manifests(root, "align", 8, 6)
            self.assertEqual(content, [p.read_bytes() for p in second])
            pressure = [json.loads(line) for line in second[1].read_text().splitlines()]
            self.assertEqual(len(pressure), 64)
            self.assertTrue(all(len(row["media"]) == 32 for row in pressure))


if __name__ == "__main__": unittest.main()

"""Real RGB+frozen VGGT forward/backward, null intervention and reload gate."""
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True); p.add_argument('--source', required=True)
    p.add_argument('--weights', required=True); p.add_argument('--manifest', required=True)
    p.add_argument('--output', required=True); a = p.parse_args()
    import torch
    from transformers import AutoProcessor, set_seed
    from spatial_intelligence.qwen3vl_geometry import load_geometry_model, GeometryCollator
    from spatial_intelligence.qwen35 import completion_loss
    from spatial_intelligence.study import dump, read_rows
    torch.set_num_threads(4); set_seed(3408)
    out = Path(a.output); out.mkdir(parents=True, exist_ok=False)
    rows = read_rows(Path(a.manifest))
    # Ordinary and maximum-frame cases, not a hand-picked tiny image only.
    selected = [rows[0], max(rows, key=lambda row: len(row['media']))]
    proc = AutoProcessor.from_pretrained(a.model)
    model = load_geometry_model(a.model, training=True).cuda()
    model.config.geometry_source = a.source; model.config.geometry_weights = a.weights
    model.geometry_adapter.sample_dropout = 0  # deterministic diagnostic, not training recipe
    collate = GeometryCollator(proc)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5)
    evidence = []
    for row in selected:
        start = time.monotonic()
        batch = {k: v.cuda() if torch.is_tensor(v) else v for k,v in collate([row]).items()}
        before = model.geometry_adapter.queries.detach().clone()
        loss, _ = completion_loss(model, batch)
        loss.backward()
        gradients = {name: float(param.grad.float().norm()) if param.grad is not None else None
                     for name,param in model.geometry_adapter.named_parameters()}
        dump(out/'gradient-audit.json', {'loss': float(loss), 'gradients': gradients,
             'parameter_finite': {n: bool(torch.isfinite(p).all()) for n,p in model.geometry_adapter.named_parameters()}})
        if not torch.isfinite(loss) or any(v is None or not torch.isfinite(torch.tensor(v)) for v in gradients.values()):
            raise ValueError('Invalid geometry gradient graph')
        if not any(v > 0 for name,v in gradients.items() if 'null' not in name):
            raise ValueError('Geometry interface receives no signal')
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        if torch.equal(before, model.geometry_adapter.queries): raise ValueError('Adapter did not update')
        evidence.append({'id': row['id'], 'frames': len(row['media']), 'loss': float(loss),
            'geometry_gradients': gradients, 'seconds': time.monotonic()-start,
            'peak_reserved_gib': torch.cuda.max_memory_reserved()/2**30})
        dump(out/'progress.json', evidence)
    model.eval()
    # Use selected completion logits only; no full-context vocabulary allocation.
    with torch.no_grad():
        _, normal = completion_loss(model, batch)
        batch['geometry_force_null'].fill_(True)
        _, null = completion_loss(model, batch)
        delta = float((normal.logits-null.logits).float().abs().max())
    if not delta > 0: raise ValueError('Scene geometry does not affect outputs')
    batch['geometry_force_null'].fill_(False)
    model.save_pretrained(out/'checkpoint'); proc.save_pretrained(out/'checkpoint')
    reference = normal.logits.detach().cpu()
    del model, optimizer, normal, null
    import gc
    gc.collect(); torch.cuda.empty_cache()
    restored = load_geometry_model(out/'checkpoint').cuda()
    with torch.no_grad(): _, result = completion_loss(restored, batch)
    reload_delta = float((reference-result.logits.cpu()).float().abs().max())
    if reload_delta > 0.05: raise ValueError(f'Reload logits mismatch {reload_delta}')
    dump(out/'acceptance.json', {'status': 'accepted', 'cases': evidence,
         'geometry_vs_null_max_logit_delta': delta, 'reload_max_logit_delta': reload_delta,
         'frozen_vggt': True, 'diagnostic_only': True})


if __name__ == '__main__': main()

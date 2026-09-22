"""Same-checkpoint/input g=0 versus g=1 GeoRoute influence diagnostic.

Requires a TRUSTED local torch bundle with {'inputs': tensor_dict,
'graph': RouteGraph field dict}. Never builds teacher graphs, changes training,
or launches distributed work. CPU/float32 is the default. GPU is opt-in.
Numerical influence is not evidence of better downstream accuracy.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def tensor_comparison(reference, actual):
    import torch
    if reference.shape != actual.shape:
        raise ValueError('Comparison shapes differ')
    left, right = reference.detach().float().reshape(-1), actual.detach().float().reshape(-1)
    if not torch.isfinite(left).all() or not torch.isfinite(right).all():
        raise ValueError('Nonfinite diagnostic tensors')
    delta = right - left
    baseline_norm, result_norm, delta_norm = float(left.norm()), float(right.norm()), float(delta.norm())
    cosine = float(torch.dot(left, right) / (left.norm() * right.norm())) if baseline_norm and result_norm else None
    return dict(shape=list(reference.shape), reference_l2=baseline_norm, actual_l2=result_norm,
                delta_l2=delta_norm, relative_delta_l2=delta_norm / baseline_norm if baseline_norm else None,
                cosine=cosine, max_abs_delta=float(delta.abs().max()) if delta.numel() else 0.,
                changed_elements=int((delta != 0).sum()), elements=delta.numel())


def graph_summary(graph):
    import torch
    graph.validate()
    destinations = torch.unique(graph.dst)
    return dict(nodes=graph.num_nodes, edges=int(graph.src.numel()), connected_nodes=int(destinations.numel()),
                connected_fraction=destinations.numel() / graph.num_nodes if graph.num_nodes else 0.,
                frames=int(torch.unique(graph.frame_ids).numel()),
                samples=int(torch.unique(graph.sample_ids).numel()))


def capture_forward(model, inputs, graph, strength):
    """Hooks bracket existing routing hooks; captures are CPU-only, then freed."""
    import torch
    states, records, handles = {}, [], []
    visual = model.model.visual
    mergers = list(visual.deepstack_merger_list) + [visual.merger]
    if len(mergers) != 4:
        raise ValueError('Expected four native merger exits')
    def capture(name, value):
        if name in states:
            raise ValueError('Repeated merger call; diagnostic expects one prefill')
        states[name] = value.detach().float().cpu().clone()
    for index, merger in enumerate(mergers):
        # prepend brackets the core pre-route hook; append observes its result.
        def before(module, args, i=index): capture(f'exit{i}.pre_before_route', args[0])
        def routed(module, args, i=index): capture(f'exit{i}.pre_merger', args[0])
        def raw_post(module, args, output, i=index): capture(f'exit{i}.post_merger_before_post_route', output)
        def post(module, args, output, i=index): capture(f'exit{i}.post_exit', output)
        handles += [merger.register_forward_pre_hook(before, prepend=True),
                    merger.register_forward_pre_hook(routed),
                    merger.register_forward_hook(raw_post, prepend=True),
                    merger.register_forward_hook(post)]
    try:
        with torch.no_grad(), model.georoute.graph_context(graph), \
                model.georoute.routing_diagnostics(strength=strength, telemetry=records.append):
            result = model(**inputs, use_cache=False, logits_to_keep=1)
        logits = result.logits.detach().float().cpu()
        return states, logits, records
    finally:
        for handle in handles:
            handle.remove()


def compare_passes(zero, one):
    import torch
    states0, logits0, telemetry0 = zero
    states1, logits1, telemetry1 = one
    if set(states0) != set(states1):
        raise ValueError('Missing/mismatched exit captures')
    comparisons = {name: tensor_comparison(states0[name], states1[name]) for name in sorted(states0)}
    exits = []
    for index in range(4):
        prefix = f'exit{index}'
        pre = comparisons[f'{prefix}.pre_merger']
        post = comparisons[f'{prefix}.post_merger_before_post_route']
        # Different widths/norm scales: this is numerical transmission, not an
        # information-theoretic retention ratio or a causal accuracy effect.
        pre_relative, post_relative = pre['relative_delta_l2'], post['relative_delta_l2']
        exits.append(dict(exit_index=index,
            pre_merger_relative_delta=pre_relative, post_merger_relative_delta=post_relative,
            post_over_pre_relative_delta=post_relative / pre_relative if pre_relative and post_relative is not None else None,
            pre_changed=pre['changed_elements'] > 0, post_changed=post['changed_elements'] > 0))
    p0, p1 = torch.log_softmax(logits0, dim=-1), torch.log_softmax(logits1, dim=-1)
    kl = (p0.exp() * (p0 - p1)).sum(-1)
    return dict(exits=exits, tensor_comparisons=comparisons,
                next_token_logits=tensor_comparison(logits0, logits1),
                next_token_kl_zero_to_one_mean=float(kl.mean()),
                argmax_zero=logits0.argmax(-1).tolist(), argmax_one=logits1.argmax(-1).tolist(),
                block_telemetry_zero=telemetry0, block_telemetry_one=telemetry1)


def gradient_probe(model, inputs, graph, target_token):
    """Gradient connectivity for one explicit next-token target; no optimizer."""
    import torch
    model.zero_grad(set_to_none=True)
    try:
        with model.georoute.graph_context(graph), model.georoute.routing_diagnostics(strength=1.):
            logits = model(**inputs, use_cache=False, logits_to_keep=1).logits.float()
            if not 0 <= target_token < logits.shape[-1]:
                raise ValueError('Target token outside model vocabulary')
            loss = -torch.log_softmax(logits, -1)[..., target_token].mean()
            loss.backward()
        report = {}
        for name, parameter in model.georoute.named_parameters():
            grad = parameter.grad
            if grad is not None and not torch.isfinite(grad).all():
                raise ValueError('Nonfinite diagnostic gradient')
            report[name] = dict(connected=grad is not None,
                               gradient_l2=None if grad is None else float(grad.detach().float().norm()),
                               dtype=str(parameter.dtype))
        return dict(target_token=target_token, loss=float(loss.detach()), gradients=report,
                    scope='Single explicit next-token connectivity, not training quality')
    finally:
        model.zero_grad(set_to_none=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--dtype', choices=['float32', 'bfloat16'], default='float32')
    parser.add_argument('--gradient-target-token', type=int)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError('Preserve prior evidence; choose a new output path')
    import torch
    from spatial_intelligence.georoute import RouteGraph, load_georoute_model
    torch.set_num_threads(4)
    bundle = torch.load(args.bundle, map_location='cpu', weights_only=True)
    inputs = dict(bundle['inputs'])
    if any(key in inputs for key in ('labels', 'past_key_values', 'inputs_embeds', 'use_cache', 'logits_to_keep')):
        raise ValueError('Bundle requires raw prompt tensor inputs only, no labels/cache/overrides')
    if not inputs or any(not isinstance(value, torch.Tensor) for value in inputs.values()):
        raise ValueError('Every input must be a tensor')
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    graph = RouteGraph(**bundle['graph']).validate().to(device)
    model = load_georoute_model(args.checkpoint, dtype=dtype, attn_implementation='sdpa').to(device).eval()
    # Freeze backbone while allowing an optional route-gradient diagnostic.
    model.requires_grad_(False)
    model.georoute.requires_grad_(args.gradient_target_token is not None)
    inputs = {key: value.to(device=device, dtype=dtype if value.is_floating_point() else value.dtype)
              for key, value in inputs.items()}
    report = compare_passes(capture_forward(model, inputs, graph, 0.),
                            capture_forward(model, inputs, graph, 1.))
    report.update(status='measured', checkpoint=args.checkpoint, bundle=str(args.bundle),
                  dtype=args.dtype, device=str(device), placement=model.georoute.settings.get('placement', 'pre'),
                  graph=graph_summary(graph),
                  alphas={name: float(parameter.detach()) for name, parameter in model.georoute.named_parameters()
                          if name.endswith('.alpha')},
                  limitations=['One fixed prompt prefill, no generation quality claim',
                               'Different merger input/output scales prevent information-retention interpretation',
                               'Float32 versus BF16 require separate same-checkpoint runs'])
    if args.gradient_target_token is not None:
        report['gradient_probe'] = gradient_probe(model, inputs, graph, args.gradient_target_token)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    main()

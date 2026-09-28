"""Independent per-GPU DSR inference for legacy and two-stage geometry SFT."""
import argparse
import os
from pathlib import Path
import sys
import json
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from spatial_intelligence.dsr_sft_eval import (read, rows, write, select_rows,
    file_identity, model_identity, validate_records, summarize, prompt_protocol)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--arm', required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--merge', action='store_true')
    args = parser.parse_args()
    plan = read(args.plan)
    instruction = prompt_protocol(plan)
    arm = next(a for a in plan['arms'] if a['name'] == args.arm)
    selected = select_rows(plan['manifest'], args.smoke)
    world = len(plan['gpus'])
    out = Path(plan['root'])/'runs'/args.arm/('smoke' if args.smoke else 'full')
    out.mkdir(parents=True, exist_ok=True)
    if args.merge:
        all_records = []
        contracts = []
        for rank in range(world):
            part = rows(out/f'predictions.rank{rank}.jsonl')
            validate_records(part, selected[rank::world], complete=True)
            receipt = read(out/f'complete.rank{rank}.json')
            if receipt.get('status') != 'complete' or receipt['count'] != len(part):
                raise ValueError('Incomplete rank receipt')
            contracts.append(read(out/f'contract.rank{rank}.json'))
            all_records.extend(part)
        if any(c != contracts[0] for c in contracts):
            raise ValueError('Ranks used different inference contracts')
        validate_records(all_records, selected, complete=True)
        report = summarize(all_records)
        report['protocol'] = contracts[0]['version']
        report['prompt'] = contracts[0]['prompt']
        # A bad format gate is a blocked evaluation, never a fabricated zero score.
        accepted = not args.smoke or (report['parsed_rate'] >= .9 and report['truncation_rate'] <= .05)
        write(out/'metrics.json', report)
        write(out/'completion.json', dict(status='complete' if accepted else 'blocked_format',
              accepted=accepted, model=arm['checkpoint'], count=len(all_records), protocol=contracts[0]))
        if not accepted:
            raise ValueError('Format gate rejected; inspect raw responses before full evaluation')
        print(json.dumps(report), flush=True)
        return

    import torch
    import transformers
    from transformers import AutoProcessor
    from spatial_intelligence.geometry_rft import spatial_prompt_inputs, PROMPT_VERSION, STRUCTURED_INSTRUCTION
    from spatial_intelligence.geometry_rft_reward import score_response, RewardConfig
    from spatial_intelligence.qwen35_video_compat import install_video_rope_compat
    rank = int(os.environ.get('RANK', 0))
    local = int(os.environ.get('LOCAL_RANK', 0))
    if int(os.environ.get('WORLD_SIZE', 1)) != world:
        raise ValueError('World size changed')
    torch.set_num_threads(4)
    torch.manual_seed(3407)
    torch.cuda.set_device(local)
    device = torch.device('cuda', local)
    contract = dict(version='dsr-four-sft-v1' if instruction is None else 'dsr-mcq-tagged-v2', model=model_identity(arm['checkpoint']),
        manifest=file_identity(plan['manifest']), ids=[r['id'] for r in selected],
        media=[dict(id=r['id'], files=[file_identity(p) for p in r['media']],
                    fps=r['fps'], frame_indices=r['frame_indices'], total_num_frames=r['total_num_frames']) for r in selected],
        architecture=arm['kind'], prompt_version=PROMPT_VERSION if instruction is None else 'mcq-tagged-v2', prompt=STRUCTURED_INSTRUCTION if instruction is None else instruction,
        answer_parser='geopsro-independent-answer-v3', max_new_tokens=512,
        max_context=16384, max_side=448, do_sample=False, seed=3407,
        dtype='bfloat16', world=world, processor=plan['processor'],
        torch=torch.__version__, transformers=transformers.__version__)
    contract_path = out/f'contract.rank{rank}.json'
    if contract_path.exists() and read(contract_path) != contract:
        raise ValueError('Changed resume contract requires a new output root')
    write(contract_path, contract)
    part = selected[rank::world]
    path = out/f'predictions.rank{rank}.jsonl'
    done = validate_records(rows(path) if path.exists() else [], part)
    processor = AutoProcessor.from_pretrained(plan['processor'])
    if arm['kind'] == 'legacy':
        from spatial_intelligence.qwen3vl_geometry import load_geometry_model, add_geometry_slots
        model = load_geometry_model(arm['checkpoint']).to(device).eval()
        # Original final-patch extractor; no on-disk feature cache expansion.
        model.config.geometry_source = plan['vggt_source']
        model.config.geometry_weights = plan['vggt_weights']
        def prepare(row):
            batch = add_geometry_slots(spatial_prompt_inputs(processor, row, instruction=instruction), processor, None, None)
            batch['geometry_media'] = [row['media']]
            return batch
    else:
        from spatial_intelligence.qwen3vl_geometry_matrix import load_matrix_model, insert_slots, preprocess_geometry
        model = load_matrix_model(arm['checkpoint'], plan['vggt_source'], adapter=arm['kind'], stage='eval').to(device).eval()
        def prepare(row):
            batch = insert_slots(spatial_prompt_inputs(processor, row, instruction=instruction), processor, [len(row['media'])], arm['kind'])
            batch['geometry_images'] = [preprocess_geometry(row['media'], plan['vggt_source'])]
            return batch
    write(out/f'video-compat.rank{rank}.json', install_video_rope_compat(model))
    eos = model.generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else eos or []
    reward_config = RewardConfig(version='geopsro-independent-answer-v3')
    def move(value):
        if isinstance(value, torch.Tensor):
            return value.to(device)
        if isinstance(value, list):
            return [move(v) for v in value]
        return value
    with path.open('a', encoding='utf-8') as stream:
        for row in part:
            if row['id'] in done:
                continue
            started = time.monotonic()
            # Ground truth never enters the model-input helper.
            input_row = {k: v for k, v in row.items() if k not in ('answer', 'ground_truth', 'response')}
            batch = {k: move(v) for k, v in prepare(input_row).items()}
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
                generated = model.generate(**batch, do_sample=False, max_new_tokens=512, use_cache=True,
                    pad_token_id=processor.tokenizer.pad_token_id)
            tokens = generated[0, batch['input_ids'].shape[1]:].tolist()
            truncated = len(tokens) >= 512 and tokens[-1] not in eos
            text = processor.decode(tokens, skip_special_tokens=True)
            score = score_response(text, row['answer'], choices=row['choices'], truncated=truncated, config=reward_config)
            record = dict(id=row['id'], question_type=row.get('question_type', 'unknown'),
                gold=row['answer'], text=text, raw_text=processor.decode(tokens, skip_special_tokens=False),
                tokens=tokens, truncated=truncated, score=score, frames=len(row['media']),
                prompt_tokens=batch['input_ids'].shape[1], seconds=time.monotonic()-started,
                peak_reserved_gib=torch.cuda.max_memory_reserved(device)/2**30)
            stream.write(json.dumps(record, ensure_ascii=False)+'\n')
            stream.flush()
            print(json.dumps(dict(id=row['id'], rank=rank, seconds=record['seconds'])), flush=True)
    validate_records(rows(path), part, complete=True)
    write(out/f'complete.rank{rank}.json', dict(status='complete', count=len(part)))


if __name__ == '__main__':
    main()

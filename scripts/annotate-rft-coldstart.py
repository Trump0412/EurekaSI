"""Explicitly authorized visual annotation only; resumable per-rank JSONL outputs."""
import argparse
import json
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spatial_intelligence.rft_coldstart import read, rows, teacher_row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', required=True)
    p.add_argument('--authorize-annotation', action='store_true')
    p.add_argument('--pilot', action='store_true')
    a = p.parse_args()
    if not a.authorize_annotation:
        p.error('Annotation is disabled until explicit user authorization')
    plan = read(a.plan); root = Path(plan['root'])
    receipt = read(plan['teacher_receipt'])
    if receipt.get('status') != 'complete' or receipt.get('revision') != plan['teacher_revision']:
        raise ValueError('Teacher download not complete at the pinned revision')
    if not a.pilot:
        review = read(root / 'human-review.json')
        pilot_ids = {r['id'] for r in rows(root / 'inputs/pilot.jsonl')}
        if not review.get('approved') or not pilot_ids <= set(review.get('reviewed_ids', [])):
            raise ValueError('Review the visual evidence of all 200 pilot examples first')
    import fcntl
    rank = int(os.getenv('RANK', '0')); local = int(os.getenv('LOCAL_RANK', '0'))
    world = int(os.getenv('WORLD_SIZE', '1'))
    tag = 'pilot' if a.pilot else 'all'
    with (root / f'annotation-{tag}-{rank}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        import torch
        from transformers import AutoProcessor, AutoTokenizer, Qwen3_5ForConditionalGeneration
        from spatial_intelligence.geometry_rft import spatial_prompt_inputs, STRUCTURED_INSTRUCTION, to_device
        from spatial_intelligence.qwen35_video_compat import install_video_rope_compat
        torch.cuda.set_device(local); torch.set_num_threads(4)
        processor = AutoProcessor.from_pretrained(plan['teacher'])
        student = AutoTokenizer.from_pretrained(plan['processor'])
        model = Qwen3_5ForConditionalGeneration.from_pretrained(plan['teacher'],
            dtype=torch.bfloat16, attn_implementation='sdpa').to(local).eval()
        install_video_rope_compat(model)
        selected = rows(root / 'inputs/pilot.jsonl') if a.pilot else (
            rows(root / 'inputs/candidates.jsonl') + rows(root / 'inputs/validation.jsonl'))
        target = root / f'annotations-{tag}.rank{rank}.jsonl'
        done = {r['id'] for r in rows(target)} if target.exists() else set()
        instruction = STRUCTURED_INSTRUCTION + (
            '\nEmit the literal <think> and </think> tags with exactly the three headings, '
            'then the literal <answer>...</answer>. Each heading needs only one concise sentence. '
            'Use only evidence visible in the supplied frames. Do not invent precise 3D measurements. '
            'Multi-view images with unknown time order are not evidence of object motion. '
            'If the evidence is insufficient, explicitly say so and use <answer>unknown</answer>. '
            'Do not include text outside the tags.')
        eos = model.generation_config.eos_token_id
        eos = {eos} if isinstance(eos, int) else set(eos or [])
        with target.open('a', encoding='utf-8') as stream:
            for row in selected[rank::world]:
                if row['id'] in done: continue
                prompt = to_device(spatial_prompt_inputs(processor, teacher_row(row),
                    instruction=instruction), torch.device('cuda', local))
                if prompt['input_ids'].shape[1] + 768 > 16384:
                    raise ValueError('Context exceeds budget; do not silently truncate frames')
                with torch.inference_mode():
                    generated = model.generate(**prompt, do_sample=False, max_new_tokens=768,
                        use_cache=True, pad_token_id=processor.tokenizer.pad_token_id)
                tokens = generated[0, prompt['input_ids'].shape[1]:].tolist()
                ended = bool(tokens and tokens[-1] in eos)
                payload = tokens[:-1] if ended else tokens
                # Preserve literal reasoning tags, including tokenizer special tags.
                text = processor.decode(payload, skip_special_tokens=False).strip()
                result = dict(id=row['id'], response=text, truncated=not ended and len(tokens) >= 768,
                    teacher='Qwen/Qwen3.5-9B', teacher_revision=plan['teacher_revision'],
                    token_ids=tokens, student_token_count=len(student.encode(text, add_special_tokens=False)),
                    teacher_input=teacher_row(row), teacher_saw_gold=False,
                    prompt_instruction=instruction, enable_thinking=False,
                    visual_grid={k: v.tolist() for k,v in prompt.items() if k.endswith('grid_thw')})
                stream.write(json.dumps(result, ensure_ascii=False) + '\n'); stream.flush()
                print(json.dumps(dict(id=row['id'], rank=rank, generated_tokens=len(tokens))), flush=True)


if __name__ == '__main__': main()

"""Publishable throughput summary: whitelist metrics, never copy infrastructure paths/logs."""
import argparse
import json
from pathlib import Path
import statistics


def summarize(suite,resume=None):
    suite=Path(suite);recommendation=json.loads((suite/'recommendation.json').read_text())
    parity=json.loads((suite/'model-parity.json').read_text())
    checkpoint_name=Path(recommendation['checkpoint']).name
    checkpoint_step=int(checkpoint_name.split('-')[-1]) if checkpoint_name.startswith('checkpoint-') and checkpoint_name.split('-')[-1].isdigit() else None
    rows=[json.loads(line) for line in (suite/'samples.jsonl').read_text().splitlines()]
    eligible={(item['backend'],item['micro_batch']) for item in parity['eligible_variants']}
    table=[]
    for key,repeats in recommendation['results'].items():
        first=repeats[0]
        if first['backend'] not in ('auto','reference','fla') or first['layout'] not in ('legacy','balanced'):
            raise ValueError('Unknown public configuration')
        table.append({'backend':first['backend'],'layout':first['layout'],'micro_batch':first['micro_batch'],
            'gradient_accumulation':first['ga'],'completed_repeats':len(repeats),
            'numerically_eligible':(first['backend'],first['micro_batch']) in eligible,
            'memory_safe':all(r['peak_reserved_gib']<.92*r['gpu_total_gib'] for r in repeats),
            'samples_per_second':[r['samples_per_second'] for r in repeats],
            'median_samples_per_second':statistics.median(r['samples_per_second'] for r in repeats),
            'effective_tokens_per_second':[r['effective_tokens_per_second'] for r in repeats],
            'padding_fraction':[r['padding_fraction'] for r in repeats],
            'peak_reserved_gib':max(r['peak_reserved_gib'] for r in repeats),
            'timed_steps':first['timed_steps'],'warmup_steps':first['warmup_steps']})
    result={'status':recommendation['status'],'seed':3407,'world_size':4,'effective_batch':64,
        'source_checkpoint_step':checkpoint_step,'samples_including_stress':len(rows),'sample_ids':[r['id'] for r in rows],
        'comparison_reference':parity['comparison_reference'],'model_parity':[
            {k:v for k,v in case.items() if k in ('backend','micro_batch','layout','loss','loss_relative_error','gradient_relative_l2','finite','passed')}
            for case in parity['cases']],
        'table':table,'training_options':recommendation['training_options'],'speed_ratio':recommendation['speed_ratio'],
        'accepted':recommendation['accepted'],'claim':'Fixed training sample throughput; not downstream accuracy or convergence'}
    if resume:
        data=json.loads((Path(resume)/'result.json').read_text())
        result['resume']={k:v for k,v in data.items() if k in ('status','parameter_relative_l2','parameter_max_absolute_difference',
            'parameter_elements','scheduler_equal','steps_and_budgets_equal','budget_optimizer_steps','interrupted_at',
            'optimizer_bitwise_equal','optimizer_relative_l2','optimizer_max_absolute_difference','shared_checkpoint_branch')}
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--suite',type=Path,required=True)
    p.add_argument('--resume',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=summarize(a.suite,a.resume);a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':result['status'],'variants':len(result['table'])}))


if __name__=='__main__':main()

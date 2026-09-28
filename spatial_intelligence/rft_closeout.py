"""Pure contracts/scoring for the paired post-RFT generalization suite.

Prompts and final extraction are adapted; never label these as leaderboard runs.
"""
import math
import re
from collections import defaultdict
from .answer_extraction import extract_answer
from .followup_benchmarks import score_prediction as followup_score, summarize as followup_summary

EXISTING = {'cvbench', 'mmsi', 'viewspatial', 'mindcube_tiny'}
BENCHMARKS = ['cvbench','mmsi','viewspatial','mindcube_tiny','spbench_si',
              'spbench_mv','sparbench','mmbench_en_dev','mmmu_validation',
              'vlm4d','stibench','videomme']
PROTOCOL = 'rft-closeout-typed-mcq512-max32-even-equivalent-options-v3'

def inference_instruction(row):
    from .dsr_sft_eval import MCQ_INSTRUCTION
    # Task type only; never consult gold or choose a prompt by accuracy.
    return MCQ_INSTRUCTION if row['answer_type']=='mcq' else None

def final_text(text, truncated=False):
    if truncated or ('<think>' in text and '</think>' not in text): return None
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.S).strip()
    values = re.findall(r'<(?:answer|final)>(.*?)</(?:answer|final)>', text, re.S|re.I)
    return values[-1].strip() if values else text

def relative_accuracy(pred, gold, spar=False):
    if not (math.isfinite(pred) and math.isfinite(gold)): return 0.
    error = abs(pred-gold) if gold == 0 else abs((pred-gold)/gold)
    if spar:
        import numpy as np
        # Preserve released SPAR implementation, including its linspace count.
        thresholds = np.linspace(.5, .95, int((.95-.5)/.05+2))
    else: thresholds = [.5+i*.05 for i in range(10)]
    return sum(error <= 1-float(t) for t in thresholds)/len(thresholds)

def score(row, response, benchmark, truncated=False):
    if benchmark in EXISTING:
        value=followup_score(row,response,benchmark,truncated)
        return dict(value,parsed=value['parse_valid'])
    final=final_text(response,truncated)
    typ=row['answer_type']; pred=None; value=0.
    if typ in ('mcq','numeric'):
        parsed=extract_answer(response,choices=row.get('choices'),truncated=truncated)
        pred=None if truncated else parsed['answer']
        if pred is not None:
            if typ=='mcq': value=float(str(pred).upper() in {str(a).upper() for a in row.get('acceptable_answers',[row['answer']])})
            else:
                try:value=relative_accuracy(float(pred),float(row['answer']),benchmark=='sparbench')
                except (ValueError,TypeError):pred=None
    elif typ=='vci':
        def parse(s):
            result={}
            for part in s.split(','):
                key,val=part.strip().split(':');result[key.strip()]=float(val)
            return result
        try:
            p=parse(final);g=parse(row['answer'])
            pairs=[('move_right','move_left'),('move_up','move_down'),('move_forward','move_backward'),('rotate_right','rotate_left'),('rotate_up','rotate_down')]
            if not p or not set(p).issubset({x for pair in pairs for x in pair}):raise ValueError('Unknown VCI action')
            # Released VCI reverses arguments to its relative metric; preserve
            # that arithmetic explicitly, rather than silently "correcting" it.
            value=sum(relative_accuracy(g.get(a,0)-g.get(b,0),p.get(a,0)-p.get(b,0),True) for a,b in pairs)/5
            pred=final
        except (ValueError,AttributeError,TypeError):pass
    elif typ=='open':
        # Deliberately conservative adapted exact match, not MMMU official fuzzy
        # open-answer scoring. Keep raw predictions for official rescoring.
        norm=lambda s:re.sub(r'\s+',' ',str(s).strip().lower()).rstrip('.')
        pred=final
        answers=row['answer'] if isinstance(row['answer'],list) else [row['answer']]
        value=float(final is not None and norm(final) in {norm(x) for x in answers})
    else:raise ValueError('Unknown answer schema')
    return dict(id=row['id'],benchmark=benchmark,score=value,parsed=pred is not None,
                prediction=pred,answer=row['answer'],acceptable_answers=row.get('acceptable_answers',[row['answer']]),question_type=row['question_type'],
                truncated=truncated,source=row.get('source','unknown'))

def summary(scores, benchmark):
    if not scores or len({s['id'] for s in scores})!=len(scores):raise ValueError('Empty/duplicate scores')
    if benchmark in EXISTING:
        r=followup_summary(scores,benchmark)
        overall=None if r['accuracy'] is None else 100*r['accuracy']
    else:
        groups=defaultdict(list)
        for s in scores:groups[s['question_type']].append(s['score'])
        means={k:sum(v)/len(v) for k,v in groups.items()}
        macro=benchmark in {'spbench_si','spbench_mv','sparbench'}
        overall=100*(sum(means.values())/len(means) if macro else sum(s['score'] for s in scores)/len(scores))
        r=dict(by_category={k:dict(count=len(groups[k]),score=100*v) for k,v in means.items()},
               aggregation='task macro' if macro else 'question micro')
    return dict(r,count=len(scores),overall_score=overall,
                parse_rate=sum(s['parsed'] for s in scores)/len(scores),
                truncation_rate=sum(s['truncated'] for s in scores)/len(scores),
                protocol=PROTOCOL,scale='0..100',
                claim='Paired adapted inference; MMMU open uses conservative final exact match; MMBench includes circular rows, not official circular aggregate')

def validate_manifest(rows):
    if not rows or len({r['id'] for r in rows})!=len(rows):raise ValueError('Empty/duplicate IDs')
    for r in rows:
        if not r.get('media') or not r.get('question'):raise ValueError('Missing visual/question input')
        if r['answer_type']=='mcq' and r['answer'] not in r['choices']:raise ValueError('Gold outside choices')
        if r['answer_type']=='mcq' and 'acceptable_answers' in r:
            options=r['acceptable_answers']
            if not options or r['answer'] not in options or any(a not in r['choices'] for a in options):raise ValueError('Invalid equivalent answers')
            if len({r['choices'][a] for a in options})!=1:raise ValueError('Equivalent answers have different text')
        if r.get('input_mode')=='video':
            ix=r['frame_indices']
            if len(ix)!=len(r['media']) or ix!=sorted(set(ix)) or r['fps']<=0:raise ValueError('Invalid video metadata')
    return rows

"""Gold-blind, fail-closed complementary MC benchmark adapters.

Strict final-letter extraction is a documented adaptation, not a claim of exact
equivalence to every benchmark's regex/LLM judge. No numeric/VSI fallback.
"""
from collections import defaultdict
import re

EXPECTED={'mmsi':1000,'mindcube_tiny':1050,'viewspatial':5712,'cvbench':2638,'site':8068}
PROTOCOL='strict-final-choice-v1'


def benchmark_name(name):
    name=name.lower().replace('-','_')
    return {'mindcubetiny':'mindcube_tiny','cv_bench':'cvbench','mmsi_bench':'mmsi'}.get(name,name)


def parse_choices(value):
    if isinstance(value,dict):return {str(k).upper():str(v) for k,v in value.items()}
    if isinstance(value,list):return {chr(65+i):str(v) for i,v in enumerate(value)}
    matches=list(re.finditer(r'(?<!\w)([A-H])[.:)]\s+',str(value)))
    result={}
    for i,m in enumerate(matches):
        if m[1] in result:raise ValueError('Repeated choice label')
        result[m[1]]=str(value)[m.end():matches[i+1].start() if i+1<len(matches) else None].strip(' ,\n')
    if not result:raise ValueError('No explicit choices')
    return result


def answer_letter(value):
    m=re.match(r'^\s*\(?([A-H])(?:\)|[.:])?(?:\s|$)',str(value))
    if not m:raise ValueError('Unsupported ground-truth choice schema')
    return m[1]


def extract_choice(response, labels):
    """Uses response and permitted alphabet only; never reads the gold answer."""
    text=str(response or '').strip()
    if '<think>' in text and '</think>' not in text:return None
    text=re.sub(r'<think>.*?</think>','',text,flags=re.S).strip()
    finals=re.findall(r'<(final|answer)>(.*?)</\1>',text,flags=re.S|re.I)
    if finals:text=finals[-1][1].strip()
    text=text.strip('`*_ \n')
    text=re.sub(r'^(?:the\s+)?(?:final\s+)?answer\s*(?:is|:)\s*','',text,flags=re.I)
    text=re.sub(r'^option\s+','',text,flags=re.I)
    m=re.fullmatch(r'[({\[]?([A-Ha-h])[)}\]]?\s*[.!]?',text)
    return m[1].upper() if m and m[1].upper() in labels else None


def validate_rows(rows, benchmark):
    benchmark=benchmark_name(benchmark)
    if benchmark in {'revsi','vsibench'}:
        ids=set()
        for row in rows:
            if row['id'] in ids:raise ValueError('Duplicate evaluation ID')
            ids.add(row['id'])
            if not row.get('media') or not row.get('question') or not row.get('question_type') or ('answer' not in row and 'ground_truth' not in row):
                raise ValueError('Missing original spatial benchmark field')
        return {'rows':len(ids),'protocol':'existing spatial_eval passthrough'}
    if benchmark not in EXPECTED:raise ValueError('Unverified benchmark protocol: '+benchmark)
    ids=set()
    for row in rows:
        if row['id'] in ids:raise ValueError('Duplicate evaluation ID')
        ids.add(row['id'])
        if not row.get('question') or not row.get('media'):raise ValueError('Missing question/media')
        choices=parse_choices(row['choices'])
        if benchmark=='site' and row.get('input_adapter_verified') is not True:
            raise ValueError('SITE requires verified image-option interleave/video input adapter')
        if len(choices)<(1 if benchmark=='site' else 2) or set(choices)!={chr(65+i) for i in range(len(choices))}:raise ValueError('Nonconsecutive choices')
        if answer_letter(row['answer']) not in choices:raise ValueError('Gold outside choices')
        if 'ground_truth' in row and answer_letter(row['ground_truth'])!=answer_letter(row['answer']):raise ValueError('Conflicting ground_truth/answer')
        if benchmark=='cvbench' and row.get('source') not in {'ADE20K','COCO','Omni3D'}:raise ValueError('Unknown CV-Bench source stratum')
    return {'rows':len(ids),'expected_rows':EXPECTED[benchmark],'complete_benchmark':len(ids)==EXPECTED[benchmark]}


def score_prediction(row,response,benchmark,truncated=False):
    benchmark=benchmark_name(benchmark)
    if benchmark in {'revsi','vsibench'}:
        from .spatial_eval import score_prediction as original
        return original(row,response,benchmark,truncated)
    if benchmark not in EXPECTED:raise ValueError('Unverified benchmark protocol: '+benchmark)
    prediction=extract_choice(response,parse_choices(row['choices']))
    gold=answer_letter(row['answer'])
    if 'ground_truth' in row and answer_letter(row['ground_truth'])!=gold:raise ValueError('Conflicting ground_truth/answer')
    correct=prediction==gold and not bool(truncated)
    return dict(id=row['id'],benchmark=benchmark,score=float(correct),correct=correct,
                prediction=prediction,answer=gold,parse_valid=prediction is not None,truncated=bool(truncated),
                question_type=row.get('question_type','unknown'),source=row.get('source','unknown'),
                num_choices=len(parse_choices(row['choices'])),protocol=PROTOCOL)


def summarize(scores,benchmark):
    benchmark=benchmark_name(benchmark)
    if benchmark in {'revsi','vsibench'}:
        from .spatial_eval import summarize as original
        return original(scores,benchmark)
    if benchmark not in EXPECTED:raise ValueError('Unverified benchmark protocol: '+benchmark)
    scores=list(scores)
    if len({s['id'] for s in scores})!=len(scores):raise ValueError('Duplicate scored IDs')
    if any(s['benchmark']!=benchmark for s in scores):raise ValueError('Mixed benchmark scores')
    n=len(scores);micro=sum(s['score'] for s in scores)/n if n else None
    groups=defaultdict(list)
    for s in scores:groups[s['question_type']].append(s['score'])
    result=dict(benchmark=benchmark,total=n,expected_total=EXPECTED[benchmark],complete_benchmark=n==EXPECTED[benchmark],
                accuracy=micro,micro_accuracy=micro,correct=sum(s['correct'] for s in scores),
                unparseable=sum(not s['parse_valid'] for s in scores),truncated=sum(s['truncated'] for s in scores),
                protocol=PROTOCOL,denominator_policy='all scored rows; invalid and truncated outputs count as wrong',
                by_category={k:dict(total=len(v),accuracy=sum(v)/len(v)) for k,v in groups.items()})
    if benchmark=='site':
        denominator=sum(1-1/s['num_choices'] for s in scores)
        numerator=sum(s['score']-1/s['num_choices'] for s in scores)
        result.update(accuracy=numerator/denominator if denominator else None,
                      metric_name='chance_adjusted_accuracy',chance_adjusted_numerator=numerator,
                      chance_adjusted_denominator=denominator,
                      aggregation='official SITE: sum(correct-1/K) / sum(1-1/K); not mean of individually normalized scores')
        for category in groups:
            selected=[s for s in scores if s['question_type']==category]
            den=sum(1-1/s['num_choices'] for s in selected)
            result['by_category'][category]['chance_adjusted_accuracy']=sum(s['score']-1/s['num_choices'] for s in selected)/den if den else None
    elif benchmark=='cvbench':
        sources=defaultdict(list)
        for s in scores:sources[s['source']].append(s['score'])
        means={k:sum(v)/len(v) for k,v in sources.items()}
        result['source_accuracy']=means
        result['accuracy']=((means['ADE20K']+means['COCO'])/2+means['Omni3D'])/2 if set(means)=={'ADE20K','COCO','Omni3D'} else None
        result['aggregation']='official source-weighted: 0.25 ADE20K + 0.25 COCO + 0.5 Omni3D'
    else:result['aggregation']='question micro accuracy; ViewSpatial category macro additionally reported, not asserted official overall'
    result['category_macro_accuracy']=sum(v['accuracy'] for v in result['by_category'].values())/len(groups) if groups else None
    return result

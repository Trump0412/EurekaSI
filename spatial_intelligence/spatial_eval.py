"""Native-video evaluation with auditable final extraction and unchanged metric arithmetic."""
from functools import lru_cache
import json
from pathlib import Path
import numpy as np
from PIL import Image
from .answer_extraction import extract_answer,VERSION
from .revsi_scoring import score_row,aggregate


def prompt_text(row,answer_format='official'):
    text=row['question']
    if row.get('choices'):text+='\nOptions:\n'+'\n'.join(f'{k}. {v}' for k,v in row['choices'].items())
    text+='\n'+row['instruction']
    if answer_format=='tagged':
        kind='one option letter' if row.get('choices') else 'one number in the units requested by the question'
        text+=f'\nReturn only <answer>{kind}</answer>, replacing the placeholder with your final answer. Do not include reasoning.'
    elif answer_format!='official':raise ValueError(answer_format)
    return text


@lru_cache(maxsize=512)
def video_metadata(source):
    import cv2
    cap=cv2.VideoCapture(source)
    try:
        if not cap.isOpened():raise ValueError(f'Cannot open {source}')
        fps=cap.get(cv2.CAP_PROP_FPS);total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps<=0 or total<=0:raise ValueError('Unknown video timestamps; refuse invented FPS')
        return fps,total
    finally:cap.release()


def native_video_inputs(processor,row,answer_format='official',max_side=448):
    frames=[]
    for path in row['media']:
        with Image.open(path) as im:
            im=im.convert('RGB');im.thumbnail((max_side,max_side),Image.Resampling.LANCZOS)
            frames.append(np.array(im))
    fps,total=video_metadata(row['source'])
    indices=row['frame_indices']
    if len(indices)!=len(frames) or indices!=sorted(indices):raise ValueError('Frame order mismatch')
    if len(set(indices))!=len(indices) or max(indices)>=total:raise ValueError('Invalid frame identities')
    messages=[{'role':'user','content':[{'type':'video'}, {'type':'text','text':prompt_text(row,answer_format)}]}]
    text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
    result=processor(text=[text],videos=[np.stack(frames)],
        video_metadata=[{'fps':fps,'total_num_frames':total,'frames_indices':indices}],
        do_sample_frames=False,return_tensors='pt',padding=True)
    if 'video_grid_thw' not in result or 'pixel_values_videos' not in result:raise ValueError('Native video not consumed')
    if int(result['video_grid_thw'][0,0])*processor.video_processor.temporal_patch_size!=len(frames):
        raise ValueError('Processor resampled/changed the prescribed frames')
    if result['input_ids'].shape[1]>16384:raise ValueError('Context exceeds locked budget')
    return dict(result)


def metric_row(row,response,benchmark):
    if benchmark=='revsi':return score_row(row,response)
    if benchmark!='vsibench':raise ValueError(benchmark)
    mapped=dict(row)
    if mapped['question_type']=='object_counting':mapped['question_type']='object_counting_single'
    if mapped['question_type']=='room_size_estimation':mapped['question_type']='room_size_estimation_single'
    return score_row(mapped,response)


def metric_aggregate(rows,benchmark):
    if benchmark=='revsi':return aggregate(rows)
    grouped={}
    for r in rows:grouped.setdefault(r['question_type'],[]).append(r['acc'])
    scores={k:float(np.mean(v)) for k,v in grouped.items()}
    names=['object_rel_direction_easy','object_rel_direction_medium','object_rel_direction_hard']
    values=[scores.pop(k) for k in names if k in scores]
    if values:scores['object_rel_direction']=float(np.mean(values))
    return {'overall_acc':float(np.mean(list(scores.values()))) if scores else 0.,'subscores':scores,'count':len(rows)}


def score_prediction(row,response,benchmark,truncated=False):
    parsed=extract_answer(response,choices=row.get('choices'),truncated=truncated)
    return {'id':row['id'],'question_type':row['question_type'],'response':response,'extraction':parsed,
        'strict_acc':metric_row(row,response,benchmark),
        'extracted_acc':metric_row(row,parsed['answer'],benchmark) if parsed['answer'] is not None else 0.,
        'pruned':row.get('pruned'),'training_scene_overlap':row.get('training_scene_overlap',False)}


def summarize(predictions,benchmark):
    report={'count':len(predictions),'extractor':VERSION,'score_scale':'0..100',
        'parse_rate':sum(x['extraction']['status']=='ok' for x in predictions)/len(predictions),
        'truncation_rate':sum(x['extraction']['truncated'] for x in predictions)/len(predictions)}
    for mode in ('strict','extracted'):
        scored=[{'question_type':x['question_type'],'acc':x[mode+'_acc']} for x in predictions]
        result=metric_aggregate(scored,benchmark)
        report[mode]={'overall_score':100*result['overall_acc'],
                      'subscores':{k:100*v for k,v in result['subscores'].items()}}
    report['claim']='Adapted native-video protocol; strict means upstream arithmetic on these raw outputs, not leaderboard-equivalent inference.'
    return report

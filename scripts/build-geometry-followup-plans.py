"""Generate private three-role GeoRoute -> GeoFits queues, without launching.

Commands are real workers, but immutable data/runtime receipts remain mandatory.
Unsupported variants and secondary evaluators are explicitly pending, not aliases.
"""
import argparse,copy,json
from pathlib import Path,PurePosixPath


def requirement(path,**equals):
    return dict(path=str(path),equals=equals or dict(status='complete',accepted=True))


def build(bindings,scientific):
    route_roles={'primary':['full','no_tip'],'control_a':['one_stb'],'control_b':['final_only','post_merger']}
    routes_done=[requirement(PurePosixPath(node['root'])/'state/route-paper-complete.json') for node in bindings['nodes'].values()]
    result={}
    for role,node in bindings['nodes'].items():
        root=PurePosixPath(node['root']);world=len(node['gpus'])
        if 64%world:raise ValueError('Global batch64 must divide assigned world')
        stages=[];inputs={}
        data_gate=requirement(bindings['data_receipt'],status='ready',media_verified=True,
            selection_name=scientific['data_selection'])
        if scientific.get('openspatial',{}).get('allow_unverified_scene_overlap') is True:
            data_gate['equals']['known_sources_leakage_checked']=True
        else:
            data_gate['equals']['leakage_checked']=True
        train_prefix=['{python}','-m','torch.distributed.run','--standalone',f'--nproc_per_node={world}']
        for variant in route_roles[role]:
            architecture=copy.deepcopy(scientific['georoute']['architecture'])
            if variant=='one_stb':architecture['blocks_per_exit']=1
            if variant=='final_only':architecture['active_exits']=[3]
            if variant=='post_merger':architecture['placement']='post'
            variant_root=root/'runs'/variant
            runtime=root/'readiness'/f'georoute-{variant}.json'
            worker=dict(bindings['common'],root=str(variant_root),variant=variant,use_lora=False,
                data_receipt=bindings['data_receipt'],runtime_acceptance=str(runtime),
                architecture=architecture,graph=scientific['georoute']['graph'],
                graph_cache=str(PurePosixPath(bindings['shared_cache'])/'georoute'),
                deepspeed=str(root/'code/configs/geometry-zero3.json'),
                tip_checkpoint=str(variant_root/'formal/tip/micro1/final'),
                tip_receipt=str(variant_root/'formal/tip/micro1/completion.json'))
            inputs[variant+'.json']=worker
            worker_path=str(root/'inputs'/f'{variant}.json')
            prior=[]
            for phase in (['sft'] if variant in ('matched_rgb_sft','no_tip') else ['tip','sft']):
                name=f'route-{variant}-{phase}'
                args=['{code}/scripts/train-georoute-stage.py','--plan',worker_path,'--stage',phase,'--micro','1']
                stages.append(dict(name=name,study='georoute',use_lora=False,after=prior,
                    requirements=[data_gate,requirement(runtime,status='ready',full_model_verified=True)],
                    commands=[train_prefix+args,['{python}']+args+['--verify-saved']],
                    receipt=str(variant_root/f'formal/{phase}/micro1/completion.json')))
                prior=[name]
            for benchmark,manifest in bindings['eval_manifests'].items():
                if benchmark not in ('revsi','vsibench'):continue
                name=f'route-{variant}-{benchmark}';output=variant_root/'eval'/benchmark
                args=['{code}/scripts/evaluate-georoute.py','--plan',worker_path,'--checkpoint',str(variant_root/'formal/sft/micro1/final'),
                    '--manifest',manifest,'--output',str(output),'--benchmark',benchmark,'--world',str(world)]
                if variant=='matched_rgb_sft':args+=['--rgb-only']
                stages.append(dict(name=name,study='georoute',use_lora=False,after=prior,
                    requirements=[data_gate,requirement(root/'readiness'/f'eval-{benchmark}.json',status='ready',full_model_verified=True)],
                    commands=[train_prefix+args,['{python}']+args+['--merge']],receipt=str(output/'completion.json')))
        # Paper completion includes secondary benchmark evidence. No fabricated
        # all-paper success when only the currently implemented two scorers run.
        all_route=[s['name'] for s in stages]
        paper_receipts=[s['receipt'] for s in stages if s['name'].endswith(('-revsi','-vsibench'))]
        secondary=[requirement(root/'results'/f'{v}-{b}.json') for v in route_roles[role]
            for b in scientific['evaluation']['benchmarks'] if b not in ('revsi','vsibench')]
        stages.append(dict(name='route-paper-complete',study='georoute',use_lora=False,after=all_route,
            requirements=[data_gate]+[requirement(x) for x in paper_receipts]+secondary,
            commands=[['{python}','{code}/scripts/accept-geometry-phase.py','--plan',str(root/'plan.json'),'--stage','route-paper-complete']],
            receipt=str(root/'state/route-paper-complete.json')))
        if role=='primary':
            fits_root=root/'runs/geofits-full';runtime=root/'readiness/geofits-full.json'
            inputs['geofits-full.json']=dict(bindings['common'],root=str(fits_root),variant='full',use_lora=False,
                architecture=bindings['geofits_architecture'],data_receipt=bindings['data_receipt'],runtime_acceptance=str(runtime),
                deepspeed=str(root/'code/configs/geometry-zero3.json'))
            args=['{code}/scripts/train-geofits-stage.py','--plan',str(root/'inputs/geofits-full.json'),'--micro','1']
            stages.append(dict(name='geofits-full-sft',study='geofits',use_lora=False,after=['route-paper-complete'],
                requirements=[data_gate,*routes_done,requirement(runtime,status='ready',full_model_verified=True)],
                commands=[train_prefix+args,['{python}']+args+['--verify-saved']],
                receipt=str(fits_root/'formal/micro1/completion.json')))
        result[role]=dict(root=str(root),python=node['python'],gpus=node['gpus'],dependencies=bindings['dependencies'],
            stages=stages,stage_plans=inputs,implementation_status='real_georoute_commands_pending_data_runtime_and_secondary_evaluators',
            authorized_by_user=True,selection=scientific['data_selection'],
            pending_work=['GeoFits non-full variants and full benchmark orchestration','actual five-source media and source-scene audit','real GPU full-model acceptance'])
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bindings',required=True);p.add_argument('--scientific',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();plans=build(json.loads(Path(a.bindings).read_text()),json.loads(Path(a.scientific).read_text()))
    output=Path(a.output);output.mkdir(parents=True,exist_ok=True)
    for role,plan in plans.items():
        target=output/(role+'.json')
        if target.exists() and json.loads(target.read_text())!=plan:raise ValueError('Different generated plan exists; use new output')
        target.write_text(json.dumps(plan,indent=2),encoding='utf-8')
    print(json.dumps(dict(status='plans_generated_not_started',roles=list(plans))))


if __name__=='__main__':main()

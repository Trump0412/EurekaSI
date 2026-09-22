"""Generate private three-role GeoRoute -> GeoFits queues, without launching.

Commands are real workers, but immutable data/runtime receipts remain mandatory.
Unsupported variants and secondary evaluators are explicitly pending, not aliases.
"""
import argparse,copy,json
from pathlib import Path,PurePosixPath
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from spatial_intelligence.geofits_recipe import architecture_for_variant


def requirement(path,**equals):
    return dict(path=str(path),equals=equals or dict(status='complete',accepted=True))


def build(bindings,scientific):
    route_roles={'primary':['full','no_tip'],'control_a':['one_stb'],'control_b':['final_only','post_merger']}
    fits_roles={'primary':['full','no_gate'],'control_a':['3d_only','4d_only'],'control_b':['dense','single_layer']}
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
            gate_name=f'route-{variant}-runtime'
            stages.append(dict(name=gate_name,study='georoute',use_lora=False,after=[],requirements=[data_gate],
                commands=[['{python}','{code}/scripts/verify-georoute-runtime.py','--plan',worker_path,
                    '--output',str(variant_root/'runtime-gate'),'--receipt',str(runtime),
                    '--gpus',','.join(map(str,node['gpus'])),'--dependency-receipt',bindings['dependencies'][0]['path'],
                    '--authorize-run']],receipt=str(runtime),receipt_equals=dict(status='ready',full_model_verified=True)))
            prior=[gate_name]
            for phase in (['sft'] if variant in ('matched_rgb_sft','no_tip') else ['tip','sft']):
                name=f'route-{variant}-{phase}'
                args=['{code}/scripts/train-georoute-stage.py','--plan',worker_path,'--stage',phase,'--micro','1']
                stages.append(dict(name=name,study='georoute',use_lora=False,after=prior,
                    requirements=[data_gate,requirement(runtime,status='ready',full_model_verified=True)],
                    commands=[train_prefix+args,['{python}']+args+['--verify-saved']],
                    receipt=str(variant_root/f'formal/{phase}/micro1/completion.json')))
                prior=[name]
            for benchmark,manifest in bindings['eval_manifests'].items():
                if benchmark not in ('revsi','vsibench','mmsi','mindcube_tiny','viewspatial','cvbench'):continue
                name=f'route-{variant}-{benchmark}';output=variant_root/'eval'/benchmark
                args=['{code}/scripts/evaluate-georoute.py','--plan',worker_path,'--checkpoint',str(variant_root/'formal/sft/micro1/final'),
                    '--manifest',manifest,'--output',str(output),'--benchmark',benchmark,'--world',str(world)]
                if variant=='matched_rgb_sft':args+=['--rgb-only']
                stages.append(dict(name=name,study='georoute',use_lora=False,after=prior,
                    requirements=[data_gate,requirement(runtime,status='ready',full_model_verified=True)],
                    commands=[train_prefix+args,['{python}']+args+['--merge']],receipt=str(output/'completion.json')))
        # Paper completion includes secondary benchmark evidence. No fabricated
        # all-paper success when only the currently implemented two scorers run.
        all_route=[s['name'] for s in stages]
        paper_receipts=[s['receipt'] for s in stages if s['name'].endswith(('-revsi','-vsibench'))]
        secondary=[requirement(root/'results'/f'{v}-{b}.json') for v in route_roles[role]
            for b in scientific['evaluation']['benchmarks'] if b not in bindings['eval_manifests'] or b=='site']
        paper_receipts=[s['receipt'] for s in stages if any(s['name'].endswith('-'+b) for b in bindings['eval_manifests'])]
        stages.append(dict(name='route-paper-complete',study='georoute',use_lora=False,after=all_route,
            requirements=[data_gate]+[requirement(x) for x in paper_receipts]+secondary+
                [requirement(root/'results/georoute-correspondence.json',status='complete',accepted=True,independent_gt=True)],
            commands=[['{python}','{code}/scripts/accept-geometry-phase.py','--plan',str(root/'plan.json'),'--stage','route-paper-complete']],
            receipt=str(root/'state/route-paper-complete.json')))
        fits_stages=[]
        for variant in fits_roles[role]:
            fits_root=root/'runs'/('geofits-'+variant);runtime=root/'readiness'/('geofits-'+variant+'.json')
            worker_path=str(root/'inputs'/('geofits-'+variant+'.json'))
            inputs['geofits-'+variant+'.json']=dict(bindings['common'],root=str(fits_root),variant=variant,use_lora=False,
                architecture=architecture_for_variant(bindings['geofits_architecture'],variant),data_receipt=bindings['data_receipt'],runtime_acceptance=str(runtime),
                deepspeed=str(root/'code/configs/geometry-zero3.json'))
            gate_name=f'geofits-{variant}-runtime'
            stages.append(dict(name=gate_name,study='geofits',use_lora=False,
                after=['route-paper-complete'],requirements=[data_gate,*routes_done],
                commands=[['{python}','{code}/scripts/verify-geofits-runtime.py','--plan',worker_path,
                    '--output',str(fits_root/'runtime-gate'),'--receipt',str(runtime),
                    '--gpus',','.join(map(str,node['gpus'])),'--dependency-receipt',str(root/'state/route-paper-complete.json'),
                    '--authorize-run']],receipt=str(runtime),receipt_equals=dict(status='ready',full_model_verified=True)))
            args=['{code}/scripts/train-geofits-stage.py','--plan',worker_path,'--micro','1']
            train_name=f'geofits-{variant}-sft'
            stages.append(dict(name=train_name,study='geofits',use_lora=False,after=[gate_name],
                requirements=[data_gate,*routes_done,requirement(runtime,status='ready',full_model_verified=True)],
                commands=[train_prefix+args,['{python}']+args+['--verify-saved']],
                receipt=str(fits_root/'formal/micro1/completion.json')))
            fits_stages.append(train_name)
            for benchmark,manifest in bindings['eval_manifests'].items():
                if benchmark not in scientific['geofits'].get('benchmarks',scientific['evaluation']['benchmarks']):continue
                if benchmark not in ('revsi','vsibench','mmsi','mindcube_tiny','viewspatial','cvbench'):continue
                output=fits_root/'eval'/benchmark
                args=['{code}/scripts/evaluate-geofits.py','--plan',worker_path,'--checkpoint',str(fits_root/'formal/micro1/final'),
                    '--manifest',manifest,'--output',str(output),'--benchmark',benchmark,'--world',str(world)]
                name=f'geofits-{variant}-{benchmark}'
                stages.append(dict(name=name,study='geofits',use_lora=False,after=[train_name],requirements=[data_gate],
                    commands=[train_prefix+args,['{python}']+args+['--merge']],receipt=str(output/'completion.json')))
                fits_stages.append(name)
        fits_secondary=[requirement(root/'results'/f'geofits-{v}-{b}.json') for v in fits_roles[role]
            for b in scientific['geofits'].get('benchmarks',scientific['evaluation']['benchmarks']) if b not in bindings['eval_manifests'] or b=='site']
        fits_receipts=[requirement(s['receipt']) for s in stages if s['name'] in fits_stages]
        stages.append(dict(name='geofits-paper-complete',study='geofits',use_lora=False,after=fits_stages,
            requirements=[data_gate,*fits_receipts,*fits_secondary],
            commands=[['{python}','{code}/scripts/accept-geometry-phase.py','--plan',str(root/'plan.json'),'--stage','geofits-paper-complete']],
            receipt=str(root/'state/geofits-paper-complete.json')))
        result[role]=dict(root=str(root),python=node['python'],gpus=node['gpus'],dependencies=bindings['dependencies'],
            stages=stages,stage_plans=inputs,implementation_status='real_georoute_commands_pending_data_runtime_and_secondary_evaluators',
            authorized_by_user=True,selection=scientific['data_selection'],
            pending_work=['Secondary benchmark adapters/manifests and official-scoring acceptance','real GPU full-model acceptance for every variant'])
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

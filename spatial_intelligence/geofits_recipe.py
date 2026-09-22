"""Torch-free, explicit GeoFits ablation recipes for queue builders.

Single-layer means post-decoder block 3, as stated in the manuscript ablation.
Dense means softmax over the same region's bank,
not spatially global attention. 4D-only retains the Pi3 temporal adapters.
"""
VARIANTS = ('full', '3d_only', '4d_only', 'dense', 'no_gate', 'single_layer')


def architecture_for_variant(base, variant='full'):
    if variant not in VARIANTS:
        raise ValueError('Unknown GeoFits variant: '+str(variant))
    result=dict(base)
    result.update(bank_sources=['vggt','pi3'],retrieval_mode='topk',gate_enabled=True,
                  fusion_layers=[1,2,3],top_k=2)
    if variant=='3d_only': result['bank_sources']=['vggt']
    if variant=='4d_only': result['bank_sources']=['pi3']
    if variant=='dense': result['retrieval_mode']='dense'
    if variant=='no_gate': result['gate_enabled']=False
    if variant=='single_layer': result['fusion_layers']=[3]
    return result


def validate_variant_architecture(architecture, variant='full'):
    """Legacy full fields omitted in saved plans imply their original defaults."""
    expected=architecture_for_variant(architecture,variant)
    defaults=dict(bank_sources=['vggt','pi3'],retrieval_mode='topk',gate_enabled=True,
                  fusion_layers=[1,2,3],top_k=2)
    for key,default in defaults.items():
        actual=architecture.get(key,default)
        if isinstance(default,list): actual=list(actual)
        if actual!=expected[key]:
            raise ValueError('GeoFits variant/architecture mismatch: '+key)
    return expected

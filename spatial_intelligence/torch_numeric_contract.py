"""Record and restore arithmetic policy across fresh-process checkpoint checks."""


def capture_numeric_contract():
    import torch
    return {
        'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
        'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
        'bf16_reduced_precision_reduction': torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        'fp16_reduced_precision_reduction': torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
    }


def restore_numeric_contract(contract):
    import torch
    expected = set(capture_numeric_contract())
    if set(contract) != expected or any(type(v) is not bool for v in contract.values()):
        raise ValueError('Incomplete or invalid arithmetic contract')
    torch.backends.cuda.matmul.allow_tf32 = contract['matmul_allow_tf32']
    torch.backends.cudnn.allow_tf32 = contract['cudnn_allow_tf32']
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = contract['bf16_reduced_precision_reduction']
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = contract['fp16_reduced_precision_reduction']
    if capture_numeric_contract() != contract:
        raise ValueError('Arithmetic contract not restored')

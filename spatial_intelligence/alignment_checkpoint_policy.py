"""Opt-in activation-memory tradeoff for frozen-language alignment only.

Call immediately before a model forward. The policy changes decoder checkpoint
flags, never autograd or ``requires_grad``. Thresholds need a real longest-sample
memory test; fake-layer tests do not establish GPU fit or a speed improvement.
"""


def apply_alignment_checkpoint_policy(model, padded_sequence_tokens, threshold=0):
    """Return checkpointing enabled, or ``None`` when the policy is disabled.

    ``padded_sequence_tokens`` is the padded sequence width (including inserted
    geometry slots), NOT non-padding tokens or the batch-wide sum. The selected
    threshold is valid only for the microbatch/world-size used in its profiling.
    Zero threshold is the default and leaves all model state unchanged.

    Flags remain selected until the next call, so backward recomputation sees
    the same setting as forward. Call for every training forward: a long sample
    re-enables checkpointing even after a short sample disabled it. Do not change
    flags with an outstanding forward awaiting backward, or between pipeline
    microbatches. This helper is for sequential Trainer forward/backward only.
    """
    if isinstance(threshold, bool) or not isinstance(threshold, int) or threshold < 0:
        raise ValueError('Checkpoint threshold must be a nonnegative integer')
    if threshold == 0:
        return None
    if (isinstance(padded_sequence_tokens, bool)
            or not isinstance(padded_sequence_tokens, int)
            or padded_sequence_tokens < 1):
        raise ValueError('Padded sequence width must be a positive integer')
    if getattr(model, '_matrix_stage', None) != 'align':
        raise ValueError('Adaptive checkpointing is restricted to alignment')
    if not getattr(model, 'training', False):
        raise ValueError('Adaptive checkpointing requires training mode')
    native = getattr(model, 'model', None)
    language = getattr(native, 'language_model', None)
    layers = getattr(language, 'layers', None)
    if language is None or layers is None or len(layers) == 0:
        raise ValueError('Expected a nonempty native language decoder layer stack')
    if not callable(getattr(language, 'parameters', None)):
        raise ValueError('Language decoder must expose parameters for ownership validation')
    parameters = list(language.parameters())
    if not parameters or any(parameter.requires_grad for parameter in parameters):
        raise ValueError('All native language weights must be frozen during alignment')
    # Validate every layer before mutating any: an incompatible library version
    # must fail closed rather than partially disable recomputation.
    for layer in layers:
        if not isinstance(getattr(layer, 'gradient_checkpointing', None), bool):
            raise ValueError('Expected boolean per-layer gradient_checkpointing flags')
        if not getattr(layer, 'training', False):
            raise ValueError('Frozen language layers must remain in training mode')
    enabled = padded_sequence_tokens > threshold
    for layer in layers:
        layer.gradient_checkpointing = enabled
    return enabled

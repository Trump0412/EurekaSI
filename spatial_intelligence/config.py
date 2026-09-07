"""Fail on unknown config keys, rather than silently ignoring an experiment knob."""
import math


SCHEMA = {
    "model": {"backend", "path", "revision", "device", "dtype", "text_only", "attention", "trust_remote_code", "adapter", "lora", "max_context", "options"},
    "teacher": {"backend", "path", "revision", "device", "dtype", "text_only", "attention", "trust_remote_code", "adapter", "lora", "max_context", "options"},
    "protocol": {"name", "seed", "max_frames", "image_max_side", "instruction", "generation", "geometry_condition"},
    "data": {"train", "eval", "heldout"},
    "train": {"mode", "steps", "gradient_accumulation", "learning_rate", "weight_decay", "max_grad_norm", "seed", "save_every", "group_size", "algorithm", "clip", "beta", "reward_plugin", "numeric_atol", "numeric_rtol", "kl_direction", "distill_temperature", "sft_weight", "privileged_answer", "teacher_refresh_steps", "batch_size"},
    "score": {"allow_missing", "numeric_atol", "numeric_rtol", "official_scorer"},
}


def validate(cfg):
    if not isinstance(cfg, dict):
        raise ValueError("Config must be a mapping")
    allowed = set(SCHEMA) | {"output"}
    if set(cfg) != allowed:
        raise ValueError(f"Config keys differ: missing={allowed-set(cfg)}, unknown={set(cfg)-allowed}")
    for section, keys in SCHEMA.items():
        if not isinstance(cfg[section], dict):
            raise ValueError(f"{section} must be a mapping")
        if set(cfg[section])-keys:
            raise ValueError(f"Unknown keys in {section}: {set(cfg[section])-keys}")
    for section in ["protocol", "data", "train", "score"]:
        required=SCHEMA[section]-({"batch_size"} if section=="train" else set())
        if not required.issubset(cfg[section]):
            raise ValueError(f"Missing keys in {section}: {SCHEMA[section]-set(cfg[section])}")
    for section in ["model", "teacher"]:
        model = cfg[section]
        for required in ["backend", "path", "revision", "device", "dtype", "text_only", "lora"]:
            if required not in model:
                raise ValueError(f"Missing {section}.{required}")
        if not isinstance(model['lora'], dict) or set(model["lora"]) != {"enabled", "rank", "alpha", "target_modules"}:
            raise ValueError("lora requires enabled/rank/alpha/target_modules")
        if model.get("backend") == "hf" and model.get("options"):
            raise ValueError("HF backend does not implement options; use a custom adapter")
    proto = cfg["protocol"]
    generation = proto["generation"]
    if not isinstance(generation, dict):
        raise ValueError('protocol.generation must be a mapping')
    if set(generation) - {"max_new_tokens", "do_sample", "temperature", "top_p", "top_k", "num_beams", "repetition_penalty"}:
        raise ValueError("Unsupported generation setting")
    if cfg["model"]["backend"] == "hf" and proto["geometry_condition"] not in {"none", "normal"}:
        raise ValueError("Vanilla HF cannot apply geometry interventions; use a geometry adapter")
    if not isinstance(cfg['data']['train'], list) or not isinstance(cfg['data']['heldout'], list):
        raise ValueError('data.train and data.heldout must be lists')
    for entry in cfg["data"]["train"]:
        if not isinstance(entry, dict) or set(entry) != {"path", "weight"}:
            raise ValueError("Each mixture entry needs path and weight only")
    t = cfg["train"]
    if t["kl_direction"] not in {"forward", "reverse"}:
        raise ValueError("Invalid distillation KL")
    for name in ("steps", "gradient_accumulation", "batch_size", "group_size"):
        value = t.get(name, 1)
        if type(value) is not int or value < 1:
            raise ValueError(f"train.{name} must be a positive integer")
    for name in ("save_every", "teacher_refresh_steps", "seed"):
        if type(t[name]) is not int or t[name] < 0:
            raise ValueError(f"train.{name} must be a nonnegative integer")
    for name in ("learning_rate", "max_grad_norm", "clip", "distill_temperature"):
        _number(t[name], f"train.{name}", positive=True)
    for name in ("weight_decay", "beta", "sft_weight", "numeric_atol", "numeric_rtol"):
        _number(t[name], f"train.{name}")
    for name in ("numeric_atol", "numeric_rtol"):
        _number(cfg["score"][name], f"score.{name}")
    for entry in cfg["data"]["train"]:
        _number(entry["weight"], "data.train.weight", positive=True)
    if t["mode"] not in {"sft", "rl", "opd", "opsd", "offline_kd"}:
        raise ValueError("Unknown training mode")
    if t["algorithm"] not in {"grpo", "gspo"}:
        raise ValueError("train.algorithm must be grpo/gspo")
    if t["mode"] == "rl" and t["group_size"] < 2:
        raise ValueError("RL requires group_size >= 2")
    if t["mode"] in {"rl", "opd", "opsd"}:
        if (generation.get("do_sample") is not True
                or generation.get("top_p", 1.0) != 1.0 or generation.get("top_k", 0) != 0
                or generation.get("repetition_penalty", 1.0) != 1.0 or generation.get("num_beams", 1) != 1):
            raise ValueError("On-policy training needs unmodified sampling: do_sample=true, top_p=1, top_k=0, num_beams=1, repetition_penalty=1")
    if "temperature" in generation:
        _number(generation["temperature"], "protocol.generation.temperature", positive=True)
    for name in ("max_frames", "image_max_side"):
        if type(proto[name]) is not int or proto[name] < 1:
            raise ValueError(f"protocol.{name} must be a positive integer")
    if type(generation.get("max_new_tokens")) is not int or generation["max_new_tokens"] < 1:
        raise ValueError("max_new_tokens must be a positive integer")
    if type(proto['seed']) is not int or proto['seed'] < 0:
        raise ValueError('protocol.seed must be a nonnegative integer')
    if 'do_sample' in generation and type(generation['do_sample']) is not bool:
        raise ValueError('do_sample must be true/false')
    if 'top_p' in generation:
        _number(generation['top_p'], 'protocol.generation.top_p', positive=True)
        if generation['top_p'] > 1:
            raise ValueError('top_p must be <= 1')
    for name, minimum in [('top_k', 0), ('num_beams', 1)]:
        if name in generation and (type(generation[name]) is not int or generation[name] < minimum):
            raise ValueError(f'{name} must be an integer >= {minimum}')
    if 'repetition_penalty' in generation:
        _number(generation['repetition_penalty'], 'protocol.generation.repetition_penalty', positive=True)


def _number(value, name, *, positive=False):
    if (type(value) not in {int, float} or not math.isfinite(value)
            or value < 0 or (positive and value == 0)):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")

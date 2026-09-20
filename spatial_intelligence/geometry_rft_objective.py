"""Masked sequence-level GSPO with a fixed SFT reference.

Equation 5--7: https://arxiv.org/abs/2507.18071 . Log probabilities are
untempered model probabilities, NOT the behavior probabilities of a top-p /
temperature sampler. Using these with truncated sampling is the conventional
approximation, not an exact importance correction. The sampled reverse-KL
estimator is an explicit reference regularizer, not a full-vocabulary KL.
"""
import torch


def response_mask(ids, eos_token_id, pad_token_id=None, attention_mask=None):
    """Response IDs only; include first EOS, exclude everything after it."""
    eos = ids.eq(eos_token_id)
    mask = (eos.long().cumsum(-1) - eos.long()).eq(0)
    if pad_token_id is not None and pad_token_id != eos_token_id:
        mask = mask & ids.ne(pad_token_id)
    if attention_mask is not None:
        if attention_mask.shape != ids.shape:
            raise ValueError("attention mask shape mismatch")
        mask = mask & attention_mask.bool()
    return mask


def group_standardized_advantages(rewards, eps=1e-6):
    """Standardize the LAST (completion-group) axis BEFORE minibatch slicing."""
    rewards = torch.as_tensor(rewards).detach()
    if not rewards.is_floating_point() or rewards.dtype in (torch.float16, torch.bfloat16):
        rewards = rewards.float()
    if rewards.ndim not in (1, 2) or rewards.shape[-1] < 2 or eps <= 0:
        raise ValueError("rewards must be [G] or [B,G], G>=2, eps>0")
    if not torch.isfinite(rewards).all():
        raise ValueError("nonfinite rewards")
    centered = rewards - rewards.mean(-1, keepdim=True)
    return centered / (rewards.std(-1, unbiased=False, keepdim=True) + eps)


def gspo_loss(new_logps, old_logps, reference_logps, advantages, response_mask,
              clip_low=0.0003, clip_high=0.0004, beta=0.02):
    """Return (scalar loss, detached scalar metrics); each sequence weighs equally.

    Logps/mask have shape [G,L] or [B,G,L], advantages shape without L.
    Empty responses are rejected. No silent log-ratio clamping is performed.
    """
    if new_logps.ndim not in (2, 3):
        raise ValueError("logps must have shape [G,L] or [B,G,L]")
    if any(x.shape != new_logps.shape for x in (old_logps, reference_logps, response_mask)):
        raise ValueError("log probability / mask shape mismatch")
    if advantages.shape != new_logps.shape[:-1]:
        raise ValueError("advantage shape mismatch")
    if not (0 <= clip_low < 1 and clip_high >= 0 and beta >= 0):
        raise ValueError("invalid clipping or KL coefficient")
    if not ((response_mask == 0) | (response_mask == 1)).all():
        raise ValueError("response mask must be binary")
    mask = response_mask.bool()
    lengths = mask.sum(-1)
    if (lengths == 0).any():
        raise ValueError("empty response")
    dtype = torch.float64 if new_logps.dtype == torch.float64 else torch.float32
    values = [torch.where(mask, x.to(dtype), 0) for x in
              (new_logps, old_logps.detach(), reference_logps.detach())]
    advantage = advantages.detach().to(dtype)
    if any(not torch.isfinite(x).all() for x in values + [advantage]):
        raise ValueError("nonfinite active log probabilities or advantages")
    new, old, ref = values
    ratio = ((new - old).sum(-1) / lengths).exp()
    surrogate = torch.minimum(ratio * advantage,
                              ratio.clamp(1 - clip_low, 1 + clip_high) * advantage)
    d = ref - new
    kl = (torch.where(mask, torch.expm1(d) - d, 0)).sum(-1) / lengths
    loss = (-surrogate + beta * kl).mean()
    if not torch.isfinite(loss):
        raise FloatingPointError("nonfinite GSPO objective; do not silently clamp ratios")
    return loss, {"loss": float(loss.detach()), "policy_loss": float(-surrogate.mean().detach()),
                  "kl": float(kl.mean().detach()), "ratio": float(ratio.mean().detach()),
                  "clip_fraction": float(((ratio < 1 - clip_low) | (ratio > 1 + clip_high)).float().mean()),
                  "response_tokens": float(lengths.float().mean())}

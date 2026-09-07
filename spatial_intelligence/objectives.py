"""Response-only objectives; tensors have shape [batch, completion, vocab]."""
import torch
import torch.nn.functional as F


def token_logps(logits, ids, temperature=1.0):
    return F.log_softmax(logits.float() / temperature, dim=-1).gather(-1, ids.unsqueeze(-1)).squeeze(-1)


def sft_loss(logits, ids):
    return -token_logps(logits, ids).mean()


def kl_loss(student, teacher, *, direction="reverse", temperature=1.0):
    if student.shape != teacher.shape:
        raise ValueError("KL requires identical vocabulary and aligned completion positions")
    p = F.log_softmax(student.float() / temperature, dim=-1)
    q = F.log_softmax(teacher.detach().to(student.device).float() / temperature, dim=-1)
    if direction == "reverse":
        loss = (p.exp() * (p-q)).sum(-1).mean()
    elif direction == "forward":
        loss = (q.exp() * (q-p)).sum(-1).mean()
    else:
        raise ValueError("KL direction must be reverse or forward")
    return loss * temperature**2


def group_advantages(rewards):
    r = torch.as_tensor(rewards, dtype=torch.float32)
    return (r-r.mean()) / r.std(unbiased=False).clamp_min(1e-6)


def policy_loss(new_logp, old_logp, reference_logp, advantage, *, algorithm, clip, beta):
    delta = new_logp - old_logp.detach()
    if algorithm == "grpo":
        ratio = delta.clamp(-20, 20).exp()
    elif algorithm == "gspo":
        ratio = delta.mean(-1, keepdim=True).clamp(-20, 20).exp()
    else:
        raise ValueError("algorithm must be grpo or gspo")
    surrogate = torch.minimum(ratio*advantage, ratio.clamp(1-clip, 1+clip)*advantage)
    # Nonnegative sampled reverse KL estimator against a frozen reference.
    d = (reference_logp.detach() - new_logp).clamp(-20, 20)
    regularizer = d.exp() - d - 1
    return -surrogate.mean() + beta * regularizer.mean()

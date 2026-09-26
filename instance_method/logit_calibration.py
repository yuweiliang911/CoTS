import torch
import torch.nn.functional as F


def compute_logit_bounds(logits):
    logit_min = logits.min(dim=-1, keepdim=True).values
    logit_max = logits.max(dim=-1, keepdim=True).values
    return logit_min, logit_max


def scale_logits_to_zs_range(logits, zs_min, zs_max, eps=1e-12):
    cur_min, cur_max = compute_logit_bounds(logits)
    denom = (cur_max - cur_min).clamp_min(eps)
    scale = (zs_max - zs_min) / denom
    return scale * (logits - cur_min) + zs_min


def range_penalty(logits, zs_min, zs_max, reduction="mean"):
    penalty = F.relu(logits - zs_max) + F.relu(zs_min - logits)
    if reduction == "none":
        return penalty
    if reduction == "sum":
        return penalty.sum()
    if reduction == "mean":
        return penalty.mean()
    raise ValueError(f"Unsupported reduction: {reduction}")


def logit_range_penalty(logits, zs_min, zs_max):
    return range_penalty(logits, zs_min, zs_max, reduction="mean")

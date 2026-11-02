"""Explicit parent-prefix retention surrogate; the native verifier is unchanged."""
import torch


def margins(logits, labels):
    """Correct-token logit minus the strongest competing token, with gradients."""
    if logits.ndim != 2 or labels.shape != (logits.shape[0],):
        raise ValueError("one label per vocabulary row required")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("finite complete vocabulary logits required")
    labels = labels.to(logits.device)
    correct = logits.gather(1, labels[:, None]).squeeze(1)
    wrong = logits.scatter(1, labels[:, None], -torch.inf).amax(1)
    return correct - wrong


def parent_reference(logits, labels, depth=5, cap=2.0, native_prefix=None):
    """Protect only positions reachable before the parent's first wrong token."""
    logits, labels = logits[:depth], labels[:depth].to(logits.device)
    with torch.no_grad():
        reachable = logits.argmax(-1).eq(labels).long().cumprod(0).bool()
        if native_prefix is not None:
            if type(native_prefix) is not int or not 0 <= native_prefix <= depth:
                raise ValueError("valid captured native accepted-prefix count required")
            reachable &= torch.arange(len(reachable), device=logits.device) < native_prefix
        return {"protected": reachable.cpu().tolist(),
                "margins": margins(logits, labels).clamp(0, cap).cpu().tolist()}


def retention_penalty(logits, labels, reference):
    count = len(reference["protected"])
    current = margins(logits[:count], labels[:count])
    protected = torch.tensor(reference["protected"], device=logits.device, dtype=torch.bool)
    target = torch.tensor(reference["margins"], device=logits.device, dtype=logits.dtype)
    if target.shape != current.shape or not bool(torch.isfinite(target).all()):
        raise ValueError("finite parent margins with matching depth required")
    # A graph-connected zero keeps failed-parent windows valid accumulation members.
    if not bool(protected.any()):
        return current.sum() * 0
    return (target[protected] - current[protected]).clamp_min(0).square().mean()

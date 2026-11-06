"""Training surrogates, deliberately separate from the native greedy verifier.

D-PAL Eq. 6/9: https://arxiv.org/html/2610.28355v1
VAT Eq. 4-7: https://arxiv.org/html/2608.30135v1
EDR motivation: https://arxiv.org/html/2610.10411v1
The latter's stochastic ratio/residual theorem is NOT asserted for these greedy
captured-prefix pilots. D5 receives supervision; the existing graph computes 7.
"""
import torch
from torch.nn import functional as F


def _check(logits, labels, positions=5):
    if logits.ndim != 2 or logits.shape[0] < positions or labels.shape != (logits.shape[0],):
        raise ValueError("complete block logits/labels required")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("finite logits required")
    return logits[:positions].float(), labels[:positions].to(logits.device)


def dpal_weights(student_logits, teacher_logits, floor=0.1):
    """Eq6, exact full-vocabulary overlap; reverse sum of prefix products."""
    if not 0 < floor < 1 or teacher_logits.shape != student_logits.shape:
        raise ValueError("same full distributions and 0<floor<1 required")
    if teacher_logits.requires_grad or not bool(torch.isfinite(teacher_logits).all()):
        raise ValueError("finite detached target logits required")
    with torch.no_grad():
        p, q = teacher_logits.float().softmax(-1), student_logits.float().softmax(-1)
        overlap = torch.minimum(p, q).sum(-1)
        prefix = (floor + (1-floor)*overlap).cumprod(0)
        weights = prefix.flip(0).cumsum(0).flip(0)
    return weights, overlap


def dpal_loss(logits, labels, teacher_logits, *, floor=0.1, ce_anchor=0.01):
    x, y = _check(logits, labels)
    teacher = teacher_logits[:5].to(x.device).detach().float()
    weights, overlap = dpal_weights(x, teacher, floor)
    logp, logq = teacher.log_softmax(-1), x.log_softmax(-1)
    # Stable Renyi divergence, order 1/2. No probability clamp in the fitting loss.
    renyi = -2*torch.logsumexp((logp+logq)*0.5, -1)
    ce = F.cross_entropy(x, y)
    value = (weights*renyi).sum()/5 + ce_anchor*ce
    return value, {"renyi_half": float(renyi.mean().detach()), "ce": float(ce.detach()),
                   "position_weights": weights.tolist(), "overlap": overlap.tolist(),
                   "weight_gradient": "detached", "floor": floor}


def vat_labels_weights(logits, labels, *, decay=0.8):
    x, y = _check(logits, labels)
    if not 0 < decay <= 1:
        raise ValueError("0<decay<=1 required")
    with torch.no_grad():
        hits = x.argmax(-1).eq(y)
        survival = hits.long().cumprod(0).float()
        wrong = torch.nonzero(~hits).flatten()
        first = int(wrong[0]) if len(wrong) else len(hits)
        slots = torch.arange(len(hits), device=x.device)
        weights = decay**(slots-first).clamp_min(0).float()
    return survival, weights, first


def vat_loss(logits, labels, hidden, head, *, decay=0.8, beta=0.1, teacher_logits=None):
    """Greedy VAT adaptation: recorded target-prefix labels, shifted CE + VH.

    Paper full objective also uses soft-label CE. When supplied authentic target
    logits this includes it; otherwise metadata explicitly declares hard-only.
    """
    x, y = _check(logits, labels)
    survival, weights, first = vat_labels_weights(logits, labels, decay=decay)
    ce = F.cross_entropy(x, y, reduction="none")
    if teacher_logits is not None:
        teacher = teacher_logits[:5].to(x.device).detach().float().softmax(-1)
        ce = ce - (teacher*x.log_softmax(-1)).sum(-1)
    verification = F.binary_cross_entropy_with_logits(head(hidden[:5]).flatten(), survival)
    task = (weights*ce).sum()/weights.sum()
    value = 0.1*task + beta*verification
    return value, {"ce": float(task.detach()), "verification_bce": float(verification.detach()),
                   "first_rejection": first, "survival_labels": survival.tolist(),
                   "position_weights": weights.tolist(), "soft_labels": teacher_logits is not None,
                   "verification_rule": "greedy_recorded_target_prefix_proxy"}


def round_transition(acceptance):
    """Probabilities of k accepted then one target token, k=0..D.

    This smooth transition assumes conditional independence; terminal clipping is
    the responsibility of the Bellman recurrence. It is not native acceptance.
    """
    if acceptance.ndim != 1 or not bool(torch.isfinite(acceptance).all()) or bool(((acceptance<0)|(acceptance>1)).any()):
        raise ValueError("one finite acceptance vector in [0,1] required")
    prefix = torch.cat([acceptance.new_ones(1), acceptance.cumprod(0)])
    return torch.cat([prefix[:-1]*(1-acceptance), prefix[-1:]])


def soft_round_values(acceptances):
    """Cross-round finite-horizon Bellman recurrence, differentiable throughout.

    acceptances[n,k] is a proxy at round start n for position k, not an exact
    stochastic EDR conditional acceptance. Every next round changes its row.
    """
    if acceptances.ndim != 2 or not acceptances.shape[0]:
        raise ValueError("dense starts x positions matrix required")
    length = acceptances.shape[0]
    values = [acceptances.new_zeros(()) for _ in range(length+1)]
    for start in range(length-1, -1, -1):
        probabilities = round_transition(acceptances[start])
        future = torch.stack([values[min(length, start+k+1)] for k in range(len(probabilities))])
        values[start] = 1 + (probabilities*future).sum()
    return torch.stack(values)


def edr_td_loss(logits, labels, continuation_values, *, ce_anchor=0.01):
    """One-row semi-gradient of a cached cross-round Bellman value.

    The six detached continuation values come from dense parent round starts,
    NOT a local accepted-length proxy. The cache must name its horizon/parent.
    """
    x, y = _check(logits, labels)
    future = torch.as_tensor(continuation_values, dtype=x.dtype, device=x.device).detach()
    if future.shape != (6,) or not bool(torch.isfinite(future).all()) or bool((future<0).any()):
        raise ValueError("six finite nonnegative cross-round values required")
    a = x.log_softmax(-1).gather(-1, y[:,None]).squeeze(-1).exp()
    bellman = 1+(round_transition(a)*future).sum()
    ce = F.cross_entropy(x, y)
    return bellman+ce_anchor*ce, {"ce": float(ce.detach()),
                                "bellman_round_proxy": float(bellman.detach()),
                                "continuation_values": future.tolist(),
                                "surrogate": "finite_horizon_recorded_token_confidence_semigradient"}

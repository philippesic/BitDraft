"""Fixed-state metrics. Teacher forcing and greedy survival are separate contracts."""
import numpy as np


def score(logits, labels, depth=5):
    x = np.asarray(logits, dtype=np.float64)[:depth]
    y = np.asarray(labels, dtype=np.int64)[:depth]
    if x.ndim != 2 or len(x) != depth or len(y) != depth or not np.isfinite(x).all():
        raise ValueError('complete finite logits required')
    if np.any(y < 0) or np.any(y >= x.shape[1]):
        raise ValueError('labels outside vocabulary')
    pred = x.argmax(-1)
    label = x[np.arange(depth), y]
    peak = x.max(-1)
    ce = peak + np.log(np.exp(x - peak[:, None]).sum(-1)) - label
    wrong = x.copy()
    wrong[np.arange(depth), y] = -np.inf
    margins = label - wrong.max(-1)
    hits = pred == y
    prefix = int(np.cumprod(hits).sum())
    return dict(ce=ce.tolist(), top1=hits.tolist(), predictions=pred.tolist(),
                margins=margins.tolist(), prefix=prefix, first_error=prefix if prefix < depth else None)


def ordinary_indices(count, take=4):
    if count < take:
        raise ValueError('insufficient ordinary eligible rounds')
    return [i * count // take for i in range(take)]

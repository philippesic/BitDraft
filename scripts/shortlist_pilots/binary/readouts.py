"""Fixed development readouts that preserve RNG and never own an optimizer."""
from contextlib import contextmanager


@contextmanager
def preserve_readout_state(model):
    from w1a1_eagle.block_training import _block_rng_state, _restore_block_rng
    from shortlist_pilots.binary.parent_loader import equal
    state = _block_rng_state(model.token_embd.device)
    training = model.training
    try:
        model.eval()
        yield
    finally:
        model.train(training)
        _restore_block_rng(state, model.token_embd.device)
        if not equal(_block_rng_state(model.token_embd.device), state):
            raise ValueError("readout changed RNG after restoration")


def disjoint_case_indices(fit_cases, training_pairs, cases, maximum=4):
    trained = {fit_cases[i]["group_id"] for pair in training_pairs for i in pair}
    selected, groups = [], set()
    for i, case in enumerate(cases):
        group = case["group_id"]
        if group not in trained and group not in groups:
            selected.append(i)
            groups.add(group)
        if len(selected) == maximum:
            break
    return selected


def development_window(fit_targets, provider, training_pairs):
    """Prefer existing probe development split; never label it fresh holdout."""
    from cohort import AnswerTargets
    from train_nine_model_qat import tensor_batch
    probe = fit_targets.plan.get("probe", {}).get("answer", [])
    if probe:
        targets = AnswerTargets(fit_targets.root, provider, role="probe")
        role = "original probe split, reused development; not fresh holdout"
    else:
        targets = fit_targets
        role = "original fit pool, excluded from this pilot's training groups; reused development"
    if targets.plan_pin != fit_targets.plan_pin:
        raise ValueError("development source plan differs")
    indices = disjoint_case_indices(fit_targets.cases, training_pairs, targets.cases)
    batches = [tensor_batch(targets.load(i)) for i in indices]
    return batches, {"role": role, "indices": indices,
                     "group_ids": [targets.cases[i]["group_id"] for i in indices],
                     "data_inputs": targets.plan_pin, "groups_disjoint_from_pilot_training": True,
                     "available": bool(indices)}


def summarize(rows):
    if not rows:
        return {"blocks": 0, "available": False}
    return {"blocks": len(rows), "available": True,
            "captured_prefix_depth5_sum": sum(r["captured_prefix_depth5"] for r in rows),
            "captured_prefix_depth5_mean": sum(r["captured_prefix_depth5"] for r in rows)/len(rows),
            "mean_ce5": sum(r["mean_ce5"] for r in rows)/len(rows)}


def compare(before, after):
    a, b = summarize(before), summarize(after)
    if len(before) != len(after) or any(x["label_ids"] != y["label_ids"] for x, y in zip(before, after)):
        raise ValueError("fixed development readout labels/window changed")
    result = {"before": a, "after": b}
    if before:
        result["delta_after_minus_before"] = {
            key: b[key]-a[key] for key in ("captured_prefix_depth5_sum", "captured_prefix_depth5_mean", "mean_ce5")}
    return result

"""Apply the original DeviceSignBaseline to unchanged native projections only."""
import copy


def measure_sign_guard(baseline, model, *, excluded=()):
    """A shallow baseline view shares authenticated immutable reference bits.

    DeviceSignBaseline.measure and its original shape/dtype/device checks run
    unchanged against a proxy exposing actual unchanged modules. Only names,
    shape/packed/reference dictionaries are subset; the historical object and
    its reference tensors are never mutated or replaced by synthetic weights.
    """
    linears = model.binary_linears()
    if tuple(linears) != baseline.names:
        raise ValueError("original sign-guard projection inventory changed")
    excluded = set(excluded)
    if not excluded <= set(baseline.names):
        raise ValueError("sign-guard exclusion is not an original projection")
    if baseline.reference is None:
        raise ValueError("original sign baseline must already be bound")
    names = tuple(name for name in baseline.names if name not in excluded)
    if not excluded:
        result = baseline.measure(model)
    elif not names:
        result = {"changed": 0, "total": 0, "fraction": None, "projections": {}}
    else:
        view = copy.copy(baseline)
        view.names = names
        view.shapes = {name: baseline.shapes[name] for name in names}
        view.packed = {name: baseline.packed[name] for name in names}
        view.reference = {name: baseline.reference[name] for name in names}
        view.baseline_device_bytes = sum(x.numel()*x.element_size() for x in view.reference.values())

        class UnchangedModel:
            def binary_linears(self):
                return {name: linears[name] for name in names}

        result = view.measure(UnchangedModel())
    return {**result, "guard_limit_fraction": .02, "guarded_projection_count": len(names),
            "reference": "original step1000 DeviceSignBaseline, unchanged projections only",
            "changed_projection_guard_not_applicable": sorted(excluded),
            "changed_projection_reason": "representation changed; original bit predicate/geometry not comparable" if excluded else None}

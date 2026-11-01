#!/usr/bin/env python3
"""Recheck frozen request evidence and bootstrap paired prompt clusters.

Run with .venv/bin/python; this reads saved responses and makes no model requests.
All five repetitions of a prompt are resampled together. Confidence intervals
describe this prompt panel, not deployment capacity or independent GPU sessions.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import numpy as np


def pin(path):
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=root / "runs/upstream-consistency-20261101")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("confirmation-summary.json"))
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    p = args.run.resolve()
    rows_path = p / "confirmation-requests-01/requests.jsonl"
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
    assert len(rows) == 1920
    by = {(r["model"], r["variant"], r["arm"], r["rep"], r["prompt_id"]): r for r in rows}
    assert len(by) == len(rows)
    models = sorted({r["model"] for r in rows})
    prompts = sorted({r["prompt_id"] for r in rows})
    assert len(models) == 4 and len(prompts) == 24
    cells = [(v, a) for v in ("conservative32", "quant4group") for a in ("target_only", "speculative")]
    signature = lambda r: {k: r["result"][k] for k in ("tokens", "content", "stop_type", "stopping_word", "truncated")}
    for r in rows:
        m, v, a, rep, prompt = (r[k] for k in ("model", "variant", "arm", "rep", "prompt_id"))
        assert signature(r) == signature(by[m, "conservative32", "target_only", rep, prompt])
        assert signature(r) == signature(by[m, v, a, 0, prompt])
        assert not r["result"]["truncated"]
        assert len(r["result"]["tokens"]) == r["output_tokens"] == r["result"]["tokens_predicted"]
        assert r["result"]["timings"]["cache_n"] == 0
        assert math.isfinite(r["full_request_seconds"]) and r["full_request_seconds"] > 0
        c = r["speculative"]
        assert all(type(x) is int and x >= 0 for x in c.values())
        assert c["accepted"] <= c["proposed"]
        if a == "speculative":
            assert c["accepted"] == r["result"]["timings"]["draft_n_accepted"]
            assert c["proposed"] == r["result"]["timings"]["draft_n"]
        else:
            assert set(c.values()) == {0}
    summary_path = p / "confirmation-requests-01/summary.json"
    raw = json.loads(summary_path.read_text())
    assert raw["completed"] and raw["all_tokens_equal"] and not raw["token_mismatches"]
    rng = np.random.default_rng(20261101)
    draws = rng.integers(0, len(prompts), size=(20000, len(prompts)))
    result = {}
    for m in models:
        result[m] = {"cells": {}, "comparisons": {}}
        times = {}
        for v, a in cells:
            rr = [by[m, v, a, rep, pr] for rep in range(5) for pr in prompts]
            token_count = sum(r["output_tokens"] for r in rr)
            wall = sum(r["full_request_seconds"] for r in rr)
            rate = token_count / wall
            name = v + "/" + a
            original = raw["cells"][m][name]
            assert original["requests"] == len(rr) == 120
            assert math.isclose(rate, original["tokens_per_second"], rel_tol=1e-12)
            reps = [sum(by[m, v, a, rep, pr]["output_tokens"] for pr in prompts) /
                    sum(by[m, v, a, rep, pr]["full_request_seconds"] for pr in prompts) for rep in range(5)]
            result[m]["cells"][name] = {"requests": len(rr), "output_tokens": token_count,
                "full_request_seconds": wall, "tokens_per_second": rate,
                "repetition_rates": reps, "repetition_min_max": [min(reps), max(reps)],
                "speculative_counts": original["speculative"]}
            times[name] = np.array([sum(by[m, v, a, rep, pr]["full_request_seconds"] for rep in range(5)) for pr in prompts])
        candidate = "quant4group/speculative"
        assert result[m]["cells"][candidate]["speculative_counts"] == result[m]["cells"]["conservative32/speculative"]["speculative_counts"]
        for reference in ("conservative32/speculative", "conservative32/target_only", "quant4group/target_only"):
            # Output token totals match in every paired cluster, so their
            # throughput ratio equals the inverse ratio of request wall times.
            percent = (times[reference].sum() / times[candidate].sum() - 1) * 100
            samples = (times[reference][draws].sum(axis=1) / times[candidate][draws].sum(axis=1) - 1) * 100
            interval = np.quantile(samples, [0.025, 0.975]).tolist()
            rep_gains = [(result[m]["cells"][candidate]["repetition_rates"][i] /
                          result[m]["cells"][reference]["repetition_rates"][i] - 1) * 100 for i in range(5)]
            result[m]["comparisons"][reference] = {"candidate": candidate,
                "throughput_gain_percent": percent, "paired_prompt_cluster_bootstrap_95_percent": interval,
                "matched_repetition_gains_percent": rep_gains,
                "prompt_clusters_faster": int((times[candidate] < times[reference]).sum()),
                "prompt_clusters": len(prompts)}
            assert interval[0] > 0 and min(rep_gains) > 0
    qualities = {}
    qrows = {}
    for v in ("original", "conservative32", "quant4group"):
        qpath = p / f"final-quality-{v}-01/requests.jsonl"
        q = [json.loads(x) for x in qpath.read_text().splitlines()]
        assert len(q) == 96
        qrows[v] = {(x["model"], x["case_id"]): x for x in q}
        assert len(qrows[v]) == 96
        qualities[v] = {m: {"passed": sum(x["judgement"]["passed"] for x in q if x["model"] == m), "cases": 24} for m in models}
    assert qrows["conservative32"].keys() == qrows["quant4group"].keys()
    for key, q in qrows["conservative32"].items():
        other = qrows["quant4group"][key]
        assert q["result"]["tokens"] == other["result"]["tokens"]
        assert q["result"]["content"] == other["result"]["content"]
        assert q["judgement"] == other["judgement"]
    output = {"scope": "Frozen Qwen3.5 MTP SM120/F16-KV matched opt-in runtime confirmation; not W1A1/EAGLE or default-backend equivalence",
        "sources": [pin(rows_path), pin(summary_path), pin(p / "final-freeze.json"), pin(p / "confirmation-inputs/receipt.json")],
        "checks": {"performance_requests": len(rows), "quality_requests": 288, "token_text_stop_and_repeat_mismatches": 0,
                   "counter_mismatches": 0, "candidate_quality_output_differences_from_conservative": 0},
        "statistics": {"method": "paired prompt-cluster percentile bootstrap; all five repetitions kept in each cluster",
                       "clusters": 24, "samples": 20000, "numpy_seed": 20261101,
                       "limits": "Panel variation only; one local GPU session, no cross-session or production capacity inference"},
        "models": result, "quality": qualities, "performance_costs_seconds": raw["costs"]}
    encoded = json.dumps(output, indent=2, sort_keys=True) + "\n"
    if args.verify:
        assert args.output.read_text() == encoded
    else:
        args.output.write_text(encoded)
    print(json.dumps({"verified": args.verify, "output": str(args.output),
        "gains": {m: x["comparisons"]["conservative32/speculative"] for m, x in result.items()}}, indent=2))


if __name__ == "__main__":
    main()

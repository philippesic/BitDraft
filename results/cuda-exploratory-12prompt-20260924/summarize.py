import hashlib
import json
import statistics
import subprocess
import tomllib
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "results/cuda-exploratory-12prompt-20260924"
CONFIG = ROOT / "configs/pytorch_w1a1.toml"
PROMPTS = ROOT / "configs/acceptance_prompts.jsonl"
MANIFEST = ROOT / "results/model-manifest-5080-20260924.json"
rows = [json.loads(line) for line in (RUN / "acceptance.jsonl").read_text().splitlines()]
targets = [json.loads(line) for line in (RUN / "target-reference.jsonl").read_text().splitlines()]
mismatches = [json.loads(line) for line in (RUN / "greedy-mismatches.jsonl").read_text().splitlines()]
expected_variants = ["ordinary", "fusion", "attention", "ffn", "head", "all"]
expected_prompts = [item["id"] for item in (
    json.loads(line) for line in PROMPTS.read_text().splitlines() if line.strip()
)]
assert len(rows) == 72
assert len(targets) == 12 and {r["prompt_id"] for r in targets} == set(expected_prompts)
assert Counter(r["variant"] for r in rows) == {v: 12 for v in expected_variants}
assert len({(r["variant"], r["prompt_id"]) for r in rows}) == 72
assert Counter((r["variant"], r["prompt_id"]) for r in mismatches) == Counter(
    (r["variant"], r["prompt_id"]) for r in rows if not r["target_greedy_match"]
)

def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def distribution(values):
    return {
        "min": min(values),
        "median": statistics.median(values),
        "max": max(values),
    }

def aggregate(items):
    accepted = sum(int(r["accepted_draft_tokens"]) for r in items)
    proposed = sum(int(r["proposed_tree_nodes"]) for r in items)
    rounds = sum(int(r["rounds"]) for r in items)
    per_prompt_round = [r["accepted_draft_tokens"] / r["rounds"] for r in items]
    per_prompt_node_rate = [r["accepted_draft_tokens"] / r["proposed_tree_nodes"] for r in items]
    outputs = [int(r["generated_tokens"]) for r in items]
    return {
        "prompts": len(items),
        "accepted_draft_tokens": accepted,
        "proposed_tree_nodes": proposed,
        "rounds": rounds,
        "accepted_per_round_weighted": accepted / rounds,
        "accepted_per_proposed_node_weighted": accepted / proposed,
        "target_greedy_match_prompts": sum(bool(r["target_greedy_match"]) for r in items),
        "target_greedy_mismatch_prompts": sum(not bool(r["target_greedy_match"]) for r in items),
        "accepted_per_round_prompt_distribution": distribution(per_prompt_round),
        "accepted_per_node_prompt_distribution": distribution(per_prompt_node_rate),
        "generated_tokens_prompt_distribution": distribution(outputs),
    }

by_variant = {v: aggregate([r for r in rows if r["variant"] == v]) for v in expected_variants}
categories = sorted({r["category"] for r in rows})
by_category = {
    v: {
        category: aggregate([r for r in rows if r["variant"] == v and r["category"] == category])
        for category in categories
    }
    for v in expected_variants
}
per_prompt = [
    {
        "variant": r["variant"],
        "prompt_id": r["prompt_id"],
        "category": r["category"],
        "prompt_tokens": r["prompt_tokens"],
        "generated_tokens": r["generated_tokens"],
        "target_greedy_checked_tokens": r["target_greedy_checked_tokens"],
        "accepted_draft_tokens": r["accepted_draft_tokens"],
        "proposed_tree_nodes": r["proposed_tree_nodes"],
        "rounds": r["rounds"],
        "accepted_per_round": r["accepted_draft_tokens"] / r["rounds"],
        "accepted_per_proposed_node": r["accepted_draft_tokens"] / r["proposed_tree_nodes"],
        "target_greedy_match": r["target_greedy_match"],
        "first_target_greedy_mismatch_index": r["first_target_greedy_mismatch_index"],
    }
    for r in rows
]
environment = json.loads((RUN / "environment.json").read_text())
config = tomllib.loads(CONFIG.read_text())
summary = {
    "classification": "exploratory same-verifier acceptance; non-target-equivalent because target/ordinary parity fails",
    "project_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    "models": {
        role: {"repo": item["repo"], "revision": item["revision"]}
        for role, item in json.loads(MANIFEST.read_text())["models"].items()
    },
    "angelslim_revision": config["models"]["angelslim_revision"],
    "precision": {"target": environment["target_dtype"], "drafter": environment["draft_dtype"]},
    "hardware": {"gpu": environment["gpu"], "cuda_runtime": environment["cuda_runtime"]},
    "config_sha256": sha(CONFIG),
    "prompt_sha256": sha(PROMPTS),
    "model_manifest_sha256": sha(MANIFEST),
    "parity": {
        "target_eagle_greedy_match": environment["target_eagle_greedy_match"],
        "greedy_mismatch_allowed": environment["greedy_mismatch_allowed"],
        "first_prompt_mismatch_index": json.loads((RUN / "greedy-parity.json").read_text())["first_mismatch_index"],
        "disabled_wrapper_match": (
            json.loads((RUN / "greedy-parity.json").read_text()).get("disabled_wrapper_token_ids")
            == json.loads((RUN / "greedy-parity.json").read_text())["ordinary_eagle_token_ids"]
        ),
    },
    "record_counts": {
        "acceptance_rows": len(rows),
        "target_reference_rows": len(targets),
        "greedy_mismatch_rows": len(mismatches),
        "variant_rows": dict(Counter(r["variant"] for r in rows)),
        "category_prompt_counts": dict(Counter(r["category"] for r in rows if r["variant"] == "ordinary")),
    },
    "by_variant": by_variant,
    "by_category": by_category,
    "per_prompt": per_prompt,
}
(RUN / "derived-metrics.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps({
    "record_counts": summary["record_counts"],
    "by_variant": by_variant,
    "by_category": by_category,
    "derived_metrics_sha256": sha(RUN / "derived-metrics.json"),
}, indent=2, sort_keys=True))

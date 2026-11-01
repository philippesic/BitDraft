#!/usr/bin/env python3
"""Recheck saved Oct 31 CUDA results and source identity; never execute their code."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    root = args.run.resolve()
    audit = json.loads(args.audit.read_text())
    exact = total = checks = flips = 0
    max_abs = 0.0
    hashed = []

    def saved(path, expected):
        path = Path(path).resolve()
        assert path.is_relative_to(root), path
        actual = digest(path)
        assert actual == expected, (path, actual, expected)
        hashed.append(str(path))
        return path

    for arm in audit["requests"]:
        rows = []
        for entry in arm["request_files"]:
            path = saved(entry["path"], entry["sha256"])
            records = [json.loads(line) for line in path.read_text().splitlines()]
            assert len(records) == 32
            indexed = {(r["prompt"], r["rep"]): r["result"] for r in records}
            assert len(indexed) == len(records)
            assert all(r["timings"]["cache_n"] == 0 for r in indexed.values())
            rows.append(indexed)
        assert len(rows) == 2 and rows[0].keys() == rows[1].keys()
        for key, result in rows[0].items():
            base = rows[1][key]
            equal = all(result[field] == base[field] for field in ["tokens", "content", "stop_type", "tokens_predicted"])
            assert equal, (arm["target"], key)
            exact += equal
            total += 1
        # The two saved no-spec repetitions must agree independently.
        for prompt in range(16):
            assert rows[1][prompt, 0]["tokens"] == rows[1][prompt, 1]["tokens"]
    for entry in audit["regressions"]:
        path = saved(entry["path"], entry["sha256"])
        summaries = re.findall(r"spec parity: checks=(\d+) greedy_flips=(\d+) max_abs=([^\s]+)", path.read_text())
        assert len(summaries) == 2, path
        for count, nflip, error in summaries:
            checks += int(count)
            flips += int(nflip)
            max_abs = max(max_abs, float(error))
    assert (total, exact, checks, flips, max_abs) == (128, 128, 2040, 0, 0.0)

    proof = json.loads((root / "live-library-proof.json").read_text())
    assert len(proof) == 1
    assert proof[0]["environment"]["GGML_CUDA_DECODE_CONSISTENT"] == "32"
    assert len(proof[0]["backend_maps"]) == 1
    library = proof[0]["backend_maps"][0]
    saved(library, proof[0]["hashes"][library])

    cache = (root / "build/CMakeCache.txt").read_text()
    unpatched = Path(re.search(r"^CMAKE_HOME_DIRECTORY:INTERNAL=(.+)$", cache, re.M)[1])
    patch = root / "cuda-decode-consistent.patch"
    paths = re.findall(r"^--- a/(.+)$", patch.read_text(), re.M)
    assert len(paths) == 5
    with tempfile.TemporaryDirectory(prefix="eagle-recovered-cuda-check-") as scratch:
        target = Path(scratch)
        for path in paths:
            dest = target / path
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes((unpatched / path).read_bytes())
        result = subprocess.run(["git", "apply", str(patch)], cwd=target, text=True, capture_output=True)
        assert result.returncode == 0, result.stderr
        for path in paths:
            assert (target / path).read_bytes() == (root / "patched-source" / path).read_bytes(), path
    print(json.dumps({"request_comparisons": total, "request_exact": exact,
                      "regression_checks": checks, "regression_greedy_flips": flips,
                      "regression_max_abs": max_abs, "evidence_hashes_verified": len(hashed),
                      "patched_source_files_matched": len(paths), "patch_sha256": digest(patch),
                      "historical_library_hash_matches": True, "live_process_proven_now": False,
                      "gpu_execution": False, "performance_acceptance": False}, sort_keys=True))


if __name__ == "__main__":
    main()

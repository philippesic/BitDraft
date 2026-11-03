"""Prepare new confirmation instances only after a selected candidate/source freeze.

CLI: modern_fresh.py prepare --freeze SELECTED.json --output NEW_DIRECTORY
No model/weight, network, GPU, inference or original sealed-data access occurs.
The frozen per-prompt exclusions must cover actual_train, development, pilot and
old_matrix. A collision or failed oracle control retains every generated case in
an INCONCLUSIVE packet; callers must require receipt.admitted before inference.

Source contract and exclusion examples are documented by validate_freeze and
load_exclusions. Outputs include matrix-compatible requests/quality-cases/controls
plus a new modern_fresh_confirmation_inputs_v1 manifest. Only item['request'] is
sent to serving; private expected answers/tests/controls never enter that body.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import itertools
import json
import random
import re
import unicodedata
from datetime import UTC, datetime
from pathlib import Path

SCHEMA = "modern_fresh_confirmation_freeze_v1"
TRUSTED_JUDGE = Path(
    "/home/philip/binary-eagle-decoding/runs/upstream-consistency-20261101/quality_judge.py"
)
TRUSTED_JUDGE_SHA = "59858c3f75048f07a686b3b4b74ae8d7983d5734e9d3432b6fe82b9b86ebe7c6"
SYSTEM = "You are a helpful assistant. Answer directly and accurately."
FAMILIES = {
    "code": ("prefix_sums", "transpose", "lower_bound", "sql_unblocked"),
    "reasoning": ("finite_sets", "calendar", "dag_route", "assignment"),
    "prose": ("chronology", "obligations", "correction", "unknown_fact"),
}
ROLES = {"actual_train", "development", "pilot", "old_matrix"}
FILE_LIMIT = 1024**2
INPUT_LIMIT = 2 * 1024**2
OUTPUT_LIMIT = 2 * 1024**2
RECORD_LIMIT = 4096
HEX = re.compile(r"[0-9a-f]{64}\Z")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def encoded(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()


def digest(value):
    return hashlib.sha256(value).hexdigest()


def normalized(text):
    return " ".join(unicodedata.normalize("NFKC", text).split())


def render(text):
    return (
        f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
        f"<|im_start|>user\n{text}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


def identities(text):
    """Raw-user and rendered exact/normalized SHA256, shared with exclusion indices."""
    match = re.fullmatch(
        r"<\|im_start\|>system\n.*?<\|im_end\|>\n"
        r"<\|im_start\|>user\n(.*?)<\|im_end\|>\n"
        r"<\|im_start\|>assistant\n.*",
        text,
        re.S,
    )
    raw = match[1] if match else text
    return {
        "raw_sha256": digest(raw.encode()),
        "normalized_raw_sha256": digest(normalized(raw).encode()),
        "rendered_sha256": digest(render(raw).encode()),
        "normalized_rendered_sha256": digest(normalized(render(raw)).encode()),
    }


def reference(path):
    path = Path(path).resolve()
    size = path.stat().st_size
    require(size <= FILE_LIMIT, "metadata/source exceeds bounded file limit")
    data = path.read_bytes()
    require(len(data) == size, "file changed while reading")
    return {"path": str(path), "bytes": size, "sha256": digest(data)}


def opaque(ref):
    """Validate identity syntax without opening or statting model/runtime binaries."""
    require(
        isinstance(ref, dict) and {"path", "bytes", "sha256"} <= ref.keys(),
        "absolute path/bytes/SHA256 pin required",
    )
    require(
        isinstance(ref["path"], str) and Path(ref["path"]).is_absolute(),
        "absolute pinned path required",
    )
    require(type(ref["bytes"]) is int and ref["bytes"] > 0, "positive pinned bytes required")
    require(
        isinstance(ref["sha256"], str) and HEX.fullmatch(ref["sha256"]), "valid SHA256 required"
    )


def checked(ref, *, suffix):
    opaque(ref)
    require(Path(ref["path"]).suffix == suffix, "wrong metadata/source file type")
    require(
        reference(ref["path"]) == {k: ref[k] for k in ("path", "bytes", "sha256")},
        "pinned source/metadata changed",
    )
    return Path(ref["path"])


def load_judge(ref):
    require(isinstance(ref, dict), "judge source pin required")
    require(
        Path(ref.get("path", "")).resolve() == TRUSTED_JUDGE.resolve()
        and ref.get("sha256") == TRUSTED_JUDGE_SHA,
        "untrusted judge source",
    )
    path = checked(ref, suffix=".py")
    spec = importlib.util.spec_from_file_location("modern_fresh_trusted_judge", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_freeze(path):
    """Required selected/source pins and complete exclusion roles precede generation.

    exclusions: [{role, format: 'payloads_v1'|'hash_index_v1', file: PIN,
                  previously_used: true, sealed: false, complete: true}, ...].
    Hash index: {schema:'modern_fresh_exclusion_index_v1', records:[identities(text)]}.
    Payloads are explicit already-used request/quality/binding/roster JSON only.
    They can open here only after selection validation; still-sealed inputs reject.
    """
    freeze_ref = reference(path)
    raw = Path(path).read_bytes()
    require(digest(raw) == freeze_ref["sha256"], "freeze changed while reading")
    data = json.loads(raw)
    require(isinstance(data, dict), "freeze must be an object")
    require(
        data.get("schema") == SCHEMA and data.get("selection_frozen") is True,
        "selected candidate/source freeze required before creating instances",
    )
    stamp = datetime.fromisoformat(data.get("selected_at_utc", "").replace("Z", "+00:00"))
    require(stamp.tzinfo is not None and stamp <= datetime.now(UTC), "invalid selection time")
    require(
        data.get("evaluation_inputs_generated", False) is False,
        "freeze must precede fresh confirmation inputs",
    )
    for key in ("candidate", "runtime"):
        opaque(data.get(key))  # intentionally no model or executable access
    source = json.loads(checked(data["source_freeze"], suffix=".json").read_bytes())
    selected = [a for a in source.get("arms", []) if a.get("id") == data.get("selected_arm_id")]
    require(len(selected) == 1, "selected arm must exist uniquely in frozen serving models")
    arm = selected[0]
    require(
        arm.get("kind") == "w1a8" and arm.get("model") == data["candidate"],
        "selected model differs from frozen serving arm",
    )
    require(
        arm.get("runtime", {}).get("server") == data["runtime"],
        "selected executable differs from frozen serving arm",
    )
    require(
        arm["runtime"].get("libraries")
        and arm["runtime"].get("environment")
        and arm["runtime"].get("source_pins"),
        "complete selected runtime contract required",
    )
    require(
        Path(data["generator"]["path"]).resolve() == Path(__file__).resolve(),
        "freeze must pin this generator source",
    )
    checked(data["generator"], suffix=".py")
    require(
        isinstance(data.get("settings"), dict)
        and type(data["settings"].get("output_limit")) is int
        and data["settings"]["output_limit"] in (256, 512),
        "freeze output_limit=256 or 512 before outputs",
    )
    exclusions = data.get("exclusions")
    require(isinstance(exclusions, list) and exclusions, "complete exclusion inventories required")
    require(all(isinstance(item, dict) for item in exclusions), "invalid exclusion entry")
    require({item.get("role") for item in exclusions} == ROLES, "missing/unknown exclusion role")
    total = 0
    for item in exclusions:
        require(
            item.get("sealed") is False
            and item.get("previously_used") is True
            and item.get("complete") is True,
            "only complete, already-used unsealed exclusions",
        )
        require(
            item.get("format") in ("payloads_v1", "hash_index_v1"), "unsupported exclusion format"
        )
        opaque(item["file"])
        total += item["file"]["bytes"]
    require(total <= INPUT_LIMIT, "exclusion inputs exceed bounded total metadata budget")
    judge = load_judge(data["judge"])
    require(reference(path) == freeze_ref, "freeze changed during validation")
    return data, freeze_ref, judge


def _texts(document):
    """Inspect only explicit prior prompt/message containers, never arbitrary dependencies."""
    if isinstance(document, str):
        yield document
    elif isinstance(document, list):
        for item in document:
            yield from _texts(item)
    elif isinstance(document, dict):
        for key in ("raw_messages", "messages"):
            if key in document:
                messages = document[key]
                require(isinstance(messages, list) and messages, "invalid prior messages")
                users = [m["content"] for m in messages if m.get("role") == "user"]
                require(
                    users and all(isinstance(t, str) for t in users), "prior user text required"
                )
                yield from users
        for key in ("prompt", "rendered_prompt", "rendered_text", "user_text"):
            if key in document:
                require(isinstance(document[key], str), "prior prompt must be text")
                yield document[key]
        for key in ("rows", "roster", "cases", "requests", "request", "prompts", "bindings"):
            if key in document:
                yield from _texts(document[key])


def load_exclusions(items):
    hashes, receipts = set(), []
    record_count = 0
    for item in items:
        path = checked(item["file"], suffix=".json")
        raw = path.read_bytes()
        require(digest(raw) == item["file"]["sha256"], "exclusion changed while reading")
        document = json.loads(raw)
        if item["format"] == "hash_index_v1":
            require(
                isinstance(document, dict)
                and document.get("schema") == "modern_fresh_exclusion_index_v1",
                "wrong exclusion index schema",
            )
            records = document.get("records")
            require(isinstance(records, list) and records, "empty exclusion index")
            fields = set(identities("schema-example"))
            require(
                all(isinstance(row, dict) and fields <= row.keys() for row in records),
                "incomplete exact/normalized raw/rendered identities",
            )
            values = [row[field] for row in records for field in fields]
            require(
                all(isinstance(v, str) and HEX.fullmatch(v) for v in values),
                "invalid exclusion identity hash",
            )
        else:
            texts = list(_texts(document))
            require(texts, "prior payload contains no explicit prompt/message records")
            records = [identities(text) for text in texts]
            values = [v for row in records for v in row.values()]
        record_count += len(records)
        require(record_count <= RECORD_LIMIT, "exclusion records exceed bounded metadata budget")
        hashes.update(values)
        receipts.append({**item, "records": len(records)})
    return hashes, receipts


def _python(family, rng, function):
    if family == "prefix_sums":
        samples = [[], [2, 3, -1], [0], [-2, 5, -4]]
        samples += [[rng.randrange(-5, 6) for _ in range(rng.randrange(1, 9))] for _ in range(8)]
        tests = [
            {"args": [x], "expected": [sum(x[: i + 1]) for i in range(len(x))]} for x in samples
        ]
        desc = "return the inclusive prefix sums of an integer list"
        good = (
            f"def {function}(x):\n    out=[]\n    total=0\n"
            "    for v in x:\n        total+=v\n        out.append(total)\n    return out\n"
        )
        bad = [
            f"def {function}(x):\n    return x\n",
            f"def {function}(x):\n    return [sum(x) for v in x]\n",
        ]
    elif family == "transpose":
        samples = [[], [[1, 2, 3], [4, 5, 6]], [[7]], [[1], [2], [3]]]
        samples += [[[rng.randrange(-9, 10) for _ in range(3)] for _ in range(2)] for _ in range(6)]
        tests = [
            {"args": [x], "expected": [[r[j] for r in x] for j in range(len(x[0]))] if x else []}
            for x in samples
        ]
        desc = "transpose a rectangular matrix (list of integer rows); [] returns []"
        good = f"def {function}(x):\n    return [list(col) for col in zip(*x)]\n"
        bad = [
            f"def {function}(x):\n    return x\n",
            f"def {function}(x):\n    return [list(col) for col in zip(*x[::-1])]\n",
        ]
    else:
        samples = [([1, 2, 2, 4], 2), ([], 3), ([2, 4], 5), ([2, 4], 0)]
        samples += [
            (sorted(rng.randrange(-5, 6) for _ in range(7)), rng.randrange(-6, 7)) for _ in range(8)
        ]
        tests = [
            {"args": [{"values": x, "key": k}], "expected": sum(v < k for v in x)}
            for x, k in samples
        ]
        desc = (
            "take one dict with sorted integer list values and integer key; return the "
            "first index whose value is >=key, or len(values) if none"
        )
        good = (
            f"def {function}(x):\n    a=x['values']\n    k=x['key']\n    lo=0\n    hi=len(a)\n"
            "    while lo<hi:\n        mid=(lo+hi)//2\n        if a[mid]<k:\n"
            "            lo=mid+1\n        else:\n            hi=mid\n    return lo\n"
        )
        bad = [
            f"def {function}(x):\n    return sum(v<=x['key'] for v in x['values'])\n",
            f"def {function}(x):\n    return 0\n",
        ]
    prompt = (
        f"Write only a Python function {function}(x) to {desc}. Do not mutate input. "
        "Use no imports, module-level code, decorators or default arguments. "
        "Return one function, optionally in one Python fence."
    )
    return {"kind": "python", "function": function, "tests": tests, "prompt": prompt}, good, bad


def _sql(rng, tag):
    ready, blocked, count = "ready_" + tag, "blocked_" + tag, rng.choice((2, 3))
    data = []
    for customer, amount in ((11, count + 1), (22, count), (33, count - 1), (55, count)):
        for j in range(amount):
            data.append([len(data) + 1, customer, ready, rng.randrange(10, 20) + j * 20])
    data.append([len(data) + 1, 22, blocked, 0])
    data.append([len(data) + 1, 44, "other_" + tag, 1])
    # A NULL blocker must not accidentally exclude every non-NULL customer.
    data.append([len(data) + 1, None, blocked, 0])
    banned = {r[1] for r in data if r[2] == blocked}
    expected = []
    for customer in sorted({r[1] for r in data if r[1] is not None}):
        amounts = [r[3] for r in data if r[1] == customer and r[2] == ready]
        if customer not in banned and len(amounts) >= count:
            expected.append([customer, min(amounts)])
    blocked_filter = (
        " AND NOT EXISTS (SELECT 1 FROM orders AS blocked_order "
        f"WHERE blocked_order.customer_id=o.customer_id AND blocked_order.status='{blocked}')"
    )
    where = f"status='{ready}'" + blocked_filter
    good = (
        f"SELECT customer_id,MIN(total) FROM orders AS o WHERE {where} GROUP BY customer_id "
        f"HAVING COUNT(*)>={count} ORDER BY customer_id"
    )
    bad = [
        good.replace("MIN(total)", "MAX(total)"),
        good.replace(blocked_filter, ""),
        good.replace(
            blocked_filter,
            f" AND customer_id NOT IN (SELECT customer_id FROM orders WHERE status='{blocked}')",
        ),
    ]
    prompt = (
        "SQLite orders(id INTEGER PRIMARY KEY,customer_id INTEGER,status TEXT,total INTEGER) "
        f"has nonnegative totals. Return customer_id and MIN(total) over status='{ready}' "
        f"for customers with at least {count} such rows and no row with status='{blocked}'. "
        "Order by customer_id ascending. Return one SELECT query only, optionally fenced."
    )
    return (
        {
            "kind": "sql",
            "schema": (
                "CREATE TABLE orders(id INTEGER PRIMARY KEY,customer_id INTEGER,"
                "status TEXT,total INTEGER)"
            ),
            "data": data,
            "expected": expected,
            "prompt": prompt,
        },
        good,
        bad,
    )


def _world(family, rng, tag):
    if family == "finite_sets":
        universe = list(range(1, 15))
        a, b = set(rng.sample(universe, 7)), set(rng.sample(universe, 8))
        expected = {"union": len(a | b), "intersection": len(a & b), "only_a": len(a - b)}
        cross = {
            "union": sum(x in a or x in b for x in universe),
            "intersection": sum(x in a and x in b for x in universe),
            "only_a": sum(x in a and x not in b for x in universe),
        }
        require(expected == cross, "set reference disagreement")
        text = (
            f"Sets A={sorted(a)} and B={sorted(b)}. "
            "Count their union, intersection and A-only elements."
        )
    elif family == "calendar":
        days, a, b = rng.randrange(35, 55), rng.choice((3, 4)), rng.choice((5, 7))
        ra, rb = rng.randrange(a), rng.randrange(b)
        matches = [d for d in range(days) if d % a == ra and d % b == rb]
        cross = sorted(set(range(ra, days, a)) & set(range(rb, days, b)))
        require(matches == cross, "calendar reference disagreement")
        expected = {"days": matches, "count": len(matches)}
        text = (
            f"Days are integers 0 through {days - 1}. List days with "
            f"d modulo {a}={ra} and d modulo {b}={rb}, plus their count."
        )
    elif family == "dag_route":
        edges = {(i, i + 1): rng.randrange(1, 10) for i in range(4)}
        edges.update({(i, j): rng.randrange(4, 20) for i in range(5) for j in range(i + 2, 5)})
        paths = []

        def visit(node, cost, route):
            if node == 4:
                paths.append((cost, route))
            for (a, b), weight in edges.items():
                if a == node:
                    visit(b, cost + weight, route + [b])

        visit(0, 0, [0])
        cost, route = min(paths)
        distance = {0: 0}
        for j in range(1, 5):
            distance[j] = min(distance[i] + edges[i, j] for i in range(j) if (i, j) in edges)
        require(cost == distance[4], "route references disagree")
        expected = {"cost": cost, "path": route}
        text = (
            f"Directed edges (start,end,cost)={[(a, b, w) for (a, b), w in edges.items()]}. "
            "Find the cheapest path 0 to 4, breaking ties by lexicographically "
            "smallest full node list."
        )
    elif family == "assignment":
        matrix = [[rng.randrange(2, 20) for _ in range(3)] for _ in range(3)]
        cost, workers = min(
            (sum(matrix[t][p[t]] for t in range(3)), p) for p in itertools.permutations(range(3))
        )
        dp = {0: 0}
        for task in range(3):
            nxt = {}
            for mask, value in dp.items():
                for worker in range(3):
                    if not mask & (1 << worker):
                        new = mask | (1 << worker)
                        nxt[new] = min(nxt.get(new, 10**9), value + matrix[task][worker])
            dp = nxt
        require(cost == dp[7], "assignment references disagree")
        expected = {"cost": cost, "workers": list(workers)}
        text = (
            f"Cost rows for tasks 0,1,2 and columns for workers 0,1,2 are {matrix}. "
            "Use each worker once; minimize total cost, tie-breaking by lexicographically "
            "smallest worker list in task order."
        )
    elif family == "chronology":
        events = [(day, f"E{tag}_{i}") for i, day in enumerate(rng.sample(range(1, 29), 4))]
        rng.shuffle(events)
        ordered = [name for day, name in sorted(events)]
        expected = {"event_ids": ordered, "latest": ordered[-1]}
        text = (
            f"Fictional dated observations (day,event_id)={events}. "
            "List all event IDs chronologically and identify the latest."
        )
    elif family == "obligations":
        records = [(f"O{tag}_{i}", "must" if i % 2 else "may") for i in range(6)]
        rng.shuffle(records)
        expected = {
            "required": [name for name, verb in records if verb == "must"],
            "optional": [name for name, verb in records if verb == "may"],
        }
        text = (
            f"Fictional policy clauses (ID,verb)={records}. 'must' is required; 'may' is optional. "
            "List IDs by category, preserving source order."
        )
    elif family == "correction":
        old, new = rng.randrange(2, 20), rng.randrange(30, 60)
        expected = {"quantity": new, "source_id": "official_" + tag}
        text = (
            f"Fictional draft note draft_{tag} states quantity={old}. Later official correction "
            f"official_{tag} states quantity={new}. Official corrections override drafts. "
            "Return the authoritative quantity and source ID."
        )
    else:
        expected = {"owner": "owner_" + tag, "recovery_minutes": None}
        text = (
            f"Fictional facts: owner=owner_{tag}; region=region_{rng.randrange(10, 99)}; "
            f"retention_days={rng.randrange(2, 14)}. Recovery duration was not recorded. "
            "Return owner and recovery_minutes; use JSON null for an unavailable fact "
            "and do not invent it."
        )
    return {
        "kind": "json",
        "expected": expected,
        "prompt": text
        + f" Return only JSON with exactly fields {list(expected)} and the values requested.",
    }


def _wrong_json(expected, version):
    wrong = copy.deepcopy(expected)
    if isinstance(wrong, list):
        wrong[0][version % len(wrong[0])] += 1
    else:
        key = list(wrong)[version % len(wrong)]
        value = wrong[key]
        if type(value) is int:
            wrong[key] = value + 1
        elif isinstance(value, list):
            if len(value) > 1 and value != list(reversed(value)):
                wrong[key] = list(reversed(value))
            else:
                wrong[key] = value + (
                    [999] if not value or type(value[0]) is int else ["unsupported_fact"]
                )
        elif value is None:
            wrong[key] = 42
        else:
            wrong[key] = value + "_unsupported"
    return json.dumps(wrong)


def _instance(seed, domain, family, phase, index):
    token = digest(f"{seed}/{family}/{phase}/{index}".encode())
    rng = random.Random(int(token, 16))
    tag = token[:10]
    case_id = f"fresh-{phase}-{family}-{index}"
    if family in ("prefix_sums", "transpose", "lower_bound"):
        case, good, bad = _python(family, rng, "fn_" + tag)
    elif family == "sql_unblocked":
        case, good, bad = _sql(rng, tag)
    else:
        case = _world(family, rng, tag)
        good = json.dumps(case["expected"])
        bad = [_wrong_json(case["expected"], v) for v in (0, 1)]
    # Reference marker prevents accidental exact reuse; it is not a novelty claim.
    case["prompt"] += f"\nInstance reference: {tag}."
    case.update(
        id=case_id,
        domain=domain,
        family=family,
        group_id=f"fresh-v1:{token}",
        source="project_synthetic_after_candidate_freeze_v1",
        phase=phase,
        instance_seed=token,
        identities=identities(case["prompt"]),
    )
    if phase == "performance":
        if case["kind"] == "json":
            case["prompt"] = case["prompt"].split(" Return only JSON with exactly fields")[0]
        else:
            case["prompt"] = case["prompt"].replace("Write only", "Write")
            case["prompt"] = case["prompt"].replace(
                "Return one function, optionally in one Python fence.",
                "Include the function implementation.",
            )
            case["prompt"] = case["prompt"].replace(
                "Return one SELECT query only, optionally fenced.", "Include the SELECT query."
            )
        case["prompt"] += f" Explain the result or approach briefly. Instance reference: {tag}."
        case["identities"] = identities(case["prompt"])
    return case, good, bad


def prepare(freeze_path, output):
    """Validate selection/source first; preserve complete prepared/inconclusive packets."""
    output = Path(output)
    require(not output.exists(), "refusing to replace existing confirmation inputs")
    freeze, freeze_pin, judge = validate_freeze(freeze_path)
    excluded, exclusion_receipts = load_exclusions(freeze["exclusions"])
    frozen_bytes = Path(freeze_path).read_bytes()
    require(digest(frozen_bytes) == freeze_pin["sha256"], "freeze changed before generation")
    seed = digest(frozen_bytes + b"modern4b-confirmation-v1")
    requests, cases, controls, collisions, groups = [], [], [], [], []
    observed = set()
    for domain, families in FAMILIES.items():
        for family in families:
            for phase in ("performance", "quality"):
                for index in range(2):
                    case, good, bad = _instance(seed, domain, family, phase, index)
                    ids = set(case["identities"].values())
                    overlap = bool(ids & (excluded | observed))
                    if overlap:
                        collisions.append(case["id"])
                    observed.update(ids)
                    request = {
                        "prompt": render(case["prompt"]),
                        "n_predict": freeze["settings"]["output_limit"],
                        "temperature": 0,
                        "seed": 1234,
                        "cache_prompt": False,
                        "return_tokens": True,
                        "stream": False,
                    }
                    groups.append(
                        {
                            k: case[k]
                            for k in (
                                "id",
                                "domain",
                                "family",
                                "group_id",
                                "source",
                                "phase",
                                "instance_seed",
                                "identities",
                            )
                        }
                    )
                    if phase == "quality":
                        case["request"] = request
                        positive = judge.judge(case, good)
                        negatives = [judge.judge(case, answer) for answer in bad]
                        controls.append(
                            {
                                "case_id": case["id"],
                                "correct_control_passed": positive["passed"],
                                "known_wrong_control_rejected": all(
                                    not r["passed"] for r in negatives
                                ),
                                "positive": {"answer": good, "judgement": positive},
                                "semantic_negatives": [
                                    {"answer": a, "judgement": r} for a, r in zip(bad, negatives)
                                ],
                            }
                        )
                        cases.append(case)
                    else:
                        requests.append(
                            {
                                **groups[-1],
                                "prompt": request["prompt"],
                                "sha256": digest(request["prompt"].encode()),
                                "request": request,
                            }
                        )
    require(
        len(requests) == len(cases) == len(controls) == 24 and len(groups) == 48,
        "family/phase counts differ",
    )
    validated = all(
        c["correct_control_passed"] and c["known_wrong_control_rejected"] for c in controls
    )
    admitted = validated and not collisions
    status = "PREPARED_NO_MODEL_EVALUATION" if admitted else "INCONCLUSIVE_INPUTS_OR_ORACLE"
    manifest = {
        "schema": "modern_fresh_confirmation_inputs_v1",
        "status": status,
        "admitted": admitted,
        "freeze": freeze_pin,
        "source_freeze": freeze["source_freeze"],
        "candidate": freeze["candidate"],
        "runtime": freeze["runtime"],
        "generator": freeze["generator"],
        "judge": freeze["judge"],
        "seed": seed,
        "groups": groups,
        "exclusions": exclusion_receipts,
        "overlap_case_ids": collisions,
        "settings": freeze["settings"],
        "rendering": {"system": SYSTEM, "thinking": "forced_empty", "interface": "/completion"},
        "model_requests_executed": 0,
        "tokenization_performed": False,
        "requirements": [
            "Runner must require admitted and authenticate all pins.",
            "Native-tokenize and freeze geometry without truncating original tasks.",
            "Send only request bodies; never expected/tests/controls to the model.",
            "Preserve all short/incorrect/inconclusive outcomes; no replacement.",
        ],
        "limits": (
            "Exact/normalized text exclusion only; no pretraining novelty or broad prose-quality "
            "claim. Grounded oracles check factual content; families are related clusters."
        ),
    }
    payloads = {
        "requests.json": requests,
        "quality-cases.json": cases,
        "quality-controls.json": controls,
        "manifest.json": manifest,
    }
    blobs = {name: encoded(value) for name, value in payloads.items()}
    require(sum(map(len, blobs.values())) <= OUTPUT_LIMIT, "generated packet exceeds disk budget")
    # Reauthenticate every consumed source before publication, never model/weights.
    require(reference(freeze_path) == freeze_pin, "selection freeze changed")
    for ref in [
        freeze["source_freeze"],
        freeze["generator"],
        freeze["judge"],
        *(item["file"] for item in freeze["exclusions"]),
    ]:
        checked(ref, suffix=Path(ref["path"]).suffix)
    receipt = {
        "schema": "modern_fresh_confirmation_receipt_v1",
        "status": status,
        "admitted": admitted,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "manifest_sha256": digest(blobs["manifest.json"]),
        "freeze": freeze_pin,
        "seed": seed,
        "performance_prompts": 24,
        "quality_tasks": 24,
        "quality_templates": 12,
        "quality_correct_controls_passed": sum(c["correct_control_passed"] for c in controls),
        "quality_known_wrong_controls_rejected": sum(
            c["known_wrong_control_rejected"] for c in controls
        ),
        "input_overlap_count": len(collisions),
        "model_requests_executed_before_freeze": 0,
        "model_requests_executed": 0,
        "files": {n: digest(b) for n, b in blobs.items()},
        "generated_bytes": sum(map(len, blobs.values())),
    }
    output.mkdir(parents=True, exist_ok=False)
    for name, blob in blobs.items():
        with (output / name).open("xb") as stream:
            stream.write(blob)
    with (output / "receipt.json").open("xb") as stream:
        stream.write(encoded(receipt))
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    prep = subs.add_parser("prepare")
    prep.add_argument("--freeze", required=True, type=Path)
    prep.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = prepare(args.freeze, args.output)
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "status",
                    "admitted",
                    "performance_prompts",
                    "quality_tasks",
                    "input_overlap_count",
                )
            }
        )
    )
    raise SystemExit(0 if result["admitted"] else 2)


if __name__ == "__main__":
    main()

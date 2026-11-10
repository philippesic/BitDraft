import math
from collections import Counter
from dataclasses import dataclass
from typing import TypeAlias, cast

Record: TypeAlias = dict[str, object]


@dataclass(frozen=True, order=True)
class Diagnostic:
    code: str
    path: str


def record(value: object) -> Record:
    if not isinstance(value, dict) or not all(isinstance(k, str) for k in value):
        raise ValueError("expected object")
    return cast(Record, value)


def sequence(value: object) -> list[object]:
    if not isinstance(value, list):
        raise ValueError("expected list")
    return cast(list[object], value)


def integer(value: object) -> int:
    if type(value) is not int:
        raise ValueError("expected integer")
    return value


def number(value: object) -> float:
    if type(value) not in (float, int):
        raise ValueError("expected finite number")
    result = float(cast(float, value))
    if not math.isfinite(result):
        raise ValueError("expected finite number")
    return result


def string(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("expected nonempty string")
    return value


def components(rows: list[Record]) -> list[list[str]]:
    parent: dict[str, str] = {}
    keys: dict[tuple[str, str, str], str] = {}
    for row in rows:
        identity = string(row["id"])
        if identity in parent:
            raise ValueError("duplicate row id")
        parent[identity] = identity

    def find(identity: str) -> str:
        while parent[identity] != identity:
            identity = parent[identity]
        return identity

    for row in rows:
        identity = string(row["id"])
        source = string(row["source"])
        for namespace in ("group", "topic"):
            value = row.get(namespace)
            if value is None and namespace == "topic":
                continue
            if not isinstance(value, str):
                raise ValueError("malformed relation")
            value = value.strip()
            if not value:
                if namespace == "topic":
                    continue
                raise ValueError("empty group")
            key = source, namespace, value
            if key in keys:
                a, b = find(identity), find(keys[key])
                parent[max(a, b)] = min(a, b)
            else:
                keys[key] = identity
    groups: dict[str, list[str]] = {}
    for identity in sorted(parent):
        groups.setdefault(find(identity), []).append(identity)
    return sorted(groups.values())


def exposure(windows: list[list[str]], visits: list[int]) -> tuple[int, float, int, float]:
    if not windows or len(windows) != len(visits) or any(n < 0 for n in visits):
        raise ValueError("invalid schedule")
    counts: Counter[str] = Counter()
    for window, count in zip(windows, visits, strict=True):
        if not window or len(window) != len(set(window)):
            raise ValueError("invalid window")
        for position in window:
            counts[position] += count
    return (
        sum(counts.values()),
        sum(counts.values()) / len(counts),
        max(counts.values()),
        sum(visits) / len(windows),
    )


def best_serving(scores: list[Record]) -> str | None:
    best: str | None = None
    score = -math.inf
    for row in scores:
        if (
            row.get("kind") != "serving"
            or row.get("complete") is not True
            or row.get("durable") is not True
        ):
            continue
        try:
            candidate = number(row["score"])
        except (ValueError, KeyError):
            continue
        if candidate > score:
            score, best = candidate, string(row["id"])
    return best


def role_allowed(row: Record, protected: set[str]) -> bool:
    identities: set[str] = {string(v) for v in sequence(row["aliases"])}
    identities.add(string(row["id"]))
    return row["role"] == "heldout" or not protected.intersection(identities)


def argmax(row: list[float]) -> int:
    if not row:
        raise ValueError("empty logits")
    best = 0
    for i in range(1, len(row)):
        if row[i] > row[best]:
            best = i
    return best


COMPARE_FIELDS = {
    "compare_clock": ("clock", "aggregation", "warmups"),
    "compare_era": ("era",),
    "compare_machine": ("machine", "software"),
    "compare_target": ("target", "tokenizer", "template", "family"),
    "compare_workload": ("workload", "prompts"),
    "compare_length": ("cap", "input_lengths"),
    "compare_role": ("role", "consumption"),
    "compare_repeats": ("processes", "attempts", "order"),
    "compare_binary": ("binary", "runtime"),
    "compare_instrument": ("instrumentation",),
    "compare_depth": ("depth",),
    "compare_kv": ("kv", "cache"),
    "compare_generation": ("seed", "greedy", "eog", "context", "batch", "concurrency"),
}


def comparison_value(key: str, value: object) -> bool:
    if key in {"warmups", "cap", "depth", "seed", "context", "batch", "concurrency"}:
        return type(value) is int and integer(value) >= (0 if key in {"warmups", "seed"} else 1)
    if key == "greedy":
        return type(value) is bool
    if key in {"prompts", "input_lengths", "processes", "attempts", "order", "eog"}:
        values = sequence(value)
        return bool(values) and len({str(v) for v in values}) == len(values)
    if key == "clock":
        return value in {
            "complete_request_http_wall",
            "server_decode_wall",
            "host_traced_draft_wall",
            "gpu_draft_event",
            "synchronized_training_update_wall",
        }
    if key == "era":
        return (
            value in {"historical_dspark", "modern_dflash_runtime04"}
            or isinstance(value, str)
            and value.startswith("new:")
        )
    return bool(string(value))


def validate_contract(rule: str, data: Record) -> list[Diagnostic]:
    valid = False
    try:
        if rule in {"null_topics", "topic_namespace", "group_properties"}:
            rows = [record(v) for v in sequence(data["rows"])]
            actual = components(rows)
            valid = actual == data["expected"]
            if rule == "group_properties":
                valid = (
                    valid
                    and components(list(reversed(rows))) == actual
                    and components(rows + [record(data["unrelated"])]) == data["extended"]
                )
        elif rule == "topic_type":
            components([record(v) for v in sequence(data["rows"])])
            valid = True
        elif rule == "heldout_role":
            protected = {string(v) for v in sequence(data["protected"])}
            valid = all(role_allowed(record(v), protected) for v in sequence(data["rows"]))
        elif rule == "unknown_ancestry":
            valid = all(
                row["role"] != "fresh-eval" or row["ancestry"] == "verified-untrained"
                for row in (record(v) for v in sequence(data["rows"]))
            )
        elif rule == "count_intent":
            valid = integer(data["minimum"]) <= integer(data["actual"]) <= integer(data["maximum"])
        elif rule == "count_conservation":
            inputs = [string(v) for v in sequence(data["inputs"])]
            terminal = [string(record(v)["id"]) for v in sequence(data["terminal"])]
            valid = (
                len(inputs) == len(set(inputs))
                and sorted(inputs) == sorted(terminal)
                and all(
                    string(record(v)["reason"]) in {"admitted", "dev", "collision", "bug"}
                    for v in sequence(data["terminal"])
                )
            )
        elif rule == "count_unique":
            unique_positions = {
                (string(record(v)["trajectory"]), integer(record(v)["position"]))
                for v in sequence(data["labels"])
            }
            valid = len(unique_positions) == integer(data["claimed"])
        elif rule == "capture_offset":
            tokens = [integer(v) for v in sequence(data["tokens"])]
            valid = all(
                integer(record(v)["label"]) == tokens[integer(record(v)["position"]) + 1]
                for v in sequence(data["labels"])
            )
        elif rule == "capture_eog":
            tokens = [integer(v) for v in sequence(data["tokens"])]
            stop = (
                tokens.index(integer(data["eog"]))
                if integer(data["eog"]) in tokens
                else len(tokens)
            )
            positions = [integer(v) for v in sequence(data["positions"])]
            anchors = [integer(v) for v in sequence(data["anchors"])]
            valid = all(0 <= p < stop for p in positions) and anchors == [
                p for p in positions if all(p + j in positions for j in range(15))
            ]
        elif rule == "exposure_schedule":
            windows = [[string(p) for p in sequence(w)] for w in sequence(data["windows"])]
            total, mean, maximum, passes = exposure(
                windows, [integer(v) for v in sequence(data["visits"])]
            )
            valid = (
                total == integer(data["total"])
                and mean <= number(data["mean_limit"])
                and maximum <= integer(data["max_limit"])
                and passes <= number(data["pass_limit"])
            )
        elif rule == "exposure_parent":
            parent, child = record(data["parent"]), record(data["child"])
            valid = integer(child["exposure"]) == integer(parent["exposure"]) + integer(
                data["added"]
            ) and all(parent[k] == child[k] for k in ("seed", "cursor", "trajectory"))
        elif rule in {"disk_reserve", "disk_backing"}:
            observations = [record(v) for v in sequence(data["devices"])]
            valid = all(
                integer(v["free"]) >= integer(v["peak"]) + integer(v["reserve"])
                for v in observations
            )
            if rule == "disk_backing":
                valid = (
                    valid
                    and {v["kind"] for v in observations} == {"wsl", "backing"}
                    and len({v["provider"] for v in observations}) == 2
                    and data["combined"] is False
                )
        elif rule in {"selection_peak", "selection_tie", "selection_partial"}:
            valid = best_serving([record(v) for v in sequence(data["scores"])]) == data["selected"]
        elif rule in COMPARE_FIELDS:
            left, right = record(data["left"]), record(data["right"])
            allowed = record(data["allowed_differences"])
            valid = all(
                k in left
                and k in right
                and comparison_value(k, left[k])
                and comparison_value(k, right[k])
                and (left[k] == right[k] or allowed.get(k) == [left[k], right[k]])
                for k in COMPARE_FIELDS[rule]
            )
            permitted = {"family", "kv", "cache"}
            if data.get("comparison_kind") == "runtime-system":
                permitted.update({"binary", "runtime"})
            valid = valid and set(allowed).issubset(permitted)
            if rule == "compare_repeats":
                valid = (
                    valid
                    and left.get("planned_processes") == left["processes"] == left["attempts"]
                    and right.get("planned_processes") == right["processes"] == right["attempts"]
                )
        elif rule == "compare_correctness":
            records = [record(v) for v in sequence(data["requests"])]
            valid = bool(records) and all(
                v["tokens"] == v["reference_tokens"]
                and integer(v["proposed"]) > 0
                and number(v["wall"]) > 0
                for v in records
            )
            total_tokens = sum(len(sequence(v["tokens"])) for v in records)
            valid = valid and math.isclose(
                number(data["throughput"]),
                total_tokens / sum(number(v["wall"]) for v in records),
                rel_tol=1e-12,
            )
        elif rule == "verify_missing":
            valid = (
                data.get("scope") == "synthetic-schema"
                and data.get("qualified") is False
                and bool(data.get("raw"))
            )
        elif rule == "verify_token":
            left, right = record(data["target"]), record(data["candidate"])
            valid = bool(sequence(left["tokens"])) and all(
                left[k] == right[k] for k in ("tokens", "text", "finish")
            )
        elif rule == "verify_label":
            logits = [[number(v) for v in sequence(row)] for row in sequence(data["logits"])]
            valid = bool(logits) and [argmax(row) for row in logits] == data["labels"]
        elif rule == "verify_activation":
            valid = (
                integer(data["proposed"]) > 0
                and data["loader"] == data["expected_loader"]
                and data["taps"] == [2, 6, 10, 14, 18, 22, 26, 30]
                and data["semantics"] == "zero_based_pre_block_input_all_tokens"
                and data["teacher"] == data["expected_teacher"]
                and data["shapes"] == [[integer(data["token_count"]), 2560] for _ in range(8)]
                and 0 < integer(data["token_count"]) <= 48
                and all(math.isfinite(number(v)) for v in sequence(data["logits"]))
            )
        elif rule == "owner_record":
            records = [record(v) for v in sequence(data["records"])]
            valid = all(v["owner"] == data["owner"] for v in records) and len(
                {(string(v["run"]), integer(v["sequence"])) for v in records}
            ) == len(records)
        elif rule in {"discovery_name", "discovery_duplicate", "discovery_schema"}:
            extensions = [record(v) for v in sequence(data["extensions"])]
            if rule == "discovery_name":
                valid = all(
                    string(v["path"]).endswith(("_adapter.py", "_workload.py", "_dataset.py"))
                    for v in extensions
                )
            elif rule == "discovery_duplicate":
                valid = len({string(v["id"]) for v in extensions}) == len(extensions)
            else:
                valid = all(
                    v["protocol"] == "1"
                    and len(string(v["digest"])) == 64
                    and all(c in "0123456789abcdef" for c in string(v["digest"]))
                    and bool(string(v["schema"]))
                    for v in extensions
                )
        else:
            raise ValueError("unknown contract")
    except (KeyError, ValueError, TypeError, IndexError, ZeroDivisionError):
        valid = False
    return [] if valid else [Diagnostic(rule.upper(), "input.json")]

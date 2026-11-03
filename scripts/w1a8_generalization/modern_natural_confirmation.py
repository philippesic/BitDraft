"""Post-freeze natural transfer confirmation: 24 unused corpus prompts + 24 oracles.

Preparation never loads models or runs inference. Every existing exclusion remains
binding, including all 512 historical TRAIN rows. Source catalogs and opaque
partition indices must be frozen before opening candidate prompt rows. Only
singleton source components are eligible: one complete prompt per independent
performance cluster, eight per domain. Native context admission is inherited
from modern_confirm before any completion; no truncation or replacement.

Freeze schema is FREEZE_SCHEMA below. natural_sources requires catalog,
partition_manifests, prior_selection, prior_inventory, current_union,
original_roster, consumed_manifest, consumed_receipt and consumed_marker pins.
Partition manifests use existing project `files` entries; every partition except
unexposed train_reserve requires all its opaque .index.jsonl files. Quality uses
the pinned original generator and judge, with no broad prose/novelty claim.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import modern_confirm as confirm
import modern_fresh as fresh
from modern_requests import atomic, read, require, signature

FREEZE_SCHEMA = "modern_natural_confirmation_freeze_v1"
MANIFEST_SCHEMA = "modern_natural_confirmation_inputs_v1"
RECEIPT_SCHEMA = "modern_natural_confirmation_receipt_v1"
CATALOG_SCHEMA = "w1a_source_catalog_v1"
MAX_FILE = 64 * 1024**2
MAX_METADATA = 192 * 1024**2
MAX_ROWS = 100000
PROJECT = Path("/home/philip/binary-eagle-decoding")
KNOWN_CATALOG = PROJECT / "data/w1a-public-sources/pinned-v1/catalog.json"
REQUIRED_PARTITIONS = (
    PROJECT / "data/w1a-public-freeze-002/manifest.json",
    PROJECT / "data/continuous-w1ax/freeze-002/manifest.json",
)
DOMAIN_COUNTS = {"code": 8, "reasoning": 8, "prose": 8}
SOURCE_FIELDS = {
    "catalog",
    "partition_manifests",
    "prior_selection",
    "prior_inventory",
    "current_union",
    "original_roster",
    "consumed_manifest",
    "consumed_receipt",
    "consumed_marker",
}


def reference(path):
    path = Path(path).resolve(strict=True)
    require(
        path.suffix in (".json", ".jsonl", ".py") and path.stat().st_size <= MAX_FILE,
        "bounded metadata/source only; model and array access forbidden",
    )
    before = path.stat()
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    after = path.stat()
    require(
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
        "source changed while hashing",
    )
    return {"path": str(path), "bytes": after.st_size, "sha256": digest}


class Metadata:
    def __init__(self):
        self.pins = {}
        self.bytes = 0

    def checked(self, ref):
        fresh.opaque(ref)
        actual = reference(ref["path"])
        require(
            actual == {key: ref[key] for key in ("path", "bytes", "sha256")}, "source pin changed"
        )
        previous = self.pins.get(actual["path"])
        require(previous is None or previous == actual, "conflicting source pins")
        if previous is None:
            self.bytes += actual["bytes"]
            require(self.bytes <= MAX_METADATA, "total metadata budget exceeded")
        self.pins[actual["path"]] = actual
        return Path(actual["path"])

    def read(self, ref):
        path = self.checked(ref)
        require(path.suffix == ".json", "JSON metadata required")
        value = json.loads(path.read_bytes())
        require(reference(path) == self.pins[str(path)], "source changed while reading")
        return value

    def rows(self, ref, *, index=False):
        path = self.checked(ref)
        require(
            path.suffix == ".jsonl" and (not index or path.name.endswith(".index.jsonl")),
            "only explicit JSONL source or opaque partition index allowed",
        )
        with path.open("rb") as stream:
            for number, raw in enumerate(stream, 1):
                require(
                    number <= MAX_ROWS and len(raw) <= fresh.FILE_LIMIT, "JSONL row budget exceeded"
                )
                if raw.strip():
                    yield number, raw, json.loads(raw)
        require(reference(path) == self.pins[str(path)], "JSONL source changed while reading")

    def recheck(self):
        for ref in self.pins.values():
            require(reference(ref["path"]) == ref, "consumed source changed before publication")


def validate_freeze(path):
    """Candidate + every serving arm/runtime/source bind before prompt metadata access."""
    audit = Metadata()
    freeze_pin = reference(path)
    data = audit.read(freeze_pin)
    require(
        data.get("schema") == FREEZE_SCHEMA
        and data.get("selection_frozen") is True
        and data.get("evaluation_inputs_generated") is False,
        "candidate/runtime selection must freeze before any new inputs",
    )
    stamp = datetime.fromisoformat(data.get("selected_at_utc", "").replace("Z", "+00:00"))
    require(
        stamp.tzinfo is not None and stamp <= datetime.now(UTC), "invalid candidate selection time"
    )
    for key in ("candidate", "runtime"):
        fresh.opaque(data.get(key))
    source = audit.read(data["source_freeze"])
    for key in ("target", "baseline"):
        fresh.opaque(source.get(key))
    arms = source.get("arms", [])
    require(
        4 <= len(arms) <= 8
        and len({a["id"] for a in arms}) == len(arms)
        and arms[0]["kind"] == "target"
        and sum(a["kind"] == "target" for a in arms) == 1
        and sum(a["kind"] == "dflash" for a in arms) == 1,
        "complete target/DFlash/candidate serving model freeze required",
    )
    selected = [a for a in arms if a["id"] == data.get("selected_arm_id")]
    require(
        len(selected) == 1
        and selected[0]["kind"] == "w1a8"
        and selected[0]["model"] == data["candidate"]
        and selected[0]["runtime"]["server"] == data["runtime"],
        "selected arm/runtime differs",
    )
    for arm in arms:
        require(
            arm["kind"] in ("target", "dflash", "w1a8", "eagle_q4"), "unknown frozen serving arm"
        )
        if arm["model"] is not None:
            fresh.opaque(arm["model"])
        runtime = arm.get("runtime", {})
        require(
            runtime.get("libraries") and runtime.get("environment") and runtime.get("source_pins"),
            "every arm must freeze complete runtime libraries/environment/sources",
        )
        for ref in [runtime["server"], *runtime["libraries"], *runtime["source_pins"]]:
            fresh.opaque(ref)  # opaque: never open models or executable/library payloads
    for key, module in (
        ("generator", Path(__file__)),
        ("quality_generator", Path(fresh.__file__)),
        ("confirmation_runner", Path(confirm.__file__)),
    ):
        require(
            Path(data[key]["path"]).resolve() == module.resolve(), "wrong generator/runner source"
        )
        audit.checked(data[key])
    require(data.get("settings") == {"output_limit": 256}, "freeze unchanged256-token protocol")
    sources = data.get("natural_sources", {})
    require(
        set(sources) == SOURCE_FIELDS and sources["partition_manifests"],
        "complete natural catalog/exclusion/partition ancestry required",
    )
    # Check identities only here; do not open any candidate source before freeze validation.
    for key, ref in sources.items():
        for value in ref if key == "partition_manifests" else [ref]:
            fresh.opaque(value)
    if Path(sources["catalog"]["path"]).resolve() == KNOWN_CATALOG.resolve():
        require(
            {str(p.resolve()) for p in REQUIRED_PARTITIONS}
            <= {str(Path(ref["path"]).resolve()) for ref in sources["partition_manifests"]},
            "known master catalog requires public and continuous opaque partition inventories",
        )
    judge = fresh.load_judge(data["judge"])
    audit.checked(data["judge"])
    audit.recheck()
    return data, freeze_pin, judge


def _identity_hashes(records):
    fields = set(fresh.identities("schema-example"))
    require(
        isinstance(records, list)
        and records
        and all(
            isinstance(r, dict)
            and fields <= r.keys()
            and all(isinstance(r[k], str) and fresh.HEX.fullmatch(r[k]) for k in fields)
            for r in records
        ),
        "complete exact/NFKC identity records required",
    )
    return {r[key] for r in records for key in fields}


def exclusion_evidence(freeze, audit):
    """No quality answer payloads: use prior authenticated prompt hash inventories."""
    sources = freeze["natural_sources"]
    prior = audit.read(sources["prior_selection"])
    inventory = audit.read(sources["prior_inventory"])
    require(
        prior.get("schema") == fresh.SCHEMA
        and prior.get("exclusion_inventory") == sources["prior_inventory"]
        and inventory.get("schema") == "modern_confirmation_explicit_prior_input_inventory_v1"
        and prior.get("candidate_selection") == inventory.get("candidate_selection"),
        "prior selection/inventory ancestry differs",
    )
    exclusions = prior.get("exclusions", [])
    require(
        {r.get("role") for r in exclusions} == fresh.ROLES
        and all(
            r.get("format") == "hash_index_v1"
            and r.get("complete") is True
            and r.get("previously_used") is True
            and r.get("sealed") is False
            for r in exclusions
        ),
        "all prior historical TRAIN/development/pilot/matrix exclusions required",
    )
    hashes, receipts = set(), []
    role_counts = Counter()
    for entry in exclusions:
        document = audit.read(entry["file"])
        require(
            document.get("schema") == "modern_fresh_exclusion_index_v1",
            "prior identity index schema differs",
        )
        hashes.update(_identity_hashes(document["records"]))
        role_counts[entry["role"]] += len(document["records"])
        receipts.append({**entry, "records": len(document["records"])})
    require(
        role_counts["actual_train"] == 512
        and all(
            role_counts[role] == inventory["counts"][role]["unique_identity_records"]
            for role in fresh.ROLES
        ),
        "prior exclusion coverage differs; all512 historical TRAIN inputs remain excluded",
    )
    bindings = [entry for entry in inventory["sources"] if entry["role"] == "actual_train"]
    historical = {}
    for entry in bindings:
        document = audit.read(entry["file"])
        for row in document["rows"]:
            require(
                row["group_id"] not in historical
                and row["source_ancestry"]["source_split"] == "TRAIN",
                "historical binding group ancestry differs",
            )
            raw = user_text(row["messages"])
            identities = fresh.identities(raw)
            require(
                set(identities.values()) <= hashes, "historical TRAIN hash inventory incomplete"
            )
            historical[row["group_id"]] = row
    require(len(historical) == 512, "complete512 source binding inventory required")
    union, original = audit.read(sources["current_union"]), audit.read(sources["original_roster"])
    require(
        union.get("schema") == "w1a8_actual_source_indexed_union_v1"
        and len(union["groups"]) == 60
        and len({g["group_id"] for g in union["groups"]}) == 60
        and Counter(g["role"] for g in union["groups"]) == {"fit": 48, "development": 12},
        "current immutable60-group union required",
    )
    require(
        original.get("schema") == "w1a8_modern_phase1_generalization_roster_v2"
        and len(original["rows"]) == 36
        and original["rows"] == original["roster"]
        and Counter(r["role"] for r in original["rows"]) == {"fit": 24, "development": 12},
        "original36 roster required",
    )
    used_groups = {g["group_id"] for g in union["groups"]}
    require(
        {r["group_id"] for r in original["rows"]} <= used_groups <= set(historical),
        "current/original groups lack frozen source ancestry",
    )
    for row in original["rows"]:
        require(
            row["messages"] == historical[row["group_id"]]["messages"],
            "original group prompt changed",
        )
        hashes.update(fresh.identities(user_text(row["messages"])).values())
    consumed = audit.read(sources["consumed_manifest"])
    receipt = audit.read(sources["consumed_receipt"])
    marker = audit.read(sources["consumed_marker"])
    require(
        consumed.get("schema") == "modern_fresh_confirmation_inputs_v1"
        and len(consumed["groups"]) == 48
        and len({r["group_id"] for r in consumed["groups"]}) == 48
        and receipt.get("manifest_sha256") == sources["consumed_manifest"]["sha256"]
        and receipt.get("performance_prompts") == receipt.get("quality_tasks") == 24
        and marker.get("schema") == "modern_fresh_cohort_consumption_v1"
        and marker.get("evaluated") is True,
        "consumed fresh48 ancestry/marker required",
    )
    require(
        receipt.get("schema") == "modern_fresh_confirmation_receipt_v1"
        and receipt.get("admitted") is True
        and consumed.get("admitted") is True
        and len(marker.get("future_exclusion_sources", [])) == 2
        and {Path(ref["path"]).name for ref in marker["future_exclusion_sources"]}
        == {"requests.json", "quality-cases.json"}
        and all(
            Path(ref["path"]).parent.resolve()
            == Path(sources["consumed_manifest"]["path"]).parent.resolve()
            and ref["sha256"] == receipt["files"][Path(ref["path"]).name]
            for ref in marker["future_exclusion_sources"]
        ),
        "consumed marker does not bind prior input packet",
    )
    for group in consumed["groups"]:
        hashes.update(_identity_hashes([group["identities"]]))
    forbidden_ids = {r["prompt_id"] for r in historical.values()}
    forbidden_groups, forbidden_topics, content_hashes = set(), set(), set()
    partition_receipts = []

    def add_partition_index(path, expected_sha, expected_count, role):
        path = Path(path).resolve()
        require(path.name.endswith(".index.jsonl"), "opaque partition index required")
        actual = reference(path)
        require(actual["sha256"] == expected_sha, "partition index pin differs")
        count = 0
        for _, _, row in audit.rows(actual, index=True):
            require(
                isinstance(row.get("id"), str)
                and isinstance(row.get("content_sha256"), str)
                and fresh.HEX.fullmatch(row["content_sha256"])
                and "messages" not in row
                and "prompt" not in row,
                "opaque index contains invalid/missing source identity",
            )
            forbidden_ids.add(row["id"])
            content_hashes.add(row["content_sha256"])
            if row.get("group"):
                forbidden_groups.add(row["group"])
            if row.get("topic"):
                forbidden_topics.add(row["topic"])
            count += 1
        require(count == expected_count, "partition index row coverage differs")
        partition_receipts.append({"role": role, "file": actual, "records": count})

    seen_roles = set()
    for ref in sources["partition_manifests"]:
        manifest = audit.read(ref)
        require(
            manifest.get("schema")
            in ("continuous_w1ax_prompt_manifest_v1", "w1a_data_manifest_v1"),
            "supported project partition manifest required",
        )
        for role, spec in manifest["files"].items():
            if role == "train_reserve":
                continue
            seen_roles.add(role)
            indexes = spec.get("shards", [spec])
            require(
                indexes and all("index" in s and "index_sha256" in s for s in indexes),
                "every exposed/dev/sealed partition needs opaque indices",
            )
            for index in indexes:
                path = (Path(ref["path"]).parent / index["index"]).resolve()
                require(
                    path.parent == Path(ref["path"]).parent.resolve()
                    and path.name.endswith(".index.jsonl"),
                    "partition must reference a local opaque index, never sealed prompts",
                )
                add_partition_index(path, index["index_sha256"], index["prompts_count"], role)
        # Continuous freezes also reference older partition indices by content hash.
        # Those opaque exclusions remain mandatory even when their prompt text is sealed.
        for legacy in manifest.get("opaque_exclusions", []):
            require(
                Path(legacy["file"]).name == legacy["file"]
                and legacy["file"].endswith(".index.jsonl")
                and fresh.HEX.fullmatch(legacy["sha256"]),
                "invalid legacy opaque index reference",
            )
            legacy_path = (
                Path(ref["path"]).parent.parent / "exclusions" / legacy["sha256"] / legacy["file"]
            )
            add_partition_index(legacy_path, legacy["sha256"], legacy["rows"], "legacy_partition")
    require(
        "dev" in seen_roles and ("final" in seen_roles or "sealed_test" in seen_roles),
        "mandatory opaque DEV and SEALED/final partitions missing",
    )
    return {
        "hashes": hashes,
        "prior": receipts,
        "forbidden_ids": forbidden_ids,
        "groups": forbidden_groups,
        "topics": forbidden_topics,
        "content_hashes": content_hashes,
        "partitions": partition_receipts,
        "current_group_ids": sorted(used_groups),
        "historical_groups": 512,
        "consumed_groups": 48,
    }


def user_text(messages):
    require(
        isinstance(messages, list)
        and len(messages) == 1
        and messages[0].get("role") == "user"
        and isinstance(messages[0].get("content"), str)
        and messages[0]["content"].strip(),
        "preserve complete single user source prompt; do not flatten chats",
    )
    return messages[0]["content"]


def select_natural(freeze, freeze_pin, audit, evidence):
    """Only called after selection freeze. No outcomes, answers or native tokenization."""
    catalog_pin = freeze["natural_sources"]["catalog"]
    catalog = audit.read(catalog_pin)
    require(
        catalog.get("schema") == CATALOG_SCHEMA and catalog.get("sources"),
        "pinned source catalog required",
    )
    # Catalog-referenced project partition manifests cannot silently be omitted.
    declared = freeze["natural_sources"]["partition_manifests"]
    require(
        all(ref in declared for ref in catalog.get("partition_manifests", [])),
        "catalog partition exclusions omitted from freeze",
    )
    rows, source_ids = [], set()
    for source in catalog["sources"]:
        require(
            source["id"] not in source_ids and source["domain"] in DOMAIN_COUNTS,
            "unique source IDs and three known domains required",
        )
        source_ids.add(source["id"])
        require(
            all(
                isinstance(source.get(k), str) and source[k]
                for k in ("id", "path", "sha256", "revision", "license", "uri")
            ),
            "complete source catalog ancestry required",
        )
        source_path = (Path(catalog_pin["path"]).parent / source["path"]).resolve()
        require(
            not re.match(r"(sealed|final|dev|holdout)([-_.]|$)", source_path.name, re.I),
            "sealed/development prompt payload cannot be a natural source",
        )
        source_pin = reference(source_path)
        require(source_pin["sha256"] == source["sha256"], "catalog source pin changed")
        fields = {
            k: source.get("fields", {}).get(k, k)
            for k in ("id", "messages", "group_id", "topic_id")
        }
        for line, raw, document in audit.rows(source_pin):
            rid = document.get(fields["id"])
            require(
                isinstance(rid, str) and rid and document.get("split", "TRAIN") == "TRAIN",
                "source row identity/TRAIN eligibility required",
            )
            prompt_id = source["id"] + ":" + rid
            group = "group:" + source["id"] + ":" + document.get(fields["group_id"], rid)
            topic = document.get(fields["topic_id"])
            topics = {"topic:" + source["id"] + ":" + topic} if topic else set()
            require(len(rows) < MAX_ROWS, "natural catalog row budget exceeded")
            if (
                prompt_id in evidence["forbidden_ids"]
                or group in evidence["groups"]
                or bool(topics & evidence["topics"])
            ):
                # Retain only opaque component links for an indexed master-source row.
                # Never inspect its messages or produce candidate prompt identities.
                rows.append(
                    {
                        "prompt_id": prompt_id,
                        "domain": source["domain"],
                        "group": group,
                        "topics": sorted(topics),
                        "identities": None,
                        "blocked": True,
                    }
                )
                continue
            messages = document.get(fields["messages"])
            text = user_text(messages)
            identities = fresh.identities(text)
            # Preserve both historical Dolly topic conventions without opening held-out text.
            if source["id"] == "dolly" and "\n\nContext:\n" in text:
                context = text.split("\n\nContext:\n", 1)[1]
                words = " ".join(re.findall(r"\w+", context.casefold()))
                topics.add("topic:dolly:context:" + fresh.digest(words.encode()))
            content_hash = fresh.digest(
                json.dumps(
                    messages, sort_keys=True, ensure_ascii=False, separators=(",", ":")
                ).encode()
            )
            blocked = bool(set(identities.values()) & evidence["hashes"]) or (
                prompt_id in evidence["forbidden_ids"]
                or group in evidence["groups"]
                or bool(topics & evidence["topics"])
                or content_hash in evidence["content_hashes"]
            )
            rows.append(
                {
                    "prompt_id": prompt_id,
                    "domain": source["domain"],
                    "messages": copy.deepcopy(messages),
                    "group": group,
                    "topics": sorted(topics),
                    "identities": identities,
                    "blocked": blocked,
                    "source_ancestry": {
                        "catalog": catalog_pin,
                        "source": source_pin,
                        "source_entry": source,
                        "line": line,
                        "line_sha256": fresh.digest(raw),
                        "source_row_id": rid,
                        "historical_source_split": (
                            "TRAIN/public source; project exposure excluded by pinned inventories"
                        ),
                    },
                }
            )
            require(len(rows) <= MAX_ROWS, "natural catalog row budget exceeded")
    require(
        len({r["prompt_id"] for r in rows}) == len(rows), "duplicate corpus source-row identity"
    )
    # Whole connected components: group/topic/normalized aliases cannot be split.
    parent = list(range(len(rows)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    links = {}
    for index, row in enumerate(rows):
        keys = [("group", row["group"])]
        if row["identities"] is not None:
            keys.append(("identity", row["identities"]["normalized_raw_sha256"]))
        keys += [("topic", topic) for topic in row["topics"]]
        for key in keys:
            if key in links:
                a, b = find(index), find(links[key])
                parent[a] = b
            else:
                links[key] = index
    components = defaultdict(list)
    for index, row in enumerate(rows):
        components[find(index)].append(row)
    seed = fresh.digest(bytes.fromhex(freeze_pin["sha256"]) + b"modern-natural-confirmation-v1")
    candidates = [
        members[0]
        for members in components.values()
        if len(members) == 1 and not members[0]["blocked"]
    ]
    selected = []
    eligible_counts = Counter(r["domain"] for r in candidates)
    for domain, count in DOMAIN_COUNTS.items():
        pool = sorted(
            (r for r in candidates if r["domain"] == domain),
            key=lambda r: fresh.digest((seed + "\0" + r["prompt_id"]).encode()),
        )
        require(len(pool) >= count, "insufficient truly unused singleton source groups: " + domain)
        for index, row in enumerate(pool[:count]):
            component = fresh.digest(row["prompt_id"].encode())
            selected.append(
                {
                    "id": f"natural-performance-{domain}-{index:02d}",
                    "domain": domain,
                    "family": "natural-source-" + component[:16],
                    "group_id": "natural-catalog-v1:" + component,
                    "source": "pinned_public_catalog_after_candidate_freeze_v1",
                    "phase": "performance",
                    "instance_seed": fresh.digest((seed + "\0" + component).encode()),
                    "identities": row["identities"],
                    "messages": row["messages"],
                    "source_ancestry": row["source_ancestry"],
                    "original_prompt_id": row["prompt_id"],
                    "original_group": row["group"],
                    "original_topics": row["topics"],
                }
            )
    return selected, {
        "catalog_rows": len(rows),
        "eligible_singletons_by_domain": dict(eligible_counts),
        "excluded_or_multimember_rows": len(rows) - len(candidates),
        "selection_seed": seed,
        "selection_rule": (
            "SHA256(frozen seed+NUL+source prompt ID) ascending within domain; "
            "no outcome replacement"
        ),
        "performance_clusters": 24,
        "cluster_unit": "whole singleton corpus source component",
        "domain_strata": DOMAIN_COUNTS,
        "synthetic_performance_family_clusters": False,
    }


def payload(text):
    return {
        "prompt": fresh.render(text),
        "n_predict": 256,
        "temperature": 0,
        "seed": 1234,
        "cache_prompt": False,
        "return_tokens": True,
        "stream": False,
    }


def build_packet(freeze, freeze_pin, judge):
    audit = Metadata()
    evidence = exclusion_evidence(freeze, audit)
    natural, selection = select_natural(freeze, freeze_pin, audit, evidence)
    seed = selection["selection_seed"]
    requests, quality, controls, groups, collisions = [], [], [], [], []
    observed = set()
    for row in natural:
        request = payload(user_text(row["messages"]))
        groups.append(copy.deepcopy(row))
        requests.append(
            {
                **row,
                "prompt": request["prompt"],
                "sha256": fresh.digest(request["prompt"].encode()),
                "request": request,
            }
        )
        observed.update(row["identities"].values())
    for domain, families in fresh.FAMILIES.items():
        for family in families:
            for index in range(2):
                case, good, wrong = fresh._instance(seed, domain, family, "quality", index)
                ids = set(case["identities"].values())
                if ids & (evidence["hashes"] | observed):
                    collisions.append(case["id"])
                observed.update(ids)
                case["request"] = payload(case["prompt"])
                positive = judge.judge(case, good)
                negatives = [judge.judge(case, answer) for answer in wrong]
                controls.append(
                    {
                        "case_id": case["id"],
                        "correct_control_passed": positive["passed"],
                        "known_wrong_control_rejected": all(not n["passed"] for n in negatives),
                        "positive": {"answer": good, "judgement": positive},
                        "semantic_negatives": [
                            {"answer": a, "judgement": r}
                            for a, r in zip(wrong, negatives, strict=True)
                        ],
                    }
                )
                quality.append(case)
                groups.append({k: case[k] for k in confirm.META})
    require(
        len(requests) == len(quality) == len(controls) == 24
        and len(groups) == 48
        and len({g["group_id"] for g in groups}) == 48,
        "complete natural/quality packet required",
    )
    admitted = not collisions and all(
        c["correct_control_passed"] and c["known_wrong_control_rejected"] for c in controls
    )
    status = "PREPARED_NO_MODEL_EVALUATION" if admitted else "INCONCLUSIVE_INPUTS_OR_ORACLE"
    audit.recheck()
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "status": status,
        "admitted": admitted,
        "freeze": freeze_pin,
        **{
            k: freeze[k]
            for k in (
                "source_freeze",
                "candidate",
                "runtime",
                "generator",
                "quality_generator",
                "confirmation_runner",
                "judge",
                "settings",
            )
        },
        "seed": seed,
        "groups": groups,
        "exclusions": evidence["prior"],
        "partition_exclusions": evidence["partitions"],
        "source_pins": list(audit.pins.values()),
        "selection": selection,
        "overlap_case_ids": collisions,
        "historical_TRAIN_inputs_excluded": 512,
        "current_groups_excluded": 60,
        "consumed_fresh_groups_excluded": 48,
        "model_requests_executed": 0,
        "tokenization_performed": False,
        "requirements": [
            "Inherited native /tokenize must admit every whole prompt "
            "with256 output reserve inside4096 before completion.",
            "No dropped/truncated/replaced groups; preserve incorrect/short/inconclusive outcomes.",
            "Five alternating full-HTTP performance repetitions; "
            "target/DFlash controls and scorer unchanged.",
        ],
        "limits": (
            "Natural performance transfer only. Public corpora may have pretraining exposure; "
            "no task/pretraining novelty or broad prose-quality claim. Quality is24 independent "
            "synthetic oracle cases,12 related templates. Semantic overlap beyond declared "
            "group/topic and exact/NFKC identities is unproven."
        ),
    }
    return {
        "manifest.json": manifest,
        "requests.json": requests,
        "quality-cases.json": quality,
        "quality-controls.json": controls,
    }


def prepare(freeze_path, output):
    output = Path(output)
    require(not output.exists(), "refusing to replace confirmation inputs")
    freeze, freeze_pin, judge = validate_freeze(freeze_path)
    packet = build_packet(freeze, freeze_pin, judge)
    manifest = packet["manifest.json"]
    blobs = {name: fresh.encoded(value) for name, value in packet.items()}
    require(
        sum(map(len, blobs.values())) <= fresh.OUTPUT_LIMIT, "prepared metadata packet exceeds2MiB"
    )
    require(reference(freeze_path) == freeze_pin, "candidate freeze changed during preparation")
    checked_freeze, checked_pin, _ = validate_freeze(freeze_path)
    require(
        checked_freeze == freeze and checked_pin == freeze_pin,
        "source freeze changed before publication",
    )
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "status": manifest["status"],
        "admitted": manifest["admitted"],
        "created_at_utc": datetime.now(UTC).isoformat(),
        "manifest_sha256": fresh.digest(blobs["manifest.json"]),
        "freeze": freeze_pin,
        "seed": manifest["seed"],
        "performance_prompts": 24,
        "quality_tasks": 24,
        "quality_templates": 12,
        "natural_performance_clusters": 24,
        "domain_strata": DOMAIN_COUNTS,
        "quality_correct_controls_passed": sum(
            c["correct_control_passed"] for c in packet["quality-controls.json"]
        ),
        "quality_known_wrong_controls_rejected": sum(
            c["known_wrong_control_rejected"] for c in packet["quality-controls.json"]
        ),
        "input_overlap_count": len(manifest["overlap_case_ids"]),
        "model_requests_executed_before_freeze": 0,
        "model_requests_executed": 0,
        "files": {name: fresh.digest(raw) for name, raw in blobs.items()},
        "generated_bytes": sum(map(len, blobs.values())),
    }
    output.mkdir(parents=True, exist_ok=False)
    for name, raw in blobs.items():
        with (output / name).open("xb") as stream:
            stream.write(raw)
    with (output / "receipt.json").open("xb") as stream:
        stream.write(fresh.encoded(receipt))
    return receipt


class NaturalConfirmation(confirm.Confirmation):
    """Honest new packet authentication; reuse existing native lifecycle/timing/judge."""

    def authenticate_inputs(self):
        refs = self.inputs
        require(self.arms[0]["kind"] == "target", "frozen target arm must be first")
        freeze, freeze_pin, self.judge = validate_freeze(self.checked(refs["selection"]))
        source = read(self.checked(freeze["source_freeze"]))
        require(
            source.get("target") == self.models.get("target")
            and source.get("baseline") == self.models.get("baseline")
            and source.get("arms") == self.arms,
            "full frozen target/DFlash/candidate runtime differs",
        )
        actual = {
            filename: read(self.checked(refs[key]))
            for key, filename in confirm.PACKET_FILES.items()
        }
        receipt = read(self.checked(refs["receipt"]))
        expected = build_packet(freeze, freeze_pin, self.judge)
        require(
            actual == expected,
            "natural packet differs from deterministic frozen source/quality ancestry",
        )
        manifest = actual["manifest.json"]
        require(
            receipt.get("schema") == RECEIPT_SCHEMA
            and manifest.get("schema") == MANIFEST_SCHEMA
            and receipt.get("admitted") is True
            and manifest.get("admitted") is True
            and receipt.get("status") == manifest.get("status") == "PREPARED_NO_MODEL_EVALUATION",
            "admitted unevaluated natural packet required",
        )
        require(
            set(receipt.get("files", {})) == set(confirm.PACKET_FILES.values())
            and all(
                refs[key]["sha256"] == receipt["files"][filename]
                for key, filename in confirm.PACKET_FILES.items()
            )
            and receipt.get("manifest_sha256") == refs["manifest"]["sha256"]
            and receipt.get("freeze") == manifest["freeze"] == freeze_pin
            and receipt.get("seed") == manifest["seed"]
            and receipt.get("model_requests_executed_before_freeze")
            == receipt.get("model_requests_executed")
            == 0
            and receipt.get("input_overlap_count") == 0,
            "natural receipt/source/unused status differs",
        )
        require(
            receipt.get("performance_prompts")
            == receipt.get("quality_tasks")
            == receipt.get("quality_correct_controls_passed")
            == receipt.get("quality_known_wrong_controls_rejected")
            == 24
            and receipt.get("quality_templates") == 12
            and receipt.get("natural_performance_clusters") == 24,
            "natural/quality receipt coverage differs",
        )
        self.selected_id, self.output_limit = freeze["selected_arm_id"], 256
        self.rows = actual["requests.json"]
        cases = actual["quality-cases.json"]
        self.payloads, self.gold = {}, {}
        for item in self.rows + cases:
            confirm.validate_payload(item["request"], 256)
            self.payloads[item["id"]] = item["request"]
        self.quality_cases = {c["id"]: c for c in cases}
        self.quality_rows = [{k: c[k] for k in confirm.META} for c in cases]
        self.warmups = refs["warmups"]
        require(
            len(self.warmups) == 2 and len({w["id"] for w in self.warmups}) == 2,
            "two distinct original FIT warmups required",
        )
        warmup_source = read(self.checked(refs["warmup_source"]))
        old_roster_ref = warmup_source["roster"]
        old_roster = read(self.checked(old_roster_ref))
        require(
            old_roster_ref == freeze["natural_sources"]["original_roster"],
            "warmup roster ancestry differs",
        )
        old_rows = {r["id"]: r for r in old_roster["rows"]}
        for item in self.warmups:
            matches = [w for w in warmup_source["warmups"] if w["id"] == item["id"]]
            require(
                item.get("role") == "fit"
                and item["id"] not in self.payloads
                and old_rows.get(item["id"], {}).get("role") == "fit"
                and len(matches) == 1
                and all(matches[0].get(k) == item[k] for k in ("id", "request", "reference")),
                "original FIT warmup ancestry differs",
            )
            request = read(self.checked(item["request"]))
            confirm.validate_payload(request, 80)
            require(
                fresh.digest(request["prompt"].encode())
                == old_rows[item["id"]]["rendered_prompt_sha256"],
                "original FIT warmup prompt differs",
            )
            self.payloads[item["id"]] = request
            self.gold[item["id"]] = signature(read(self.checked(item["reference"])))
        self.manifest, self.receipt = manifest, receipt
        self.request_schema = "modern_natural_confirmation_requests_v1"
        self.request_scope = manifest["limits"]
        self.input_source_refs = [
            freeze_pin,
            freeze["source_freeze"],
            freeze["generator"],
            freeze["quality_generator"],
            freeze["confirmation_runner"],
            freeze["judge"],
            refs["warmup_source"],
            old_roster_ref,
            *manifest["source_pins"],
            *(refs[key] for key in confirm.PACKET_FILES),
            refs["receipt"],
        ]
        self.current_phase, self.geometry = "performance", None
        (self.output / "gold").mkdir()
        atomic(
            self.output / "authenticated-controls.json",
            {
                "all_positive_passed": True,
                "all_negative_rejected": True,
                "records": actual["quality-controls.json"],
                "natural_source_audit": manifest["selection"],
            },
        )

    def summarize(self):
        performance, quality = super().summarize()
        for row in performance:
            row["statistical_unit"] = "whole natural corpus prompt/source component"
            row["performance_clusters"] = 24
            row["domain_strata"] = DOMAIN_COUNTS
            row["synthetic_family_cluster_count_applicable"] = False
        return performance, quality


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    sub = cli.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--freeze", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    serve = sub.add_parser("serve")
    serve.add_argument("--models", type=Path, required=True)
    serve.add_argument("--inputs", type=Path, required=True)
    serve.add_argument("--output", type=Path, required=True)
    serve.add_argument("--repetitions", type=int, default=5)
    serve.add_argument("--allow-cuda", action="store_true")
    args = cli.parse_args()
    if args.command == "prepare":
        print(json.dumps(prepare(args.freeze, args.output), indent=2))
    else:
        NaturalConfirmation(args).run()


if __name__ == "__main__":
    main()

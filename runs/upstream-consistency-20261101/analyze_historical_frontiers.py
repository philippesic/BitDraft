#!/usr/bin/env python3
"""Join the retained development serving traces without reading replay outcomes.

Output positions identify common observable token histories, not equal hidden
states. No GPU use, model loading, training, capture, or subprocess execution.
Only compact metadata is written: prompt text and token arrays stay in sources.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from itertools import combinations
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ARMS = ("fresh_best", "parent3504", "q4_k_m")
DOMAINS = ("code", "prose", "reasoning")


def need(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def pin(path):
    path = Path(path).resolve(strict=True)
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": digest(data)}


def lines(path):
    return [(number, json.loads(line)) for number, line in enumerate(Path(path).read_text().splitlines(), 1)
            if line.strip()]


def eligible(row):
    return row["status"] == "complete" and row["n_proposed"] == 5 and not row["terminal"]


def stats(rows):
    rows = list(rows)
    count = len(rows)
    accepted = sum(r["n_accepted"] for r in rows)
    return {"frontiers": count, "accepted": accepted,
            "usable_accepted": sum(r["n_accepted_usable_prefix"] for r in rows),
            "proposed": sum(r["n_proposed"] for r in rows),
            "mean_accepted": accepted / count if count else None,
            "first_error_histogram": dict(sorted(Counter(r["n_accepted"] for r in rows).items())),
            "survival_counts_1_to_5": [sum(r["n_accepted"] >= depth for r in rows) for depth in range(1, 6)]}


def pair_summary(data, metadata, a, b):
    shared = []
    only_a = []
    only_b = []
    shared_any = 0
    per_prompt_delta = []
    for prompt, meta in metadata.items():
        aa, bb = data[a, prompt], data[b, prompt]
        common = aa.keys() & bb.keys()
        shared_any += len(common)
        pairs = [(aa[k], bb[k], meta) for k in sorted(common) if eligible(aa[k]) and eligible(bb[k])]
        shared.extend(pairs)
        per_prompt_delta.append(sum(x["n_accepted"] - y["n_accepted"] for x, y, _ in pairs))
        only_a.extend(aa[k] for k in sorted(aa.keys() - bb.keys()) if eligible(aa[k]))
        only_b.extend(bb[k] for k in sorted(bb.keys() - aa.keys()) if eligible(bb[k]))

    def summarize(pairs):
        pairs = list(pairs)
        x, y = stats(p[0] for p in pairs), stats(p[1] for p in pairs)
        return {a: x, b: y, "accepted_delta_a_minus_b": x["accepted"] - y["accepted"]}

    result = {"shared_any_anchor": shared_any, "nonterminal_full_D5": summarize(shared),
              "a_only_nonterminal_full_D5": stats(only_a),
              "b_only_nonterminal_full_D5": stats(only_b),
              "prompt_gain_loss_tie": {"gain": sum(x > 0 for x in per_prompt_delta),
                                       "loss": sum(x < 0 for x in per_prompt_delta),
                                       "tie": sum(x == 0 for x in per_prompt_delta)},
              "by_domain": {d: summarize(p for p in shared if p[2]["domain"] == d) for d in DOMAINS},
              "by_output_position": {
                  name: summarize(p for p in shared if lo <= p[0]["anchor"] < hi)
                  for name, lo, hi in (("early_1_to_31", 1, 32), ("middle_32_to_63", 32, 64),
                                       ("late_64_plus", 64, sys.maxsize))}}
    if (a, b) == ("fresh_best", "parent3504"):
        result["stratified_by_parent_acceptance"] = {
            "parent_unsaturated": summarize(p for p in shared if p[1]["n_accepted"] < 5),
            "parent_full_five": summarize(p for p in shared if p[1]["n_accepted"] == 5),
            "parent_zero_or_one": summarize(p for p in shared if p[1]["n_accepted"] <= 1),
            "parent_two_to_four": summarize(p for p in shared if 2 <= p[1]["n_accepted"] <= 4)}
    return result


def analyze(source):
    source = source.resolve(strict=True)
    config_path = source / "config.json"
    request_path = source / "requests.jsonl"
    config = json.loads(config_path.read_text())
    request_lines = lines(request_path)
    requests = [r for _, r in request_lines]
    need(len(requests) == 72, "expected the retained 72-request diagnostic")
    need(set(r["arm"] for r in requests) == set(ARMS), "unexpected arm set")
    need(set(config["panels"]) == {"development"}, "development-only input required")
    need(all(r["panel"] == "development" and r["repetition"] == 0 and r["warmup_excluded"]
             for r in requests), "request panel/repetition/warmup role changed")
    data = {}
    metadata = {}
    request_by = {}
    join_details = {}
    trace_pins = {}
    checks = Counter()
    complications = {}
    for arm in ARMS:
        trace_path = source / "development" / "rep-00" / arm / "rounds.jsonl"
        trace_pins[arm] = pin(trace_path)
        groups = {}
        previous = None
        for number, row in lines(trace_path):
            task = row["task_id"]
            need(task == previous or task not in groups, "interleaved task trace at concurrency one")
            groups.setdefault(task, []).append((number, row))
            previous = task
            need(not row.get("replay") and row["status"] in ("complete", "no_proposal"),
                 "unexpected replay or round status")
            need(row["n_draft_configured"] == 5, "draft limit changed")
        selected = [(number, q) for number, q in request_lines if q["arm"] == arm]
        need(len(selected) == 24 and len(groups) == 26, "24 measured and two warmup groups required")
        need(len({q["prompt_id"] for _, q in selected}) == 24, "duplicate measured prompt")
        grouped = list(groups.items())
        warmups = []
        for index, (task, group) in enumerate(grouped[:2]):
            emitted = [t for _, row in group for t in row["emitted_token_ids"]]
            ids = selected[index][1]["result"]["generated_token_ids"]
            need(emitted in (ids, ids[1:]), "warmup group fails independent token join")
            warmups.append({"task_id": task, "matches_measured_prompt_id": selected[index][1]["prompt_id"]})
            checks["warmup_groups_independently_joined"] += 1
        details = []
        terminal_issues = []
        status_counts = Counter()
        offsets = Counter()
        for (request_line, q), (task, group) in zip(selected, grouped[2:]):
            prompt = q["prompt_id"]
            ids = q["result"]["generated_token_ids"]
            need(q["result"]["completion_tokens"] == len(ids), "response token count differs")
            need([r["round_index"] for _, r in group] == list(range(len(group))), "round index discontinuity")
            emitted = [t for _, r in group for t in r["emitted_token_ids"]]
            if emitted == ids:
                offset = 0
            elif emitted == ids[1:]:
                offset = 1
            else:
                raise ValueError("trace emission does not equal measured output or output without initial seed")
            offsets[offset] += 1
            anchor = offset
            frontiers = {}
            for index, (line_number, r) in enumerate(group):
                need(r["n_emitted"] == len(r["emitted_token_ids"]), "emission count differs")
                need(r["n_proposed"] == len(r["proposed_token_ids"]), "proposal count differs")
                need(0 <= r["n_accepted_usable_prefix"] <= r["n_accepted"] <= r["n_proposed"] <= 5,
                     "invalid acceptance bounds")
                need(r["emitted_token_ids"] == ids[anchor:anchor + r["n_emitted"]], "anchor emission mismatch")
                if r["status"] == "complete":
                    need(r["emitted_token_ids"] == r["verified_token_ids"][:r["n_emitted"]],
                         "consumed verifier prefix mismatch")
                    checks["complete_round_verifier_joins"] += 1
                else:
                    need(r["n_proposed"] == r["n_accepted"] == 0 and r["n_emitted"] == 1,
                         "invalid no-proposal round")
                    checks["no_proposal_emission_joins"] += 1
                usable = r["n_accepted_usable_prefix"]
                need(r["proposed_token_ids"][:usable] == r["emitted_token_ids"][:usable],
                     "accepted proposal prefix mismatch")
                need(anchor not in frontiers, "duplicate observable output frontier")
                item = {k: r[k] for k in ("status", "n_proposed", "n_accepted", "n_accepted_usable_prefix", "n_emitted")}
                item.update(task_id=task, round_index=r["round_index"], trace_line=line_number,
                            terminal=index == len(group) - 1, anchor=anchor)
                item["nonterminal_full_D5_eligible"] = eligible(item)
                frontiers[anchor] = item
                status_counts[r["status"]] += 1
                if (r["n_accepted"] != usable or
                        (r["status"] == "complete" and r["n_emitted"] != r["n_accepted"] + 1)):
                    need(item["terminal"], "nonterminal emission/acceptance complication needs investigation")
                    terminal_issues.append({"prompt_id": prompt, **item})
                anchor += r["n_emitted"]
            need(anchor == len(ids), "trace fails complete output coverage")
            complete = [r for r in frontiers.values() if r["status"] == "complete"]
            expected = {"rounds": len(complete), "proposed": sum(r["n_proposed"] for r in complete),
                        "accepted": sum(r["n_accepted"] for r in complete)}
            need(all(q["result"]["speculative"][k] == v for k, v in expected.items()),
                 "trace counts differ from measured native counters")
            need(all(q["round_costs"][k] == v for k, v in expected.items()),
                 "trace counts differ from retained round summary")
            checks["measured_request_emission_joins"] += 1
            checks["request_native_counter_joins"] += 1
            data[arm, prompt] = frontiers
            request_by[arm, prompt] = q
            meta = {"domain": q["domain"], "output_tokens": len(ids),
                    "output_token_ids_sha256": digest(canonical(ids)),
                    "group_id": q.get("group_id")}
            if prompt in metadata:
                need(meta == metadata[prompt], "different output tokens/domain/group across matched arms")
            metadata[prompt] = meta
            details.append({"prompt_id": prompt, "task_id": task, "source_request_line": request_line,
                            "initial_seed_offset": offset, **expected})
        join_details[arm] = {"warmups": warmups, "measured_requests": details}
        complications[arm] = {"status_counts": dict(status_counts), "initial_seed_offsets": dict(offsets),
                              "terminal_emission_exceptions": terminal_issues}
    need(len(metadata) == 24, "matched prompt coverage differs")
    checks["matched_three_arm_output_sequences"] = len(metadata)
    pair_results = {f"{a}_vs_{b}": pair_summary(data, metadata, a, b) for a, b in combinations(ARMS, 2)}
    triple = []
    for prompt in metadata:
        for anchor in sorted(set.intersection(*(set(data[a, prompt]) for a in ARMS))):
            if all(eligible(data[a, prompt][anchor]) for a in ARMS):
                triple.append((prompt, anchor))
    table = []
    for prompt in sorted(metadata):
        for anchor in sorted(set.union(*(set(data[a, prompt]) for a in ARMS))):
            row = {"frontier_id": f"{prompt}@{anchor}", "prompt_id": prompt,
                   "domain": metadata[prompt]["domain"], "output_relative_anchor": anchor,
                   "observable_output_prefix_sha256": digest(canonical(request_by[ARMS[0], prompt]["result"]["generated_token_ids"][:anchor])),
                   "arms": {a: data[a, prompt][anchor] for a in ARMS if anchor in data[a, prompt]}}
            if metadata[prompt]["group_id"] is not None:
                row["group_id"] = metadata[prompt]["group_id"]
            table.append(row)
    comparison = [r for r in table if all(a in r["arms"] and eligible(r["arms"][a])
                                         for a in ("fresh_best", "parent3504"))]
    damaged = [r for r in comparison if r["arms"]["parent3504"]["n_accepted"] == 5
               and r["arms"]["fresh_best"]["n_accepted"] < 5]
    pools = {
        "unchanged_full_five": [r for r in comparison if r["arms"]["parent3504"]["n_accepted"] ==
                                r["arms"]["fresh_best"]["n_accepted"] == 5],
        "improved": [r for r in comparison if r["arms"]["fresh_best"]["n_accepted"] >
                     r["arms"]["parent3504"]["n_accepted"]]}
    need(len(damaged) == 32, "retained damaged-frontier count differs from verified 32")
    used = set()
    matches = []
    for damaged_row in sorted(damaged, key=lambda r: (r["domain"], r["prompt_id"], r["output_relative_anchor"])):
        record = {"damaged_frontier": damaged_row["frontier_id"], "controls": {}}
        for category, pool in pools.items():
            candidates = [r for r in pool if r["frontier_id"] not in used]
            need(candidates, "insufficient unique matched controls")
            same_domain = [r for r in candidates if r["domain"] == damaged_row["domain"]]
            if same_domain:
                candidates = same_domain
            distinct_group = [r for r in candidates if r.get("group_id") is not None
                              and damaged_row.get("group_id") is not None
                              and r["group_id"] != damaged_row["group_id"]]
            if distinct_group:
                candidates = distinct_group
            distinct_prompt = [r for r in candidates if r["prompt_id"] != damaged_row["prompt_id"]]
            if distinct_prompt:
                candidates = distinct_prompt
            chosen = min(candidates, key=lambda r: (abs(r["output_relative_anchor"] - damaged_row["output_relative_anchor"]),
                                                    r["prompt_id"], r["output_relative_anchor"]))
            used.add(chosen["frontier_id"])
            record["controls"][category] = {"frontier_id": chosen["frontier_id"],
                "same_domain": chosen["domain"] == damaged_row["domain"],
                "different_prompt": chosen["prompt_id"] != damaged_row["prompt_id"],
                "group_relation": "different" if chosen in distinct_group else "unavailable",
                "absolute_output_offset_distance": abs(chosen["output_relative_anchor"] - damaged_row["output_relative_anchor"])}
        matches.append(record)
    selected = {r["frontier_id"] for r in damaged} | used
    need(len(used) == 64 and len(selected) == 96, "matched controls are not unique")
    domain_visits = {}
    for a in ARMS:
        all_complete = [r for p in metadata for r in data[a, p].values() if r["status"] == "complete"]
        domain_visits[a] = {}
        for domain in DOMAINS:
            rr = [r for p in metadata if metadata[p]["domain"] == domain for r in data[a, p].values()
                  if r["status"] == "complete"]
            tokens = sum(m["output_tokens"] for m in metadata.values() if m["domain"] == domain)
            domain_visits[a][domain] = {"all_complete": stats(rr),
                "nonterminal_full_D5": stats(r for r in rr if eligible(r)), "output_tokens": tokens,
                "round_share": len(rr) / len(all_complete),
                "output_token_share": tokens / sum(m["output_tokens"] for m in metadata.values())}
    return {"schema": "historical_observable_frontier_analysis_v1", "scope": "reused development, not held-out",
            "source": {"analyzer": pin(__file__), "config": pin(config_path), "requests": pin(request_path),
                       "traces": trace_pins},
            "source_contract": {k: config[k] for k in ("binary", "build", "target", "models", "source_root",
                                                      "source_files", "runtime_libraries", "environment")},
            "join_assertions": {"all_passed": True, "counts": dict(checks),
                "asserted": ["development-only 72 requests, three arms, 24 unique prompts each",
                             "contiguous concurrency-one task groups, two warmups per arm",
                             "warmup emissions independently match corresponding measured prompts",
                             "sequential round indices and no replay records",
                             "trace emitted IDs equal generated IDs or generated IDs excluding seed",
                             "every emission is the exact output-token slice at its inferred anchor",
                             "complete-round consumed verifier prefix and usable accepted proposal match",
                             "proposal/emission counts and accepted bounds valid",
                             "trace totals equal both native request counters and retained round summaries",
                             "all three arms have identical output sequence hashes and lengths",
                             "32 damaged frontiers and 64 unique pre-replay controls selected"]},
            "request_joins": join_details, "prompt_metadata": metadata, "complications": complications,
            "pairs": pair_results,
            "triple_common_nonterminal_full_D5": {a: stats(data[a, p][k] for p, k in triple) for a in ARMS},
            "domain_visit_weighting": domain_visits, "frontiers": table,
            "counterfactual_selection": {"damaged_count": len(damaged), "control_count": len(used),
                "selected_total": len(selected), "control_pool_counts": {k: len(v) for k, v in pools.items()},
                "policy": "Process damage sorted by domain,prompt_id,anchor; choose unused controls, preferring same domain, then distinct known group, then distinct prompt, then nearest output offset; ties by prompt_id,anchor. Unchanged controls require both arms accepted five; improved controls require fresh accepted more than parent.",
                "future_replay_outputs_read": False,
                "group_identity_available": all(m["group_id"] is not None for m in metadata.values()),
                "matched_controls": matches,
                "selected_frontier_ids": sorted(selected)},
            "limitations": ["Shared output-token prefixes do not establish identical target/draft hidden states or cache layout.",
                "Observed arm-specific visited anchors are not counterfactual scores for the other arm.",
                "Terminal, partial-width and no-proposal events are retained but excluded from full-D5 nonterminal comparisons.",
                "No-proposal events have no verifier-token array; their emitted token is checked against actual output.",
                "Original group IDs are absent from these request rows; distinct prompts do not prove group disjointness.",
                "No join between retained static training windows and this request panel is asserted.",
                "Hard-frontier gains versus formerly saturated-frontier losses are observed associations, not proof of their training cause."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=HERE / "historical-component-01")
    parser.add_argument("--output", type=Path, default=HERE / "historical-frontiers.json")
    parser.add_argument("--verify", action="store_true", help="require an identical existing output; never write")
    args = parser.parse_args()
    result = analyze(args.input)
    encoded = canonical(result) + b"\n"
    if args.output.exists():
        need(args.output.read_bytes() == encoded, "existing output differs; preserve it and choose a new output")
        action = "verified_existing"
    else:
        need(not args.verify, "verification output does not exist")
        with args.output.open("xb") as stream:
            stream.write(encoded)
        action = "created"
    chosen = result["counterfactual_selection"]
    print(json.dumps({"status": action, "all_join_assertions_passed": True,
                      "frontier_rows": len(result["frontiers"]), "damaged": chosen["damaged_count"],
                      "controls": chosen["control_count"], "selected_total": chosen["selected_total"],
                      "output": pin(args.output)}, indent=2))


if __name__ == "__main__":
    main()

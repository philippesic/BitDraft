"""Zero-update replay of native deterministic DSpark round trajectories."""
import argparse
import hashlib
import json
from pathlib import Path


def replay_collector(rows):
    """Independently reproduce saved native collector's greedy D5 transitions.

    These authentic seven-row captures include native verifier IDs, target labels,
    and next anchors, so correctness is checkable without the draft weight files.
    This validates historical trajectories, not current-candidate request speed.
    """
    chains = {}
    accepted = 0
    for row in rows:
        if (row.get("conditioning") != "native_greedy_correct_captured_prefix"
                or row.get("submitted_rows") != 5 or row.get("computed_noise_rows") != 7):
            raise ValueError("native D5/seven-row captured-prefix collector required")
        predictions = [r["argmax"] for r in row["rows"]]
        labels = [r["label"] for r in row["rows"]]
        if len(predictions) != 7 or len(labels) != 7:
            raise ValueError("all seven native argmax/label rows required")
        first = next((i for i,(a,b) in enumerate(zip(predictions,labels)) if a != b),7)
        count = min(first,5)
        if (first != row["first_rejection7"] or count != row["accepted_drafts"]
                or row["next_anchor"] != row["anchor"]+count+1
                or row["verifier_token_ids"] != labels[:count+1]):
            raise ValueError("native argmax/target/acceptance/anchor reconstruction differs")
        previous = chains.get(row["chain_id"])
        if previous and (row["round"] != previous["round"]+1 or row["anchor"] != previous["next_anchor"]):
            raise ValueError("cross-round occupancy chain has a gap")
        chains[row["chain_id"]] = row
        accepted += count
    if not rows:
        raise ValueError("nonempty collector required")
    return {"schema":"greedy_cross_round_replay_v1","source_kind":"historical_native_round_collector",
            "rounds":len(rows),"chains":len(chains),"accepted":accepted,"proposed":5*len(rows),
            "native_counters_checked":True,"target_reference_checked":True,
            "exact_scope":"saved native argmax, verifier IDs and next-anchor reproduction; no current-model inference or throughput claim",
            "stochastic_EDR_theorem_claimed":False}


def replay_rounds(rows, reference_tokens=None, expected_counters=None):
    """Validate one request's authentic trace and reconstruct occupied starts.

    References may contain the initial target seed omitted by the native trace.
    Actual emitted counts handle EOS/output truncation; accepted+1 is NOT assumed.
    """
    emitted, occupied, replay_count = [], [], 0
    totals = {"rounds": 0, "accepted": 0, "proposed": 0}
    if not rows:
        raise ValueError("nonempty native trace required")
    task = {(r.get("task_id"), r.get("repetition")) for r in rows}
    if len(task) != 1:
        raise ValueError("pass exactly one request/task trajectory")
    for i, row in enumerate(rows):
        if row.get("schema") != "w1ax_eagle_round_v1":
            raise ValueError("native trace schema required")
        status = row.get("status")
        if status not in ("complete", "no_proposal", "checkpoint_replay"):
            raise ValueError("unknown native round status")
        proposals, tokens = row["proposed_token_ids"], row["emitted_token_ids"]
        a, p = row["n_accepted"], row["n_proposed"]
        if any(type(v) is not int or v < 0 for v in [a, p, *proposals, *tokens]):
            raise ValueError("nonnegative native counts/token IDs required")
        if not 0 <= a <= p <= 5 or p != len(proposals) or row["n_emitted"] != len(tokens):
            raise ValueError("D5 native count/token mismatch")
        if status == "checkpoint_replay":
            if tokens:
                raise ValueError("checkpoint replay cannot emit quality tokens")
            replay_count += 1
            continue
        if row.get("replay") is not False or not 0 < len(tokens) <= a+1:
            raise ValueError("invalid native quality emission")
        if status == "no_proposal" and (p or a or len(tokens) != 1):
            raise ValueError("invalid no-proposal round")
        if tokens[:min(a, len(tokens))] != proposals[:min(a, len(tokens))]:
            raise ValueError("accepted emitted prefix differs from proposals")
        if len(tokens) <= a and any(r["status"] != "checkpoint_replay" for r in rows[i+1:]):
            raise ValueError("terminal output clipping before another quality round")
        occupied.append({"round_index": row["round_index"], "start": len(emitted),
                         "accepted": a, "advance": len(tokens), "status": status})
        emitted.extend(tokens)
        # Server drafts_total counts proposing rounds; no-proposal fallback is
        # separately reported, since it can have different server accounting.
        if status == "complete":
            totals["rounds"] += 1
            totals["accepted"] += a
            totals["proposed"] += p
    omitted_seed = None
    if reference_tokens is not None:
        if emitted == reference_tokens:
            omitted_seed = 0
        elif emitted == reference_tokens[1:]:
            omitted_seed = 1
        else:
            raise ValueError("native emitted IDs differ from exact target reference")
    if expected_counters is not None and any(totals[k] != expected_counters[k] for k in totals):
        raise ValueError("reconstructed counters differ from native request counters")
    return {"schema": "greedy_cross_round_replay_v1", "counters": totals,
            "quality_rounds": len(occupied), "checkpoint_replays": replay_count,
            "occupied_starts": occupied, "emitted_tokens": len(emitted),
            "omitted_target_seed": omitted_seed, "target_reference_checked": reference_tokens is not None,
            "native_counters_checked": expected_counters is not None,
            "exact_scope": "saved native trajectory accounting; not offline counterfactual model inference",
            "stochastic_EDR_theorem_claimed": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--request", type=Path, help="JSON with generated_token_ids and speculative counters")
    parser.add_argument("--collector", action="store_true",help="saved native collector format instead of server trace")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(s) for s in args.trace.read_text().splitlines() if s.strip()]
    request = json.loads(args.request.read_text()) if args.request else {}
    result = replay_collector(rows) if args.collector else replay_rounds(rows, request.get("generated_token_ids"), request.get("speculative"))
    result["trace"] = {"path": str(args.trace), "sha256": hashlib.sha256(args.trace.read_bytes()).hexdigest()}
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)


if __name__ == "__main__":
    main()

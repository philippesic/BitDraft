"""Serving joins must protect committed ancestry and retain successful states."""
import copy
import importlib.util
from pathlib import Path

import unittest
import tempfile
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

MODULE = Path(__file__).resolve().parents[1] / "scripts/w1a8_generalization/data.py"
spec = importlib.util.spec_from_file_location("generalization_data", MODULE)
data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data)


def history():
    rows = []
    tokens = list(range(40))
    for position, token in enumerate(tokens):
        rows.extend([{"event": "decoded_row", "feature_row": position, "task_id": 8,
                      "position": position, "token_id": token, "prefix_token_ids": tokens[:position + 1],
                      "decode_ordinal": position // 4, "feature_dim": data.FEATURE_DIM,
                      "target_layer_ids": data.TAPS, "feature_dtype": "float32_native_endian"},
                     {"event": "disposition", "feature_row": position, "task_id": 8,
                      "retained_input": True, "reason": "accepted_prefix"}])
    # A rejected speculative row at the same position must never replace the
    # committed feature or become a label, despite having a valid feature row.
    rows.extend([{"event": "decoded_row", "feature_row": 40, "task_id": 8, "position": 5,
                  "token_id": 999, "prefix_token_ids": [0, 1, 2, 3, 4, 999], "decode_ordinal": 1,
                  "feature_dim": data.FEATURE_DIM, "target_layer_ids": data.TAPS,
                  "feature_dtype": "float32_native_endian"},
                 {"event": "disposition", "feature_row": 40, "task_id": 8,
                  "retained_input": False, "reason": "rejected_suffix"}])
    return rows


def rounds():
    result = []
    for index, anchor in enumerate(range(8, 24)):
        accepted = 0 if index < 4 else 5
        proposal = list(range(anchor + 1, anchor + 6))
        if not accepted:
            proposal[0] = 999
        result.append({"task_id": 8, "round_index": index, "prefix_token_ids": list(range(anchor)),
                       "seed_token_id": anchor, "pos0": anchor, "draft_token_ids": proposal,
                       "accepted_drafts": accepted,
                       "verifier_token_ids": list(range(anchor + 1, anchor + accepted + 2))})
    return result


class ServingDataTests(unittest.TestCase):
    def test_rejected_rows_cannot_supply_context_or_labels(self):
        tasks, count = data.ledger(history())
        candidates = data.eligible_windows(rounds(), tasks[8], 8)
        assert count == 41
        assert all(40 not in row["feature_rows"] for row in candidates)
        assert all(999 not in row["labels"] and 999 not in row["prefix_tokens"] for row in candidates)
        assert candidates[0]["feature_rows"][5] == 5


    def test_corrupted_or_ambiguous_native_ancestry_is_rejected(self):
        for corruption in ["duplicate", "gap", "prefix", "disposition", "task"]:
            with self.subTest(corruption=corruption):
                rows = history()
                if corruption == "duplicate":
                    rows.append(copy.deepcopy(rows[0]))
                elif corruption == "gap":
                    rows = [r for r in rows if r["feature_row"] != 3]
                elif corruption == "prefix":
                    rows[12]["prefix_token_ids"][-1] = 999
                elif corruption == "disposition":
                    rows.pop(1)
                else:
                    rows[1]["task_id"] = 7
                with self.assertRaises(ValueError):
                    data.ledger(rows)


    def test_round_label_and_native_history_mismatches_are_rejected(self):
        for corruption in ["accepted", "rejection", "prefix", "pos0", "duplicate", "cross_task", "gap"]:
            with self.subTest(corruption=corruption):
                tasks, _ = data.ledger(history())
                rr = rounds()
                if corruption == "accepted":
                    rr[4]["draft_token_ids"][0] = 999
                elif corruption == "rejection":
                    rr[0]["draft_token_ids"][0] = rr[0]["verifier_token_ids"][0]
                elif corruption == "prefix":
                    rr[0]["prefix_token_ids"][0] = 999
                elif corruption == "pos0":
                    rr[0]["pos0"] += 1
                elif corruption == "duplicate":
                    rr.append(copy.deepcopy(rr[0]))
                elif corruption == "gap":
                    rr.pop(1)
                else:
                    rr[0]["task_id"] = 7
                with self.assertRaises(ValueError):
                    data.eligible_windows(rr, tasks[8], 8)


    def test_occupancy_policy_recovers_successes_and_preserves_round_mass(self):
        tasks, _ = data.ledger(history())
        candidates = data.eligible_windows(rounds(), tasks[8], 8)
        legacy = data.select(candidates, "legacy_errors", key="group")
        occupancy = data.select(candidates, "occupancy", key="group")
        assert len(legacy) == len(occupancy) == 4
        assert all(w["accepted_drafts"] == 0 for w in legacy)
        assert any(w["accepted_drafts"] == 5 for w in occupancy)
        assert occupancy == data.select(candidates, "occupancy", key="group")
        assert sum(w["occupancy_weight"] for w in occupancy) == len(candidates)
        assert all(w["inclusion_probability"] == 4 / len(candidates) for w in occupancy)


    def test_group_split_leakage_rejected_before_sampling(self):
        tasks, _ = data.ledger(history())
        cases = [{"index": i, "role": role, "source_split": "TRAIN", "group_id": "same",
                  "domain": "code", "chain_id": "chain", "prompt_tokens": list(range(8))}
                 for i, role in enumerate(("fit", "development"))]
        complete = {"status": "SERVING_CAPTURE_COMPLETE", "source_split": "TRAIN", "request_panels_used": False,
                    "records": [{"case_index": i, "task_id": i, "rounds": 16} for i in range(2)]}
        with self.assertRaisesRegex(ValueError, "split leakage"):
            data.join({"cases": cases, "request_panels_used": False}, complete, tasks, rounds(),
                      {"path": "unused", "sha256": "unused"}, policy="occupancy")


    def test_materialization_limit_prevents_large_allocation(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            manifest = {"windows": [{"anchor": 1000000}]}
            with self.assertRaisesRegex(ValueError, "byte cap"):
                data.materialize(manifest, tmp_path / "huge", max_bytes=1024)
            assert not (tmp_path / "huge").exists()

    def test_only_terminal_bonus_may_lack_a_decoded_feature(self):
        tasks, _ = data.ledger(history())
        rr = rounds()
        rr.append({"task_id": 8, "round_index": 16, "prefix_token_ids": list(range(36)),
                   "seed_token_id": 36, "pos0": 36, "draft_token_ids": [37, 38, 39],
                   "accepted_drafts": 3, "verifier_token_ids": [37, 38, 39, 40]})
        selected = data.eligible_windows(rr, tasks[8], 8)
        assert len(selected) == 16
        rr[-1]["verifier_token_ids"][0] = 999
        with self.assertRaises(ValueError):
            data.eligible_windows(rr, tasks[8], 8)

    def test_shared_rows_materialize_and_block_preserve_native_context(self):
        import numpy as np
        tasks, _ = data.ledger(history())
        candidate = data.eligible_windows(rounds(), tasks[8], 8)[0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = np.arange(41 * data.FEATURE_DIM, dtype="<f4").reshape(41, data.FEATURE_DIM)
            path = root / "shared.f32"
            raw.tofile(path)
            window = {**candidate, "features": data.pin(path), "chain_id": "chain", "case_index": 0}
            manifest = {"windows": [window]}
            result = data.block(window, {"sha256": "manifest"})
            np.testing.assert_array_equal(result.context_features, raw[:8].reshape(8, 5, 2560))
            np.testing.assert_array_equal(result.labels, list(range(9, 16)))
            materialized = data.materialize(manifest, root / "materialized", max_bytes=8 * data.FEATURE_DIM * 4)
            saved = data.read(materialized["path"])
            assert "feature_rows" not in saved["windows"][0]
            np.testing.assert_array_equal(data.feature_array(saved["windows"][0]), result.context_features)


if __name__ == "__main__":
    unittest.main()

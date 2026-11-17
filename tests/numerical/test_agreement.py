import tempfile
import unittest
from pathlib import Path

import numpy as np

from w1a1_eagle.contracts import Record, record
from w1a1_eagle.data import Artifact
from w1a1_eagle.numerical.agreement import agreement
from w1a1_eagle.records import canonical, identity, reference

TAPS = (2, 6, 10, 14, 18, 22, 26, 30)
PROMPT = [5, 6, 7, 8, 9, 10]
ANSWER = [11, 12, 13, 14, 15, 16]
EOS = 19


def cell(
    root: Path,
    features: "np.ndarray[tuple[int, ...], np.dtype[np.float16]]",
    metadata: Record,
    extra: dict[str, Record],
) -> tuple[Artifact, list[Record]]:
    root.mkdir(parents=True)
    files = []
    for index, tap in enumerate(TAPS):
        path = root / f"layer-{tap}.f16"
        path.write_bytes(features[index].tobytes())
        files.append(reference(Artifact.capture(path)))
    for name, value in extra.items():
        path = root / name
        path.write_bytes(canonical(value))
        files.append(reference(Artifact.capture(path)))
    path = root / "training-metadata.json"
    path.write_bytes(canonical(metadata))
    return Artifact.capture(path), files


def corpus(root: Path, metadata: Artifact, files: list[Record]) -> Artifact:
    path = root / "corpus.json"
    path.write_bytes(
        canonical(
            {
                "schema": 1,
                "target_sha256": "a" * 64,
                "protected": ["heldout24"],
                "expected_train": 1,
                "records": [
                    {
                        "id": "row",
                        "source": "fixture",
                        "group": "row-group",
                        "topic": "code",
                        "aliases": [],
                        "role": "train",
                        "ancestry": "verified",
                        "prior_use": "unknown",
                        "metadata": reference(metadata),
                        "files": files,
                    }
                ],
            }
        )
    )
    return Artifact.capture(path)


class TestReplayAgreement(unittest.TestCase):
    def test_agreement_reports_identity_labels_and_feature_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tokens = PROMPT + ANSWER
            features = (
                np.random.default_rng(3)
                .standard_normal((len(TAPS), len(tokens), 2560))
                .astype(np.float16)
            )
            source: Record = {"id": "row", "category": "code", "thinking": False}
            original: Record = {
                "taps": list(TAPS),
                "feature_semantics": "zero_based_pre_block_input_all_tokens",
                "feature_dtype": "float16",
                "input_tokens": tokens,
                "prompt_tokens": len(PROMPT),
                "logit_positions": list(range(len(PROMPT) - 1, len(tokens))),
                "greedy_ids": ANSWER + [EOS],
                "generated_tokens": ANSWER,
                "eog_ids": [EOS],
                "window_policy": "up-to-15",
                "target_sha256": "a" * 64,
            }
            reference_metadata, reference_files = cell(
                root / "reference" / "row", features, original, {"source.json": source}
            )
            reference_corpus = corpus(root / "reference", reference_metadata, reference_files)
            stream_sha = identity({"tokens": tokens})

            def replayed(name: str, values: object, changes: Record) -> Artifact:
                metadata = {
                    **original,
                    "supervised_spans": [[len(PROMPT) - 1, len(tokens) - 1]],
                    "replay": {"route": "token-replay-v1", "chunk": 32, "outputs": 512},
                    "generated_tokens": [],
                    "token_replay": {
                        "source_metadata": reference(reference_metadata),
                        "tokens_sha256": stream_sha,
                    },
                    **changes,
                }
                ref, files = cell(
                    root / name / "row",
                    np.asarray(values, dtype=np.float16),
                    metadata,
                    {"source.json": source, "token-stream.json": {"tokens_sha256": stream_sha}},
                )
                return corpus(root / name, ref, files)

            def compare(
                name: str, replay_corpus: Artifact, base: Artifact = reference_corpus
            ) -> Record:
                specification = root / f"{name}.json"
                specification.write_bytes(
                    canonical(
                        {
                            "schema": "replay-agreement-v1",
                            "reference": reference(base),
                            "replay": reference(replay_corpus),
                            "bands": [8, 64],
                            "rows": 5,
                        }
                    )
                )
                return agreement(specification, root / f"{name}-out")

            copy = replayed("copy", features, {})
            same = compare("same", copy)
            self.assertEqual(same["decision"], "bit-identical")
            self.assertTrue(same["identities_exact"] and same["labels_valid"])
            groups = record(same["groups"])
            self.assertEqual(list(groups), ["code|8-64"])
            windows = record(record(groups["code|8-64"])["windows_by_labels"])
            self.assertEqual(windows["reference"], windows["replay"])
            drifted = features.copy()
            drifted[3, 7, 11] += np.float16(0.5)
            labels = ANSWER[:2] + [99] + ANSWER[3:] + [EOS]
            moved = compare("moved", replayed("drift", drifted, {"greedy_ids": labels}))
            self.assertEqual(
                moved["decision"], "values differ: matched training and serving check required"
            )
            self.assertFalse(moved["bit_identical"] or moved["labels_identical"])
            summary = record(moved["summary"])
            self.assertEqual((summary["taps"], summary["bit_identical_taps"]), (8, 7))
            self.assertEqual(summary["labels_equal"], len(ANSWER))
            self.assertGreater(float(str(summary["maximum_absolute"])), 0.4)
            paired = compare("paired", replayed("drift-2", drifted, {}), copy)
            self.assertTrue(paired["identities_exact"])
            self.assertEqual(record(paired["summary"])["bit_identical_taps"], 7)
            other = list(tokens)
            other[2] = 70
            refused = compare("refused", replayed("tokens", features, {"input_tokens": other}))
            self.assertEqual(refused["decision"], "refused: identities or labels")

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from typing import TypeAlias

import gguf
import numpy as np

from w1a1_eagle.contracts import Record, record
from w1a1_eagle.data import Artifact
from w1a1_eagle.numerical.vocabulary_variants import (
    ROWS_ONLY_KEY,
    ROWS_TENSOR,
    SHORTLIST_KEY,
    TIED_KEY,
    shortlist,
    write_variants,
)
from w1a1_eagle.records import canonical, reference

ROWS = 8
COLUMNS = 256
QUANTIZER = """#!/usr/bin/env python3
import sys
widths = {"Q2_K": 84, "Q3_K": 110, "Q4_K": 144, "Q6_K": 210}
source, target, rows, columns, path, output = sys.argv[1:]
open(output, "wb").write(bytes(int(rows) * int(columns) // 256 * widths[target]))
"""


def authenticated(path: Path) -> Artifact:
    return Artifact(path, hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_size)


def write(path: Path, value: object) -> Artifact:
    path.write_bytes(canonical(value))
    return authenticated(path)


def request(prompt: list[int], answer: list[int]) -> Record:
    tokens = prompt + answer
    return {
        "input_tokens": tokens,
        "prompt_tokens": len(prompt),
        "logit_positions": list(range(len(prompt) - 1, len(tokens) - 1)),
        "greedy_ids": answer,
        "eog_ids": [7],
        "window_policy": "up-to-15",
    }


def mixture(root: Path, answers: list[list[int]], role: str = "train") -> Artifact:
    root.mkdir(parents=True, exist_ok=True)
    records = [
        {
            "id": f"r{index}",
            "group": f"g{index}",
            "topic": "code",
            "role": role,
            "metadata": reference(write(root / f"r{index}.json", request([1, 2], answer))),
            "files": [],
        }
        for index, answer in enumerate(answers)
    ]
    corpus = write(root / "corpus.json", {"target_sha256": "t", "records": records})
    return write(
        root / "mixture.json",
        {
            "schema": "corpus-mixture-v1",
            "target_sha256": "t",
            "components": [{"kind": "corpus-v1", "source": reference(corpus)}],
        },
    )


ROW_QUANTIZER = """#!/usr/bin/env python3
import sys
source, target, rows, columns, path, output = sys.argv[1:7]
single = sys.argv[7:] == ["1"]
data = bytearray()
for row in range(int(rows)):
    value = BROKEN if single and row == 2 else row
    data += bytes([value]) * (int(columns) // 256 * 144)
open(output, "wb").write(bytes(data))
"""


def fused_mixture(root: Path, answers: list[list[int]], role: str = "train") -> Artifact:
    root.mkdir(parents=True, exist_ok=True)
    records = [
        {
            "id": f"r{index}",
            "group": f"g{index}",
            "topic": "code",
            "role": role,
            "metadata": reference(write(root / f"r{index}.json", request([1, 2], answer))),
            "files": [],
        }
        for index, answer in enumerate(answers)
    ]
    corpus = write(
        root / "corpus.json",
        {"schema": "fused-corpus-v1", "target_sha256": "t", "records": records},
    )
    return write(
        root / "mixture.json",
        {
            "schema": "fused-mixture-v1",
            "target_sha256": "t",
            "components": [{"kind": "fused-corpus-v1", "source": reference(corpus)}],
        },
    )


def q6k_rows() -> "np.ndarray[tuple[int, ...], np.dtype[np.uint8]]":
    block = np.zeros(210, dtype=np.uint8)
    block[:128] = np.arange(128, dtype=np.uint8) * 3
    block[192:208] = 1
    block[208:210] = np.frombuffer(np.float16(0.5).tobytes(), dtype=np.uint8)
    return np.tile(block, (ROWS, COLUMNS // 256))


Payload: TypeAlias = "np.ndarray[tuple[int, ...], np.dtype[np.uint8 | np.int32 | np.float32]]"


def gguf_file(
    path: Path, tensors: dict[str, tuple[Payload, gguf.GGMLQuantizationType]], tied: bool = True
) -> Artifact:
    writer = gguf.GGUFWriter(str(path), "dflash")
    writer.add_string("dflash.w1ax.config_hash", "recipe")
    writer.add_uint32("dflash.block_size", 16)
    writer.add_array("dflash.target_layers", [2, 6])
    writer.add_bool(TIED_KEY, tied)
    for name, (data, kind) in tensors.items():
        writer.add_tensor(name, data, raw_dtype=kind)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return authenticated(path)


def export(root: Path, head: int = 0) -> Artifact:
    kinds = gguf.GGMLQuantizationType
    return gguf_file(
        root / "export.gguf",
        {
            "output_norm.weight": (np.ones(COLUMNS, dtype=np.float32), kinds.F32),
            "blk.0.ffn_gate.w1a1_packed": (np.arange(16, dtype=np.int32).reshape(2, 8), kinds.I32),
            "blk.0.attn_q.weight": (np.zeros((2, 18 * 8), dtype=np.uint8), kinds.Q4_0),
            "output.weight": (np.full((ROWS, 144), head, dtype=np.uint8), kinds.Q4_K),
        },
    )


def tensors(path: Path) -> dict[str, tuple[str, bytes]]:
    reader = gguf.GGUFReader(str(path))
    return {
        tensor.name: (tensor.tensor_type.name, np.asarray(tensor.data).tobytes())
        for tensor in reader.tensors
    }


def fields(path: Path) -> dict[str, object]:
    reader = gguf.GGUFReader(str(path))
    return {key: field.contents() for key, field in reader.fields.items()}


def specification(root: Path, value: Record) -> Path:
    path = root / "spec.json"
    path.write_text(json.dumps(value))
    return path


def tables(root: Path, variants: list[Record]) -> Record:
    script = root / "quantizer"
    script.write_text(QUANTIZER)
    script.chmod(0o755)
    built = write(root / "quantizer.json", {"binary": reference(authenticated(script))})
    source = gguf_file(
        root / "target.gguf", {"token_embd.weight": (q6k_rows(), gguf.GGMLQuantizationType.Q6_K)}
    )
    return {
        "source": reference(source),
        "tensor": "token_embd.weight",
        "quantizer": reference(built),
        "variants": variants,
    }


class TestShortlist(unittest.TestCase):
    def test_every_train_label_or_the_most_frequent_in_token_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = mixture(root / "train", [[5, 3, 7], [5, 4, 7], [6, 5, 7]])
            every = shortlist(source, None)
            self.assertEqual(every.tokens, (3, 4, 5, 6, 7))
            self.assertEqual((every.distinct, every.labels, every.size), (5, 9, None))
            frequent = shortlist(source, 2)
            self.assertEqual(frequent.tokens, (5, 7))
            self.assertAlmostEqual(frequent.coverage or 0.0, 6 / 9)
            self.assertEqual(every.payload(), np.asarray([3, 4, 5, 6, 7], dtype="<i4").tobytes())
            self.assertEqual(
                every.record()["tokens_sha256"], hashlib.sha256(every.payload()).hexdigest()
            )

    def test_refuses_development_data_and_invalid_sizes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            development = mixture(root / "dev", [[5, 7]], role="dev")
            with self.assertRaisesRegex(ValueError, "TRAIN"):
                shortlist(development, None)
            train = mixture(root / "train", [[5, 7]])
            with self.assertRaisesRegex(ValueError, "positive"):
                shortlist(train, 0)
            output = root / "out"
            spec = specification(
                root,
                {
                    "schema": "vocabulary-variants-v1",
                    "export": reference(export(root)),
                    "shortlist": {"mixture": reference(development), "size": None},
                },
            )
            with self.assertRaisesRegex(ValueError, "TRAIN"):
                write_variants(spec, output)
            self.assertFalse((output / "drafter-shortlist.gguf").exists())


class TestShortlistRoundTrip(unittest.TestCase):
    def test_shortlist_metadata_round_trip_keeps_every_tensor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = export(root)
            train = mixture(root / "train", [[5, 3, 7], [5, 4, 7]])
            spec = specification(
                root,
                {
                    "schema": "vocabulary-variants-v1",
                    "export": reference(source),
                    "shortlist": {"mixture": reference(train), "size": None},
                },
            )
            result = write_variants(spec, root / "out")
            produced = root / "out" / "drafter-shortlist.gguf"
            self.assertEqual(tensors(produced), tensors(source.path))
            values = fields(produced)
            self.assertEqual(values[SHORTLIST_KEY], [3, 4, 5, 7])
            self.assertEqual(values[SHORTLIST_KEY + ".size"], 0)
            self.assertEqual(values[SHORTLIST_KEY + ".mixture"], train.sha256)
            self.assertEqual(values["dflash.w1ax.config_hash"], "recipe")
            self.assertEqual(values["dflash.target_layers"], [2, 6])
            self.assertIs(values[TIED_KEY], True)
            sidecar = (root / "out" / "shortlist.i32").read_bytes()
            self.assertEqual(sidecar, np.asarray([3, 4, 5, 7], dtype="<i4").tobytes())
            summary = record(record(result["outputs"])["shortlist"])
            self.assertEqual(summary["rows"], 4)
            again = specification(
                root,
                {
                    "schema": "vocabulary-variants-v1",
                    "export": reference(authenticated(produced)),
                    "shortlist": {"mixture": reference(train), "size": 2},
                },
            )
            with self.assertRaisesRegex(ValueError, "without a shortlist"):
                write_variants(again, root / "out-again")


class TestTableVariants(unittest.TestCase):
    def test_shared_and_scoring_layouts_read_back(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = export(root)
            spec = specification(
                root,
                {
                    "schema": "vocabulary-variants-v1",
                    "export": reference(source),
                    "tables": tables(
                        root,
                        [
                            {"type": "Q3_K", "layout": "shared"},
                            {"type": "Q2_K", "layout": "scoring"},
                        ],
                    ),
                },
            )
            result = write_variants(spec, root / "out")
            original = tensors(source.path)
            shared = tensors(root / "out" / "table-q3k-shared.gguf")
            self.assertEqual(shared["output.weight"], ("Q3_K", bytes(ROWS * 110)))
            self.assertNotIn("token_embd.weight", shared)
            self.assertEqual(
                {k: v for k, v in shared.items() if k != "output.weight"},
                {k: v for k, v in original.items() if k != "output.weight"},
            )
            self.assertIs(fields(root / "out" / "table-q3k-shared.gguf")[TIED_KEY], True)
            scoring_path = root / "out" / "table-q2k-scoring.gguf"
            scoring = tensors(scoring_path)
            self.assertEqual(scoring["output.weight"], ("Q2_K", bytes(ROWS * 84)))
            self.assertEqual(scoring["token_embd.weight"], original["output.weight"])
            values = fields(scoring_path)
            self.assertIs(values[TIED_KEY], False)
            self.assertEqual(values["dflash.vocabulary.type"], "Q2_K")
            self.assertEqual(values["dflash.vocabulary.layout"], "scoring")
            report = record(record(result["outputs"])["tables"])
            self.assertIs(report["reproduced_export_table"], True)
            self.assertEqual(record(report["relative_error"])["Q2_K"], 1.0)
            self.assertFalse((root / "out" / "scratch").exists())

    def test_refuses_a_source_that_does_not_reproduce_the_export_table(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = specification(
                root,
                {
                    "schema": "vocabulary-variants-v1",
                    "export": reference(export(root, head=1)),
                    "tables": tables(root, [{"type": "Q2_K", "layout": "shared"}]),
                },
            )
            with self.assertRaisesRegex(ValueError, "reproduce"):
                write_variants(spec, root / "out")
            self.assertFalse((root / "out" / "table-q2k-shared.gguf").exists())
            unsupported = specification(
                root,
                {
                    "schema": "vocabulary-variants-v1",
                    "export": reference(export(root)),
                    "tables": tables(root, [{"type": "Q4_0", "layout": "shared"}]),
                },
            )
            with self.assertRaisesRegex(ValueError, "Q2_K or Q3_K"):
                write_variants(unsupported, root / "out-2")


def indexed_export(root: Path) -> Artifact:
    kinds = gguf.GGMLQuantizationType
    head = np.repeat(np.arange(ROWS, dtype=np.uint8)[:, None], 144, axis=1)
    return gguf_file(
        root / "export.gguf",
        {
            "output_norm.weight": (np.ones(COLUMNS, dtype=np.float32), kinds.F32),
            "blk.0.ffn_gate.w1a1_packed": (np.arange(16, dtype=np.int32).reshape(2, 8), kinds.I32),
            "output.weight": (head, kinds.Q4_K),
        },
    )


def rows_only(root: Path, broken: int) -> Record:
    script = root / "row-quantizer"
    script.write_text(ROW_QUANTIZER.replace("BROKEN", str(broken)))
    script.chmod(0o755)
    built = write(root / "row-quantizer.json", {"binary": reference(authenticated(script))})
    source = gguf_file(
        root / "target.gguf", {"token_embd.weight": (q6k_rows(), gguf.GGMLQuantizationType.Q6_K)}
    )
    return {
        "source": reference(source),
        "tensor": "token_embd.weight",
        "quantizer": reference(built),
    }


class TestFusedLabels(unittest.TestCase):
    def test_fused_mixture_labels_and_refusals(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fused = fused_mixture(root / "fused", [[5, 3, 7], [5, 4, 7]])
            self.assertEqual(shortlist(fused, None).tokens, (3, 4, 5, 7))
            with self.assertRaisesRegex(ValueError, "TRAIN"):
                shortlist(fused_mixture(root / "dev", [[5, 7]], role="dev"), None)
            plain = mixture(root / "plain", [[5, 7]])
            raw = json.loads(plain.path.read_text())
            raw["schema"] = "fused-mixture-v1"
            with self.assertRaisesRegex(ValueError, "fused-corpus-v1"):
                shortlist(write(root / "mixed.json", raw), None)


class TestRowsOnly(unittest.TestCase):
    def test_shortlist_only_file_holds_the_export_rows_and_no_full_table(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = indexed_export(root)
            train = mixture(root / "train", [[5, 3, 7], [5, 4, 7]])
            spec = specification(
                root,
                {
                    "schema": "vocabulary-variants-v1",
                    "export": reference(source),
                    "shortlist": {"mixture": reference(train), "size": None},
                    "rows_only": rows_only(root, 2),
                },
            )
            result = write_variants(spec, root / "out")
            produced = root / "out" / "drafter-shortlist-only.gguf"
            original = tensors(source.path)
            stored = tensors(produced)
            self.assertNotIn("output.weight", stored)
            self.assertEqual(
                stored[ROWS_TENSOR], ("Q4_K", b"".join(bytes([t]) * 144 for t in (3, 4, 5, 7)))
            )
            self.assertEqual(
                {k: v for k, v in stored.items() if k != ROWS_TENSOR},
                {k: v for k, v in original.items() if k != "output.weight"},
            )
            values = fields(produced)
            self.assertIs(values[ROWS_ONLY_KEY], True)
            self.assertIs(values[TIED_KEY], True)
            self.assertEqual(values[SHORTLIST_KEY], [3, 4, 5, 7])
            self.assertEqual(values["dflash.w1ax.config_hash"], "recipe")
            self.assertEqual(
                fields(root / "out" / "drafter-shortlist.gguf")[SHORTLIST_KEY], [3, 4, 5, 7]
            )
            report = record(record(result["outputs"])["rows_only"])
            self.assertEqual(record(report["readback"])["rows_compared"], 4)
            self.assertEqual(report["rows_bytes"], 4 * 144)
            self.assertEqual(report["full_table_bytes"], ROWS * 144)
            proof = json.loads((root / "out" / "row-independence.json").read_text())
            self.assertEqual((proof["rows_checked"], proof["differing_rows"]), (ROWS, 0))
            self.assertFalse((root / "out" / "scratch-rows").exists())

    def test_refuses_rows_that_change_when_converted_alone(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = mixture(root / "train", [[5, 3, 7]])
            spec = specification(
                root,
                {
                    "schema": "vocabulary-variants-v1",
                    "export": reference(indexed_export(root)),
                    "shortlist": {"mixture": reference(train), "size": None},
                    "rows_only": rows_only(root, 99),
                },
            )
            with self.assertRaisesRegex(ValueError, "not independent"):
                write_variants(spec, root / "out")
            self.assertFalse((root / "out" / "drafter-shortlist-only.gguf").exists())
            alone = specification(
                root,
                {
                    "schema": "vocabulary-variants-v1",
                    "export": reference(indexed_export(root)),
                    "tables": tables(root, [{"type": "Q2_K", "layout": "shared"}]),
                    "rows_only": rows_only(root, 2),
                },
            )
            with self.assertRaisesRegex(ValueError, "needs a shortlist"):
                write_variants(alone, root / "out-2")

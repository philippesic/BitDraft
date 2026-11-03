import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch

PATH = Path(__file__).resolve().parents[1] / "scripts/w1a8_generalization/recovery.py"
definition = importlib.util.spec_from_file_location("generalization_recovery", PATH)
recovery = importlib.util.module_from_spec(definition)
definition.loader.exec_module(recovery)


def publish(path, state, verify):
    temporary = path.with_suffix(".partial")
    torch.save(state, temporary)
    verify(torch.load(temporary, weights_only=False))
    temporary.rename(path)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size}


class RecoveryTests(unittest.TestCase):
    def test_failed_publication_preserves_last_usable_state(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            writer = recovery.RecoveryCheckpoints(out, publish, lambda a, b: self.assertEqual(a, b),
                                                   bound_bytes=1024**2, floor_bytes=0)
            ref = writer.publish(4, {"local_updates": 4, "cursor": 2})
            def interrupted(*args):
                raise RuntimeError("injected interruption")
            writer.publisher = interrupted
            with self.assertRaises(RuntimeError):
                writer.publish(8, {"local_updates": 8, "cursor": 4})
            self.assertEqual(json.loads((out / "latest.json").read_text())["path"], ref["path"])
            self.assertTrue(Path(ref["path"]).exists())
            self.assertEqual(torch.load(ref["path"], weights_only=False)["cursor"], 2)

    def test_retention_preserves_newest_two_after_verified_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            writer = recovery.RecoveryCheckpoints(out, publish, lambda a, b: self.assertEqual(a, b),
                                                   bound_bytes=1024**2, floor_bytes=0)
            refs = [writer.publish(i, {"local_updates": i, "cursor": i // 2}) for i in (4, 8, 12)]
            self.assertFalse(Path(refs[0]["path"]).exists())
            self.assertTrue(all(Path(r["path"]).exists() for r in refs[1:]))
            self.assertEqual(json.loads((out / "latest.json").read_text())["local_updates"], 12)
            history = json.loads((out / "checkpoint-history.json").read_text())
            self.assertTrue(history[0]["retired"])
            self.assertEqual(history[0]["replacement"]["sha256"], refs[2]["sha256"])

    def test_failed_verification_keeps_old_pointer_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            writer = recovery.RecoveryCheckpoints(out, publish, lambda a, b: self.assertEqual(a, b),
                                                   bound_bytes=1024**2, floor_bytes=0)
            old = writer.publish(4, {"local_updates": 4})
            writer.verify = lambda a, b: (_ for _ in ()).throw(ValueError("bad roundtrip"))
            with self.assertRaises(ValueError):
                writer.publish(8, {"local_updates": 8})
            self.assertTrue(Path(old["path"]).exists())
            self.assertEqual(json.loads((out / "latest.json").read_text())["sha256"], old["sha256"])

    def test_chunked_hash_matches_original_bytes_including_strided_tensor(self):
        model = SimpleNamespace(weight=torch.arange(96, dtype=torch.float32).reshape(8, 12)[:, ::2])
        expected = hashlib.sha256(model.weight.contiguous().numpy().tobytes()).hexdigest()
        refs = {"weight": {"actor_f32_sha256": expected, "shape": [8, 6]}}
        slots = lambda m: {"weight": (m, "weight")}
        result = recovery.chunked_frozen(model, refs, slots, chunk_bytes=48)
        self.assertEqual(result["weight"]["sha256"], expected)
        model.weight[3, 2] = -1
        with self.assertRaises(ValueError):
            recovery.chunked_frozen(model, refs, slots, chunk_bytes=48)


if __name__ == "__main__":
    unittest.main()

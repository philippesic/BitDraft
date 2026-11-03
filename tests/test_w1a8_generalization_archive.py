import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

DIRECTORY = Path(__file__).resolve().parents[1] / "scripts/w1a8_generalization"
sys.path.insert(0, str(DIRECTORY))
definition = importlib.util.spec_from_file_location("generalization_archive", DIRECTORY / "archive.py")
archive = importlib.util.module_from_spec(definition)
definition.loader.exec_module(archive)


class ArchiveTests(unittest.TestCase):
    def fixture(self, root):
        output = root / "run"
        output.mkdir()
        source = output / "checkpoint-00000008.pt"
        source.write_bytes(bytes(range(256)) * 2000)
        state = root / "supervisor" / "state.json"
        state.parent.mkdir()
        state.write_text(json.dumps({"status": "finished", "exit_code": 0,
                                     "command": ["train.py", "--output", str(output)]}))
        state.with_name("identity.json").write_text(json.dumps({
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            "supervisor": {"pid": 999999991, "start_ticks": 1},
            "child": {"pid": 999999992, "start_ticks": 1}}))
        return source, state, {"path": str(source), "sha256": archive.digest(source)}

    def test_paid_state_is_recoverable_after_explicit_archival(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, state, ref = self.fixture(root)
            receipt = archive.archive(ref, root / "saved.pt.gz", owned_root=root,
                                      owner_state=state, remove_source=True)
            self.assertFalse(source.exists())
            self.assertTrue(receipt["gzip_roundtrip_verified"])
            result = archive.restore(receipt, root / "restored.pt")
            self.assertEqual(result["sha256"], ref["sha256"])
            self.assertEqual(hashlib.sha256((root / "restored.pt").read_bytes()).hexdigest(), ref["sha256"])
            self.assertTrue(json.loads((root / "saved.pt.gz.json").read_text())["source_removed"])

    def test_changed_original_or_archive_never_replaces_good_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, state, ref = self.fixture(root)
            bad = {**ref, "sha256": "0" * 64}
            with self.assertRaises(ValueError):
                archive.archive(bad, root / "bad.pt.gz", owned_root=root, owner_state=state, remove_source=True)
            self.assertTrue(source.exists())
            receipt = archive.archive(ref, root / "saved.pt.gz", owned_root=root, owner_state=state)
            (root / "saved.pt.gz").write_bytes(b"corrupt")
            with self.assertRaises(ValueError):
                archive.restore(receipt, root / "restored.pt")
            self.assertFalse((root / "restored.pt").exists())
            self.assertTrue(source.exists())

    def test_another_run_or_live_owner_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, state, ref = self.fixture(root)
            changed = json.loads(state.read_text())
            changed["command"][-1] = str(root / "other")
            state.write_text(json.dumps(changed))
            with self.assertRaises(ValueError):
                archive.archive(ref, root / "saved.pt.gz", owned_root=root, owner_state=state, remove_source=True)
            changed["command"][-1] = str(source.parent)
            state.write_text(json.dumps(changed))
            owner = json.loads(state.with_name("identity.json").read_text())
            birth = int(Path(f"/proc/{os.getpid()}/stat").read_text().rsplit(")", 1)[1].split()[19])
            owner["child"] = {"pid": os.getpid(), "start_ticks": birth}
            state.with_name("identity.json").write_text(json.dumps(owner))
            with self.assertRaises(ValueError):
                archive.archive(ref, root / "saved.pt.gz", owned_root=root, owner_state=state, remove_source=True)
            self.assertTrue(source.exists())


if __name__ == "__main__":
    unittest.main()

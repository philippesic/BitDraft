"""CPU/stdlb fixtures only: never import torch or execute the real trainer."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("diagnostic_phase_trace", HERE / "phase_trace_runner.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class PhaseTraceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.script = self.root / "scripts" / "fixture.py"
        self.script.parent.mkdir()
        self.journal = self.root / "trace.jsonl"

    def tearDown(self):
        self.temp.cleanup()

    def run_fixture(self, body, **kwargs):
        self.script.write_text(body)
        runner.execute(self.script, self.journal, **kwargs)
        return self.records()

    def records(self):
        return [json.loads(line) for line in self.journal.read_text().splitlines()]

    def test_nested_begin_is_durable_before_body_and_end_order(self):
        code = '''
import json
from pathlib import Path
journal = Path(JOURNAL)
def block_inputs():
    records = [json.loads(line) for line in journal.read_text().splitlines()]
    assert records[-1]['event'] == 'BEGIN'
    assert records[-1]['phase'] == 'block_inputs'
    assert not any(r['event'] == 'END' for r in records)
def run_block():
    block_inputs()
run_block()
'''.replace("JOURNAL", repr(str(self.journal)))
        syncs = []
        real_fsync = os.fsync
        def sync(fd):
            real_fsync(fd)
            syncs.append(self.journal.read_bytes())
        with patch.object(runner.os, "fsync", sync):
            records = self.run_fixture(code)
        phases = [(r["event"], r.get("phase")) for r in records if r["event"] in {"BEGIN", "END"}]
        self.assertEqual(phases, [("BEGIN", "run_block"), ("BEGIN", "block_inputs"),
                                  ("END", "block_inputs"), ("END", "run_block")])
        self.assertEqual(len(syncs), len(records))
        self.assertEqual(json.loads(syncs[3].splitlines()[-1])["phase"], "block_inputs")
        self.assertEqual(records[3]["parent_call_id"], records[2]["call_id"])
        self.assertEqual(records[0]["script_sha256"], runner.digest(self.script))
        self.assertEqual(records[0]["wrapper_sha256"], runner.digest(HERE / "phase_trace_runner.py"))

    def test_exception_identity_propagation_finally_and_argv_restoration(self):
        original_argv, original_path = sys.argv, sys.path[:]
        original_error = RuntimeError("private exception detail")
        support = types.ModuleType("phase_fixture_support")
        support.original_error = original_error
        support.seen = []
        self.script.write_text('''
import sys
from phase_fixture_support import original_error, seen
def block_inputs():
    try:
        raise original_error
    finally:
        seen.append('finally')
def run_block():
    block_inputs()
seen.append(sys.argv[:])
sys.argv = ['changed by fixture']
run_block()
''')
        with patch.dict(sys.modules, {"phase_fixture_support": support}):
            with self.assertRaises(RuntimeError) as caught:
                runner.execute(self.script, self.journal, ["--secret", "private-argument"])
        self.assertIs(caught.exception, original_error)
        self.assertIs(sys.argv, original_argv)
        self.assertEqual(sys.path, original_path)
        self.assertEqual(support.seen, [[str(self.script.resolve()), "--secret", "private-argument"], "finally"])
        records = self.records()
        events = [(r["event"], r.get("phase")) for r in records if r.get("phase")]
        self.assertEqual(events, [("BEGIN", "run_block"), ("BEGIN", "block_inputs"),
                                  ("EXCEPTION", "block_inputs"), ("EXCEPTION", "run_block")])
        self.assertEqual(records[-1]["event"], "SCRIPT_EXCEPTION")
        self.assertNotIn("private-argument", self.journal.read_text())
        self.assertNotIn("private exception detail", self.journal.read_text())
        self.assertIsNone(sys.getprofile())
        self.assertIsNone(sys.gettrace())

    def test_caught_exception_return_none_is_end(self):
        records = self.run_fixture('''
def block_inputs():
    try:
        raise ValueError('not logged')
    except ValueError:
        return None
block_inputs()
''')
        self.assertEqual([r["event"] for r in records if r.get("phase")], ["BEGIN", "END"])

    def test_exception_replaced_in_finally(self):
        self.script.write_text('''
def block_inputs():
    try:
        raise ValueError('not logged')
    finally:
        raise LookupError('not logged either')
block_inputs()
''')
        with self.assertRaises(LookupError):
            runner.execute(self.script, self.journal)
        self.assertEqual([r for r in self.records() if r["event"] == "EXCEPTION"][0]["exception_type"],
                         "builtins.LookupError")

    def test_event_and_byte_caps_are_visible_and_bounded(self):
        for event_limit, byte_limit in [(8, 50000), (10000, 4096)]:
            with self.subTest(event_limit=event_limit):
                self.journal.unlink(missing_ok=True)
                records = self.run_fixture('''
def block_inputs():
    return None
for _ in range(200):
    block_inputs()
''', max_events=event_limit, max_bytes=byte_limit)
                self.assertEqual(records[-1]["event"], "LOG_CAP")
                self.assertLessEqual(len(records), event_limit)
                self.assertLessEqual(self.journal.stat().st_size, byte_limit)
                self.assertEqual(records[-1]["coverage_after_cap"], "none; pending BEGIN markers are inconclusive")

    def test_journal_reentrancy_and_write_failure_dont_replace_target_error(self):
        journal = runner.Journal(self.journal)
        original_encode = journal._encode
        def encode(record):
            self.assertFalse(journal.emit("REENTRANT"))
            return original_encode(record)
        journal._encode = encode
        self.assertTrue(journal.emit("OUTER"))
        self.assertEqual(journal.count, 1)
        journal.close()
        self.assertEqual(self.records()[0]["event"], "OUTER")
        self.journal.unlink()
        original_emit = runner.Journal.emit
        def fail_after_start(instance, event, **fields):
            if event == "BEGIN":
                with patch.object(runner.os, "fsync", side_effect=OSError("fixture failure")):
                    return original_emit(instance, event, **fields)
            return original_emit(instance, event, **fields)
        self.script.write_text("def block_inputs():\n    raise KeyError('original')\nblock_inputs()\n")
        with patch.object(runner.Journal, "emit", fail_after_start):
            with self.assertRaisesRegex(KeyError, "original"):
                runner.execute(self.script, self.journal)
        self.assertIsNone(sys.getprofile())
        self.assertIsNone(sys.gettrace())

    def test_rejects_existing_observer_and_existing_journal(self):
        self.script.write_text("raise AssertionError('must not run')\n")
        self.journal.write_text("untouched")
        with self.assertRaises(FileExistsError):
            runner.execute(self.script, self.journal)
        self.assertEqual(self.journal.read_text(), "untouched")
        def observer(frame, event, arg):
            pass
        sys.setprofile(observer)
        try:
            with self.assertRaisesRegex(RuntimeError, "existing profile/trace"):
                runner.execute(self.script, self.root / "fresh.jsonl")
            self.assertIs(sys.getprofile(), observer)
        finally:
            sys.setprofile(None)

    def test_module_cuda_and_tensor_selection_without_torch(self):
        # Synthetic Python code verifies matching, with no torch dependency/import.
        records = self.run_fixture('''
class BlockDrafter:
    def __init__(self):
        pass
BlockDrafter()
__name__ = 'torch.nn.modules.module'
class Module:
    def _apply(self):
        pass
    def to(self):
        self._apply()
Module().to()
__name__ = 'torch.cuda'
def _lazy_init():
    pass
_lazy_init()
''')
        self.assertEqual([r["phase"] for r in records if r["event"] == "BEGIN"],
                         ["BlockDrafter.__init__", "Module.to", "Module._apply", "torch.cuda._lazy_init"])
        # Direct classification fixture: no repr/property or tensor metadata read.
        class HostileTensor:
            __module__ = "torch"
            def __repr__(self):
                raise AssertionError("no repr")
        fake_builtin = types.SimpleNamespace(__name__="copy_", __module__=None, __self__=HostileTensor())
        self.assertEqual(runner.PhaseTracer.c_phase(fake_builtin), "Tensor.copy_")

    def test_preclock_smoke_resource_and_rng_source_phases(self):
        records = self.run_fixture("""
def _block_rng_state():
    pass
def rng_state():
    _block_rng_state()
def _restore_block_rng():
    pass
def restore_rng():
    _restore_block_rng()
def cuda_memory_snapshot():
    pass
def resources():
    cuda_memory_snapshot()
def block_transaction_resources():
    resources()
def smoke_block():
    resources()
def smoke_with_training_memory():
    smoke_block()
    block_transaction_resources()
def run_block():
    rng_state()
    smoke_with_training_memory()
    restore_rng()
run_block()
""")
        expected = ["run_block", "rng_state", "_block_rng_state",
                    "smoke_with_training_memory", "smoke_block", "resources",
                    "cuda_memory_snapshot", "block_transaction_resources", "resources",
                    "cuda_memory_snapshot", "restore_rng", "_restore_block_rng"]
        self.assertEqual([r["phase"] for r in records if r["event"] == "BEGIN"], expected)
        begins = {r["call_id"]: r["phase"] for r in records if r["event"] == "BEGIN"}
        ends = {r["call_id"]: r["phase"] for r in records if r["event"] == "END"}
        self.assertEqual(begins, ends)

    def test_fake_autograd_tensor_backward_and_cuda_query_boundaries(self):
        # Each synthetic function has independent module globals and a filename
        # outside source_root, so this checks the PyTorch matcher specifically.
        code = r"""
import json
from pathlib import Path
journal = Path(JOURNAL)
def fixture_function(module, name, expected_phase):
    namespace = {'__name__': module, 'journal': journal, 'json': json,
                 'expected_phase': expected_phase}
    body = ('def ' + name + '():\n'
            '    records = [json.loads(line) for line in journal.read_text().splitlines()]\n'
            '    assert records[-1]["event"] == "BEGIN"\n'
            '    assert records[-1]["phase"] == expected_phase\n')
    exec(compile(body, '<cpu_fake_torch_module>', 'exec'), namespace)
    return namespace[name]
for module, name, phase in [
    ('torch.autograd', 'grad', 'torch.autograd.grad'),
    ('torch.autograd', 'backward', 'torch.autograd.backward'),
    ('torch._tensor', 'backward', 'Tensor.backward'),
    ('torch.cuda.memory', 'mem_get_info', 'torch.cuda.mem_get_info'),
    ('torch.cuda', 'mem_get_info', 'torch.cuda.mem_get_info'),
    ('torch.cuda', 'get_device_properties', 'torch.cuda.get_device_properties'),
]:
    fixture_function(module, name, phase)()
""".replace("JOURNAL", repr(str(self.journal)))
        records = self.run_fixture(code)
        expected = ["torch.autograd.grad", "torch.autograd.backward", "Tensor.backward",
                    "torch.cuda.mem_get_info", "torch.cuda.mem_get_info",
                    "torch.cuda.get_device_properties"]
        self.assertEqual([r["phase"] for r in records if r["event"] == "BEGIN"], expected)
        self.assertEqual([r["phase"] for r in records if r["event"] == "END"], expected)

    def test_cuda_c_memoryinfo_and_deviceprops_classification_without_queries(self):
        for name, module in [("cudaMemGetInfo", "torch._C._cudart"),
                             ("_cuda_getDeviceProperties", "torch._C")]:
            function = types.SimpleNamespace(__name__=name, __module__=module, __self__=None)
            self.assertEqual(runner.PhaseTracer.c_phase(function), "torch._C." + name)
        class OpaqueCudartOwner:
            __module__ = "torch._C._cudart"
            def __getattribute__(self, name):
                raise AssertionError("no owner attributes or tensor metadata")
        function = types.SimpleNamespace(__name__="cudaMemGetInfo", __module__=None,
                                         __self__=OpaqueCudartOwner())
        self.assertEqual(runner.PhaseTracer.c_phase(function), "torch._C.cudaMemGetInfo")
        unrelated = types.SimpleNamespace(__name__="cudaMemGetInfo", __module__="unrelated",
                                          __self__=None)
        self.assertIsNone(runner.PhaseTracer.c_phase(unrelated))

    def test_actual_cpu_c_call_end_and_exception_pairing(self):
        # Test-only classifier selects a real CPU builtin for profile c_* events.
        # Production classifier and immutable source functions remain untouched.
        with patch.object(runner.PhaseTracer, "c_phase", staticmethod(
                lambda fn: "cpu_fixture.index" if fn.__name__ == "index" else None)):
            records = self.run_fixture("'abc'.index('b')\ntry:\n    'abc'.index('z')\nexcept ValueError:\n    pass\n")
        events = [r for r in records if r.get("phase") == "cpu_fixture.index"]
        self.assertEqual([r["event"] for r in events], ["BEGIN", "END", "BEGIN", "EXCEPTION"])
        self.assertEqual(events[0]["call_id"], events[1]["call_id"])
        self.assertEqual(events[2]["call_id"], events[3]["call_id"])

    def test_cli_preserves_script_arguments_and_systemexit(self):
        self.script.write_text("import sys\nassert sys.argv[1:] == ['--token', 'redacted-value']\nraise SystemExit(7)\n")
        result = subprocess.run([sys.executable, str(HERE / "phase_trace_runner.py"),
                                 "--script", str(self.script), "--journal", str(self.journal),
                                 "--", "--token", "redacted-value"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 7, result.stderr)
        self.assertEqual(self.records()[-1]["exception_type"], "builtins.SystemExit")
        self.assertNotIn("redacted-value", self.journal.read_text())

    def test_import_has_no_target_execution_or_torch_import(self):
        result = subprocess.run([sys.executable, "-c",
            "import importlib.util, sys; "
            f"s=importlib.util.spec_from_file_location('phase', {str(HERE / 'phase_trace_runner.py')!r}); "
            "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
            "assert 'torch' not in sys.modules; assert sys.getprofile() is None; assert sys.gettrace() is None"],
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)

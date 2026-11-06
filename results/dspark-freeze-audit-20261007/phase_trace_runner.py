#!/usr/bin/env python3
"""Diagnostic instrumentation, never source-neutral production/performance evidence.

Importing this stdlib-only module never imports torch or executes a target. The
CLI executes the requested script; launching the real training script requires
separate human authorization and resource-owner coordination. No tensor values,
metadata, argument values, environment, or exception messages are inspected.
"""
from __future__ import annotations

import argparse
import dis
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import threading
import time

PHASES = frozenset({
    "run_block", "block_inputs", "block_optimizer", "load_block_checkpoint",
    "block_train_batch", "save_block_checkpoint", "validate_source",
    "_runtime", "_validate_optimizer", "_validate_block_rng",
    "training_source_identity", "configure_cuda", "validate_initializer_reference",
    "training_runtime_identity", "record_runtime_observation",
    "validate_optimization_readiness",
    "smoke_with_training_memory", "smoke_block", "resources",
    "cuda_memory_snapshot", "block_transaction_resources",
    "rng_state", "restore_rng", "_block_rng_state", "_restore_block_rng",
})
CUDA_PYTHON = frozenset({
    "init", "_lazy_init", "synchronize", "empty_cache",
    "mem_get_info", "get_device_properties",
})
CUDA_C = frozenset({
    "_cuda_init", "_cuda_synchronize", "_cuda_emptyCache",
    "_cuda_getDeviceProperties", "cudaMemGetInfo",
})
TENSOR_C = frozenset({"to", "copy_", "copy", "_to_copy"})
NORMAL_RETURN = frozenset({"RETURN_VALUE", "RETURN_CONST"})
CAP_RESERVE = 1024


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def boot_identity():
    # Read-only kernel file, no subprocess, CUDA query, or environment scrape.
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError:
        return None  # Non-Linux: explicitly unavailable, never invent an identity.


class Journal:
    """Exclusive new file; each complete marker is flushed and fsynced."""
    def __init__(self, path, *, max_events=20000, max_bytes=8 * 1024 * 1024):
        if max_events < 4 or max_bytes < 4096:
            raise ValueError("journal limits require >= 4 events and >= 4096 bytes")
        self.path = Path(path)
        self.stream = self.path.open("xb", buffering=0)
        self.max_events, self.max_bytes = max_events, max_bytes
        self.count = self.bytes = 0
        self.capped = self.failed = False
        self.busy = False
        self.owner_thread = threading.get_ident()
        self.pid, self.boot = os.getpid(), boot_identity()

    def _encode(self, record):
        return (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode()

    def _record(self, event, fields):
        return dict(schema="diagnostic_phase_trace_v1", event=event,
                    sequence=self.count + 1, pid=self.pid, boot_id=self.boot,
                    wall_time_ns=time.time_ns(), monotonic_ns=time.monotonic_ns(),
                    **fields)

    def _write(self, payload):
        # FileIO.write may return a short count. Complete the line before fsync.
        offset = 0
        while offset < len(payload):
            written = self.stream.write(payload[offset:])
            if not written:
                raise OSError("short journal write")
            offset += written
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.count += 1
        self.bytes += len(payload)

    def emit(self, event, **fields):
        if self.busy or self.failed or self.capped:
            return False
        self.busy = True
        try:
            payload = self._encode(self._record(event, fields))
            if self.count >= self.max_events - 1 or self.bytes + len(payload) > self.max_bytes - CAP_RESERVE:
                cap = self._encode(self._record("LOG_CAP", {
                    "reason": "event_or_byte_limit", "max_events": self.max_events,
                    "max_bytes": self.max_bytes, "events_before_cap": self.count,
                    "bytes_before_cap": self.bytes,
                    "coverage_after_cap": "none; pending BEGIN markers are inconclusive",
                }))
                self._write(cap)
                self.capped = True
                return False
            self._write(payload)
            return True
        except Exception:
            # A callback must not replace the target's original exception.
            # Fail open but announce lost coverage; never present a failed trace as complete.
            self.failed = True
            try:
                os.write(2, b"PHASE_TRACE_JOURNAL_ERROR: tracing disabled; target continues\n")
            except OSError:
                pass
            return False
        finally:
            self.busy = False

    def close(self):
        self.stream.close()


class PhaseTracer:
    def __init__(self, journal, source_root):
        self.journal = journal
        self.source_root = str(Path(source_root).resolve()) + os.sep
        self.frames = {}
        self.c_frames = []
        self.next_call = 0
        self.busy = False

    def python_phase(self, frame):
        code = frame.f_code
        name = code.co_name
        module = frame.f_globals.get("__name__", "")
        filename = code.co_filename
        qualname = getattr(code, "co_qualname", name)
        if filename.startswith(self.source_root):
            if name in PHASES or qualname.endswith("BlockDrafter.__init__"):
                return qualname
        if module == "torch.nn.modules.module" and name in {"_apply", "to"}:
            return "Module." + name
        if module in {"torch.cuda", "torch.cuda.memory"} and name in CUDA_PYTHON:
            return "torch.cuda." + name
        if module == "torch.autograd" and name in {"grad", "backward"}:
            return "torch.autograd." + name
        if module == "torch._tensor" and name == "backward":
            return "Tensor.backward"
        if module.startswith("w1a1_eagle.") and name in PHASES:
            return qualname
        return None

    @staticmethod
    def c_phase(function):
        # CPython supplies builtin functions/methods for c_call; access only
        # builtin name/module and the owner's TYPE. Never access Tensor properties.
        name = function.__name__
        module = function.__module__ or ""
        owner = getattr(function, "__self__", None)
        owner_type = type(owner)
        if name in CUDA_C and (module.startswith("torch") or owner_type.__module__.startswith("torch")):
            return "torch._C." + name
        if name in TENSOR_C and owner_type.__module__.startswith("torch"):
            return "Tensor." + name
        return None

    def _begin(self, phase, frame, kind):
        self.next_call += 1
        call = self.next_call
        # No arguments or locals are serialized. Source line identifies the
        # function for Python and caller for C, without model/data path values.
        parent = self.frames.get(id(frame.f_back)) if kind == "python" else self.frames.get(id(frame))
        if self.journal.emit("BEGIN", call_id=call, parent_call_id=parent[0] if parent else None,
                             phase=phase, kind=kind, source_file=frame.f_code.co_filename,
                             source_line=frame.f_lineno):
            return [call, phase, None]
        return None

    def trace(self, frame, event, arg):
        if self.busy or self.journal.busy or self.journal.capped or self.journal.failed:
            return None
        if event == "call":
            if self.python_phase(frame):
                frame.f_trace_lines = False
                frame.f_trace_opcodes = False
                return self.trace
            return None
        state = self.frames.get(id(frame))
        if state and event == "exception":
            exc_type = arg[0]
            state[2] = exc_type.__module__ + "." + exc_type.__qualname__
        return self.trace if state else None

    def profile(self, frame, event, arg):
        if self.busy or self.journal.busy:
            return
        if self.journal.capped or self.journal.failed:
            self.frames.clear()
            self.c_frames.clear()
            return
        self.busy = True
        try:
            if event == "call":
                phase = self.python_phase(frame)
                if phase:
                    state = self._begin(phase, frame, "python")
                    if state:
                        self.frames[id(frame)] = state
            elif event == "return":
                state = self.frames.pop(id(frame), None)
                if state:
                    opcode = frame.f_code.co_code[frame.f_lasti] if frame.f_lasti >= 0 else 0
                    opname = dis.opname[opcode]
                    exceptional = state[2] is not None and opname not in NORMAL_RETURN and not opname.startswith("YIELD")
                    fields = dict(call_id=state[0], phase=state[1], kind="python")
                    if exceptional:
                        fields["exception_type"] = state[2]
                    elif opname.startswith("YIELD"):
                        fields["exit_kind"] = "suspend"
                    self.journal.emit("EXCEPTION" if exceptional else "END", **fields)
            elif event == "c_call":
                phase = self.c_phase(arg)
                if phase:
                    state = self._begin(phase, frame, "c")
                    if state:
                        self.c_frames.append((id(frame), phase, state))
            elif event in {"c_return", "c_exception"}:
                if self.c_frames and self.c_frames[-1][0] == id(frame) and self.c_frames[-1][1] == self.c_phase(arg):
                    state = self.c_frames.pop()[2]
                    self.journal.emit("EXCEPTION" if event == "c_exception" else "END",
                                      call_id=state[0], phase=state[1], kind="c")
        except Exception:
            # Unexpected observer errors also fail open without inspecting arg.
            self.journal.failed = True
            try:
                os.write(2, b"PHASE_TRACE_OBSERVER_ERROR: tracing disabled; target continues\n")
            except OSError:
                pass
        finally:
            self.busy = False


def execute(script, journal_path, script_args=(), *, max_events=20000, max_bytes=8 * 1024 * 1024):
    """Explicit execution only. Leaves functions, model, config and RNG untouched."""
    script = Path(script).resolve(strict=True)
    if sys.getprofile() is not None or sys.gettrace() is not None:
        raise RuntimeError("existing profile/trace callbacks: refuse to replace instrumentation")
    wrapper = Path(__file__).resolve(strict=True)
    # Hash before script import; parent separately authenticates the immutable checkout.
    pins = {"script_path": str(script), "script_sha256": digest(script),
            "wrapper_path": str(wrapper), "wrapper_sha256": digest(wrapper)}
    journal = Journal(journal_path, max_events=max_events, max_bytes=max_bytes)
    tracer = PhaseTracer(journal, script.parent.parent)
    if not journal.emit("PROVENANCE", **pins, instrumentation="sys.setprofile + narrow exception sys.settrace",
                        classification="diagnostic; not source-neutral production performance",
                        thread_coverage="calling thread only", tensor_metadata="omitted",
                        argv_logged=False, max_events=max_events, max_bytes=max_bytes):
        journal.close()
        raise OSError("cannot durably initialize phase trace")
    original_argv, original_path = sys.argv, sys.path[:]
    sys.argv = [str(script), *script_args]
    sys.path.insert(0, str(script.parent))
    terminal_event, terminal_fields = "SCRIPT_END", {}
    try:
        if not journal.emit("SCRIPT_BEGIN"):
            raise OSError("cannot durably mark script entry")
        sys.settrace(tracer.trace)
        sys.setprofile(tracer.profile)
        runpy.run_path(str(script), run_name="__main__")
    except BaseException as error:
        terminal_event = "SCRIPT_EXCEPTION"
        terminal_fields = {"exception_type": type(error).__module__ + "." + type(error).__qualname__}
        raise
    finally:
        sys.setprofile(None)
        sys.settrace(None)
        sys.argv = original_argv
        sys.path[:] = original_path
        journal.emit(terminal_event, **terminal_fields)
        journal.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--script", required=True, type=Path)
    parser.add_argument("--journal", required=True, type=Path)
    parser.add_argument("--max-events", type=int, default=20000)
    parser.add_argument("--max-bytes", type=int, default=8 * 1024 * 1024)
    parser.add_argument("script_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    script_args = args.script_args[1:] if args.script_args[:1] == ["--"] else args.script_args
    execute(args.script, args.journal, script_args, max_events=args.max_events, max_bytes=args.max_bytes)


if __name__ == "__main__":
    main()

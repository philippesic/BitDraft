import hashlib
import math
import threading
import zipfile
from contextlib import contextmanager
from dataclasses import asdict
from typing import BinaryIO, Callable, Iterator, Literal, Protocol, cast

import numpy as np
import torch
from torch import Tensor

from w1a1_eagle.checkpoints.store import StateRef, Store
from w1a1_eagle.clock import Clock, Monotonic
from w1a1_eagle.contracts import Record, integer, number, record, sequence, string
from w1a1_eagle.data import Supervision
from w1a1_eagle.device import DeviceClock
from w1a1_eagle.model import HE4, Projection, clamp_latents
from w1a1_eagle.numerical.accounting import Accounting
from w1a1_eagle.numerical.captures import Capture
from w1a1_eagle.records import canonical
from w1a1_eagle.training import PARAMETER_GROUPS, Budget, Optimization, Progress, train

HEAD_ROWS = 128
FINITE_SCAN_UPDATES = 32


class Samples(Protocol):
    def sample(self, cursor: int) -> tuple[Capture, tuple[Supervision, ...]]: ...


def label_weights(lengths: tuple[int, ...]) -> Tensor:
    parts = []
    for length in lengths:
        weights = torch.exp(-torch.arange(length) / 7)
        parts.append(weights / weights.sum() / len(lengths))
    return torch.cat(parts)


def optimizer_options(group: object) -> Record:
    return {
        key: value for key, value in record(group).items() if key not in {"foreach", "fused", "lr"}
    }


class TensorWorker:
    def __init__(
        self,
        model: HE4,
        samples: Samples,
        identity: str,
        dataset: str | None = None,
        clock: Clock | None = None,
        optimization: Optimization | None = None,
    ) -> None:
        if not identity or model.geometry.slots != 16:
            raise ValueError("explicit HE4 recipe and sixteen slots required")
        self.accounting = Accounting(dataset=dataset)
        self.accounting.validate()
        self.clock = clock or Monotonic()
        self.update_clock = DeviceClock(
            self.clock, torch.cuda.synchronize if model.vocabulary.device.type == "cuda" else None
        )
        self.model = model
        self.samples = samples
        self.identity = identity
        self.updates = 0
        self.cursor = 0
        self.optimizer_updates = 0
        self.progress = Progress()
        self.lineage: tuple[str, ...] = ()
        self.ready = True
        self.last_publication: StateRef | None = None
        self.lock = threading.RLock()
        self.optimization = optimization or Optimization()
        rates = self.optimization.step_sizes
        suffixes = {
            "latent_sign": "sign",
            "scale": "scale",
            "midpoint": "midpoint",
            "log_gain": "gain",
            "log_clip": "clip",
        }
        groups: dict[str, list[Tensor]] = {key: [] for key in PARAMETER_GROUPS}
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                groups[suffixes.get(name.rsplit(".", 1)[-1], "norm")].append(parameter)
        self.optimizer = torch.optim.AdamW(
            [
                {"params": values, "lr": rates[key], "name": key}
                for key, values in groups.items()
                if values
            ],
            weight_decay=0,
            fused=True,
        )
        self.last_gradient_norm = 0.0
        self.frozen = {name: (value, value._version) for name, value in model.named_buffers()}
        digest = hashlib.sha256()
        for name, (value, _) in self.frozen.items():
            if value.dtype != torch.float32:
                raise ValueError("frozen HE4 execution operands must be F32")
            digest.update(canonical([name, str(value.dtype), list(value.shape)]))
            array = cast(
                np.ndarray[tuple[int, ...], np.dtype[np.float32]],
                value.detach().cpu().contiguous().numpy(),
            )
            data = array.data.cast("B")
            for offset in range(0, len(data), 1024 * 1024):
                digest.update(data[offset : offset + 1024 * 1024])
        self.frozen_sha256 = digest.hexdigest()
        self.native_codes = model.native_codes

    def verify_frozen(self) -> None:
        current = dict(self.model.named_buffers())
        if (
            self.model.native_codes is not self.native_codes
            or set(current) != set(self.frozen)
            or any(
                current[name] is not value or value._version != version
                for name, (value, version) in self.frozen.items()
            )
        ):
            self.ready = False
            raise ValueError("frozen source or vocabulary changed after construction")

    @contextmanager
    def offload(
        self,
        release_gradients: bool = False,
        retain_frozen: bool = False,
        publication: StateRef | None = None,
    ) -> Iterator[Record]:
        with self.lock:
            if not self.ready or self.progress.needs_recovery:
                raise ValueError("offload requires a complete update")
            self.verify_frozen()
            if release_gradients:
                self.optimizer.zero_grad(set_to_none=True)
            parameters = list(self.model.parameters())
            tensors = parameters + (list(self.model.buffers()) if not retain_frozen else [])
            tensors += [p.grad for p in parameters if p.grad is not None]
            optimizer = cast(dict[str, object], self.optimizer.state_dict())
            states = cast(dict[object, object], optimizer["state"])
            tensors.extend(
                value
                for state in states.values()
                for value in record(state).values()
                if torch.is_tensor(value)
            )
            unique = {id(value): value for value in tensors}
            backing: dict[int, Tensor] = {}
            if publication is not None:
                if publication is not self.last_publication or publication.updates != self.updates:
                    raise ValueError("offload requires this worker's current durable publication")
                if not release_gradients or not retain_frozen:
                    raise ValueError("mapped offload releases gradients and retains frozen buffers")
                publication.artifact.verify()
                saved = record(
                    cast(
                        object,
                        torch.load(
                            publication.artifact.path,
                            map_location="cpu",
                            weights_only=True,
                            mmap=True,
                        ),
                    )
                )
                if saved["config"] != self.identity or saved["cursor"] != self.cursor:
                    raise ValueError("mapped offload publication identity differs")
                saved_parameters = record(saved["model"])
                for name, parameter in self.model.named_parameters():
                    source = saved_parameters[name]
                    if not torch.is_tensor(source):
                        raise ValueError("mapped offload requires tensor parameters")
                    backing[id(parameter)] = source
                saved_states = cast(dict[object, object], record(saved["optimizer"])["state"])
                for key, state in states.items():
                    for name, value in record(state).items():
                        if torch.is_tensor(value):
                            source = record(saved_states[key])[name]
                            if not torch.is_tensor(source):
                                raise ValueError("mapped offload requires tensor optimizer state")
                            backing[id(value)] = source
                if any(
                    source.shape != unique[key].shape or source.dtype != unique[key].dtype
                    for key, source in backing.items()
                ):
                    raise ValueError("mapped offload tensor geometry differs")
            versions = {key: value._version for key, value in unique.items()}
            moved = [
                (value, value.device) for value in unique.values() if value.device.type == "cuda"
            ]
            cpu_rng = torch.get_rng_state()
            cuda_rng = torch.cuda.get_rng_state_all() if moved else None
            self.ready = False
            try:
                for value, _ in moved:
                    value.data = (
                        backing[id(value)] if publication is not None else value.detach().to("cpu")
                    )
                if moved:
                    self.update_clock.now()
                    torch.cuda.empty_cache()
                    self.update_clock.now()
                yield {
                    "cpu_tensors": len(unique),
                    "offloaded_tensors": len(moved),
                    "cuda_allocated_bytes": torch.cuda.memory_allocated() if moved else 0,
                    "cuda_reserved_bytes": torch.cuda.memory_reserved() if moved else 0,
                    "cpu_tensor_bytes": sum(
                        value.numel() * value.element_size() for value in unique.values()
                    ),
                    "retained_frozen": retain_frozen,
                    "backing_state": publication.artifact.sha256 if publication else None,
                }
            finally:
                for value, device in moved:
                    value.data = value.detach().to(device)
                torch.set_rng_state(cpu_rng)
                if cuda_rng is not None:
                    torch.cuda.set_rng_state_all(cuda_rng)
                    self.update_clock.now()
                self.verify_frozen()
                if any(value._version != versions[key] for key, value in unique.items()):
                    raise ValueError("training tensors changed during offload")
                self.ready = True

    def step(self, cursor: int) -> float:
        with self.lock:
            if not self.ready or cursor != self.cursor:
                raise ValueError("worker requires exact cursor or durable recovery")
            self.verify_frozen()
            self.ready = False
            self.last_publication = None
            started = self.clock.now()
            try:
                capture, windows = self.samples.sample(cursor)
            finally:
                self.accounting.data_seconds += self.elapsed(started)
            began = self.update_clock.now()
            complete = False
            try:
                loss = self.update(capture, windows)
                complete = True
                labels = sum(len(window.labels) for window in windows)
                self.accounting.dataset_labels += labels
                self.accounting.observed_labels += labels
                return loss
            finally:
                elapsed = self.elapsed(began, self.update_clock)
                if complete:
                    self.accounting.successful_update_seconds += elapsed
                else:
                    self.accounting.failed_update_seconds += elapsed

    def elapsed(self, started: float, clock: Clock | None = None) -> float:
        elapsed = (clock or self.clock).now() - started
        if not math.isfinite(started) or not math.isfinite(elapsed) or elapsed < 0:
            self.ready = False
            raise ValueError("invalid model/data measurement clock")
        return elapsed

    def verify_finite(self) -> None:
        checks = [torch.isfinite(parameter).all() for parameter in self.model.parameters()]
        if not bool(torch.stack(checks).all()):
            self.ready = False
            raise ValueError("nonfinite post-update state")

    def group_loss(self, capture: Capture, windows: tuple[Supervision, ...]) -> Tensor:
        if not windows or any(not window.labels for window in windows):
            raise ValueError("an update requires supervised windows")
        model = self.model
        device = model.vocabulary.device
        slots = model.geometry.slots
        hidden = model.group_hidden(
            capture.features, capture.tokens, tuple(window.anchor for window in windows)
        )
        indices: list[int] = [
            index * slots + 1 + offset
            for index, window in enumerate(windows)
            for offset in range(len(window.labels))
        ]
        targets: list[int] = [label for window in windows for label in window.labels]
        rows = torch.tensor(indices, device=device)
        labels = torch.tensor(targets, dtype=torch.int64, device=device)
        weights = label_weights(tuple(len(window.labels) for window in windows)).to(device)
        selected = hidden.view(-1, hidden.shape[-1]).index_select(0, rows)
        head = selected.detach().requires_grad_()
        total = torch.zeros((), device=device)
        for start in range(0, len(targets), HEAD_ROWS):
            end = start + HEAD_ROWS
            losses = torch.nn.functional.cross_entropy(
                torch.nn.functional.linear(head[start:end], model.vocabulary),
                labels[start:end],
                reduction="none",
            )
            loss = (losses * weights[start:end]).sum()
            torch.autograd.backward(loss)
            total = total + loss.detach()
        if not bool(torch.isfinite(total)) or head.grad is None:
            raise ValueError("nonfinite loss")
        torch.autograd.backward(selected, head.grad)
        return total

    def update(self, capture: Capture, windows: tuple[Supervision, ...]) -> float:
        feature_version = capture.features._version
        multiplier = self.optimization.schedule.multiplier(self.updates)
        self.optimizer.zero_grad(set_to_none=True)
        loss = self.group_loss(capture, windows)
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0, error_if_nonfinite=True)
        self.last_gradient_norm = float(norm)
        for group in cast(list[dict[str, object]], self.optimizer.param_groups):
            group["lr"] = self.optimization.step_sizes[str(group["name"])] * multiplier
        self.optimizer.step()
        with torch.no_grad():
            for module in self.model.modules():
                if isinstance(module, Projection):
                    module.scale.clamp_(min=0)
        if self.optimization.latent_multiple is not None:
            clamp_latents(self.model)
        if (self.updates + 1) % FINITE_SCAN_UPDATES == 0:
            self.verify_finite()
        self.verify_frozen()
        if capture.features._version != feature_version:
            raise ValueError("teacher features changed during an update")
        self.optimizer_updates += 1
        self.updates += 1
        self.cursor += 1
        self.ready = True
        return float(loss.detach())

    def run(self, clock: Clock, budget: Budget, stop: Callable[[], bool]) -> str:
        with self.lock:
            if self.progress.updates != self.updates or self.progress.cursor != self.cursor:
                raise ValueError("worker and budget accounting diverged")
            return train(self, clock, budget, self.progress, stop)

    def checkpoint(self, store: Store, max_bytes: int) -> StateRef:
        with self.lock:
            result = store.commit_with(self.write_snapshot, self.identity, self.updates, max_bytes)
            self.last_publication = result
            return result

    def write_snapshot(self, output: BinaryIO) -> None:
        with self.lock:
            if (
                not self.ready
                or self.progress.needs_recovery
                or self.progress.updates != self.updates
                or self.progress.cursor != self.cursor
            ):
                raise ValueError("cannot publish an incomplete update or budget record")
            self.verify_frozen()
            self.verify_finite()
            self.progress.validate()
            state: dict[str, object] = {
                "schema": 4,
                "accounting": self.accounting.encode(),
                "config": self.identity,
                "updates": self.updates,
                "cursor": self.cursor,
                "optimizer_updates": self.optimizer_updates,
                "gradient_norm": self.last_gradient_norm,
                "progress": cast(dict[str, object], asdict(self.progress)),
                "geometry": cast(dict[str, object], asdict(self.model.geometry)),
                "mask": self.model.mask,
                "model": {
                    name: value.detach()
                    for name, value in self.model.named_parameters(remove_duplicate=False)
                },
                "frozen_sha256": self.frozen_sha256,
                "optimizer": cast(dict[str, object], self.optimizer.state_dict()),
                "rng": torch.get_rng_state(),
                "device": self.model.vocabulary.device.type,
                "cuda_rng": torch.cuda.get_rng_state_all()
                if self.model.vocabulary.device.type == "cuda"
                else None,
                "lineage": list(self.lineage),
                "origin": list(self.model.origin) if self.model.origin else None,
            }
            torch.save(state, output)

    def restore(
        self,
        reference: StateRef,
        memory_budget_bytes: int,
        mode: Literal["resume", "branch", "continue"] = "resume",
    ) -> None:
        with self.lock:
            self.ready = False
            self.last_publication = None
            if (
                mode not in {"resume", "branch", "continue"}
                or reference.artifact.size > memory_budget_bytes
            ):
                raise ValueError("restore mode or memory budget invalid")
            reference.artifact.verify()
            with zipfile.ZipFile(reference.artifact.path) as archive:
                if sum(item.file_size for item in archive.infolist()) * 2 > memory_budget_bytes:
                    raise ValueError("tensor state logical size exceeds memory budget")
            state = record(
                cast(
                    object,
                    torch.load(
                        reference.artifact.path, map_location="cpu", weights_only=True, mmap=True
                    ),
                )
            )
            required = {
                "schema",
                "config",
                "updates",
                "cursor",
                "optimizer_updates",
                "gradient_norm",
                "progress",
                "geometry",
                "mask",
                "model",
                "optimizer",
                "rng",
                "device",
                "cuda_rng",
                "lineage",
                "origin",
                "frozen_sha256",
            }
            schema = integer(state["schema"])
            if schema not in {3, 4} or set(state) != (
                required | {"accounting"} if schema == 4 else required
            ):
                raise ValueError("unsupported tensor state schema")
            accounting = Accounting.parse(state["accounting"]) if schema == 4 else Accounting()
            dataset = self.accounting.dataset
            if (
                mode == "resume"
                and accounting.dataset is not None
                and accounting.dataset != dataset
            ):
                raise ValueError("resume dataset identity mismatch")
            if (
                state["geometry"] != cast(dict[str, object], asdict(self.model.geometry))
                or state["mask"] != self.model.mask
            ):
                raise ValueError("state model semantics mismatch")
            if state["origin"] != (list(self.model.origin) if self.model.origin else None):
                raise ValueError("source/vocabulary origin mismatch")
            if string(state["config"]) != reference.config:
                raise ValueError("state manifest and payload identity disagree")
            if state["device"] not in {"cpu", "cuda"}:
                raise ValueError("unsupported checkpoint device")
            if (
                mode in {"resume", "continue"}
                and state["device"] != self.model.vocabulary.device.type
            ):
                raise ValueError("exact resume device mismatch")
            if mode == "resume" and (
                state["config"] != self.identity or reference.config != self.identity
            ):
                raise ValueError("exact resume recipe mismatch")
            updates, cursor = integer(state["updates"]), integer(state["cursor"])
            if updates != reference.updates or updates < 0 or cursor < 0:
                raise ValueError("state accounting mismatch")
            self.verify_frozen()
            if state["frozen_sha256"] != self.frozen_sha256:
                raise ValueError("frozen initialization fingerprint mismatch")
            saved = record(state["model"])
            expected = dict(self.model.named_parameters(remove_duplicate=False))
            if set(saved) != set(expected):
                raise ValueError("state tensor coverage mismatch")
            tensors: dict[str, Tensor] = {}
            for name, value in saved.items():
                if (
                    not torch.is_tensor(value)
                    or value.shape != expected[name].shape
                    or value.dtype != expected[name].dtype
                    or not bool(torch.isfinite(value).all())
                ):
                    raise ValueError("state tensor shape, dtype or finiteness mismatch")
                tensors[name] = value
            aliases: dict[int, str] = {}
            for name, parameter in self.model.named_parameters(remove_duplicate=False):
                previous = aliases.setdefault(id(parameter), name)
                if not torch.equal(tensors[name], tensors[previous]):
                    raise ValueError("shared tensor alias conflict")
            optimizer_updates = integer(state["optimizer_updates"])
            optimizer_state = record(state["optimizer"])
            if not 0 <= optimizer_updates <= updates:
                raise ValueError("optimizer update accounting mismatch")
            saved_groups = sequence(optimizer_state["param_groups"])
            current_optimizer = cast(dict[str, object], self.optimizer.state_dict())
            fresh_groups = sequence(current_optimizer["param_groups"])
            if [optimizer_options(group) for group in saved_groups] != [
                optimizer_options(group) for group in fresh_groups
            ]:
                raise ValueError("optimizer parameter groups or rates mismatch")
            raw_states = optimizer_state["state"]
            if not isinstance(raw_states, dict):
                raise ValueError("invalid optimizer states")
            states = cast(dict[object, object], raw_states)
            expected_count = sum(
                1 for parameter in self.model.parameters() if parameter.requires_grad
            )
            if len(states) != (expected_count if optimizer_updates else 0):
                raise ValueError("optimizer state coverage mismatch")
            identifiers = [
                integer(value)
                for group in saved_groups
                for value in sequence(record(group)["params"])
            ]
            groups = cast(list[dict[str, object]], self.optimizer.param_groups)
            parameters = [item for group in groups for item in cast(list[Tensor], group["params"])]
            if any(type(key) is not int for key in states) or set(states) != (
                set(identifiers) if optimizer_updates else set()
            ):
                raise ValueError("optimizer parameter identity mismatch")
            for identifier, optimizer_parameter in zip(identifiers, parameters, strict=True):
                if not optimizer_updates:
                    continue
                value = states[identifier]
                fields = record(value)
                if set(fields) != {"step", "exp_avg", "exp_avg_sq"}:
                    raise ValueError("optimizer field coverage mismatch")
                step = fields["step"]
                if not torch.is_tensor(step) or step.ndim != 0 or float(step) != optimizer_updates:
                    raise ValueError("optimizer step mismatch")
                for key in ("exp_avg", "exp_avg_sq"):
                    field = fields[key]
                    if (
                        not torch.is_tensor(field)
                        or field.shape != optimizer_parameter.shape
                        or field.dtype != optimizer_parameter.dtype
                    ):
                        raise ValueError("optimizer moment geometry mismatch")
                for field in fields.values():
                    if torch.is_tensor(field) and not bool(torch.isfinite(field).all()):
                        raise ValueError("nonfinite optimizer state")
            progress_data = record(state["progress"])
            if (
                set(progress_data)
                != {
                    "updates",
                    "cursor",
                    "attempts",
                    "failed",
                    "discarded",
                    "needs_recovery",
                    "wall_seconds",
                    "last_loss",
                }
                or progress_data["needs_recovery"] is not False
            ):
                raise ValueError("incomplete budget record")
            progress = Progress(
                updates=integer(progress_data["updates"]),
                cursor=integer(progress_data["cursor"]),
                attempts=integer(progress_data["attempts"]),
                failed=integer(progress_data["failed"]),
                discarded=integer(progress_data["discarded"]),
                wall_seconds=number(progress_data["wall_seconds"]),
                last_loss=number(progress_data["last_loss"])
                if progress_data["last_loss"] is not None
                else None,
            )
            progress.validate()
            if progress.updates != updates or progress.cursor != cursor:
                raise ValueError("tensor and budget counters disagree")
            if reference.config == self.identity and self.progress.attempts > progress.attempts:
                discarded = max(0, self.progress.updates - progress.updates)
                progress.failed = max(progress.failed, self.progress.failed)
                progress.discarded = max(progress.discarded, self.progress.discarded) + discarded
                progress.attempts = progress.updates + progress.failed + progress.discarded
                progress.wall_seconds = max(progress.wall_seconds, self.progress.wall_seconds)
                progress.validate()
            with torch.no_grad():
                for name, parameter in expected.items():
                    parameter.copy_(tensors[name])
            if mode in {"continue", "branch"} and (
                dataset is None or accounting.dataset != dataset
            ):
                accounting.dataset_start_updates = updates
                accounting.dataset_labels = 0
            accounting.dataset = dataset
            if reference.config == self.identity:
                accounting.successful_update_seconds = max(
                    accounting.successful_update_seconds, self.accounting.successful_update_seconds
                )
                accounting.failed_update_seconds = max(
                    accounting.failed_update_seconds, self.accounting.failed_update_seconds
                )
                accounting.data_seconds = max(accounting.data_seconds, self.accounting.data_seconds)
                accounting.observed_labels = max(
                    accounting.observed_labels, self.accounting.observed_labels
                )
                if dataset == self.accounting.dataset:
                    accounting.dataset_labels = max(
                        accounting.dataset_labels, self.accounting.dataset_labels
                    )
            accounting.validate()
            self.accounting = accounting
            lineage = tuple(string(v) for v in sequence(state["lineage"]))
            if mode in {"resume", "continue"}:
                implementation = [
                    {
                        **record(saved),
                        "foreach": record(fresh)["foreach"],
                        "fused": record(fresh)["fused"],
                    }
                    for saved, fresh in zip(saved_groups, fresh_groups, strict=True)
                ]
                self.optimizer.load_state_dict(
                    cast(dict[str, object], {**optimizer_state, "param_groups": implementation})
                )
                self.optimizer_updates = optimizer_updates
                rng = state["rng"]
                if not torch.is_tensor(rng) or rng.dtype != torch.uint8:
                    raise ValueError("invalid RNG state")
                torch.set_rng_state(rng)
                cuda_rng = state["cuda_rng"]
                if self.model.vocabulary.device.type == "cuda":
                    values = sequence(cuda_rng)
                    states_on_cuda: list[Tensor] = []
                    for value in values:
                        if not torch.is_tensor(value) or value.dtype != torch.uint8:
                            raise ValueError("invalid CUDA RNG state")
                        states_on_cuda.append(value)
                    torch.cuda.set_rng_state_all(states_on_cuda)
                elif cuda_rng is not None:
                    raise ValueError("CPU state contains CUDA RNG history")
                self.cursor = cursor if mode == "resume" else 0
                self.lineage = (
                    lineage if mode == "resume" else (*lineage, reference.artifact.sha256)
                )
            else:
                if bool(cast(object, self.optimizer.state)):
                    raise ValueError("branch requires a fresh optimizer")
                self.cursor = 0
                self.optimizer_updates = 0
                self.lineage = (*lineage, reference.artifact.sha256)
            progress.cursor = self.cursor
            self.progress = progress
            self.last_gradient_norm = number(state["gradient_norm"])
            if self.last_gradient_norm < 0:
                raise ValueError("invalid gradient norm")
            self.updates = updates
            self.ready = True


def checkpoint_accounting(reference: StateRef, memory_budget_bytes: int) -> Accounting | None:
    reference.artifact.verify()
    if reference.artifact.size > memory_budget_bytes:
        raise ValueError("checkpoint accounting exceeds memory budget")
    with zipfile.ZipFile(reference.artifact.path) as archive:
        if sum(item.file_size for item in archive.infolist()) * 2 > memory_budget_bytes:
            raise ValueError("checkpoint logical size exceeds memory budget")
    state = record(
        cast(
            object,
            torch.load(reference.artifact.path, map_location="cpu", weights_only=True, mmap=True),
        )
    )
    if integer(state["updates"]) != reference.updates:
        raise ValueError("checkpoint progress differs from its manifest")
    schema = integer(state["schema"])
    if schema == 3:
        return None
    if schema != 4:
        raise ValueError("unsupported checkpoint accounting schema")
    result = Accounting.parse(state["accounting"])
    if result.dataset_start_updates > reference.updates:
        raise ValueError("dataset exposure origin is ahead of checkpoint")
    return result

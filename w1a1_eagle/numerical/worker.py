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
from w1a1_eagle.model import HE4, DeferredChecks, Layer, Projection, bound_latents, clamp_latents
from w1a1_eagle.numerical.accounting import Accounting, Telemetry
from w1a1_eagle.numerical.captures import Capture
from w1a1_eagle.records import canonical
from w1a1_eagle.round_states import RoundStateWeighting
from w1a1_eagle.training import (
    MAXIMUM_SLOTS,
    SOFT_FAMILY,
    TABLE_GROUP,
    Arithmetic,
    Blocks,
    Budget,
    Features,
    GradualPrecision,
    Optimization,
    Progress,
    reach_weights,
    train,
)

HEAD_ROWS = 128
FINITE_SCAN_UPDATES = 32
NONFINITE_NORM = "nonfinite gradient norm; the update cannot be clipped"
FEATURE_PARAMETERS = {
    "scale": "scale",
    "clip": "log_clip",
    "gain": "log_gain",
    "midpoint": "midpoint",
}


def feature_parameters(model: HE4, group: str) -> list[torch.nn.Parameter]:
    attribute = FEATURE_PARAMETERS[group]
    found: dict[int, torch.nn.Parameter] = {}
    for module in model.modules():
        if isinstance(module, Projection):
            parameter = cast(torch.nn.Parameter, getattr(module, attribute))
            found[id(parameter)] = parameter
    return list(found.values())


def latent_parameters(model: HE4) -> list[torch.nn.Parameter]:
    return [module.latent_sign for module in model.modules() if isinstance(module, Projection)]


class Samples(Protocol):
    def sample(self, cursor: int) -> tuple[Capture, tuple[Supervision, ...]]: ...


def label_weights(
    lengths: tuple[int, ...],
    positions: tuple[float, ...] | None = None,
    shares: tuple[float, ...] | None = None,
) -> Tensor:
    if shares is not None and len(shares) != len(lengths):
        raise ValueError("one window share per window required")
    parts = []
    for index, length in enumerate(lengths):
        weights = (
            torch.exp(-torch.arange(length) / 7)
            if positions is None
            else torch.tensor(positions[:length], dtype=torch.float32)
        )
        parts.append(
            weights / weights.sum() / len(lengths)
            if shares is None
            else weights / weights.sum() * shares[index]
        )
    return torch.cat(parts)


def position_weights(value: object) -> tuple[float, ...] | None:
    if value is None:
        return None
    weights = tuple(number(item) for item in sequence(value))
    if len(weights) != MAXIMUM_SLOTS - 1 or any(
        not math.isfinite(weight) or weight <= 0 for weight in weights
    ):
        raise ValueError("position weights need 15 positive finite values")
    return weights


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
        blocks: Blocks | None = None,
        block_slots: Callable[[int], int] | None = None,
        features: Features | None = None,
        precision: GradualPrecision | None = None,
        telemetry: bool = False,
        arithmetic: Arithmetic = "reference",
        round_states: RoundStateWeighting | None = None,
    ) -> None:
        if not identity or model.geometry.slots != MAXIMUM_SLOTS:
            raise ValueError("explicit HE4 recipe and sixteen slots required")
        self.blocks = blocks or Blocks()
        fixed = self.blocks.fixed
        if block_slots is None and fixed is None:
            raise ValueError("random block sizes require the sampler's block size per cursor")
        self.block_slots: Callable[[int], int] = (
            block_slots if block_slots is not None else lambda _: cast(int, fixed)
        )
        self.position_weights: tuple[float, ...] | None = None
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
        self.features = features or Features()
        self.precision = precision
        self.arithmetic: Arithmetic = arithmetic
        self.phase: Literal["soft", "hard"] = "hard" if precision is None else "soft"
        self.latent_step_offset = 0
        for group in self.features.untrained:
            for parameter in feature_parameters(model, group):
                parameter.requires_grad_(False)
        self.verify_features()
        rates = self.optimization.step_sizes
        suffixes = {
            "latent_sign": "sign",
            "scale": "scale",
            "midpoint": "midpoint",
            "log_gain": "gain",
            "log_clip": "clip",
            "operand": TABLE_GROUP,
        }
        groups: dict[str, list[Tensor]] = {key: [] for key in self.optimization.groups}
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                key = suffixes.get(name.rsplit(".", 1)[-1], "norm")
                if key not in groups:
                    raise ValueError("trained tables require an explicit table step size")
                groups[key].append(parameter)
        if TABLE_GROUP in groups and not groups[TABLE_GROUP]:
            raise ValueError("a table step size requires trained tables")
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
        if model.frozen_format != "Q4_0":
            digest.update(
                canonical([model.frozen_format, model.frozen_baseline, list(model.native_types)])
            )
            for name, shape, codes in model.native_codes:
                digest.update(canonical([name, list(shape), hashlib.sha256(codes).hexdigest()]))
        self.frozen_sha256 = digest.hexdigest()
        self.native_codes = model.native_codes
        self.native_types = model.native_types
        self.frozen_configuration = (
            model.frozen_format,
            model.frozen_baseline,
            model.parent_frozen_sha256,
        )
        self.frozen_transition: Record | None = None
        self.telemetry = Telemetry(model) if telemetry else None
        self.fusion_sha256: str | None = None
        self.round_states = round_states
        self.round_state_counts = [0, 0]

    def verify_features(self) -> None:
        for group in self.features.untrained:
            for parameter in feature_parameters(self.model, group):
                if parameter.requires_grad:
                    raise ValueError("an untrained feature parameter requires gradients")
                if group in self.features.off and bool((parameter != 0).any()):
                    raise ValueError(group + " is off but its parameters are not at identity")

    def temperature(self) -> float | None:
        if self.precision is None or self.phase == "hard":
            return None
        return self.precision.temperature(self.updates)

    def precision_record(self) -> Record | None:
        if self.precision is None:
            return None
        temperature = self.temperature()
        return {
            "phase": self.phase,
            "temperature": temperature,
            "switch_update": self.precision.soft_updates,
            "evaluated": "hard signs of the current weights"
            if self.phase == "soft"
            else "hard signs",
        }

    def harden(self) -> None:
        with torch.no_grad():
            if self.optimization.latent_multiple is not None:
                bound_latents(self.model, self.optimization.latent_multiple)
            states = cast(dict[Tensor, dict[str, Tensor]], self.optimizer.state)
            for parameter in latent_parameters(self.model):
                for value in states.get(parameter, {}).values():
                    value.zero_()
        self.latent_step_offset = self.optimizer_updates
        self.phase = "hard"

    def telemetry_record(self) -> Record | None:
        if self.telemetry is None or self.telemetry.signs is None:
            return None
        return self.telemetry.record(
            self.updates, self.accounting, self.precision_record(), self.phase == "soft"
        )

    def verify_frozen(self) -> None:
        current = dict(self.model.named_buffers())
        if (
            self.model.native_codes is not self.native_codes
            or self.model.native_types is not self.native_types
            or (
                self.model.frozen_format,
                self.model.frozen_baseline,
                self.model.parent_frozen_sha256,
            )
            != self.frozen_configuration
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
                slots = self.block_slots(cursor)
                capture, windows = self.samples.sample(cursor)
            finally:
                self.accounting.data_seconds += self.elapsed(started)
            began = self.update_clock.now()
            complete = False
            try:
                loss = self.update(capture, windows, slots)
                complete = True
                if self.round_states is not None:
                    flags = self.round_states.states.flags(windows)
                    self.round_state_counts[0] += len(flags)
                    self.round_state_counts[1] += sum(flags)
                labels = sum(min(len(window.labels), slots - 1) for window in windows)
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

    def refresh_position_weights(self, measured: Record) -> tuple[float, ...]:
        positions = record(measured["positions"])
        weights = reach_weights(
            tuple(integer(value) for value in sequence(positions["windows"])),
            tuple(integer(value) for value in sequence(positions["survivors"])),
        )
        self.position_weights = weights
        return weights

    def round_state_record(self) -> Record | None:
        if self.round_states is None:
            return None
        windows, starts = self.round_state_counts
        self.round_state_counts = [0, 0]
        return {
            "windows": windows,
            "round_starts": starts,
            "share": starts / windows if windows else None,
            "weight": self.round_states.weight,
        }

    def label_tensors(
        self, windows: tuple[Supervision, ...], size: int, positions: tuple[float, ...] | None
    ) -> tuple[Tensor, Tensor, Tensor]:
        device = self.model.vocabulary.device
        lengths = tuple(min(len(window.labels), size - 1) for window in windows)
        indices: list[int] = [
            index * size + 1 + offset
            for index, length in enumerate(lengths)
            for offset in range(length)
        ]
        targets: list[int] = [
            label
            for window, length in zip(windows, lengths, strict=True)
            for label in window.labels[:length]
        ]
        rows = torch.tensor(indices, device=device)
        labels = torch.tensor(targets, dtype=torch.int64, device=device)
        shares = None if self.round_states is None else self.round_states.shares(windows)
        return rows, labels, label_weights(lengths, positions, shares).to(device)

    def group_loss(
        self,
        capture: Capture,
        windows: tuple[Supervision, ...],
        slots: int | None = None,
        checks: DeferredChecks | None = None,
    ) -> Tensor:
        if not windows or any(not window.labels for window in windows):
            raise ValueError("an update requires supervised windows")
        model = self.model
        size = model.block(slots)
        if size < 2:
            raise ValueError("an update requires at least one drafted position")
        positions: tuple[float, ...] | None = None
        if self.blocks.position_weights == "reach":
            if self.position_weights is None:
                raise ValueError("reach weights require a development refresh before updating")
            positions = self.position_weights
        if capture.fusion is not None and capture.fusion != self.fusion_sha256:
            raise ValueError("fused capture and model fusion table identities differ")
        prepared = None if checks is None else self.label_tensors(windows, size, positions)
        hidden = model.group_hidden(
            capture.features,
            capture.tokens,
            tuple(window.anchor for window in windows),
            size,
            capture.fusion is not None,
        )
        rows, labels, weights = (
            self.label_tensors(windows, size, positions) if prepared is None else prepared
        )
        targets = labels.shape[0]
        selected = hidden.view(-1, hidden.shape[-1]).index_select(0, rows)
        head = selected.detach().requires_grad_()
        total = torch.zeros((), device=hidden.device)
        for start in range(0, targets, HEAD_ROWS):
            end = start + HEAD_ROWS
            losses = torch.nn.functional.cross_entropy(
                torch.nn.functional.linear(head[start:end], model.vocabulary),
                labels[start:end],
                reduction="none",
            )
            loss = (losses * weights[start:end]).sum()
            torch.autograd.backward(loss)
            total = total + loss.detach()
        if checks is None:
            if not bool(torch.isfinite(total)) or head.grad is None:
                raise ValueError("nonfinite loss")
        else:
            checks.require(torch.isfinite(total), "nonfinite loss")
            if head.grad is None:
                checks.read()
                raise ValueError("nonfinite loss")
        torch.autograd.backward(selected, head.grad)
        return total

    def update(
        self, capture: Capture, windows: tuple[Supervision, ...], slots: int | None = None
    ) -> float:
        feature_version = capture.features._version
        multiplier = self.optimization.schedule.multiplier(self.updates)
        if (
            self.precision is not None
            and self.phase == "soft"
            and self.updates >= self.precision.soft_updates
        ):
            self.harden()
        temperature = self.temperature()
        telemetry = self.telemetry
        if telemetry is not None:
            telemetry.begin(self.updates, self.accounting)
            telemetry.attach()
        projections = [module for module in self.model.modules() if isinstance(module, Projection)]
        layers = [module for module in self.model.modules() if isinstance(module, Layer)]
        checks = None if self.arithmetic == "reference" else DeferredChecks()
        for module in projections:
            module.temperature = temperature
            module.checks = checks
            module.single_product = checks is not None
            module.tensor_cores = self.arithmetic == "fast-exact-tf32ff"
        for layer in layers:
            layer.checks = checks
        self.optimizer.zero_grad(set_to_none=True)
        try:
            loss = self.group_loss(capture, windows, slots, checks)
        finally:
            for module in projections:
                module.temperature = None
                module.checks = None
                module.single_product = False
                module.tensor_cores = False
            for layer in layers:
                layer.checks = None
            if telemetry is not None:
                telemetry.detach()
        groups = cast(list[dict[str, object]], self.optimizer.param_groups)
        families = {
            SOFT_FAMILY
            if str(group["name"]) == "sign" and self.phase == "soft"
            else str(group["name"]): [
                value.grad
                for value in cast(list[Tensor], group["params"])
                if value.grad is not None
            ]
            for group in groups
        }
        measured_loss: float | None = None
        if checks is None:
            if telemetry is not None:
                telemetry.gradients(families)
            norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 1.0, error_if_nonfinite=True
            )
            norm_value = float(norm)
        else:
            gradients = [p.grad for p in self.model.parameters() if p.grad is not None]
            norm = torch.nn.utils.get_total_norm(gradients, 2.0, False)
            measured_loss, norm_value = checks.read(loss, norm)
            if not math.isfinite(norm_value):
                raise RuntimeError(NONFINITE_NORM)
            if telemetry is not None:
                telemetry.gradients(families)
            torch.nn.utils.clip_grads_with_norm_(self.model.parameters(), 1.0, norm)
        if telemetry is not None:
            telemetry.clipping(norm)
        self.last_gradient_norm = norm_value
        for group in groups:
            name = str(group["name"])
            rate = self.optimization.step_sizes[name]
            if name == "sign" and self.phase == "soft" and self.precision is not None:
                rate = self.precision.step_size
            group["lr"] = rate * multiplier
        self.optimizer.step()
        with torch.no_grad():
            for module in projections:
                module.scale.clamp_(min=0)
        if self.optimization.latent_multiple is not None and self.phase == "hard":
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
        return float(loss.detach()) if measured_loss is None else measured_loss

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
                "schema": 5,
                "position_weights": None
                if self.position_weights is None
                else list(self.position_weights),
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
                **(
                    {"frozen_transition": self.frozen_transition}
                    if self.frozen_transition is not None
                    else {}
                ),
                **(
                    {
                        "precision": {
                            "phase": self.phase,
                            "latent_step_offset": self.latent_step_offset,
                        }
                    }
                    if self.precision is not None or self.latent_step_offset
                    else {}
                ),
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
            if schema not in {3, 4, 5} or set(state) != (
                required
                | ({"frozen_transition"} if "frozen_transition" in state else set())
                | ({"precision"} if "precision" in state else set())
                | ({"accounting"} if schema >= 4 else set())
                | ({"position_weights"} if schema == 5 else set())
            ):
                raise ValueError("unsupported tensor state schema")
            accounting = Accounting.parse(state["accounting"]) if schema >= 4 else Accounting()
            saved_weights = position_weights(state["position_weights"]) if schema == 5 else None
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
            saved_precision = record(state["precision"]) if "precision" in state else None
            if saved_precision is not None and (
                set(saved_precision) != {"phase", "latent_step_offset"}
                or saved_precision["phase"] not in ("soft", "hard")
            ):
                raise ValueError("invalid gradual precision state")
            phase: Literal["soft", "hard"] = (
                "soft"
                if saved_precision is not None and saved_precision["phase"] == "soft"
                else "hard"
            )
            offset = (
                0 if saved_precision is None else integer(saved_precision["latent_step_offset"])
            )
            if phase == "soft" and self.precision is None:
                raise ValueError("a soft-phase state requires its gradual precision recipe")
            if self.precision is not None and (
                (phase == "hard" and updates <= self.precision.soft_updates)
                or (phase == "soft" and mode == "resume" and updates > self.precision.soft_updates)
            ):
                raise ValueError("state phase disagrees with the gradual precision schedule")
            self.verify_frozen()
            transition = (
                record(state["frozen_transition"]) if "frozen_transition" in state else None
            )
            unfreezing = False
            if state["frozen_sha256"] != self.frozen_sha256:
                if (
                    mode != "branch"
                    or (self.model.frozen_format != "baseline-k" and not self.model.trained_tables)
                    or state["frozen_sha256"] != self.model.parent_frozen_sha256
                ):
                    raise ValueError("frozen initialization fingerprint mismatch")
                unfreezing = self.model.trained_tables
                transition = {
                    "parent": reference.artifact.sha256,
                    "from": state["frozen_sha256"],
                    "to": self.frozen_sha256,
                    "format": self.model.frozen_format,
                    "baseline": self.model.frozen_baseline,
                    **({"tables": "trained"} if unfreezing else {}),
                }
            self.frozen_transition = transition
            saved = record(state["model"])
            expected = dict(self.model.named_parameters(remove_duplicate=False))
            initial = {name for name in expected if unfreezing and name.endswith(".operand")}
            if set(saved) != set(expected) - initial:
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
                if name in initial:
                    continue
                previous = aliases.setdefault(id(parameter), name)
                if not torch.equal(tensors[name], tensors[previous]):
                    raise ValueError("shared tensor alias conflict")
            optimizer_updates = integer(state["optimizer_updates"])
            optimizer_state = record(state["optimizer"])
            if not 0 <= optimizer_updates <= updates or not 0 <= offset <= optimizer_updates:
                raise ValueError("optimizer update accounting mismatch")
            saved_groups = sequence(optimizer_state["param_groups"])
            current_optimizer = cast(dict[str, object], self.optimizer.state_dict())
            fresh_groups = sequence(current_optimizer["param_groups"])
            regrouped = mode == "branch" and [
                optimizer_options(group) for group in saved_groups
            ] != [optimizer_options(group) for group in fresh_groups]
            if unfreezing or regrouped:
                saved_groups = fresh_groups
                optimizer_updates = 0
                optimizer_state = {"state": {}, "param_groups": fresh_groups}
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
            latents = {id(parameter) for parameter in latent_parameters(self.model)}
            for identifier, optimizer_parameter in zip(identifiers, parameters, strict=True):
                if not optimizer_updates:
                    continue
                value = states[identifier]
                fields = record(value)
                if set(fields) != {"step", "exp_avg", "exp_avg_sq"}:
                    raise ValueError("optimizer field coverage mismatch")
                step = fields["step"]
                expected_step = optimizer_updates - (
                    offset if id(optimizer_parameter) in latents else 0
                )
                if not torch.is_tensor(step) or step.ndim != 0 or float(step) != expected_step:
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
                    if name not in initial:
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
                self.latent_step_offset = offset
            else:
                if bool(cast(object, self.optimizer.state)):
                    raise ValueError("branch requires a fresh optimizer")
                self.cursor = 0
                self.optimizer_updates = 0
                self.latent_step_offset = 0
                self.lineage = (*lineage, reference.artifact.sha256)
            self.phase = phase
            self.verify_features()
            progress.cursor = self.cursor
            self.progress = progress
            self.last_gradient_norm = number(state["gradient_norm"])
            if self.last_gradient_norm < 0:
                raise ValueError("invalid gradient norm")
            self.position_weights = (
                saved_weights if self.blocks.position_weights == "reach" else None
            )
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
    if schema not in {4, 5}:
        raise ValueError("unsupported checkpoint accounting schema")
    result = Accounting.parse(state["accounting"])
    if result.dataset_start_updates > reference.updates:
        raise ValueError("dataset exposure origin is ahead of checkpoint")
    return result

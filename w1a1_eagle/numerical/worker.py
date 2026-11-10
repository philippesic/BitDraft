import threading
import zipfile
from dataclasses import asdict
from typing import BinaryIO, Callable, Literal, Protocol, cast

import torch
from torch import Tensor

from w1a1_eagle.checkpoints.store import StateRef, Store
from w1a1_eagle.clock import Clock
from w1a1_eagle.contracts import integer, number, record, sequence, string
from w1a1_eagle.data import Window
from w1a1_eagle.model import HE4, Projection
from w1a1_eagle.numerical.captures import Capture
from w1a1_eagle.training import Budget, Progress, train


class Samples(Protocol):
    def sample(self, cursor: int) -> tuple[Capture, Window]: ...


class TensorWorker:
    def __init__(self, model: HE4, samples: Samples, identity: str) -> None:
        if not identity or model.geometry.slots != 16:
            raise ValueError("explicit HE4 recipe and sixteen slots required")
        self.model = model
        self.samples = samples
        self.identity = identity
        self.updates = 0
        self.cursor = 0
        self.optimizer_updates = 0
        self.progress = Progress()
        self.lineage: tuple[str, ...] = ()
        self.ready = True
        self.lock = threading.RLock()
        rates = {
            "sign": 1e-3,
            "scale": 1e-5,
            "midpoint": 1e-5,
            "gain": 1e-5,
            "clip": 1e-5,
            "norm": 1e-5,
        }
        suffixes = {
            "latent_sign": "sign",
            "scale": "scale",
            "midpoint": "midpoint",
            "log_gain": "gain",
            "log_clip": "clip",
        }
        groups: dict[str, list[Tensor]] = {key: [] for key in rates}
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
            foreach=False,
        )
        self.last_gradient_norm = 0.0

    def step(self, cursor: int) -> float:
        with self.lock:
            if not self.ready or cursor != self.cursor:
                raise ValueError("worker requires exact cursor or durable recovery")
            self.ready = False
            capture, window = self.samples.sample(cursor)
            self.optimizer.zero_grad(set_to_none=True)
            logits = self.model.forward(capture.features, capture.tokens, window.anchor)[1:]
            labels = torch.tensor(window.labels, dtype=torch.int64, device=logits.device)
            losses = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
            weights = torch.exp(-torch.arange(15, device=logits.device) / 7)
            loss = (losses * weights).sum() / weights.sum()
            if not bool(torch.isfinite(loss)):
                raise ValueError("nonfinite loss")
            torch.autograd.backward(loss)
            norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 1.0, error_if_nonfinite=True
            )
            self.last_gradient_norm = float(norm)
            self.optimizer.step()
            with torch.no_grad():
                for module in self.model.modules():
                    if isinstance(module, Projection):
                        module.scale.clamp_(min=0)
            if any(
                not bool(torch.isfinite(parameter).all()) for parameter in self.model.parameters()
            ):
                raise ValueError("nonfinite post-update state")
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
            return store.commit_with(self.write_snapshot, self.identity, self.updates, max_bytes)

    def write_snapshot(self, output: BinaryIO) -> None:
        with self.lock:
            if (
                not self.ready
                or self.progress.needs_recovery
                or self.progress.updates != self.updates
                or self.progress.cursor != self.cursor
            ):
                raise ValueError("cannot publish an incomplete update or budget record")
            self.progress.validate()
            state: dict[str, object] = {
                "schema": 2,
                "config": self.identity,
                "updates": self.updates,
                "cursor": self.cursor,
                "optimizer_updates": self.optimizer_updates,
                "gradient_norm": self.last_gradient_norm,
                "progress": cast(dict[str, object], asdict(self.progress)),
                "geometry": cast(dict[str, object], asdict(self.model.geometry)),
                "mask": self.model.mask,
                "model": cast(dict[str, Tensor], self.model.state_dict()),
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
        mode: Literal["resume", "branch"] = "resume",
    ) -> None:
        with self.lock:
            self.ready = False
            if mode not in {"resume", "branch"} or reference.artifact.size > memory_budget_bytes:
                raise ValueError("restore mode or memory budget invalid")
            reference.artifact.verify()
            with zipfile.ZipFile(reference.artifact.path) as archive:
                if sum(item.file_size for item in archive.infolist()) * 2 > memory_budget_bytes:
                    raise ValueError("tensor state logical size exceeds memory budget")
            state = record(
                cast(
                    object,
                    torch.load(reference.artifact.path, map_location="cpu", weights_only=True),
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
            }
            if set(state) != required or type(state["schema"]) is not int or state["schema"] != 2:
                raise ValueError("unsupported tensor state schema")
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
            if mode == "resume" and state["device"] != self.model.vocabulary.device.type:
                raise ValueError("exact resume device mismatch")
            if mode == "resume" and (
                state["config"] != self.identity or reference.config != self.identity
            ):
                raise ValueError("exact resume recipe mismatch")
            updates, cursor = integer(state["updates"]), integer(state["cursor"])
            if updates != reference.updates or updates < 0 or cursor < 0:
                raise ValueError("state accounting mismatch")
            saved = record(state["model"])
            expected = cast(dict[str, Tensor], self.model.state_dict())
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
            for name, value in self.model.named_buffers():
                if not torch.equal(value.detach().cpu(), tensors[name]):
                    raise ValueError("frozen source or vocabulary changed")
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
            if saved_groups != fresh_groups:
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
            self.model.load_state_dict(tensors, strict=True)
            lineage = tuple(string(v) for v in sequence(state["lineage"]))
            if mode == "resume":
                self.optimizer.load_state_dict(optimizer_state)
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
                self.cursor = cursor
                self.lineage = lineage
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

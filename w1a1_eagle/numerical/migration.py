import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import cast

from w1a1_eagle.checkpoints.store import Store
from w1a1_eagle.contracts import Record, string
from w1a1_eagle.device import device_lease
from w1a1_eagle.intent import Intent
from w1a1_eagle.numerical.entry import admission, geometry, read_corpus
from w1a1_eagle.numerical.legacy import import_state, materialize
from w1a1_eagle.numerical.weights import load_he4
from w1a1_eagle.numerical.worker import TensorWorker
from w1a1_eagle.processes import ProcessIdentity
from w1a1_eagle.records import artifact, canonical, read, reference
from w1a1_eagle.storage import output_path, relocated


def migrate(specification: Path, output: Path) -> Record:
    spec = read(specification)
    if (
        set(spec) - {"raw_checkpoint"} != {"schema", "intent", "archive", "receipt"}
        or spec["schema"] != 1
    ):
        raise ValueError("unsupported migration specification")
    base = specification.resolve().parent
    intent = Intent.load(base / string(spec["intent"]))
    archive = artifact(spec["archive"], base)
    receipt_ref = artifact(spec["receipt"], base)
    receipt_ref.verify()
    receipt = read(receipt_ref.path)
    if output.exists():
        raise ValueError("migration destination must be new; preserve previous attempts")
    with device_lease(str(intent.raw["device"])):
        store = Store(output, intent.disk_reserve_bytes)
        store.atomic(
            store.root / "owner.json",
            canonical(cast(Record, asdict(ProcessIdentity.read(os.getpid())))),
        )
        store.atomic(store.root / "specification.json", specification.read_bytes())
        corpus = read_corpus(intent)
        admission(intent, corpus)
        cfg = geometry(intent)
        if spec.get("raw_checkpoint") is not None:
            archive.verify()
            raw = artifact(spec["raw_checkpoint"], base)
            if raw.sha256 != receipt["raw_sha256"] or raw.size != receipt["raw_bytes"]:
                raise ValueError("materialized legacy state differs from archive receipt")
            raw.verify()
        else:
            raw = materialize(
                archive, receipt, Store(store.root / "legacy", intent.disk_reserve_bytes)
            )
        model = load_he4(intent.source, intent.vocabulary, intent.memory_bytes, cfg)
        model.to(str(intent.raw["device"]))
        worker = TensorWorker(model, corpus, intent.recipe, intent.corpus.sha256)
        state = import_state(
            raw, worker, Store(store.root / "states", intent.disk_reserve_bytes), intent.state_bytes
        )
        result: Record = {
            "status": "imported",
            "archive": reference(archive),
            "state": reference(state.artifact),
            "updates": state.updates,
            "recipe": state.config,
            "qualification": "state mapping only; matched next-update and native qualification still required",
        }
        store.atomic(store.root / "result.json", canonical(result))
        return result


class Arguments(argparse.Namespace):
    specification: Path
    output: Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("specification", type=Path)
    parser.add_argument("output", type=Path, nargs="?")
    args = parser.parse_args(namespace=Arguments())
    args.specification = relocated(args.specification).resolve()
    args.output = output_path(args.output, "migration", args.specification.stem)
    print(json.dumps(migrate(args.specification, args.output), indent=2))


if __name__ == "__main__":
    main()

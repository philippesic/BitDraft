import datetime
import hashlib
import json
from pathlib import Path

from w1a1_eagle.block_shard_lifecycle import FrozenShardPlan, validate_replay_admission


plan_path = Path(
    "/mnt/c/Users/philip/binary-eagle-decoding-artifacts/5080-training-20261009/"
    "dflash-w1a8-screen-20261009-01/phase1/logical-plan.json"
)
plan_sha256 = "205961eed921f39cb90efe2aa5fe9966af0ed25786cc8adb9ae157eb5d359888"
plan = FrozenShardPlan.load({"path": str(plan_path), "sha256": plan_sha256})

admission_path = Path(
    "/home/philip/binary-eagle-decoding/runs/5080-dflash-replay-admission-20261010-01/"
    "replay-admission.json"
)
admission_sha256 = hashlib.sha256(admission_path.read_bytes()).hexdigest()
producer = {
    "path": "/home/philip/binary-eagle-decoding/runs/5080-training-20261009/"
    "source-dflash-preparation-20261009/scripts/replay_dspark_shard.py",
    "sha256": "d9dc8a79810f4339783a3caa8609e033d62ee777e123d506d68bc9cf79eda0e3",
}

validate_replay_admission(
    plan,
    {"path": str(admission_path), "sha256": admission_sha256},
    producer=producer,
)
print(
    json.dumps(
        {
            "schema": "originalS_replay_admission_validation_v1",
            "result": "PASS",
            "validated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "plan_sha256": plan.sha256,
            "admission": {"path": str(admission_path), "sha256": admission_sha256},
            "producer": producer,
        },
        sort_keys=True,
    )
)

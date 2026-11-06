#!/usr/bin/env bash
set -euo pipefail
PROJECT=/home/philip/binary-eagle-decoding
F=/mnt/c/Users/philip/binary-eagle-decoding-artifacts/5080-resume-20261010/dspark-w1a8-finalist-20261009-01
SOURCE=$PROJECT/runs/5080-training-20261009/source-937af03
INPUTS=$PROJECT/runs/dspark-paid-shard0034-inputs-20261011-01/inputs
GUARD=$INPUTS/restore_dspark_admission_bootstrap_20261009.py
VALIDATOR=$INPUTS/validate_dspark_unpublished_shard_20261009.py
PUBLISHER=$INPUTS/publish_dspark_validated_shard_20261009.py
OLD_RESOURCE=$PROJECT/runs/dspark-admission-bootstrap-inspect-20261011-01/inputs/old-resource-observation.json
RUN_ID=dspark-paid-shard0034-validate-20261011-01
RUN=$PROJECT/runs/$RUN_ID
CONTROLS=$PROJECT/runs/${RUN_ID}-controls
AVAILABILITY=$CONTROLS/availability.json
REMOTE_JOB=$F/metadata/autonomy-contracts-v1/scripts/remote_job.py
OUT=$RUN/validated-0034-20261011-01.json
LANE=$F/lane.json
CONFIG=$F/initial-finalist-packet/configs/dspark_a8.json
PUBLICATION=$F/lane/attempts/0f3032fd71294fa2badc82492ed655c6/shard-phases/988dfc8ccabd45388ad0e0033da82774/publication.json
REQUEST=$F/lane/attempts/0f3032fd71294fa2badc82492ed655c6/shard-phases/988dfc8ccabd45388ad0e0033da82774/request.json
CACHE=$F/train-cache-admitted/cache-state.json
META=$F/original-source-seals-admitted/chunk-0826/original-metadata.json
# The initial publication-bound lease is created once, with exclusive create and short expiry.
python3 -B -S -c 'import json,os,time,hashlib,pathlib,subprocess; project=pathlib.Path("/home/philip/binary-eagle-decoding"); run=project/"runs/dspark-paid-shard0034-validate-20261011-01"; controls=project/"runs/dspark-paid-shard0034-validate-20261011-01-controls"; f=pathlib.Path("/mnt/c/Users/philip/binary-eagle-decoding-artifacts/5080-resume-20261010/dspark-w1a8-finalist-20261009-01"); lane=json.loads((f/"lane.json").read_text()); gpu_control=json.loads(pathlib.Path(lane["gpu_control_path"]).read_text()); assert gpu_control["rtx5080"]["pause_requested"] is False and gpu_control["rtx5080"].get("stop_all_requested",False) is False; assert not run.exists() and not controls.exists(); assert not (f/"lane/STOP").exists(); controls.mkdir(mode=0o700); os.chmod(controls,0o700); p=controls/"availability.json"; t=time.time(); lease={"schema":"nine_model_gpu_lease_v1","host":"rtx5080","user_announced_available":True,"sole_owner":True,"pause_requested":False,"bundle_sha256":"eae17abd1f441d9b723d5f38e24f56155e27acb506baf254bf776e33c9c28fa6","gpu_uuid":lane["gpu_uuid"],"granted_unix":t,"expires_unix":t+300}; fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600); stream=os.fdopen(fd,"wb"); data=(json.dumps(lease,separators=(",",":"))+"\n").encode(); stream.write(data); stream.flush(); os.fsync(stream.fileno()); stream.close(); st=p.stat(); assert st.st_mode & 0o777==0o600 and st.st_nlink==1 and json.loads(p.read_text())==lease; print(json.dumps({"path":str(p),"sha256":hashlib.sha256(data).hexdigest(),"mode":oct(st.st_mode&0o777),"nlink":st.st_nlink,"granted_unix":t,"expires_unix":t+300,"gpu_uuid":lane["gpu_uuid"],"publication_sha256":lease["bundle_sha256"],"control_pause":gpu_control["rtx5080"]["pause_requested"]},sort_keys=True))'
# Existing detached supervisor, direct validator child, normal admitted Python -B.
env CUDA_VISIBLE_DEVICES=GPU-44ceb8b5-b67a-a317-fee3-f01c9201994e PYTHONPATH=/home/philip/binary-eagle-decoding/third_party/llama.cpp/gguf-py OMP_NUM_THREADS=2 WSL_INTEROP=/run/WSL/51623_interop WSL_DISTRO_NAME=Ubuntu \
  tmux -L binary-eagle-runtime new-session -d -s "$RUN_ID" -c "$PROJECT" \
  "exec env CUDA_VISIBLE_DEVICES=GPU-44ceb8b5-b67a-a317-fee3-f01c9201994e PYTHONPATH=/home/philip/binary-eagle-decoding/third_party/llama.cpp/gguf-py OMP_NUM_THREADS=2 WSL_INTEROP=/run/WSL/51623_interop WSL_DISTRO_NAME=Ubuntu $PROJECT/.venv/bin/python -B -S $REMOTE_JOB $RUN_ID --stop-grace-seconds 90 --output-root $PROJECT/runs --command-root $INPUTS --command-script $VALIDATOR -- $PROJECT/.venv/bin/python -B $VALIDATOR validate --source-root $SOURCE --lane $LANE --lane-sha256 fd10e9918eafd2b7edff154017a490957472eb01b606859b35c59c201e1c383c --config $CONFIG --publication $PUBLICATION --publication-sha256 eae17abd1f441d9b723d5f38e24f56155e27acb506baf254bf776e33c9c28fa6 --request $REQUEST --request-sha256 55d1bb36fcf79c4685a37a93da44a92f67e2ebad03e7394b97807421cb264e76 --cache-state $CACHE --cache-state-sha256 a8896fdb91784654fe9bb64a9403c01c4d4b223a8ce52dfef80b13bc8af60655 --metadata-0826 $META --metadata-0826-sha256 89eac5bfa4cd84599951b0226c5b4b7ae5a477f352841611a14b6aca844e871b --guard-helper $GUARD --guard-helper-sha256 9cc200f99e3ca9597a4ead3670991c82812e5b48460b964e481af2fa25e5ad04 --output $OUT --availability $AVAILABILITY --supervisor-state $RUN/state.json --old-resource-observation $OLD_RESOURCE --old-resource-observation-sha256 3f6de736326d92213f2c3defd053df35ea3faef9241578acef5eddd012f9e48d --protected-json $F/lane/state.json cc4fb56979962c1e8522b8089c284c1b48cc778dd535c60a5dc16da7b6a4ad9d --protected-json $F/lane/training/checkpoints/latest.json ea53b5f9a1df96dfa395018db437508c52dff5c1f146d94b0d2f9acd3e433849 --protected-json $F/lane/training/budget-used.json d31ba176e299cc542b19fc9b4989e08d9dfcc7250ff3a0672ced1817f8f9124f"
tmux -L binary-eagle-runtime list-sessions

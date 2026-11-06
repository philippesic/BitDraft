"""Unselected process-local actual-byte reuse for explicit immutable locators.

Only hash acquisition is shared. Original validators and expected-SHA checks
still execute on every invocation. No metadata schema/result is cached.
"""
import hashlib
import json
import os
from pathlib import Path
import stat
import threading
import types

HELPER_SHA='5b0ffcbe472fa25275c49c610333777972a8681de24732f616f445e65e17bc10'

class SharedBytes:
    def __init__(self, records, fingerprint, *, trial_records=()):
        self.allowed={}
        for record in records:
            if not isinstance(record,dict) or set(record)!={'path','sha256'}:
                raise ValueError('exact immutable locator required')
            path=Path(record['path'])
            digest=record['sha256']
            if not path.is_absolute() or path!=path.resolve() or not isinstance(digest,str) or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):
                raise ValueError('canonical pinned immutable locator required')
            if str(path) in self.allowed and self.allowed[str(path)]!=digest:
                raise ValueError('immutable locator conflicts')
            self.allowed[str(path)]=digest
        self.trial_allowed = set()
        for record in trial_records:
            if (not isinstance(record, dict) or set(record) != {"path", "sha256"}
                    or not isinstance(record["path"], str) or record["path"] not in self.allowed
                    or self.allowed[record["path"]] != record["sha256"]
                    or record["path"] in self.trial_allowed):
                raise ValueError("exact selected trial array subset required")
            self.trial_allowed.add(record["path"])
        self.fingerprint=fingerprint;self.pid=os.getpid();self.entries={}
        self.reads=self.hits=self.bytes_read=0;self.lock=threading.RLock()

    def digest(self,path):
        path=Path(path)
        if str(path) not in self.allowed:raise ValueError('unselected shared payload')
        # A fork can inherit a lock held by a vanished thread. Discard both
        # authority and that lock before acquiring anything in the child.
        if self.pid!=os.getpid():
            self.lock=threading.RLock();self.entries={};self.pid=os.getpid()
            self.reads=self.hits=self.bytes_read=0
        with self.lock:
            before=path.lstat()
            if (path!=path.resolve() or not stat.S_ISREG(before.st_mode)
                    or before.st_nlink < 1
                    or (before.st_nlink != 1 and str(path) not in self.trial_allowed)):
                raise ValueError('one-link canonical shared payload required')
            identity=self.fingerprint(before)
            fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
            with os.fdopen(fd,'rb') as stream:
                if self.fingerprint(os.fstat(stream.fileno()))!=identity:raise ValueError('shared payload replaced on open')
                prior=self.entries.get(str(path))
                if prior is not None and prior[0]==identity:
                    actual=prior[1];self.hits+=1
                else:
                    actual=hashlib.sha256()
                    for block in iter(lambda:stream.read(1024*1024),b''):actual.update(block)
                    actual=actual.hexdigest();self.reads+=1;self.bytes_read+=before.st_size
                if self.fingerprint(os.fstat(stream.fileno()))!=identity or path!=path.resolve() or self.fingerprint(path.lstat())!=identity:
                    raise ValueError('shared payload changed during check')
            # Actual bytes are retained. The original caller compares EACH
            # caller's expected SHA, even when it differs from this selection.
            self.entries[str(path)]=(identity,actual)
            return actual


def install(helper,records,fingerprint,*,trial_records=()):
    source=Path(helper.__file__).read_bytes()
    if hashlib.sha256(source).hexdigest()!=HELPER_SHA:raise ValueError('accepted helper source differs')
    compiled=compile(source,str(helper.__file__),'exec')
    classes={c.co_name:c for c in compiled.co_consts if isinstance(c,types.CodeType)}
    for name in ('PolicyHashes','ReplayAuditHashes'):
        cls=getattr(helper,name)
        methods={c.co_name:c for c in classes[name].co_consts if isinstance(c,types.CodeType)}
        for method in ('__init__','__call__'):
            fn=getattr(cls,method)
            if fn.__globals__ is not helper.__dict__ or fn.__code__!=methods[method]:
                raise ValueError('original memo callable differs')
    if not isinstance(fingerprint,types.FunctionType):raise ValueError('original nine-field fingerprint required')
    pipeline_source=Path(fingerprint.__globals__['__file__']).read_bytes()
    if hashlib.sha256(pipeline_source).hexdigest()!='a27e9121261938175d88fb78651b2f02fa0a641d4b5afccc2197e561399979af':
        raise ValueError('original fingerprint source differs')
    pipeline_code=compile(pipeline_source,fingerprint.__globals__['__file__'],'exec')
    files=next(c for c in pipeline_code.co_consts if isinstance(c,types.CodeType) and c.co_name=='Files')
    expected=next(c for c in files.co_consts if isinstance(c,types.CodeType) and c.co_name=='fingerprint')
    if fingerprint.__code__!=expected or fingerprint.__defaults__ is not None or fingerprint.__kwdefaults__ is not None:
        raise ValueError('original nine-field fingerprint callable differs')
    store=SharedBytes(records,fingerprint,trial_records=trial_records)
    original_policy=helper.PolicyHashes;original_replay=helper.ReplayAuditHashes
    def eligible(path):
        path=Path(path)
        # Preserve original accepted hardlinks through ordinary full checking.
        return str(path) in store.allowed and (path.lstat().st_nlink == 1
                or (str(path) in store.trial_allowed and path.lstat().st_nlink > 0))
    class Policy(original_policy):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs);self._pid=os.getpid()
        def __call__(self,path):
            if self._pid!=os.getpid():self.entries.clear();self._pid=os.getpid()
            if not eligible(path):return super().__call__(path)
            path=Path(path)
            if not path.is_absolute() or path!=path.resolve():raise ValueError('canonical policy artifact required')
            before=path.lstat()
            if not stat.S_ISREG(before.st_mode):raise ValueError('regular policy artifact required')
            identity=self.fingerprint(before);previous=self.entries.get(str(path))
            if previous is not None and previous[0]!=identity:raise ValueError('policy artifact identity changed')
            reads,hits,size=store.reads,store.hits,store.bytes_read
            digest=store.digest(path)
            if self.fingerprint(path.lstat())!=identity or path!=path.resolve():raise ValueError('policy artifact changed during cache check')
            result=helper.PolicyDigest(digest);self.entries[str(path)]=(identity,result)
            self.reads+=store.reads-reads;self.hits+=store.hits-hits;self.bytes_read+=store.bytes_read-size
            return result
    class Replay(original_replay):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs);self._pid=os.getpid()
        def __call__(self,path):
            if self._pid!=os.getpid():self.entries.clear();self._pid=os.getpid()
            if not eligible(path):return super().__call__(path)
            path=Path(path)
            if not path.is_absolute() or path!=path.resolve():raise ValueError('canonical replay audit file required')
            before=path.lstat()
            if not stat.S_ISREG(before.st_mode):raise ValueError('regular replay audit file required')
            identity=self.fingerprint(before);previous=self.entries.get(str(path))
            if previous is not None and previous[0]!=identity:raise ValueError('replay audit artifact identity changed')
            reads,hits,size=store.reads,store.hits,store.bytes_read
            digest=store.digest(path)
            if self.fingerprint(path.lstat())!=identity or path!=path.resolve():raise ValueError('replay audit artifact changed during cache check')
            self.entries[str(path)]=(identity,digest)
            self.reads+=store.reads-reads;self.hits+=store.hits-hits;self.bytes_read+=store.bytes_read-size
            return digest
    helper.PolicyHashes=Policy;helper.ReplayAuditHashes=Replay
    return store


PROFILE = "original_immutable_models_process_actual_bytes_v1"


def select_model_locators(config, *, metadata, checked):
    """Select model roles from fresh authenticated JSON, never open payloads.

    This selector grants no admission or deletion authority. All scientific
    consumers still validate their complete graphs, including pending results.
    """
    if config.get("startup_byte_proof") != {"profile": PROFILE}:
        raise ValueError("explicit process actual-byte profile required")
    observations=[];records={};roles={}
    def require(ok,message):
        if not ok:raise ValueError(message)
    def locator(record, *, project=False):
        require(isinstance(record,dict) and (project or set(record)=={"path","sha256"}),
                "exact model path/SHA required")
        value={key:record[key] for key in ("path","sha256")}
        path=Path(value["path"]);digest=value["sha256"]
        require(path.is_absolute() and os.path.normpath(str(path))==str(path) and isinstance(digest,str)
                and len(digest)==64 and all(c in "0123456789abcdef" for c in digest),
                "canonical model locator required")
        return value
    def read(record):
        record=locator(record)
        value=checked(record)
        fresh,digest=metadata(record["path"])
        require(digest==record["sha256"] and fresh==value,"selector metadata changed during read")
        observations.append((record,value))
        return value
    def model(role,record,*,project=False):
        record=locator(record,project=project);path=record["path"]
        require(path not in records or records[path]==record,"immutable model locator conflicts")
        records[path]=record;roles[role]=record
    lane=read(config["lane"])
    require(lane.get("schema")=="nine_model_training_lane_v1","selector lane schema differs")
    spec=read(lane["config"]);plan=read(config["evaluation_plan"])
    candidate=lane["candidate"]
    require(spec.get("schema")=="nine_model_qat_training_v1" and spec.get("candidate")==candidate
            and plan.get("schema")=="nine_model_timed_evaluation_plan_v1"
            and plan.get("candidate")==candidate and plan.get("config")==lane["config"]
            and lane.get("timed_evaluation_plan")==config["evaluation_plan"],
            "selector config/plan/lane joins differ")
    model("spec.model",spec["model"])
    if spec.get("initialization") is not None:model("spec.initialization",spec["initialization"],project=True)
    if spec.get("original_initializer") is not None:
        model("spec.original_initializer.checkpoint",spec["original_initializer"]["checkpoint"],project=True)
    for name,record in (("plan.target",plan["target"]),("plan.control.model",plan["control"]["model"]),
                        ("plan.initial.model",plan["initial"]["model"])):model(name,record)
    directory=Path(config["run_dir"])/"training"
    latest,digest=metadata(directory/"checkpoints/latest.json")
    observations.append(({"path":str(directory/"checkpoints/latest.json"),"sha256":digest},latest))
    require(latest.get("schema")=="block_qat_checkpoint_v1" and latest.get("committed") is True,
            "selector committed latest schema differs")
    model("latest.checkpoint",latest,project=True)
    state,digest=metadata(directory/"timed-evaluation-state.json")
    observations.append(({"path":str(directory/"timed-evaluation-state.json"),"sha256":digest},state))
    contract=state.get("contract",{})
    require(state.get("schema")=="nine_model_timed_evaluation_state_v1"
            and isinstance(state.get("completed"),list)
            and contract.get("candidate")==candidate
            and contract.get("bundle_sha256")==config["lane"]["sha256"]
            and contract.get("config_sha256")==lane["config"]["sha256"]
            and contract.get("protocol")==plan["protocol"],"selector timed contract differs")
    for index,entry in enumerate(state["completed"]):
        result=read(entry["receipt"]);report=read(result["evaluation"])
        require(result.get("schema")=="nine_model_timed_evaluation_receipt_v1"
                and result.get("completed") is True and result.get("request")==entry["request"]
                and result.get("candidate")==candidate
                and result.get("bundle_sha256")==config["lane"]["sha256"]
                and result.get("config_sha256")==lane["config"]["sha256"]
                and result.get("protocol")==plan["protocol"]
                and report.get("schema")=="nine_model_timed_matched_report_v1"
                and report.get("candidate")==candidate
                and report.get("target")==plan["target"]
                and report.get("strict_quality_status")==result.get("strict_quality_status")
                and report.get("continuation_status")==result.get("continuation_status")
                and report.get("execution_context",{}).get("checkpoint")==result["checkpoint"],
                "selector completed receipt/report joins differ")
        model(f"completed.{index}.checkpoint",result["checkpoint"])
        model(f"completed.{index}.report.target",report["target"])
        require(isinstance(report.get("model_ancestry"),dict)
                and set(report["model_ancestry"])=={candidate,"initial","eagle_q4"},
                "selector explicit model ancestry differs")
        for name,record in report["model_ancestry"].items():model(f"completed.{index}.model_ancestry.{name}",record)
    for record,value in observations:
        current,digest=metadata(record["path"])
        require(digest==record["sha256"] and current==value,"selector metadata changed before install")
    return list(records.values()), {"profile":PROFILE,"roles":roles,"metadata": [v[0] for v in observations],
                                   "payloads_read":0,"scientific_admission_granted":False}


REPLAY_PROFILE = "original_immutable_models_and_admitted_trial_arrays_process_actual_bytes_v1"
TRIAL_HARDLINK_PROFILE = "original_immutable_models_and_admitted_trial_arrays_multilink_process_actual_bytes_v1"


def select_replay_trial_locators(record, *, metadata, checked):
    """Select only typed arrays in the authenticated original replay proof.

    This is metadata selection, never replay admission. Original replay and
    BlockDataset validators still execute, with their original physical views.
    """
    observations = []; records = {}; roles = {}
    def require(ok, message):
        if not ok: raise ValueError(message)
    def loc(value):
        require(isinstance(value, dict) and set(value) == {"path", "sha256"},
                "exact trial metadata locator required")
        path = Path(value["path"]); digest = value["sha256"]
        require(path.is_absolute() and str(path) == os.path.normpath(str(path))
                and isinstance(digest, str) and len(digest) == 64
                and all(c in "0123456789abcdef" for c in digest),
                "canonical trial locator required")
        return dict(value)
    def read(value):
        value = loc(value); result = checked(value)
        current, digest = metadata(value["path"])
        require(current == result and digest == value["sha256"],
                "trial metadata changed during selection")
        observations.append((value, result)); return result
    admission = read(record)
    require(admission.get("schema") == "block_shard_replay_admission_v1"
            and admission.get("status") == "PASS" and admission.get("original_identity")
            and admission["original_identity"] == admission.get("replay_identity")
            and set(admission.get("domains", [])) == {"prose", "code", "reasoning"},
            "typed original replay selector contract differs")
    original = read(admission["original_metadata"])
    publication = read(admission["trial_publication"])
    require(original.get("schema") == "block_shard_original_metadata_v1"
            and publication.get("schema") == "block_shard_publication_v1"
            and publication.get("status") == "PASS"
            and all(x.get("plan_sha256") == admission["plan_sha256"]
                    and x.get("shard") == admission["shard"] for x in (original, publication))
            and original.get("identity") == admission["original_identity"],
            "trial original/publication joins differ")
    manifest = read(publication["manifest"]); completed = read(publication["admission"])
    require(manifest.get("schema") == "native_block_train_v1"
            and completed.get("schema") == "block_data_completed_admission_v1"
            and completed.get("manifest_sha256") == publication["manifest"]["sha256"]
            and completed.get("source_sha256") == admission["source"].get("module:src/w1a1_eagle/block_data.py")
            and manifest.get("producer", {}).get("target_sha256") == admission["source"].get("target_sha256"),
            "trial manifest/admission/source joins differ")
    artifacts = completed.get("artifacts", [])
    require(isinstance(artifacts, list) and all(isinstance(x, dict) for x in artifacts),
            "trial completed artifacts required")
    bypath = {x["path"]: x for x in artifacts}
    require(len(bypath) == len(artifacts), "duplicate trial admission artifact path")
    chains = manifest.get("chains", [])
    require(isinstance(chains, list) and len(chains) == len(admission["original_identity"])
            and {x["chain_id"] for x in chains} == set(admission["original_identity"])
            and {x["domain"] for x in chains} == set(admission["domains"]),
            "trial chain identity/domain joins differ")
    for chain in chains:
        identity = admission["original_identity"][chain["chain_id"]]
        require(all(chain.get(k) == identity.get(k) for k in ("anchors", "prompt_length", "logits_indices")),
                "trial prefix identity differs")
        for role in ("tokens", "features", "logits"):
            value = loc({k: chain[role][k] for k in ("path", "sha256")})
            artifact = bypath.get(value["path"], {})
            require(value["sha256"] == identity["artifacts"][role]
                    and artifact.get("sha256") == value["sha256"]
                    and isinstance(artifact.get("bytes"), int) and artifact["bytes"] > 0,
                    "trial typed array artifact join differs")
            require(value["path"] not in records, "trial array path alias differs")
            records[value["path"]] = value
            roles[f"trial.{chain['chain_id']}.{role}"] = value
    for value, result in observations:
        current, digest = metadata(value["path"])
        require(current == result and digest == value["sha256"], "trial metadata changed before install")
    return list(records.values()), {"roles": roles, "metadata": [x[0] for x in observations],
                                   "payloads_read": 0, "scientific_admission_granted": False}


def select_immutable_locators(config, *, metadata, checked):
    profile = config.get("startup_byte_proof")
    if profile == {"profile": PROFILE}:
        return select_model_locators(config, metadata=metadata, checked=checked)
    if profile not in ({"profile": REPLAY_PROFILE}, {"profile": TRIAL_HARDLINK_PROFILE}):
        raise ValueError("explicit process actual-byte profile required")
    # Reuse the unchanged model-only selector and all its metadata joins.
    model_config = dict(config, startup_byte_proof={"profile": PROFILE})
    records, provenance = select_model_locators(model_config, metadata=metadata, checked=checked)
    lane = checked(config["lane"]); spec = checked(lane["config"])
    record = spec["data"]["replay_admission"]
    if spec["shard_lifecycle"].get("replay_admission") != record:
        raise ValueError("trial controller/data replay admission differs")
    logical_record = {k: spec["data"][k] for k in ("path", "sha256")}
    logical = checked(logical_record); replay = checked(record)
    if (spec["shard_lifecycle"]["plan"] != logical_record
            or replay["plan_sha256"] != hashlib.sha256(json.dumps(logical, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            or replay["source"] != logical.get("source")
            or set(replay["original_identity"]) != set(logical["shards"][replay["shard"]])
            or replay["producer"]["sha256"] != logical["source"].get("module:scripts/replay_dspark_shard.py")):
        raise ValueError("trial frozen logical/source membership differs")
    trial, extra = select_replay_trial_locators(record, metadata=metadata, checked=checked)
    if checked(logical_record) != logical or checked(record) != replay:
        raise ValueError("trial logical/replay changed before install")
    # Conflicts are rejected by the same canonical locator consumer as install.
    all_records = {x["path"]: x for x in records}
    for value in trial:
        if value["path"] in all_records:
            raise ValueError("model/trial locator overlap")
        all_records[value["path"]] = value
    provenance = dict(provenance, profile=profile["profile"],
                      roles={**provenance["roles"], **extra["roles"]},
                      metadata=provenance["metadata"] + extra["metadata"],
                      selected_models=len(records), selected_trial_arrays=len(trial))
    if profile == {"profile": TRIAL_HARDLINK_PROFILE}:
        provenance["multilink_trial_arrays"] = trial
    # Recheck the lane/spec at the outer boundary as well.
    if checked(config["lane"]) != lane or checked(lane["config"]) != spec:
        raise ValueError("trial lane/spec changed before install")
    return list(all_records.values()), provenance


def install_from_config(helper,config,*,metadata,checked,fingerprint):
    """Explicit operational hook; no model bytes are read until consumed."""
    if config.get("startup_byte_proof") is None:return None
    records,provenance=select_immutable_locators(config,metadata=metadata,checked=checked)
    pool=install(helper,records,fingerprint,
                 trial_records=provenance.get("multilink_trial_arrays", ()))
    import atexit
    import json
    import sys
    def emit():
        try:
            print(json.dumps({"startup_shared_actual_byte_pool":{
                "profile":provenance["profile"],"pid":os.getpid(),"owner_pid":pool.pid,
                "counters_for_current_pid":pool.pid==os.getpid(),
                "hash_reads":pool.reads if pool.pid==os.getpid() else 0,
                "hash_reuses":pool.hits if pool.pid==os.getpid() else 0,
                "bytes_hashed":pool.bytes_read if pool.pid==os.getpid() else 0,
                "selected_models":provenance.get("selected_models",len(records)),
                "selected_trial_arrays":provenance.get("selected_trial_arrays",0),
                "scientific_admission_granted":False}}),file=sys.stderr,flush=True)
        except BaseException:pass
    atexit.register(emit)
    pool.provenance=provenance
    return pool

"""Source-authenticated original budget settlement before launch preparation.

No new clock, refund, cursor write or model read. Caller retains canonical flock.
"""
import fcntl
import hashlib
import math
import os
from pathlib import Path
import stat
import types

BUDGET_SHA='9cbc775931b1f2739c8a9b7cad9e98f746449000cce06dd8b9b6792b74eb655e'
PIPELINE_SHA='a27e9121261938175d88fb78651b2f02fa0a641d4b5afccc2197e561399979af'


def require(ok,reason):
    if not ok:raise ValueError(reason)


def _original(record,expected):
    path=Path(record['path']);before=path.lstat()
    require(record['sha256']==expected and path.is_absolute() and path.resolve()==path
            and stat.S_ISREG(before.st_mode) and before.st_nlink==1 and before.st_size<1024**2,
            'original accounting source role differs')
    fields=('st_dev','st_ino','st_mode','st_uid','st_gid','st_nlink','st_size','st_mtime_ns','st_ctime_ns')
    identity=lambda value:tuple(getattr(value,key) for key in fields)
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    with os.fdopen(fd,'rb') as stream:
        require(identity(os.fstat(stream.fileno()))==identity(before),'accounting source changed on open')
        raw=stream.read(1024**2)
        require(identity(os.fstat(stream.fileno()))==identity(before)
                and identity(path.lstat())==identity(before),'accounting source changed during read')
    require(hashlib.sha256(raw).hexdigest()==expected,'original accounting source bytes differ')
    # Execute exactly the authenticated bytes, never re-open via importlib.
    module=types.ModuleType('_original_prelaunch_'+path.stem)
    module.__file__=str(path);module.__package__='w1a1_eagle'
    exec(compile(raw,str(path),'exec',dont_inherit=True),module.__dict__)
    return module


def settle_original_budget(config,lane,directory,*,config_path,owner_fd,metadata,checked,current_control):
    """Call original9cbc load once under the actual existing launch flock."""
    config_path=Path(config_path);directory=Path(directory)
    require(config_path.is_absolute() and config_path.resolve()==config_path
            and directory.is_absolute() and directory.resolve()==directory,'canonical launch paths required')
    require(type(owner_fd) is int,'canonical launch lock descriptor required')
    lock=Path.home()/'.config/binary-eagle-decoding/rtx5080-campaign.lock'
    held,current=os.fstat(owner_fd),lock.lstat()
    require(stat.S_ISREG(held.st_mode) and held.st_nlink==1 and held.st_uid==os.getuid()
            and (held.st_dev,held.st_ino)==(current.st_dev,current.st_ino),'canonical launch lock differs')
    fcntl.flock(owner_fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
    canonical,config_sha=metadata(config_path)
    require(canonical==config,'canonical config changed before settlement')
    require(checked(config['lane'])==lane,'launch lane changed')
    spec=checked(lane['config'])
    maximum=spec['limits']['max_seconds']
    require(spec.get('candidate')==lane['candidate'] and maximum==43200,'original lane/allocation required')
    latest_path=directory/'training/checkpoints/latest.json';budget_path=directory/'training/budget-used.json'
    latest,latest_sha=metadata(latest_path);ledger,ledger_sha=metadata(budget_path)
    elapsed=latest.get('cursor',{}).get('elapsed_seconds')
    require(latest.get('schema')=='block_qat_checkpoint_v1' and latest.get('committed') is True
            and type(elapsed) in (int,float) and math.isfinite(elapsed) and elapsed>=0,
            'committed original checkpoint cursor required')
    root=Path(config['native_source_root'])
    sources=lane['source']
    for role in ('src/w1a1_eagle/continuous_budget.py','src/w1a1_eagle/nine_model_pipeline.py'):
        require(Path(sources[role]['path'])==root/role,'accounting source role path differs')
    budget=_original(sources['src/w1a1_eagle/continuous_budget.py'],BUDGET_SHA)
    api=_original(sources['src/w1a1_eagle/nine_model_pipeline.py'],PIPELINE_SHA)
    def authorize():
        current_control(lane)
        require(not (directory/'STOP').exists() and not (directory/'training/STOP').exists(),'human STOP must remain')
        require(metadata(config_path)==(canonical,config_sha),'canonical config changed during settlement')
        require(checked(config['lane'])==lane,'launch lane changed during settlement')
        require(metadata(latest_path)==(latest,latest_sha),'latest checkpoint changed during settlement')
        require(metadata(budget_path)==(ledger,ledger_sha),'budget changed during settlement')
    authorize()
    def publish(path,value):
        require(Path(path)==budget_path,'original budget write path differs')
        authorize()
        api.atomic_json(path,value)
    original=budget.TrainingBudget(budget_path,config['lane']['sha256'],maximum,publish)
    used=original.load(elapsed)
    after,after_sha=metadata(budget_path)
    require(after.get('active_attempt') is None,'original budget remains active')
    return dict(profile='original_prelaunch_budget_load_v1',source_sha256=config['lane']['sha256'],
                maximum=maximum,checkpoint_elapsed=elapsed,original_load_used=used,
                ledger_before={'path':str(budget_path),'sha256':ledger_sha},
                ledger_after={'path':str(budget_path),'sha256':after_sha},
                recovered_attempt=after.get('recovered_attempt'),
                accounting_outcome='uncertain_recovery_charge' if after.get('recovered_attempt') and used>elapsed else 'checkpoint_accounted',
                allocation_exhausted_by_recovery=bool(after.get('recovered_attempt') and used>=maximum and elapsed<maximum),
                new_clock_started=False,
                optimizer_updates_performed=0,effective_run_dir=str(directory),
                canonical_config={'path':str(config_path),'sha256':config_sha})

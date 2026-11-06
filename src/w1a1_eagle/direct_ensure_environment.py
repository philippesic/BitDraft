"""Specialized prelaunch host ensure; no TRAIN/teacher/GPU operations.

Call only from launch after pause rejection, under its existing campaign flock.
Status/check paths must never call this module. Frozen active owners are untouched.
"""
import copy
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pwd
import shutil
import stat
import subprocess
import time
import uuid

KEEPER_SHA='e16455fdad78bd3bbabf61f79ed3cd8b86c295aba846416a8326a5d96b684d7c'
LAUNCHER_SHA='7ae86393a2b31af8108cafcc38de2031db917d72b52b8d854f10d84b21ab6aa5'
AUDITOR_SHA='a16a1d53d2b9f4098f8d3377e3f4829e17666bbf2eb1a33e831c34221cd93add'
READER_SHA='d8531872a1641c7b35e822107a3186abf9e16542cabbe83e6c0dfc0bdfcd5544'
PIPELINE_SHA='a27e9121261938175d88fb78651b2f02fa0a641d4b5afccc2197e561399979af'
SELECTOR_SHA='98168c149ef7afffcdf9902c2c5a7d94eabff6887521925ed243d3c632e57ead'
REFRESH_SHA='c060794652e5ff6bb007bafa2b9609f8781bd12a4eb98bdf3fbff27e785a98ad'


def require(ok,reason):
    if not ok:raise ValueError(reason)


def pin(path):
    path=Path(path)
    require(path.is_absolute() and path.resolve()==path and not path.is_symlink(),'canonical file required')
    before=path.stat();require(stat.S_ISREG(before.st_mode) and before.st_nlink==1 and before.st_size<=16*1024**2,'bounded regular single-link control required')
    raw=path.read_bytes();after=path.stat()
    fields=('st_dev','st_ino','st_mode','st_uid','st_gid','st_nlink','st_size','st_mtime_ns','st_ctime_ns')
    require(tuple(getattr(before,k) for k in fields)==tuple(getattr(after,k) for k in fields),'control changed')
    return {'path':str(path),'sha256':hashlib.sha256(raw).hexdigest()}


def authenticated(path,expected):
    result=pin(path);require(result['sha256']==expected,'original operational source differs: '+str(path));return result


def create(path,value):
    path=Path(path)
    require(path.is_absolute() and path.parent.resolve()==path.parent,'canonical output parent required')
    path.parent.mkdir(parents=True,exist_ok=True)
    require(path.parent.resolve()==path.parent,'canonical output parent changed')
    with path.open('x') as stream:
        json.dump(value,stream,indent=2,sort_keys=True);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    return pin(path)


def json_record(record):
    require(pin(record['path'])==record,'control pin differs')
    return json.loads(Path(record['path']).read_text())


def original_tools(record):
    authenticated(record['path'],REFRESH_SHA)
    require(record['sha256']==REFRESH_SHA,'original admission helper selection differs')
    spec=importlib.util.spec_from_file_location('_environment_original_refresh',record['path'])
    tools=importlib.util.module_from_spec(spec);spec.loader.exec_module(tools)
    require(tools.READER_SHA256==READER_SHA,'original admission reader differs');return tools



INLINE_DESCRIPTOR_SHA = 'cba834e3ad337970e9904ee5233f63953a915d73d84ef00fc32952d46f8ea10a'
INLINE_COMMAND_SHA = 'ac570693abe6a704eb618f4a1193b15839fa9427538beeafb1e58cd08004b9b1'
INLINE_TEMPLATE = "$ErrorActionPreference='Stop'; $stage='{WINDOWS_LAUNCHER}'; $expectedLauncher='7ae86393a2b31af8108cafcc38de2031db917d72b52b8d854f10d84b21ab6aa5'; $actualLauncher=(Get-FileHash -LiteralPath $stage -Algorithm SHA256).Hash.ToLowerInvariant(); if($actualLauncher -ne $expectedLauncher){throw 'staged launcher SHA256 mismatch'}; $run='{RUN}'; $keeper='{KEEPER}'; $WslExecutable=Join-Path $env:SystemRoot 'System32\\wsl.exe'; $probeB64='{PREFLIGHT_B64}'; $q=[char]39; $probePython='import base64;exec(compile(base64.b64decode('+ $q + $probeB64 + $q + ',validate=True),'+$q+'<keeper-preflight>'+$q+','+$q+'exec'+$q+'))'; $probeOutput=@(& $WslExecutable -d Ubuntu -u {USER} -- /usr/bin/python3 -B -c $probePython 2>&1); $probeExit=$LASTEXITCODE; if($probeExit -ne 0){throw ('Linux keeper preflight failed, exit '+$probeExit)}; if($probeOutput.Count -ne 1){throw ('Unexpected Linux preflight output count '+$probeOutput.Count)}; $preflight=ConvertFrom-Json $probeOutput[0]; if($preflight.schema -ne 'keeper_namespace_preflight_v1' -or $preflight.namespace_absent -ne $true -or $preflight.lexists -ne $false -or $preflight.source_sha256 -ne 'e16455fdad78bd3bbabf61f79ed3cd8b86c295aba846416a8326a5d96b684d7c'){throw 'Linux source/lexists preflight did not match exact expectations'}; $existing=@(Get-CimInstance -ClassName Win32_Process | Where-Object { $_.Name -eq 'wsl.exe' -and $_.CommandLine -like ('*'+$keeper+' --run-dir '+$run+'*') }); if($existing.Count -ne 0){throw 'matching Windows WSL keeper process already exists'}; $KeeperCommand=([char]34)+$WslExecutable+([char]34)+' -d Ubuntu -u {USER} -- /usr/bin/python3 -B '+$keeper+' --run-dir '+$run; $KeeperCreation=Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=$KeeperCommand}; if($KeeperCreation.ReturnValue -ne 0){throw ('Keeper process creation failed: '+$KeeperCreation.ReturnValue)}; $KeeperWindowsPid=[uint32]$KeeperCreation.ProcessId; $KeeperWindowsProcess=Get-CimInstance -ClassName Win32_Process -Filter ('ProcessId = {0}' -f $KeeperWindowsPid); if($null -eq $KeeperWindowsProcess){throw ('Created keeper process already absent: '+$KeeperWindowsPid)}; [ordered]@{schema='wsl_interop_keeper_windows_launch_v1'; observed_utc=[DateTime]::UtcNow.ToString('o'); return_value=[uint32]$KeeperCreation.ReturnValue; windows_pid=$KeeperWindowsPid; creation_date=$KeeperWindowsProcess.CreationDate.ToUniversalTime().ToString('o'); command_line=$KeeperWindowsProcess.CommandLine; linux_keeper_source=$keeper; linux_run_path=$run; gpu_queried=$false; model_loaded=$false} | ConvertTo-Json -Depth 4 -Compress"


def inline_keeper_command(windows_launcher, keeper, target, user):
    """Literal adaptation of the accepted source-pinned inline-CIM caller."""
    import base64
    import re
    for value in (keeper['path'],str(target)):
        require(re.fullmatch(r'/[A-Za-z0-9._/-]+',value) is not None,
                'original launcher Linux path grammar required')
    require(re.fullmatch(r'[a-z][a-z0-9_-]*',user) is not None,
            'original launcher Linux user grammar required')
    preflight = ('import os, json, hashlib, time\n'
        + 'source = '+repr(keeper['path'])+'\n'
        + 'run = '+repr(str(target))+'\n'
        + 'expected = '+repr(KEEPER_SHA)+'\n'
        + 'actual = hashlib.sha256(open(source, "rb").read()).hexdigest()\n'
        + 'exists = os.path.lexists(run)\n'
        + 'assert actual == expected\nassert not exists\n'
        + 'print(json.dumps({"schema":"keeper_namespace_preflight_v1","observed_unix":time.time(),"boot_id":open("/proc/sys/kernel/random/boot_id").read().strip(),"source":source,"source_sha256":actual,"run_namespace":run,"namespace_absent":True,"lexists":exists}))\n')
    compile(preflight,'<keeper-preflight>','exec')
    encoded=base64.b64encode(preflight.encode()).decode('ascii')
    replacements={'{WINDOWS_LAUNCHER}':windows_launcher.replace("'","''"),
                  '{RUN}':str(target),'{KEEPER}':keeper['path'],
                  '{USER}':user,'{PREFLIGHT_B64}':encoded}
    body=INLINE_TEMPLATE
    for old,new in replacements.items():body=body.replace(old,new)
    require(body.count('Invoke-CimMethod')==1 and '-ExecutionPolicy' not in body,
            'one original inline CIM launch required')
    return body,dict(schema='original_inline_cim_keeper_caller_v1',
        descriptor_sha256=INLINE_DESCRIPTOR_SHA,original_command_sha256=INLINE_COMMAND_SHA,
        command_sha256=hashlib.sha256(body.encode()).hexdigest(),linux_preflight=preflight,
        preflight_sha256=hashlib.sha256(preflight.encode()).hexdigest(),keeper=keeper,
        target=str(target),user=user,launcher_sha256=LAUNCHER_SHA,
        windows_launcher=windows_launcher,gpu_queried=False,model_loaded=False)

def ensure_keeper(config,operation,repo,authorize,keeper_environment,*,execute=subprocess.run):
    try:
        return config['keeper'],keeper_environment(config['keeper']),None
    except (OSError,ValueError,KeyError) as error:
        # Never edit a dead keeper/deadline. A NEW CPU-only original keeper must
        # establish its own boot/birth/socket; all later GPU gates remain intact.
        previous={'type':type(error).__name__,'reason':str(error)}
    authorize()
    endpoint=os.environ.get('WSL_INTEROP','')
    require(endpoint and Path(endpoint).is_socket() and os.environ.get('WSL_DISTRO_NAME')=='Ubuntu',
            'working Ubuntu Windows interop required; this Linux entrypoint cannot boot WSL or invent an endpoint')
    executable=shutil.which('powershell.exe');converter=shutil.which('wslpath')
    require(executable and converter,'Windows PowerShell and wslpath must be available in this WSL terminal')
    keeper=authenticated(Path(repo)/'scripts/wsl_interop_keeper_20261011.py',KEEPER_SHA)
    launcher=authenticated(Path(repo)/'scripts/start_wsl_interop_keeper_20261011.ps1',LAUNCHER_SHA)
    windows_path=execute([converter,'-w',launcher['path']],check=True,capture_output=True,text=True).stdout.strip()
    target=Path(operation)/'keeper'
    require(not os.path.lexists(target),'fresh keeper namespace required')
    command,caller=inline_keeper_command(windows_path,keeper,target,pwd.getpwuid(os.getuid()).pw_name)
    argv=[executable,'-NoProfile','-NonInteractive','-Command',command]
    caller['argv']=argv
    create(Path(operation)/'keeper-inline-command.json',caller)
    authorize()
    try:
        result=execute(argv,check=True,capture_output=True,text=True)
    except BaseException as failure:
        try:
            create(Path(operation)/'keeper-inline-failure.json',dict(
                schema='original_inline_cim_keeper_failure_v1',type=type(failure).__name__,
                reason=str(failure),stdout=getattr(failure,'stdout',None),
                stderr=getattr(failure,'stderr',None),caller=pin(Path(operation)/'keeper-inline-command.json')))
        except BaseException:pass
        raise
    create(Path(operation)/'keeper-inline-result.json',dict(returncode=getattr(result,'returncode',0),stdout=result.stdout,stderr=getattr(result,'stderr',None)))
    launched=json.loads(result.stdout)
    require(launched.get('schema')=='wsl_interop_keeper_windows_launch_v1'
            and launched.get('return_value')==0 and type(launched.get('windows_pid')) is int
            and launched.get('linux_keeper_source')==keeper['path']
            and launched.get('linux_run_path')==str(target)
            and launched.get('gpu_queried') is False and launched.get('model_loaded') is False,
            'original Windows CIM keeper launch response differs')
    create(Path(operation)/'keeper-windows-launch.json',launched)
    deadline=time.monotonic()+30
    while time.monotonic()<deadline:
        authorize()
        if (target/'keeper.json').exists():
            record=pin(target/'keeper.json')
            environment=keeper_environment(record)
            return record,environment,previous
        time.sleep(.25)
    raise TimeoutError('original CPU keeper did not publish authenticated readiness; preserve partial operation')


def selected_inputs(config,runtime):
    """Auditor tuples only; bind original package BEFORE scientific imports."""
    runtime.bind_scientific_sources(config)
    selector_path=Path(__file__).with_name('direct_environment_inputs.py');authenticated(selector_path,SELECTOR_SHA)
    loader=importlib.util.spec_from_file_location('_environment_auditor_inputs',selector_path)
    selector=importlib.util.module_from_spec(loader);loader.loader.exec_module(selector)
    from w1a1_eagle import sustained_block_handoff as core
    from w1a1_eagle import direct_prepared_recovery as publication
    from w1a1_eagle import block_shard_lifecycle as original_api
    from types import SimpleNamespace
    native=Path(config['native_source_root'])
    authenticated(native/'src/w1a1_eagle/block_shard_lifecycle.py','a2ad87283999fe1268cd933dce59388c0c0ab544a7a27dcc3797967f305ece4e')
    api=original_api
    lane=runtime.checked(config['lane']);spec=runtime.checked(lane['config'])
    plan=api.FrozenShardPlan.load({key:spec['data'][key] for key in ('path','sha256')})
    replay=runtime.checked(spec['data']['replay_admission']);trial=runtime.checked(replay['trial_publication'])
    rows=[dict(origin='global_trial',manifest=trial['manifest'],admission=trial['admission'])]
    state,state_sha=runtime.metadata(Path(config['run_dir'])/'state.json');observations={}
    def read(path):
        record=pin(Path(path));value=json_record(record);observations[record['path']]=record;return value
    predecessor=runtime.prior_active_window(config,state)
    catalog=runtime.checked(predecessor['catalog']) if predecessor else None
    old_roots={str(Path(row['path']).parent) for row in catalog['source_cache_states']} if catalog else set()
    registrations={root:row for root,row in state.get('prepared_pair_registrations',{}).items() if root not in old_roots}
    require(set(registrations)<=set(state.get('prepared_owned_pair_roots',[])), 'registered auditor ownership absent')
    require(predecessor is not None or registrations,'authenticated predecessor or registrations required')
    deferred=[]
    if registrations:
        def validate(row):
            runtime.prior_active_window(config,{'direct_training_runtime':row['runtime']})
            previous=runtime.checked(row['runtime'])
            expected=[digest for path,digest in previous['runtime_sources'].items()
                      if Path(path).parts[-3:]==('src','w1a1_eagle','sustained_block_handoff.py')]
            require(expected==[row['producer_source']['sha256']] and pin(row['producer_source']['path'])==row['producer_source'],
                    'registered original producer source differs')
        groups,identities=runtime.recover_owned_process_inventory(Path(config['run_dir']))
        rows.extend(selector.registered_inputs(plan,registrations,run_dir=config['run_dir'],
            config=spec['shard_lifecycle'],source=lane['source'],runner=SimpleNamespace(process_groups=groups,process_identities=identities),
            validate_registration=validate,api=api,read=read,publication=publication,deferred_intents=deferred))
    if catalog is not None:
        latest,_=runtime.metadata(Path(config['run_dir'])/'training/checkpoints/latest.json')
        cursor=latest['cursor']['data_cursor'];reservation=latest['cursor'].get('batch_reservation')
        if registrations:
            current=selector.completed_retirement_states(plan,cursor,predecessor['catalog'],state,
                canonical_root=spec['data']['cache_root'],reservation=reservation,registrations=registrations,
                expected_producer=predecessor['producer_source'],
                core=core,api=api,read=read);deleting=set()
        elif catalog['producer_source']['sha256'] != selector.CORE_SHA:
            current=selector.unchanged_predecessor_states(plan,predecessor['catalog'],
                expected_producer=predecessor['producer_source'],canonical_root=spec['data']['cache_root'],
                core=core,api=api,read=read)
            deleting=set()
        else:
            current,deleting,_=selector.surviving_catalog_states(plan,cursor,predecessor['catalog'],
                reservation=reservation,api=api,core=core,read=read)
        for name,(path,value) in current.items():
            for shard,record in value['resident'].items():
                if (name,shard) in deleting:continue
                directory=Path(record['directory'])
                require(directory.is_dir() and directory.resolve().is_relative_to(path.parent),
                        'current surviving publication directory differs')
                rows.append(dict(origin='surviving:'+shard,manifest=record['manifest'],admission=record['admission']))
    unique={}
    for row in rows:
        key=(row['manifest']['path'],row['manifest']['sha256'],row['admission']['path'],row['admission']['sha256'])
        unique.setdefault(key,row)
    require(runtime.metadata(Path(config['run_dir'])/'state.json')[1]==state_sha,'canonical state changed during selection')
    for record in observations.values():require(pin(record['path'])==record,'selector metadata changed')
    selector.recheck_preservation_inputs(deferred,publication=publication,api=api)
    if deferred:rows[0]['deferred_preservation']=deferred
    return list(unique.values()),lane,state_sha


def census(row,tools):
    manifest=json_record(row['manifest']);old=json_record(row['admission'])
    records=tools.records(old,row['manifest']['sha256']);changes=[]
    for path,record in records.items():
        actual=tools.fingerprint(path)
        require(all(actual[key]==value for key,value in record.items() if key not in {'sha256','device'}),
                'nondevice artifact change or missing data; original error must remain')
        if actual['device']!=record['device']:
            changes.append(dict(path=path,old_device=record['device'],new_device=actual['device']))
    return old,changes


def audit_row(row,directory,*,python,auditor,tools,recovery,authorize,run_cpu):
    """Reuse only a fully source/report/stat-validated actual original audit."""
    old,changes=census(row,tools)
    if not changes:return None,'original_admission_current'
    directory=Path(directory)
    require(directory.is_absolute() and directory.resolve()==directory,'canonical audit row path required')
    completed=directory/'completed.json'
    if completed.exists():
        record=json_record(pin(completed))
        require(record['manifest']==row['manifest'] and record['old_admission']==row['admission']
                and record['auditor']==auditor,'completed audit belongs to different source/input')
        entry=record['entry'];new=json_record(entry['new_admission']);report=json_record(entry['audit_report'])
        recovery.validate_values({},entry,old,new,report,tools)
        for path,value in tools.records(new,row['manifest']['sha256']).items():
            require(tools.fingerprint(path)=={k:v for k,v in value.items() if k!='sha256'},'completed audit current stat differs')
        return entry,'completed_audit_reused'
    # Existing partial output is never overwritten or falsely marked complete.
    directory.mkdir(parents=True,exist_ok=True)
    attempt=directory/('attempt-'+uuid.uuid4().hex);attempt.mkdir(exist_ok=False)
    admission=attempt/'completed-admission.json';report=attempt/'audit-report.json'
    argv=[python,'-B',auditor['path'],'--manifest',row['manifest']['path'],'--manifest-sha256',row['manifest']['sha256'],
          '--require-soft-teachers','--admission-output',str(admission),'--output',str(report)]
    authorize();require(pin(auditor['path'])==auditor,'auditor changed before invocation')
    run_cpu(argv,attempt/'process')
    authorize()
    entry=dict(manifest=row['manifest'],old_admission=row['admission'],new_admission=pin(admission),audit_report=pin(report),device_changes=changes)
    recovery.validate_values({},entry,old,json_record(entry['new_admission']),json_record(entry['audit_report']),tools)
    for path,value in tools.records(json_record(entry['new_admission']),row['manifest']['sha256']).items():
        require(tools.fingerprint(path)=={k:v for k,v in value.items() if k!='sha256'},'new audit artifact changed')
    create(completed,dict(manifest=row['manifest'],old_admission=row['admission'],auditor=auditor,entry=entry,trainer_seconds_charged=0))
    return entry,'original_full_audit_completed'


def ensure_environment(config_path,*,repo,runtime,recovery,lock_fd,run_dir=None,expected_config=None,run_cpu=None,execute=subprocess.run):
    """Candidate launch-only ensure. Caller keeps canonical lock for GPU launch."""
    config_path=Path(config_path);before=pin(config_path)
    if expected_config is not None:
        require(before==expected_config,'canonical config changed after budget settlement')
    public_config=json_record(before)
    run=Path(run_dir or public_config['run_dir']).resolve(strict=True)
    require(run.is_absolute() and run.resolve()==run,'canonical effective run directory required')
    overridden=run!=Path(public_config['run_dir']).resolve(strict=True)
    config=copy.deepcopy(public_config);config['run_dir']=str(run)
    lane=runtime.checked(config['lane']);runtime.current_control(lane)
    lock=Path.home()/'.config/binary-eagle-decoding/rtx5080-campaign.lock'
    info=os.fstat(lock_fd);current=lock.lstat()
    require(stat.S_ISREG(info.st_mode) and info.st_nlink==1 and info.st_uid==os.getuid()
            and (info.st_dev,info.st_ino)==(current.st_dev,current.st_ino),'actual canonical campaign lock required')
    fcntl.flock(lock_fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
    budget_before,initial_budget=runtime.metadata(run/'training/budget-used.json')
    def authorize():
        runtime.current_control(lane)
        require(not (run/'STOP').exists() and not (run/'training/STOP').exists(),'human STOP must remain')
        require(pin(config_path)==before,'canonical config changed during ensure')
        require(runtime.metadata(run/'training/budget-used.json')[1]==initial_budget,'TRAIN accounting changed during ensure')
    authorize()
    require(budget_before.get('active_attempt') is None,'original dead-attempt reconciliation must precede CPU audits')
    rows,_,state_sha=selected_inputs(config,runtime)
    tools_record=config.get('physical_admission_recovery',{}).get('existing_helper') or authenticated(Path(repo)/'scripts/refresh_block_replay_publication_admission.py',REFRESH_SHA)
    tools=original_tools(tools_record)
    # Validate every cheap row BEFORE launching a keeper or auditing arrays.
    for row in rows:census(row,tools)
    operation=run/'direct-runtime'/('environment-recovery-'+uuid.uuid4().hex)
    require(operation.resolve()==operation,'canonical environment operation required')
    operation.mkdir(parents=True,exist_ok=False)
    create(operation/'selected-auditor-inputs.json',dict(rows=rows,authority='CPU-auditor-inputs-only'))
    backup=operation/'config.before.json'
    with backup.open('xb') as stream:
        stream.write(config_path.read_bytes());stream.flush();os.fsync(stream.fileno())
    require(pin(backup)['sha256']==before['sha256'],'exact config backup changed')
    keeper=config['keeper'];environment={};previous=None
    known={}
    selected=config.get('physical_admission_recovery')
    if selected is not None:
        require(selected.get('profile')==recovery.PROFILE,'selected original admission profile differs')
        previous_view=json_record(selected['view'])
        require(previous_view.get('schema')==recovery.PROFILE
                and previous_view.get('reader_sha256')==READER_SHA,'original view source/profile differs')
        valid=previous_view.get('boot_id')==recovery.boot_id()
        if valid:
            view=recovery.load(selected['view'],tools)
            known={(e['old_admission']['path'],e['old_admission']['sha256'],e['manifest']['path'],e['manifest']['sha256']):e for e in view['entries']}
            for row in rows:
                _,changes=census(row,tools)
                key=(row['admission']['path'],row['admission']['sha256'],row['manifest']['path'],row['manifest']['sha256'])
                if changes and key not in known:
                    valid=False # Audit this new tuple; retain genuine known rows.
                if key in known:
                    recovery.load(selected['view'],tools,selected=key)
    else:valid=False
    updated=copy.deepcopy(config);updated['keeper']=keeper
    decisions=[]
    if not valid:
        require(budget_before.get('active_attempt') is None or not any(census(row,tools)[1] for row in rows),
                'original dead-attempt reconciliation must precede CPU audits; no uncertain TRAIN time may be relabeled')
        native=Path(config['native_source_root']);auditor=authenticated(native/'scripts/prepare_block_qat_data.py',AUDITOR_SHA)
        authenticated(native/'src/w1a1_eagle/block_data.py',READER_SHA)
        if run_cpu is None and any(census(row,tools)[1] for row in rows):
            source=lane['source']['src/w1a1_eagle/nine_model_pipeline.py']
            require(source['sha256']==PIPELINE_SHA,'original CPU supervision selection differs')
            authenticated(source['path'],PIPELINE_SHA)
            loader=importlib.util.spec_from_file_location('_environment_original_cpu_supervision',source['path'])
            supervision=importlib.util.module_from_spec(loader);loader.loader.exec_module(supervision)
            runner=supervision.SubprocessRunner(native,{**lane.get('environment',{}),
                'OMP_NUM_THREADS':'2'},authorization=authorize)
            def run_cpu(argv,directory):
                with supervision.stop_signals():
                    runner.run(argv,directory=directory,stop_path=run/'STOP',wall_seconds=18000)
        # Retired rows remain structural only; exact active tuples reuse their
        # original completed audit even if it lives outside our marker directory.
        entries=list(known.values())
        for row in rows:
            old_key=(row['admission']['path'],row['admission']['sha256'],row['manifest']['path'],row['manifest']['sha256'])
            if old_key in known:
                decisions.append(dict(origin=row['origin'],decision='existing_current_boot_audit_reused'))
                continue
            key=hashlib.sha256(json.dumps([row['manifest'],row['admission']],sort_keys=True).encode()).hexdigest()
            # Deterministic CURRENT-BOOT path: interrupted identical completed
            # rows can be reused by a new invocation, not swept or repinned.
            boot=recovery.boot_id();auditroot=run/'direct-runtime'/('environment-audits-'+boot)
            entry,decision=audit_row(row,auditroot/('row-'+key),python=config['python'],auditor=auditor,
                tools=tools,recovery=recovery,authorize=authorize,run_cpu=run_cpu)
            decisions.append(dict(origin=row['origin'],decision=decision))
            if entry:entries.append(entry)
        if entries:
            view=dict(schema=recovery.PROFILE,reader_sha256=READER_SHA,boot_id=recovery.boot_id(),entries=entries,
                      original_full_auditor=auditor,trainer_seconds_charged=0)
            view_pin=create(operation/'view.json',view)
            recovery.load(view_pin,tools) # Structural check, including dormant rows.
            keys={(entry['old_admission']['path'],entry['old_admission']['sha256'],entry['manifest']['path'],entry['manifest']['sha256']) for entry in entries}
            for row in rows:
                key=(row['admission']['path'],row['admission']['sha256'],row['manifest']['path'],row['manifest']['sha256'])
                if key in keys:recovery.load(view_pin,tools,selected=key)
            updated['physical_admission_recovery']=dict(profile=recovery.PROFILE,existing_helper=tools_record,view=view_pin)
        else:
            # Every original admission already matches current physical stats.
            # Keep default ORIGINAL readers rather than select an obsolete view.
            updated.pop('physical_admission_recovery',None)
    from w1a1_eagle import direct_prepared_recovery as publication
    authenticated(Path(__file__).with_name('direct_environment_inputs.py'),SELECTOR_SHA)
    loader=importlib.util.spec_from_file_location('_environment_final_input_recheck',Path(__file__).with_name('direct_environment_inputs.py'))
    selector=importlib.util.module_from_spec(loader);loader.loader.exec_module(selector)
    from w1a1_eagle import block_shard_lifecycle as original_api
    selector.recheck_preservation_inputs([intent for row in rows for intent in row.get('deferred_preservation',[])],publication=publication,api=original_api)
    authorize();require(runtime.metadata(run/'state.json')[1]==state_sha,'canonical run state changed during ensure')
    # Audits and independent source/reference checks precede a replacement
    # keeper, so a failed long audit does not create an unused CPU keeper.
    keeper,environment,previous=ensure_keeper(config,operation,repo,authorize,runtime.keeper_environment,execute=execute)
    updated['keeper']=keeper
    authorize();require(runtime.metadata(run/'state.json')[1]==state_sha,'canonical run state changed before config publication')
    if overridden:
        effective_config=create(operation/'effective-config.json',updated)
    else:
        if updated!=public_config:
            temporary=config_path.with_name('.'+config_path.name+'-environment-'+uuid.uuid4().hex+'.tmp')
            create(temporary,updated);authorize();os.replace(temporary,config_path)
        effective_config=pin(config_path)
    create(operation/'result.json',dict(schema='direct_environment_ensure_v1',config_before=before,config_backup=pin(backup),config_after=pin(config_path),keeper=keeper,
        effective_run_dir=str(run),public_run_dir_preserved=public_config['run_dir'],effective_config=effective_config,
        previous_keeper_failure=previous,audits=decisions,trainer_seconds_charged=0))
    return updated,environment,operation

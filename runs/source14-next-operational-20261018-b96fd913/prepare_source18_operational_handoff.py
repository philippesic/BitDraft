#!/usr/bin/env python3
"""One-off, metadata-only source18 -> reviewed host packet14 handoff input producer.

Run only after the existing source18 public --stop completed. This script never
stops/launches a job, queries a GPU, touches a model/tensor, changes public config,
or creates a historical scientific/GPU verdict. Normal training uses train.py.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import types
import json
import math
import os
from pathlib import Path
import stat
import time

CANDIDATE_MANIFEST_SHA = '22ac955c7f1097d18007a9e6e165803e508955b05add33a37a8b7f3874e75567'
CANDIDATE_HELPER_SHA = '111ceb87'  # Full expected digest is authenticated by the exact manifest.
BASE_ARCHIVE_SHA = '40db1b9af8a4c49b4e26a2059c871abdd2d27fe05ba5c658498aabb478ba5807'
OVERLAYS = frozenset('src/w1a1_eagle/'+name+'.py' for name in (
    'training_supervisor','continuous_budget','direct_checkpoint_retention',
    'direct_continuation_contract','continuation_evaluation','direct_export_retention'))
HELPER = 'src/w1a1_eagle/source18_operational_adoption.py'
TEST_ROLES = frozenset('tests/'+name+'.py' for name in (
    'test_continuation_boundary_reuse','test_continuation_final_gate_reuse',
    'test_direct_scale_only_inputs','test_source18_declared_history','test_source18_history_scope',
    'test_source18_operational_adoption','test_source18_supervisor_inheritance','test_timed_history_single_pass',
    'test_supervisor_host_environment'))


def require(ok,message):
    if not ok:raise ValueError(message)


def canonical(path):
    path=Path(path)
    require(path.is_absolute() and path == path.resolve() and not any(p.is_symlink() for p in (path,*path.parents)),
            'canonical no-symlink path required')
    return path


def metadata_bytes(path,maximum=64*1024**2):
    path=canonical(path);before=path.lstat()
    require(stat.S_ISREG(before.st_mode) and before.st_size<=maximum,'bounded regular metadata required')
    fp=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mode,s.st_mtime_ns,s.st_ctime_ns)
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    with os.fdopen(fd,'rb') as stream:
        require(fp(os.fstat(stream.fileno()))==fp(before),'metadata replaced before read')
        raw=stream.read(maximum+1)
        require(len(raw)<=maximum and fp(os.fstat(stream.fileno()))==fp(before)
                and fp(path.lstat())==fp(before),'metadata changed while reading')
    return raw


def pin(path,raw=None):
    path=canonical(path);raw=metadata_bytes(path) if raw is None else raw
    return {'path':str(path),'sha256':hashlib.sha256(raw).hexdigest()}


def read(locator,files=None):
    require(isinstance(locator,dict) and set(locator)=={'path','sha256'},'exact metadata pin required')
    require(Path(locator['path']).suffix=='.json','JSON control role required; model/tensor paths refused')
    path=canonical(locator['path']) if files is None else files.check(locator)
    raw=metadata_bytes(path)
    require(hashlib.sha256(raw).hexdigest()==locator['sha256'],'metadata SHA mismatch')
    return json.loads(raw),raw


def live(path,files=None):
    require(Path(path).suffix=='.json','JSON control role required; model/tensor paths refused')
    if files is not None and hasattr(files,'live'):
        locator=files.live(str(path))  # CPU receiver transport only; never CLI configurable.
        value,raw=read(locator,files);return value,raw,locator
    raw=metadata_bytes(path);return json.loads(raw),raw,pin(path,raw)


def load_pinned_source(path,digest,name):
    # Execute authenticated source bytes, never a pre-existing .pyc cache.
    raw=metadata_bytes(path,16*1024**2)
    require(hashlib.sha256(raw).hexdigest()==digest,'import source SHA mismatch')
    module=types.ModuleType(name);module.__file__=str(path)
    exec(compile(raw,str(path),'exec'),module.__dict__)
    return module


def verify_candidate(root,manifest_pin,old_manifest):
    root=canonical(root)
    require(manifest_pin=={'path':str(root/'source-manifest.json'),'sha256':CANDIDATE_MANIFEST_SHA},
            'only reviewed host packet14 manifest accepted')
    manifest,_=read(manifest_pin)
    require(manifest.get('base_packet',{}).get('sha256')==BASE_ARCHIVE_SHA,'candidate frozen18 base differs')
    old=old_manifest['files'];current=manifest['files']
    require(set(old)<=set(current) and set(current)-set(old)<=TEST_ROLES|{HELPER},'candidate source roles differ')
    changed={name for name,digest in current.items() if old.get(name)!=digest}
    require(changed==OVERLAYS|TEST_ROLES|{HELPER},'candidate CPU composition differs')
    for name,digest in current.items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts,'source role escapes packet')
        raw=metadata_bytes(root/name,16*1024**2)
        require(hashlib.sha256(raw).hexdigest()==digest,'candidate source SHA mismatch: '+name)
    require(manifest['files'][HELPER].startswith(CANDIDATE_HELPER_SHA),'reviewed helper differs')
    helper=load_pinned_source(root/HELPER,manifest['files'][HELPER],'_reviewed_source18_handoff_helper')
    return manifest,helper


def settled_checkpoint(config,latest,selected,segment,budget):
    require(latest.get('schema')=='block_qat_checkpoint_v1' and latest.get('committed') is True
            and latest.get('cursor',{}).get('completed_endpoint') is None,'committed non-evaluation checkpoint required')
    elapsed=latest['cursor'].get('elapsed_seconds')
    require(type(elapsed) in (int,float) and math.isfinite(elapsed)
            and selected=={'segment':segment[0]} and segment[0]['sha256']=='9a71fdd873ae0aa4556887856212b508fb72591c6a0c86a2b8d23c59f3536c28',
            'first source18 finite selection differs')
    record=segment[1]
    require(budget.get('schema')=='continuous_training_budget_v1'
            and budget.get('source_sha256')==config['lane']['sha256']
            and budget.get('max_seconds')==record['maximum']
            and budget.get('active_attempt') is None
            and budget.get('training_seconds')==elapsed
            and record['start_seconds']<=elapsed<record['start_seconds']+14400,
            'checkpoint/selected budget unsettled or first native boundary reached')


def validate_empty_journal(journal,lane,config,segment):
    expected=dict(candidate=lane['candidate'],bundle_sha256=config['lane']['sha256'],
        config_sha256=config['original_spec']['sha256'],protocol=config['evaluation_protocol'],
        milestones_seconds=[segment['start_seconds']+14400,segment['maximum']],max_seconds=segment['maximum'])
    require(journal.get('schema')=='nine_model_timed_evaluation_state_v1' and journal.get('contract')==expected
            and journal.get('completed')==[] and journal.get('pending') is None,
            'first segment journal is foreign, pending or already evaluated')


def validate_stop(record,config,public_pin,state_pin,state,boot_id,now):
    argv=[config['python'],'-B',str(Path(config['source_root'])/'train.py'),'--config',public_pin['path'],'--stop']
    requested,completed=record.get('requested_unix'),record.get('completed_unix')
    require(record.get('schema')=='source18_public_stop_result_v1' and record.get('configuration')==public_pin
            and record.get('state')==state_pin and record.get('boot_id')==boot_id
            and record.get('argv')==argv and record.get('exit_code')==0
            and record.get('source')=={'path':str(Path(config['source_root'])/'train.py'),
                                      'sha256':config['source_files']['train.py']},'original public stop provenance differs')
    proof=state.get('owned_release',{})
    require(record.get('result')==dict(stop_requested=True,release_verified=True,owned_release=proof)
            and proof.get('owned_process_groups_absent') is True and proof.get('owned_cuda_pids_absent') is True
            and proof.get('other_context_pids')==[], 'original public stop release was not verified')
    require(type(requested) in (int,float) and type(completed) in (int,float)
            and math.isfinite(requested) and math.isfinite(completed)
            and requested<=state.get('observed_unix',-1)<=completed<=now+5
            and 0<=now-completed<=300,'stale or unordered public stop observation')


def stage_journal_absent(path,files=None):
    # A missing directory entry is distinct from a dangling symlink, FIFO or
    # present malformed journal. The latter never permits predecessor fallback.
    physical=(files.journal_path(str(path)) if files is not None and hasattr(files,'journal_path') else path)
    physical=canonical(physical)
    require(physical.parent.parent.is_dir(),'unstarted stage directory is absent')
    try:physical.lstat()
    except FileNotFoundError:return True
    return False


def unstarted_training_provenance(state,config,latest,latest_pin,attempt,files=None,*,source14_transition=False):
    stage=state.get('active_stage') or {};directory=Path(stage.get('directory',''))
    require(state.get('status')=='stopped' and state.get('pending_training') is None
            and stage.get('kind')=='training' and directory.parent==Path(attempt)/'training'
            and len(directory.name)==32 and all(c in '0123456789abcdef' for c in directory.name)
            and stage.get('receipt')==str(directory/'receipt.json'),
            'unstarted TRAIN requires exact stopped original runtime stage')
    journal=directory/'process/process.json'
    require(stage_journal_absent(journal,files),'unstarted TRAIN journal is present or unsafe')
    require(stage_journal_absent(directory/'receipt.json',files),
            'unstarted TRAIN already has a completion receipt')
    previous=state.get('train_receipt');require(isinstance(previous,dict),'preceding DATA receipt required')
    require(Path(previous['path']).name=='receipt.json'
            and Path(previous['path']).parent.parent==Path(attempt)/'training'
            and Path(previous['path']).parent!=directory,'preceding DATA receipt namespace differs')
    endpoint,_=read(previous,files);required=endpoint.get('data_required',{})
    expected_operation={'path':str(Path(attempt)/'operation.json'),
        'sha256':'4afdd2dbbf61015fb32ae70832d94b9687aac37815112087fb86a9c59b96c23e'}
    if source14_transition:
        require(config.get('operation_config',{}).get('sha256')==
            'cefdf01030d672a61ba4bd1afac923c3faf725d2586f658852e0ab8e58a10db7'
            and config.get('operational_context',{}).get('sha256')==
            'c26042e348541d95eb86ab96280f864b89dae293be58c63b976f98807dac0556',
            'source14 unstarted TRAIN requires exact predecessor operation/context')
        expected_operation=config['operation_config']
        require(expected_operation['path']==str(Path(attempt)/'operation.json'),
                'source14 preceding TRAIN operation escaped runtime')
    require(endpoint.get('schema')=='nine_model_stage_receipt_v1' and endpoint.get('status')=='PASS'
            and endpoint.get('committed') is True and endpoint.get('artifact_kind')=='production'
            and endpoint.get('stage')==state.get('candidate')+'/train'
            and endpoint.get('completion_reason')=='shard_required'
            and endpoint.get('timed_evaluation_request') is None
            and endpoint.get('bundle_sha256')==config['lane']['sha256']
            and endpoint.get('config_sha256')==config['original_spec']['sha256']
            and endpoint.get('operation_config')==expected_operation
            and endpoint.get('counters')==latest.get('cursor')
            and endpoint.get('checkpoint')=={k:latest[k] for k in ('path','sha256')}
            and required.get('schema')=='retained_training_data_required_v1'
            and required.get('start_checkpoint')==latest_pin
            and required.get('reservation')==latest['cursor'].get('batch_reservation')
            and required.get('plan_sha256')==latest['cursor'].get('data_cursor',{}).get('dataset_sha256')
            and required.get('window')==endpoint.get('window')
            and required.get('window_admission')==endpoint.get('window_admission'),
            'preceding DATA receipt/latest/original operation differs')
    window,_=read(state['current_window'],files);admission,_=read(state['current_window_admission'],files)
    require(state['current_window']!=endpoint.get('window')
            and window.get('schema')=='prepared_training_window_v1'
            and window.get('plan_sha256')==required.get('plan_sha256')
            and window.get('original_start_checkpoint')==latest_pin
            and window.get('start_checkpoint',{}).get('sha256')==latest_pin['sha256']
            and window.get('start_cursor')==latest['cursor'].get('data_cursor')
            and window.get('start_reservation')==latest['cursor'].get('batch_reservation')
            and window.get('data_admission')==state['current_window_admission'],
            'unstarted TRAIN new window does not bind preceding saved checkpoint')
    expected=dict(schema='prepared_training_window_data_admission_v1',status='PASS',scope='owned_ready_shards_only',
        store=window['store'],start_checkpoint=window['start_checkpoint'],start_cursor=window['start_cursor'],
        start_reservation=window['start_reservation'],known_seals=window['known_seals'],
        shards=window['shards'],coverage=window['coverage'],optimizer_updates=0)
    require(admission==expected,'unstarted TRAIN new window CPU admission differs')
    return dict(schema='source18_unstarted_training_provenance_v1',stage=copy.deepcopy(stage),
        absent_journal=str(journal),preceding_receipt=previous,preceding_window=endpoint['window'],
        preceding_window_admission=endpoint['window_admission'],latest=latest_pin,
        current_window=state['current_window'],current_window_admission=state['current_window_admission'])


def select_training_process(state,files=None,attempt=None,unstarted=None):
    stage=state.get('active_stage') or {}
    if stage.get('kind')=='training':
        require(Path(stage['receipt']).parent==Path(stage['directory']),'training stage receipt scope differs')
        process=Path(stage['directory'])/'process/process.json'
        if stage_journal_absent(process,files) and unstarted is not None:
            proof=unstarted_training_provenance(state,unstarted['config'],unstarted['latest'],
                unstarted['latest_pin'],attempt,files,source14_transition=unstarted.get('source14_transition',False))
            unstarted['provenance']=proof
            process=Path(proof['preceding_receipt']['path']).parent/'process/process.json'
    else:
        receipt_pin=state.get('train_receipt')
        require(isinstance(receipt_pin,dict),'no exact training stage or saved training receipt')
        receipt,_=read(receipt_pin,files)
        require(receipt.get('schema')=='nine_model_stage_receipt_v1' and receipt.get('stage','').endswith('/train'),
                'saved receipt is not a training endpoint')
        process=Path(receipt_pin['path']).parent/'process/process.json'
    if attempt is not None:
        require(process.is_relative_to(Path(attempt)/'training'),'training process escaped original runtime attempt')
    return live(process,files)


def validate_process(process,config,state,resolved,unstarted=None):
    argv=process.get('argv',[])
    require(argv[:4]==[config['python'],'-B','-m','w1a1_eagle.static_training'],
            'training process entrypoint differs')
    def flag(name):
        require(argv.count(name)==1 and argv.index(name)+1<len(argv),'missing/duplicate process flag: '+name)
        return argv[argv.index(name)+1]
    for name,locator in (('--operation-config',resolved['operation_config']),
                         ('--operational-context',resolved['operational_context'])):
        require(flag(name)==locator['path'] and flag(name+'-sha256')==locator['sha256'],
                'training process operation/context differs')
    require(flag('--run-dir')==str(Path(config['run_dir'])/'training')
            and flag('--bundle-sha256')==config['lane']['sha256']
            and flag('--config')==config['original_spec']['path']
            and flag('--admission')==resolved['operation_admission']['path']
            and process.get('pid')==process.get('kernel_identity',{}).get('pid'),
            'training process original config/admission/identity differs')
    stage=state.get('active_stage') or {}
    if unstarted is not None:
        require(flag('--completion-output')==unstarted['preceding_receipt']['path'],
                'preceding process DATA receipt differs')
        for name,key in (('--window','preceding_window'),('--window-admission','preceding_window_admission')):
            require(flag(name)==unstarted[key]['path'] and flag(name+'-sha256')==unstarted[key]['sha256'],
                    'preceding process DATA window differs')
        return
    if stage.get('kind')!='training':
        require(flag('--completion-output')==state['train_receipt']['path'],
                'training process differs from selected saved receipt')
    if stage.get('kind')=='training':
        require(flag('--completion-output')==stage['receipt'],'training process receipt differs')
        for name,key in (('--window','current_window'),('--window-admission','current_window_admission')):
            require(flag(name)==state[key]['path'] and flag(name+'-sha256')==state[key]['sha256'],
                    'training process current window differs')


def require_dead_owners(state,process,boot_id,identity_active):
    owners=[*state.get('owner',{}).values(),process.get('kernel_identity',{})]
    require(owners and all(isinstance(i,dict) and {'pid','start_ticks','boot_id'}<=set(i)
            and i['boot_id']==boot_id and not identity_active(i) for i in owners),
            'source18 owner remains active or boot differs')


def unchanged_observations(observations,files=None):
    for path,raw in observations:
        require(live(path,files)[1]==raw,'source18 boundary drift before publication: '+str(path))


def pause_controls(root):
    controls=[]
    for path in (Path(root)/'STOP',Path(root)/'training/STOP'):
        raw=metadata_bytes(path,1024**2)
        st=path.lstat()
        controls.append(dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest(),
            identity=[st.st_dev,st.st_ino,st.st_size,st.st_mode,st.st_mtime_ns,st.st_ctime_ns]))
    return controls


def host_binding_inputs(locator, *, files, boot_id):
    binding,raw=read(locator,files)
    require(binding.get('schema')=='source18_original_host_binding_preflight_v1',
            'original host binding preflight required')
    root=canonical(binding['tools_root'])
    expected={'scripts/wsl_interop_keeper_20261011.py':
        'e16455fdad78bd3bbabf61f79ed3cd8b86c295aba846416a8326a5d96b684d7c',
        'scripts/start_wsl_interop_keeper_20261011.ps1':
        '7ae86393a2b31af8108cafcc38de2031db917d72b52b8d854f10d84b21ab6aa5'}
    require(set(binding.get('tools',{}))==set(expected),'host tool closure differs')
    for name,digest in expected.items():
        tool=binding['tools'][name]
        require(tool=={'path':str(root/name),'sha256':digest},'original host tool pin differs')
        path=canonical(tool['path']) if files is None else files.check(tool)
        require(hashlib.sha256(metadata_bytes(path,16*1024**2)).hexdigest()==digest,
                'original host tool bytes differ')
    keeper,_=read(binding['keeper'],files)
    require(keeper.get('schema')=='bounded_wsl_interop_keeper_v2'
            and keeper.get('status')=='running' and keeper.get('boot_id')==boot_id
            and keeper.get('wsl_distro_name')=='Ubuntu'
            and binding.get('environment')=={'WSL_INTEROP':keeper['wsl_interop'],
                                            'WSL_DISTRO_NAME':'Ubuntu'},
            'initial keeper binding differs from stopped boot')
    # Launch rechecks actual PID birth/socket/UID/deadline with the original
    # keeper consumer. This metadata selection grants no process/GPU authority.
    return binding,raw


def prepare_handoff(public_config,runtime_pin,source_manifest,candidate_root,candidate_manifest,stop_result,
                    output_dir,*,host_binding=None,files=None,identity_active=None,boot_id=None,now=None):
    """Produce fresh immutable inputs only; injectable readers are CPU test seams."""
    now=time.time() if now is None else now
    directory=canonical(output_dir)
    require(not directory.exists(),'fresh output directory required; existing snapshots preserved')
    config,configraw=read(public_config,files);runtime,runtimeraw=read(runtime_pin,files)
    require(config.get('schema')=='unattended_training_v1' and runtime.get('config')==public_config,
            'public configuration/runtime differs')
    root=Path(config['run_dir'])
    require(Path(runtime_pin['path'])==Path(runtime['attempt'])/'runtime.json'
            and Path(runtime['attempt']).parent==root/'unattended','original runtime attempt namespace differs')
    protected=(root/'training',Path(config['source_root']),canonical(candidate_root),
               Path(config['bootstrap']['output_dir']),Path(public_config['path']).parent)
    require(directory!=root and all(not directory.is_relative_to(path) for path in protected),
            'snapshot directory overlaps paid/config/source data')
    state,stateraw,statepin=live(root/'state.json',files)
    require(state.get('status')=='stopped' and state.get('pending_training') is None,'source18 is not stopped without pending evaluation')
    current,currentraw,currentpin=live(root/'unattended/current.json',files)
    require(current==runtime_pin,'current runtime selector differs from supplied runtime')
    selected,selectedraw,selectedpin=live(root/'training/continuation/selected.json',files)
    segment_pin=selected.get('segment');segment,_=read(segment_pin,files)
    budget,budgetraw,budgetpin=live(Path(segment_pin['path']).parent/'budget-used.json',files)
    latest,latestraw,latestpin=live(root/'training/checkpoints/latest.json',files)
    settled_checkpoint(config,latest,selected,(segment_pin,segment),budget)
    old_manifest,_=read(source_manifest,files)
    require(source_manifest['path']==str(Path(config['source_root'])/'source-manifest.json'),
            'predecessor source manifest namespace differs')
    manifest,helper=verify_candidate(candidate_root,candidate_manifest,old_manifest)
    require(config['source_files']=={k:v for k,v in old_manifest['files'].items() if k in config['source_files']},
            'predecessor source inventory differs')
    unstarted=dict(config=config,latest=latest,latest_pin=latestpin)
    process,processraw,processpin=select_training_process(state,files,attempt=runtime['attempt'],unstarted=unstarted)
    provenance=unstarted.get('provenance')
    if identity_active is None:
        path=Path(candidate_root)/'src/w1a1_eagle/nine_model_pipeline.py'
        api=load_pinned_source(path,manifest['files']['src/w1a1_eagle/nine_model_pipeline.py'],
                               '_reviewed_handoff_identity');identity_active=api.identity_active
    boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip() if boot_id is None else boot_id
    require_dead_owners(state,process,boot_id,identity_active)
    stopped,stopraw=read(stop_result,files)
    validate_stop(stopped,config,public_config,statepin,state,boot_id,now)
    stop_source=stopped.get('source',{})
    require(set(stop_source)=={'path','sha256'},'stop source pin required')
    stop_path=canonical(stop_source['path']) if files is None else files.check(stop_source)
    require(hashlib.sha256(metadata_bytes(stop_path,16*1024**2)).hexdigest()==stop_source['sha256'],
            'original stop source bytes changed')
    pauses=pause_controls(root)
    bootstrap_pointer,bootstrap_pointerraw,bootstrap_pointerpin=live(root/'unattended/bootstrap-complete.json',files)
    bootstrap,bootstrapraw=read(bootstrap_pointer,files)
    resolved_pin=bootstrap['resolved_config'];resolved,_=read(resolved_pin,files)
    require(resolved.get('operation_config',{}).get('sha256')==helper.OPERATION_SHA
            and resolved.get('operational_context',{}).get('sha256')==helper.CONTEXT_SHA,'not the declared source18 runtime')
    validate_process(process,config,state,resolved,unstarted=provenance)
    require(host_binding is not None,'initial authenticated host binding required')
    binding,bindingraw=host_binding_inputs(host_binding,files=files,boot_id=boot_id)
    timed,timedraw,timedpin=live(root/'training/timed-evaluation-state.json',files)
    journalpath=Path(segment_pin['path']).parent/'timed-evaluation-state.json'
    journal=None
    if journalpath.exists() or (files is not None and hasattr(files,'exists') and files.exists(str(journalpath))):
        journal,journalraw,journalpin=live(journalpath,files)
        lane,_=read(config['lane'],files);validate_empty_journal(journal,lane,config,segment)
    directory.mkdir(parents=True,exist_ok=False)
    snapshots={}
    def snapshot(name,raw):
        path=directory/(name+'.json')
        with path.open('xb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
        snapshots[name]=pin(path,raw);return snapshots[name]
    contract=dict(schema=helper.PROFILE,runtime=snapshot('runtime',runtimeraw),bootstrap=snapshot('bootstrap-receipt',bootstrapraw),
        resolved_configuration=resolved_pin,configuration=snapshot('public-configuration-backup',configraw),
        operation=resolved['operation_config'],context=resolved['operational_context'],admission=resolved['operation_admission'],
        source_manifest=source_manifest,state=snapshot('stopped-state',stateraw),checkpoint=snapshot('latest',latestraw),
        timed=snapshot('original-timed-journal',timedraw),selected=snapshot('selected',selectedraw),segment=segment_pin,
        budget=snapshot('selected-budget',budgetraw),original_recovery=config['original_recovery'],
        window=state['current_window'],window_admission=state['current_window_admission'],
        recovery_adoption=state['checkpoint_recovery_adoption'],trainer_process=snapshot('trainer-process',processraw),
        segment_journal=None if journal is None else snapshot('empty-segment-journal',journalraw))
    if provenance is not None:
        snapshot('unstarted-train-provenance',json.dumps(provenance,sort_keys=True).encode())
        snapshot('preceding-data-receipt',read(provenance['preceding_receipt'],files)[1])
    snapshot('public-stop-result',stopraw)
    snapshot('host-environment-binding',bindingraw)
    snapshot('current-runtime-selector',currentraw)
    snapshot('bootstrap-complete-pointer',bootstrap_pointerraw)
    release=dict(schema='source18_operational_release_v1',**{k:contract[k] for k in ('checkpoint','selected','budget','state')},
        owned_release=copy.deepcopy(state['owned_release']),original_public_stop=snapshots['public-stop-result'],boot_id=boot_id)
    contract['release']=snapshot('observed-release-binding',json.dumps(release,sort_keys=True).encode())
    contractpin=snapshot('source18-adoption',json.dumps(contract,sort_keys=True).encode())
    require(all(value is None or not isinstance(value,dict) or set(value)!={'path','sha256'}
                or Path(value['path']).suffix=='.json' for value in contract.values()),
            'adoption control role must remain JSON metadata')
    helper.validate_predecessor(contractpin,files=files,require_released=True,identity_active=identity_active)
    # A second independent observation closes publication drift; no mutable path is changed.
    unchanged_observations(((root/'unattended/current.json',currentraw),
        (root/'unattended/bootstrap-complete.json',bootstrap_pointerraw),(root/'state.json',stateraw),(root/'training/checkpoints/latest.json',latestraw),
        (root/'training/continuation/selected.json',selectedraw),(Path(segment_pin['path']).parent/'budget-used.json',budgetraw),
        (root/'training/timed-evaluation-state.json',timedraw)),files)
    require((journalpath.exists() if files is None else files.exists(str(journalpath)))==(journal is not None),
            'segment journal appeared or vanished during preparation')
    if journal is not None:require(live(journalpath,files)[1]==journalraw,'segment journal changed before publication')
    require(read(public_config,files)[1]==configraw and read(runtime_pin,files)[1]==runtimeraw,
            'public configuration/runtime drift before publication')
    require(read(source_manifest,files)[0]==old_manifest,'predecessor source manifest drift before publication')
    # Bind the current released metadata once more before creating a usable config.
    helper.validate_predecessor(contractpin,files=files,require_released=True,identity_active=identity_active)
    if provenance is not None:
        require(unstarted_training_provenance(state,config,latest,latestpin,runtime['attempt'],files)==provenance
                and live(Path(provenance['preceding_receipt']['path']).parent/'process/process.json',files)[1]==processraw,
                'unstarted TRAIN provenance changed during preparation')
    require(pause_controls(root)==pauses,'public STOP flags changed during preparation')
    successor=copy.deepcopy(config)
    successor.update(source_root=str(canonical(candidate_root)),source_files=manifest['operational_source_files'],
        supervisor={'path':str(canonical(candidate_root)/'scripts/remote_job.py'),'sha256':manifest['files']['scripts/remote_job.py']},
        source18_adoption=contractpin)
    successor.update(host_environment_tools_root=binding['tools_root'],keeper=binding['keeper'])
    candidatepin=snapshot('candidate-training-configuration',json.dumps(successor,sort_keys=True,indent=2).encode())
    require(live(root/'unattended/current.json',files)[0]==runtime_pin,
            'current runtime selector changed after candidate publication')
    require(live(root/'unattended/bootstrap-complete.json',files)[1]==bootstrap_pointerraw,
            'bootstrap selector changed after candidate publication')
    unchanged_observations(((root/'training/checkpoints/latest.json',latestraw),
        (root/'training/continuation/selected.json',selectedraw),
        (Path(segment_pin['path']).parent/'budget-used.json',budgetraw)),files)
    if provenance is not None:
        require(unstarted_training_provenance(state,config,latest,latestpin,runtime['attempt'],files)==provenance,
                'unstarted TRAIN provenance changed after publication')
    require(pause_controls(root)==pauses,'public STOP flags changed after candidate publication')
    record=dict(schema='source18_handoff_preparation_v1',adoption=contractpin,candidate_configuration=candidatepin,
        candidate_manifest=candidate_manifest,source_manifest=source_manifest,snapshots=snapshots,
        producer=pin(Path(__file__).resolve()),optimizer_updates_performed=0,public_configuration_changed=False,
        unstarted_training=provenance,pause_controls=pauses,rollback=dict(eligible_only_if_no_new_budget_reservation_or_ledger_latest_selected_changes=True,predecessor_public_configuration=contract['configuration'],
            checkpoint=contract['checkpoint'],selected=contract['selected'],budget=contract['budget'],
            runtime_selector=snapshots['current-runtime-selector'],bootstrap_selector=snapshots['bootstrap-complete-pointer'],
            note='No rollback action is implemented. Config-only rollback eligibility requires original clock bytes/no new reservation, latest and selected all unchanged with dead owners. Never restore or overwrite paid bytes after any ledger/reservation/selection/checkpoint change.'),
        full_native_history_validated=False,current_gpu_admission_produced=False)
    result=snapshot('preparation-receipt',json.dumps(record,sort_keys=True,indent=2).encode())
    return dict(receipt=result,adoption=contractpin,configuration=candidatepin)


def verify_source14_candidate(root,manifest_pin,old_manifest):
    """Only the four reviewed transition roles may differ; no packet mutations."""
    root=canonical(root)
    require(manifest_pin['path']==str(root/'source-manifest.json'),'source14-next manifest namespace differs')
    manifest,_=read(manifest_pin)
    require(set(manifest.get('files',{}))==set(old_manifest['files']), 'source14-next inventory differs')
    changed={k for k,v in manifest['files'].items() if old_manifest['files'][k]!=v}
    allowed={HELPER,'src/w1a1_eagle/training_supervisor.py','src/w1a1_eagle/continuous_budget.py',
             'src/w1a1_eagle/direct_checkpoint_retention.py'}
    require(changed==allowed,'source14-next exact four-role composition differs')
    require(manifest['files']['src/w1a1_eagle/continuous_budget.py']==
        '020925bf91fd845f92aabc9c2c82f93d198a15c0885ccaff1cd602f189d47376',
        'source14-next reviewed budget optimization differs')
    for name,digest in manifest['files'].items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts,'source14-next role escapes root')
        require(hashlib.sha256(metadata_bytes(root/name,16*1024**2)).hexdigest()==digest,
                'source14-next source SHA differs: '+name)
    helper=load_pinned_source(root/HELPER,manifest['files'][HELPER],'_reviewed_source14_next_handoff_helper')
    return manifest,helper


def source14_stop_configuration(stop_config,runtime,bootstrap,*,public_default=None,files=None):
    """STOP and descriptor use immutable resolved4e, never public/private aliases."""
    resolved=bootstrap.get('resolved_config')
    require(isinstance(resolved,dict) and stop_config==resolved
        and stop_config.get('sha256')=='4e9c31886c26247430e5709b50e3b94401a6e4aaab7daa657eaa6fdb9d640c8b',
        'source14 STOP requires exact immutable resolved4e configuration')
    config,raw=read(resolved,files)
    configurations=[bootstrap.get('configuration'),resolved]
    if public_default is not None:
        source14_public_default_bytes(public_default,raw,files=files)
        configurations.append(public_default)
    require(runtime.get('config') in configurations,
            'source14 runtime is not its original or exact resolved configuration')
    return config,raw


def source14_public_default_bytes(locator,raw,*,files=None):
    if locator is None:return None
    value,public_raw=read(locator,files)
    require(locator.get('sha256')=='4e9c31886c26247430e5709b50e3b94401a6e4aaab7daa657eaa6fdb9d640c8b'
        and public_raw==raw,'source14 public default differs from immutable resolved configuration')
    return public_raw


def prepare_source14_handoff(public_config,runtime_pin,source_manifest,candidate_root,candidate_manifest,
                             stop_result,output_dir,*,host_binding=None,public_default=None,files=None,
                             identity_active=None,boot_id=None,now=None):
    """Snapshot an actual released post16h boundary, including its old journal.

    This never clears a journal or chooses when to stop. Real16h metadata and a
    genuine new-source admission are external prerequisites, never fabricated.
    """
    now=time.time() if now is None else now
    directory=canonical(output_dir)
    require(not directory.exists(),'fresh output directory required; historical controls preserved')
    config,configraw=read(public_config,files)
    require(public_config['sha256']=='4e9c31886c26247430e5709b50e3b94401a6e4aaab7daa657eaa6fdb9d640c8b',
            'only exact source14 resolved configuration may transition')
    runtime,runtimeraw=read(runtime_pin,files)
    root=Path(config['run_dir'])
    require(config.get('schema')=='unattended_training_v1'
        and Path(runtime_pin['path'])==Path(runtime['attempt'])/'runtime.json'
        and Path(runtime['attempt']).parent==root/'unattended','source14 runtime/configuration namespace differs')
    protected=(root/'training',Path(config['source_root']),canonical(candidate_root),
               Path(config['bootstrap']['output_dir']),Path(public_config['path']).parent)
    require(directory!=root and all(not directory.is_relative_to(p) for p in protected),
            'snapshot directory overlaps paid/config/source data')
    state,stateraw,statepin=live(root/'state.json',files)
    require(state.get('status')=='stopped' and state.get('pending_training') is None,
            'source14 is not stopped without pending evaluation')
    selector,selectorraw,_=live(root/'unattended/current.json',files)
    require(selector==runtime_pin,'source14 runtime selector differs')
    bootstrap_pin,bootstrap_pointerraw,_=live(root/'unattended/bootstrap-complete.json',files)
    bootstrap,_=read(bootstrap_pin,files)
    resolved_pin=bootstrap.get('resolved_config')
    require(source14_stop_configuration(public_config,runtime,bootstrap,
        public_default=public_default,files=files)[1]==configraw,
            'source14 original bootstrap/resolved/runtime binding differs')
    public_default_raw=source14_public_default_bytes(public_default,configraw,files=files)
    latest,latestraw,_=live(root/'training/checkpoints/latest.json',files)
    selected,selectedraw,_=live(root/'training/continuation/selected.json',files)
    require(set(selected)=={'segment'},'source14 selected shape differs')
    segment_pin=selected['segment'];segment,_=read(segment_pin,files)
    segment_dir=Path(segment_pin['path']).parent
    budget,budgetraw,_=live(segment_dir/'budget-used.json',files)
    journal,journalraw,_=live(segment_dir/'timed-evaluation-state.json',files)
    timed,timedraw,_=live(root/'training/timed-evaluation-state.json',files)
    old_manifest,_=read(source_manifest,files)
    manifest,helper=verify_source14_candidate(candidate_root,candidate_manifest,old_manifest)
    unstarted=dict(config=config,latest=latest,latest_pin=pin_bytes(root/'training/checkpoints/latest.json',latestraw),
                   source14_transition=True)
    process,processraw,_=select_training_process(state,files,attempt=runtime['attempt'],unstarted=unstarted)
    provenance=unstarted.get('provenance')
    # A normal postDATA loader already has its own process journal, so there is
    # no need to manufacture an earlier TRAIN process or a native result.
    validate_process(process,config,state,config,unstarted=provenance)
    if identity_active is None:
        api=load_pinned_source(Path(candidate_root)/'src/w1a1_eagle/nine_model_pipeline.py',
            manifest['files']['src/w1a1_eagle/nine_model_pipeline.py'],'_source14_handoff_identity')
        identity_active=api.identity_active
    boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip() if boot_id is None else boot_id
    require_dead_owners(state,process,boot_id,identity_active)
    stopped,stopraw=read(stop_result,files)
    validate_stop(stopped,config,public_config,statepin,state,boot_id,now)
    source=stopped.get('source',{})
    require(set(source)=={'path','sha256'},'source14 public stop source required')
    stop_source=canonical(source['path']) if files is None else files.check(source)
    require(hashlib.sha256(metadata_bytes(stop_source,16*1024**2)).hexdigest()==source['sha256'],
            'source14 public stop source changed')
    require(host_binding is not None,'authenticated host binding required')
    binding,bindingraw=host_binding_inputs(host_binding,files=files,boot_id=boot_id)
    require(binding['tools_root']==config.get('host_environment_tools_root') and binding['keeper']==config.get('keeper'),
            'source14 host keeper identity changed')
    pauses=pause_controls(root)
    directory.mkdir(parents=True,exist_ok=False)
    snapshots={}
    def snapshot(name,raw):
        path=directory/(name+'.json')
        with path.open('xb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
        snapshots[name]=pin(path,raw);return snapshots[name]
    operation,_=read(config['operation_config'],files)
    contract=dict(schema=helper.SOURCE14_PROFILE,configuration=resolved_pin,operation=config['operation_config'],
        context=config['operational_context'],admission=config['operation_admission'],source_manifest=source_manifest,
        adoption=config['source18_adoption'],adoption_receipt=state['source18_operational_adoption'],
        history_validation=operation['source18_history_validation'],
        state=snapshot('stopped-source14-state',stateraw),checkpoint=snapshot('source14-latest',latestraw),
        selected=snapshot('source14-selected',selectedraw),segment=segment_pin,
        budget=snapshot('source14-budget',budgetraw),segment_journal=snapshot('source14-completed-journal',journalraw),
        timed=snapshot('original-four-journal',timedraw),window=state['current_window'],
        window_admission=state['current_window_admission'],trainer_process=snapshot('source14-trainer-process',processraw),
        candidate_root=str(canonical(candidate_root)),candidate_manifest=candidate_manifest)
    snapshot('public-stop-result',stopraw);snapshot('host-binding',bindingraw)
    snapshot('source14-runtime',runtimeraw);snapshot('source14-runtime-selector',selectorraw)
    snapshot('source14-bootstrap-pointer',bootstrap_pointerraw)
    if public_default_raw is not None:snapshot('public-default-backup',public_default_raw)
    if provenance is not None:
        snapshot('source14-unstarted-training',json.dumps(provenance,sort_keys=True).encode())
    release=dict(schema='source14_operational_release_v1',**{k:contract[k] for k in
        ('checkpoint','selected','budget','state','segment_journal')},owned_release=copy.deepcopy(state['owned_release']),
        original_public_stop=snapshots['public-stop-result'],boot_id=boot_id,
        unstarted_training=snapshots.get('source14-unstarted-training'))
    contract['release']=snapshot('source14-release',json.dumps(release,sort_keys=True).encode())
    transition=snapshot('source14-transition',json.dumps(contract,sort_keys=True).encode())
    helper.validate_source14_transition(transition,files=files,require_released=True,identity_active=identity_active)
    # Rebind future readers to ORIGINAL111c, not the executing next helper.
    helper.validate_history_transaction(contract['history_validation'],contract['adoption'],files=files,
                                        source14_transition=transition)
    observations=[(root/'state.json',stateraw),(root/'unattended/current.json',selectorraw),
        (root/'unattended/bootstrap-complete.json',bootstrap_pointerraw),
        (root/'training/checkpoints/latest.json',latestraw),(root/'training/continuation/selected.json',selectedraw),
        (segment_dir/'budget-used.json',budgetraw),(segment_dir/'timed-evaluation-state.json',journalraw),
        (root/'training/timed-evaluation-state.json',timedraw)]
    unchanged_observations(observations,files)
    if provenance is not None:
        require(unstarted_training_provenance(state,config,latest,unstarted['latest_pin'],runtime['attempt'],files,
            source14_transition=True)==provenance,'source14 unstarted TRAIN changed before publication')
    require(source14_public_default_bytes(public_default,configraw,files=files)==public_default_raw,
            'source14 public default changed before publication')
    require(pause_controls(root)==pauses,'source14 pause controls changed before publication')
    successor=copy.deepcopy(config)
    for key in ('operation_config','operation_admission','operational_context','initial_window','initial_window_admission'):
        successor.pop(key,None)
    successor.update(source_root=str(canonical(candidate_root)),source_files=manifest['operational_source_files'],
        supervisor={'path':str(canonical(candidate_root)/'scripts/remote_job.py'),
                    'sha256':manifest['files']['scripts/remote_job.py']},source14_transition=transition)
    candidate=snapshot('candidate-training-configuration',json.dumps(successor,sort_keys=True,indent=2).encode())
    unchanged_observations(observations,files)
    if provenance is not None:
        require(unstarted_training_provenance(state,config,latest,unstarted['latest_pin'],runtime['attempt'],files,
            source14_transition=True)==provenance,'source14 unstarted TRAIN changed after publication')
    require(source14_public_default_bytes(public_default,configraw,files=files)==public_default_raw,
            'source14 public default changed after publication')
    require(pause_controls(root)==pauses,'source14 pause controls changed after publication')
    record=dict(schema='source14_handoff_preparation_v1',transition=transition,configuration=candidate,
        previous_adoption_receipt=contract['adoption_receipt'],original_adoption=contract['adoption'],
        original_history_validation=contract['history_validation'],completed_entries=journal['completed'],
        candidate_manifest=candidate_manifest,snapshots=snapshots,producer=pin(Path(__file__).resolve()),
        optimizer_updates_performed=0,actual_training_seconds_added=0,public_configuration_changed=False,
        public_default_backup=snapshots.get('public-default-backup'),
        public_default_path=None if public_default is None else public_default['path'],
        supplementary_source14_validation_produced=False,current_gpu_admission_produced=False)
    receipt=snapshot('preparation-receipt',json.dumps(record,sort_keys=True,indent=2).encode())
    return dict(receipt=receipt,transition=transition,configuration=candidate)


def pin_bytes(path,raw):
    # Namespace of a live control for the reader seam, without opening it twice.
    return {'path':str(path),'sha256':hashlib.sha256(raw).hexdigest()}



def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('public-config','runtime','source-manifest','candidate-manifest','stop-result','host-binding'):
        p.add_argument('--'+name,type=Path,required=True);p.add_argument('--'+name+'-sha256',required=True)
    p.add_argument('--mode',choices=('source18','source14-next'),default='source18',
        help='source14-next requires --public-config to name immutable resolved4e used for STOP')
    p.add_argument('--public-default',type=Path,help='optional mutable public default to back up/check; never a history input')
    p.add_argument('--public-default-sha256')
    p.add_argument('--candidate-root',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args(argv)
    locator=lambda name:{'path':str(getattr(a,name)),'sha256':getattr(a,name+'_sha256')}
    producer=prepare_source14_handoff if a.mode=='source14-next' else prepare_handoff
    require(bool(a.public_default)==bool(a.public_default_sha256),'public default requires path/SHA together')
    require(a.mode=='source14-next' or a.public_default is None,'public-default backup applies only to source14-next')
    options={} if a.public_default is None else dict(public_default=locator('public_default'))
    result=producer(locator('public_config'),locator('runtime'),locator('source_manifest'),a.candidate_root,
        locator('candidate_manifest'),locator('stop_result'),a.output_dir,host_binding=locator('host_binding'),**options)
    print(json.dumps(result));return 0

if __name__=='__main__':raise SystemExit(main())

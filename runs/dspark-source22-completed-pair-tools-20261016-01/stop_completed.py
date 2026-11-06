"""Run-specific completed-retirement + fully-published-pending SOURCE20 ->22 stop.

Parent supplies an immutable fresh binding JSON and its SHA256. This operation
never inventories payload trees. Existing terminal deploy retains the full census.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import signal
import sys
import time

BASE = Path('/home/philip/binary-eagle-decoding')
RUN = Path('/mnt/c/Users/philip/binary-eagle-decoding-artifacts/5080-training-20261013/dspark-fresh-scale-only-12h-20261015-04/lane')
OP = RUN / 'direct-runtime/direct-dspark-30e1728b885a4caa9202ad9b783b271d'
BOOT = '808f6149-6c87-47f4-890d-3a9aebe15dca'
EXPECTED = {'controller': {'boot_id': BOOT, 'pid':9292, 'start_ticks':2782878},
            'supervisor': {'boot_id': BOOT, 'pid':9291, 'start_ticks':2782865}}
RUNTIME_SHA = '866c72fba3aa89e24f33332e20f7202e0be0001e8ef26472eb8fe89807d7fdfe'
CORE_SHA = '1e62c9610f3c02a4a1fcba44696d50fc866a08ea9ade1c8ece202d0cbb7afefb'
CHECK22_SHA = '64eac15bf39fd509c8f91b331bcb6c98f478e81bf6705ccf863d420c706bfb2b'


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def identity(pid):
    fields = (Path('/proc') / str(pid) / 'stat').read_text().rsplit(') ',1)[1].split()
    return {'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            'pid':pid, 'start_ticks':int(fields[19])}


def fingerprint(path):
    info = Path(path).lstat()
    return tuple(getattr(info, name) for name in ('st_dev','st_ino','st_mode','st_uid','st_gid',
                'st_nlink','st_size','st_mtime_ns','st_ctime_ns'))


def _same_original_seal(left, right):
    """Recovery evidence may grow; original tensors/anchors/metadata cannot change."""
    return left["identity"] == right["identity"] and (
        left.get("original_metadata") is None
        or right.get("original_metadata") is None
        or left["original_metadata"] == right["original_metadata"]
    )


def endpoint_eligibility(binding, values, identities, children, stops, now):
    """Pure run-specific metadata checks; production acquisition remains rt.metadata."""
    require(binding['schema']=='source22_completed_pair_endpoint_binding_v1', 'binding schema')
    require(0 <= now-binding['observed_unix'] <= 60 and now < binding['expires_unix']
            <= binding['observed_unix']+60, 'binding expired or future')
    require(identities==EXPECTED, 'kernel birth/boot differs')
    require(not children, 'live controller child')
    require(not stops, 'preserve STOP')
    state, latest, budget, endpoint, status, canonical, active, catalog, progress = (
        values[n] for n in ('state','latest','budget','receipt','status','canonical','active','catalog','progress'))
    require(state['status']=='restoring_shards' and not state.get('failure')
            and state.get('pending_shard_resume') is True, 'not original shard boundary')
    require(state['direct_training_runtime']=={'path':str(OP/'runtime.json'),'sha256':RUNTIME_SHA}, 'state runtime differs')
    require(state['train_receipt']==binding['files']['receipt'], 'current receipt differs')
    require(budget['active_attempt'] is None and budget['max_seconds']==43200
            and budget['source_sha256']==state['bundle_sha256'], 'active clock or budget contract')
    require(latest['committed'] is True and endpoint['committed'] is True
            and latest['cursor']==endpoint['counters']
            and budget['training_seconds']==latest['cursor']['elapsed_seconds'], 'fresh endpoint/latest/budget differs')
    require(endpoint['schema']=='nine_model_stage_receipt_v1' and endpoint['status']=='PASS'
            and endpoint['completion_reason']=='shard_required' and endpoint['artifact_kind']=='production'
            and endpoint['stage']=='dspark_a8/train'
            and endpoint['bundle_sha256']==state['bundle_sha256']
            and endpoint['config_sha256']==binding['scientific_config_sha256']
            and endpoint['hardware']['gpu_uuid']==binding['gpu_uuid']
            and endpoint['hardware']['compute_capability']==[12,0], 'original production endpoint differs')
    require(endpoint['checkpoint']=={k:latest[k] for k in ('path','sha256')}, 'checkpoint locator differs')
    require(status['status']=='shard_required' and status['cursor']==endpoint['counters']
            and status['shard_request']==endpoint['shard_request']
            and endpoint['shard_request']['reservation']['cursor']==endpoint['counters']['data_cursor'], 'original shard request differs')
    require(progress['controller_pid']==9292 and progress['bundle_sha256']==state['bundle_sha256'], 'resource progress owner differs')
    last=progress['observations'][-1]
    require(last['status']=='PASS' and last['training_budget_active'] is False
            and last['observed_unix']>=binding['minimum_release_unix']
            and last['owned_release']['owned_process_groups_absent'] is True
            and last['owned_release']['owned_cuda_pids_absent'] is True
            and last['owned_release']['other_context_pids']==[], 'original release missing')
    return {'step':latest['cursor']['step'],'training_seconds':budget['training_seconds'],
            'catalog':active['catalog']}


def completed_contract(binding, values, get):
    """Finite metadata eligibility only; original post-stop consumers still admit bytes."""
    state=values['state']; latest=values['latest']; active=values['active']; catalog=values['catalog']
    canonical=values['canonical']; cursor=latest['cursor']['data_cursor']
    locator=binding['files']['catalog']; plan=catalog['plan_sha256']
    require(locator['sha256']=='e5673234f28ff2f4046692c097e7905e30ecf6a3408aed6e8359ffeb9fc133e8',
            'run-specific old catalog SHA differs')
    require(active['catalog']==locator and active['plan_sha256']==plan
            and catalog['producer_source']=={'path':str(OP/'source/src/w1a1_eagle/sustained_block_handoff.py'),'sha256':CORE_SHA},
            'old catalog/source differs')
    old_roots={str(Path(row['path']).parent) for row in catalog['source_cache_states']}
    registrations=state.get('prepared_pair_registrations',{})
    pending={root:row for root,row in registrations.items() if root not in old_roots}
    require(bool(pending),'completed seam requires pending registrations')
    canonical_root=str(Path(binding['files']['canonical']['path']).parent)
    roots={canonical_root,*state.get('prepared_cache_roots',[])}
    require(set(pending)==roots-old_roots-{canonical_root}
            and set(pending)<=set(state.get('prepared_owned_pair_roots',[])), 'pending owned inventory differs')
    matches=[row for row in state.get('prepared_window_retirements',[])
             if row.get('original_catalog')==locator and row.get('cursor')==cursor]
    require(bool(matches),'completed retirement missing')
    retired=matches[-1]
    require(retired.get('schema')=='prepared_block_window_retirement_v1'
            and set(retired['owned_pair_roots'])==set(catalog['owned_pair_roots'])
            and {str(Path(row['path']).parent) for row in retired['owned_cache_states']}==set(catalog['owned_pair_roots']),
            'completed retirement owned inventory differs')
    archives=retired['source_state_archives']
    require([row['original'] for row in archives]==catalog['source_cache_states'],'retirement archive original inventory differs')
    directory=Path(archives[0]['snapshot']['path']).parent
    require(directory.parent==Path(locator['path']).parent and directory.name.startswith('retirement-')
            and binding['files']['retirement_receipt']['path']==str(directory/'receipt.json')
            and binding['files']['retirement_context']['path']==str(directory/'context.json')
            and binding['files']['archive_index']['path']==str(directory/'source-state-archives.json')
            and values['retirement_receipt']==retired and values['archive_index']==archives,
            'durable retirement receipt/index differs')
    context=values['retirement_context']
    require(context['schema']=='prepared_retirement_context_v1' and context['catalog']==locator
            and context['cursor']==cursor and context['reservation']==latest['cursor'].get('batch_reservation')
            and context['owned_pair_roots']==sorted(catalog['owned_pair_roots']), 'retirement cursor/reservation differs')
    snapshots={}
    for i,row in enumerate(archives):
        require(row['snapshot']['sha256']==row['original']['sha256']
                and Path(row['snapshot']['path'])==directory/('cache-state-'+str(i)+'.json'), 'archive source identity/scope differs')
        snapshots[row['original']['path']]=get(row['snapshot'])
    require(set(snapshots)=={binding['files']['canonical']['path']}|{row['path'] for row in retired['owned_cache_states']},
            'unaccounted original source role')
    # Match exact frozen9134 union order and replacement, not an approximate merge.
    expected=json.loads(json.dumps(snapshots[binding['files']['canonical']['path']]))
    def merge(cache):
        require(cache['schema']=='block_shard_cache_v1' and cache['owner'] is None and cache['plan_sha256']==plan,
                'cache owner/plan differs')
        for shard,seal in cache['seals'].items():
            old=expected['seals'].get(shard)
            require(old is None or _same_original_seal(old,seal),'immutable original seal differs')
            expected['seals'][shard]=json.loads(json.dumps(seal))
    for row in retired['owned_cache_states']:merge(get(row))
    controls=binding['pending_controls']
    require(set(binding['pending_states'])==set(pending),'pending state inventory differs')
    require(set(controls)==set(pending),'every pending pair needs native controls')
    # Refuse any partial/owned pair before inspecting another pair's publication.
    for root,registration in pending.items():
        row=binding['pending_states'][root]
        require(row['path']==str(Path(root)/'cache-state.json'),'pending state path differs')
        cache=get(row)
        require(cache.get('owner') is None and set(cache['resident'])==set(registration['requested_shards']),
                'pending pair active/partial; preserve producer')
    for root,registration in pending.items():
        pair=Path(root);window=Path(registration['window_root'])
        require(registration.get('schema')=='prepared_pair_registration_v1'
                and pair.parent==window and window.parent==OP/'prepared-data'
                and pair.name.startswith('pair-') and len(pair.name)==37
                and all(c in '0123456789abcdef' for c in pair.name[5:])
                and registration['runtime']=={'path':str(OP/'runtime.json'),'sha256':RUNTIME_SHA}
                and registration['producer_source']==catalog['producer_source'], 'pending registration source/scope differs')
        initial=get(registration['initial_state']);row=binding['pending_states'][root]
        require(row['path']==str(pair/'cache-state.json'),'pending state path differs')
        cache=get(row);requested=registration['requested_shards']
        require(initial.get('schema')=='block_shard_cache_v1' and initial['plan_sha256']==plan
                and initial['owner'] is None and initial['resident']=={}
                and isinstance(requested,list) and 0<len(requested)<=2 and len(set(requested))==len(requested)
                and set(cache)==set(initial) and cache['schema']==initial['schema'] and cache['plan_sha256']==plan
                and cache['owner'] is None and set(cache['resident'])==set(requested)
                and all(cache['seals'].get(shard)==seal for shard,seal in initial['seals'].items())
                and set(cache['seals'])-set(initial['seals'])==set(requested), 'pending pair active/partial/seal transition differs')
        require(set(controls[root])==set(requested),'pending native control shard inventory differs')
        for shard in requested:
            target=str(pair/shard);record=cache['resident'][shard];seal=cache['seals'][shard];c=controls[root][shard]
            original=get(seal['original_metadata'])
            require(original['schema']=='block_shard_original_metadata_v1' and original['plan_sha256']==plan
                    and original['shard']==shard and original['identity']==seal['identity']
                    and original['original_directory']==target,'pending original metadata differs')
            publication=get(c['publication']);request=get(c['request']);exit=get(c['exit']);process=get(c['process'])
            manifest=get(record['manifest']);admission=get(record['admission'])
            phase=Path(c['publication']['path']).parent
            require(phase.parent==RUN/'attempts'/OP.name/'shard-phases'
                    and c['publication']['path']==str(phase/'publication.json')
                    and c['request']['path']==str(phase/'request.json')
                    and c['exit']['path']==str(phase/'producer/process-exit.json')
                    and c['process']['path']==str(phase/'producer/process.json'), 'native publication role path differs')
            require(publication['schema']=='block_shard_publication_v1' and publication['status']=='PASS'
                    and publication['plan_sha256']==plan and publication['shard']==shard
                    and publication['directory']==target and publication['request']==c['request']
                    and not publication.get('recovery') and not publication.get('reconstruction_join')
                    and record=={k:publication[k] for k in ('directory','manifest','admission')}
                    and request['schema']=='block_shard_restore_request_v1' and request['plan_sha256']==plan
                    and request['directory']==target and request['shard']==shard
                    and request['original_identity'] is None and request['original_metadata'] is None
                    and process['pid']==process['pgid']==exit['pid']
                    and process['kernel_identity']['pid']==process['pid']
                    and process['argv']==[c['request']['path'] if arg=='{request}' else c['publication']['path']
                                          if arg=='{output}' else arg for arg in values['_capture']['argv']]
                    and request['plan']==values['_shard_plan']
                    and exit['exit_code']==0 and exit['cleanup_status']=='PASS'
                    and exit['owned_group_cleaned'] is True and exit['cleanup_errors']==[], 'native PASS publication/request/exit differs')
            require(Path(record['manifest']['path']).is_relative_to(pair/shard)
                    and Path(record['admission']['path']).is_relative_to(pair/shard)
                    and admission['schema']=='block_data_completed_admission_v1'
                    and admission['manifest_sha256']==record['manifest']['sha256']
                    and manifest['schema']=='native_block_train_v1', 'native manifest/admission join differs')
        merge(cache)
    require(set(binding['pending_states'])==set(pending),'pending state inventory differs')
    require(canonical==expected,'canonical differs from exact original completed union; unmerged pair refused')
    return {'completed_retirement':binding['files']['retirement_receipt'],
            'pending_pairs':len(pending),'pending_shards':sum(len(r['requested_shards']) for r in pending.values())}


def acquire(record, rt, stable):
    path=Path(record['path']);before=fingerprint(path)
    value,digest=rt.metadata(path)
    require(digest==record['sha256'] and fingerprint(path)==before,'pinned metadata changed: '+str(path))
    stable[str(path)]=before
    return value


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binding',type=Path)
    parser.add_argument('--binding-sha256')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--check',action='store_true',help='Read-only; never writes intent or signals')
    args=parser.parse_args(argv)
    if not args.check and not all((args.binding,args.binding_sha256,args.output)):
        parser.error('--binding, --binding-sha256 and --output are required to arm')
    sys.dont_write_bytecode=True
    sys.path.insert(0,str(OP/'source/src'))
    from w1a1_eagle import direct_training_runtime as rt
    runtime,cfg=rt.authenticate_runtime(OP/'runtime.json')
    require(rt.metadata(OP/'runtime.json')[1]==RUNTIME_SHA,'runtime SHA differs')
    require(hashlib.sha256((OP/'source/src/w1a1_eagle/sustained_block_handoff.py').read_bytes()).hexdigest()
            == CORE_SHA,'original seal comparator source differs')
    lane=rt.checked(cfg['lane']);rt.current_control(lane);rt.keeper_environment(cfg['keeper'])
    require(rt.metadata(BASE/'runs/dspark-direct-environment-20261016-source-22/source-check.json')[1]==CHECK22_SHA,'source22 private check differs')
    require(rt.metadata(OP/'owned-identity.json')[0]==EXPECTED,'owned identity differs')
    # Cheap actual current-process/clock refusal precedes all catalog acquisition.
    require({k:identity(v['pid']) for k,v in EXPECTED.items()}==EXPECTED,'kernel birth/boot differs')
    require(not Path('/proc/9292/task/9292/children').read_text().strip(),'live controller child')
    require(rt.metadata(RUN/'training/budget-used.json')[0]['active_attempt'] is None,'active clock')
    require(args.binding is not None and args.binding_sha256 is not None,'fresh actual binding required')
    raw=args.binding.read_bytes()
    require(hashlib.sha256(raw).hexdigest()==args.binding_sha256,'binding SHA differs')
    binding=json.loads(raw)
    if not args.check:
        require(args.output.is_absolute() and args.output.parent==BASE/'runs'
                and args.output.name.startswith('dspark-source22-completed-pair-stop-'),'run-specific output required')
    require(binding['scientific_config_sha256']==lane['config']['sha256'] and binding['gpu_uuid']==lane['gpu_uuid'],'lane binding differs')
    expected_paths={'state':RUN/'state.json','latest':RUN/'training/checkpoints/latest.json',
                    'budget':RUN/'training/budget-used.json','status':RUN/'training/status.json',
                    'active':OP/'active-window.json','progress':RUN/'resource-handoff-progress.json'}
    spec=rt.checked(lane['config']);expected_paths['canonical']=Path(spec['data']['cache_root'])/'cache-state.json'
    require(set(binding['files'])==set(expected_paths)|{'receipt','catalog','retirement_receipt','retirement_context','archive_index'},'binding roles differ')
    for name,path in expected_paths.items():require(binding['files'][name]['path']==str(path),'bound role path differs')
    receipt_path=Path(binding['files']['receipt']['path'])
    require(receipt_path.is_relative_to(RUN/'attempts'/OP.name/'training-attempts') and receipt_path.name=='train-receipt.json','receipt role path differs')
    stable={};values={}
    for name,record in binding['files'].items():values[name]=acquire(record,rt,stable)
    values['_capture']=spec['shard_lifecycle']['capture'];values['_shard_plan']=spec['shard_lifecycle']['plan']
    require(values['_capture']['producer'] in lane['source'].values(),'capture producer role differs')
    producer=values['_capture']['producer']
    require(hashlib.sha256(Path(producer['path']).read_bytes()).hexdigest()==producer['sha256'],
            'capture producer source differs')
    for field,name in (('original_provider_source','block_shard_lifecycle.py'),('original_data_source','block_data.py')):
        source=values['catalog'][field]
        require(source==lane['source']['src/w1a1_eagle/'+name]
                and hashlib.sha256(Path(source['path']).read_bytes()).hexdigest()==source['sha256'],
                'original scientific source role differs')
    require(binding['minimum_release_unix'] >= stable[str(receipt_path)][-2]/1e9,
            'release threshold predates original endpoint publication')
    catalog_path=Path(binding['files']['catalog']['path'])
    require(catalog_path.is_relative_to(OP/'prepared-data') and catalog_path.name=='catalog.json','catalog role path differs')
    metadata=binding['metadata']
    require(isinstance(metadata,list) and len(metadata)<=256,'finite metadata inventory required')
    require(len({row['path'] for row in metadata})==len(metadata),'duplicate auxiliary metadata')
    require(sum(fingerprint(row['path'])[6] for row in metadata)<=256*1024**2,'metadata inventory exceeds256MiB')
    auxiliary={row['path']:(row,acquire(row,rt,stable)) for row in metadata}
    def get(record):
        require(record['path'] in auxiliary and auxiliary[record['path']][0]==record,
                'missing or changed explicit metadata pin')
        return auxiliary[record['path']][1]
    sup=rt.live_metadata(runtime['supervisor_state'])[0]
    require(sup['status']=='running' and sup.get('received_signal') is None and sup['stop_grace_seconds']==90,'original supervisor differs')
    def check():
        rt.current_control(lane)
        supervisor=rt.live_metadata(runtime['supervisor_state'])[0]
        require(supervisor['status']=='running' and supervisor.get('received_signal') is None
                and supervisor['stop_grace_seconds']==90,'original supervisor changed')
        for path,before in stable.items():require(fingerprint(path)==before,'metadata identity changed: '+path)
        summary=endpoint_eligibility(binding,values,{k:identity(v['pid']) for k,v in EXPECTED.items()},
                           Path('/proc/9292/task/9292/children').read_text().strip(),
                           (RUN/'STOP').exists() or (RUN/'training/STOP').exists(),time.time())
        return summary
    proof=check()
    proof.update(completed_contract(binding,values,get))
    for pair in binding['pending_controls'].values():
        for controls in pair.values():
            birth=get(controls['process'])['kernel_identity']
            try:require(identity(birth['pid'])!=birth,'pending native producer still alive')
            except FileNotFoundError:pass
    if args.check:
        print(json.dumps({'status':'ELIGIBLE_METADATA_ONLY','proof':proof,'signal_sent':False}))
        return
    args.output.mkdir(mode=0o700,exist_ok=False)
    before_raw=Path(binding['files']['state']['path']).read_bytes()
    require(hashlib.sha256(before_raw).hexdigest()==binding['files']['state']['sha256'],
            'pre-stop original state SHA differs')
    with (args.output/'pre-stop-state.json').open('xb') as before:
        before.write(before_raw);before.flush();os.fsync(before.fileno())
    require(hashlib.sha256((args.output/'pre-stop-state.json').read_bytes()).hexdigest()
            ==binding['files']['state']['sha256'],'pre-stop state copy differs')
    rt.write_new(args.output/'intent.json',{'pre_stop_state':{'path':str(args.output/'pre-stop-state.json'),
                 'sha256':binding['files']['state']['sha256']},'schema':'source22_completed_pair_stop_intent_v1',
                 'binding_sha256':args.binding_sha256,'proof':proof,'owned':EXPECTED,
                 'post_terminal_original_full_inventory_and_resource_gate_required':True})
    fd=os.pidfd_open(9291)
    try:
        check()  # Last bounded metadata/kernel/STOP/control checks, after durable intent.
        signal.pidfd_send_signal(fd,signal.SIGTERM)
        rt.write_new(args.output/'signal-issued.json',{'pid':9291,'signal':'SIGTERM','via':'pidfd',
                     'original_stop_grace_seconds':90,'at_unix':time.time()})
    finally:os.close(fd)


if __name__=='__main__':main()

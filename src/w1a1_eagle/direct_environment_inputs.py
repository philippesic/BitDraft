"""Read-only original metadata selects CPU auditor inputs, never cache validity.

Every returned tuple still needs original full audit + constructor view, followed
by unchanged original PREP/reconcile/publish consumers under real released custody.
"""
import ast
import copy
import inspect
import json
from pathlib import Path
import types

CORE_SHA = "20da5b0faa1cd8e789179be71e1eff6221da7983c596cb702b7fc0b51b157476"
SOURCE22_CORE_SHA = "2a22a651a0e3366173eb4a7d77113e9a43f1239caadad501a3efc2d945c2603c"
SOURCE20_CORE_SHA = "1e62c9610f3c02a4a1fcba44696d50fc866a08ea9ade1c8ece202d0cbb7afefb"
SOURCE15_CORE_SHA = "ec85e3821248f6c9d27b23af98b4e6a6d6f0ad0301d5f738e19626ac23ee4927"
PREVIOUS_CORE_SHA = "024159411d9c1180b8074e83a036c6eaf73edbc8cf6402914ccbb6dc591782c4"
PUBLICATION_SHA = "6b4b1a26ffe19dbb1dd7a198a0a2394d108b2aa59634b9148704f677d931c11d"


def require(ok, reason):
    if not ok: raise ValueError(reason)


def _catalog_metadata(plan, catalog_locator, *, expected_producer, core, api, read):
    """Authenticate metadata roles without opening data or adopting a producer."""
    require(api.file_sha256(Path(core.__file__)) == CORE_SHA, 'current metadata selector source differs')
    catalog = read(api.checked(catalog_locator))
    require(catalog.get('schema') == core.SCHEMA and catalog.get('plan_sha256') == plan.sha256,
            'predecessor catalog logical identity differs')
    producer = catalog['producer_source']
    require(producer == expected_producer and producer.get('sha256') in
            {CORE_SHA, SOURCE22_CORE_SHA, SOURCE20_CORE_SHA, SOURCE15_CORE_SHA, PREVIOUS_CORE_SHA},
            'authenticated original predecessor producer required')
    api.checked(producer)
    require(set(catalog['publications']) <= set(plan.shards)
            and set(catalog['publications']) <= set(catalog['seals']),
            'predecessor catalog membership/seals differ')
    for field, cls in (('original_provider_source', api.ShardCache),
                       ('original_data_source', api.BlockDataset)):
        api.checked(catalog[field])
        require(Path(catalog[field]['path']).resolve() == Path(inspect.getfile(cls)).resolve(),
                'predecessor catalog original scientific role differs')
    require(0 <= catalog['payload_bytes'] <= catalog['max_window_bytes'],
            'predecessor catalog disk accounting differs')
    return catalog


def unchanged_predecessor_states(plan, catalog_locator, *, expected_producer, canonical_root, core, api, read):
    """Auditor metadata only for an unchanged authenticated predecessor.

    Caller authenticates expected_producer through prior_active_window. No new
    catalog, dataset admission, release or source pin is granted here. The actual
    post-view migrate_catalog must still revalidate every surviving publication.
    Stale original cache-state pins refuse; partial changed-source recovery is
    not inferred from this specific before-eviction handoff.
    """
    catalog = _catalog_metadata(plan, catalog_locator, expected_producer=expected_producer,
                                core=core, api=api, read=read)
    require(expected_producer.get('sha256') in {SOURCE22_CORE_SHA, SOURCE20_CORE_SHA, SOURCE15_CORE_SHA, PREVIOUS_CORE_SHA},
            'authenticated original predecessor producer required')
    source_roots = {str(Path(row['path']).parent) for row in catalog['source_cache_states']}
    require(source_roots == {str(Path(canonical_root)), *catalog['owned_pair_roots']},
            'predecessor source roles differ from primary and owned pairs')
    states, publications = {}, {}
    for locator in catalog['source_cache_states']:
        path = api.checked(locator)
        value = read(path)
        require(path.is_absolute() and path.resolve() == path and path.name == 'cache-state.json'
                and str(path) not in states and value.get('schema') == 'block_shard_cache_v1'
                and value.get('plan_sha256') == plan.sha256 and value.get('owner') is None
                and len(value['resident']) <= 2 and set(value['resident']) <= set(plan.shards)
                and set(value['resident']) <= set(value['seals']),
                'unchanged predecessor cache-state identity or owner differs')
        for shard, seal in value['seals'].items():
            require(shard in catalog['seals'] and core._same_original_seal(seal,catalog['seals'][shard]),
                    'predecessor immutable historical seal differs')
        for shard, record in value['resident'].items():
            directory = Path(record['directory'])
            require(directory.is_absolute() and directory.resolve() == directory
                    and directory.is_dir() and directory.is_relative_to(path.parent),
                    'predecessor original publication directory differs')
            for role in ('manifest','admission'):
                control = api.checked(record[role])
                require(control.resolve().is_relative_to(directory),
                        'predecessor original publication control outside directory')
                read(control)
            require(shard not in publications or publications[shard] == record,
                    'predecessor conflicting original publication')
            publications[shard] = record
        states[str(path)] = (path, value)
    require(all(publications.get(shard) == record for shard,record in catalog['publications'].items()),
            'predecessor selected original publication missing or changed')
    return states


def surviving_catalog_states(plan, cursor, catalog_locator, *, reservation, api, core, read):
    """Only original024 metadata prefix; no release/admission/adoption/eviction.

    The ONE omitted set_recovery call would validate replay admission; it remains
    mandatory in the unmodified post-view reconcile consumer. This prefix grants
    no source-cache pin, recovered catalog, release or scientific PASS.
    """
    source = Path(core.__file__)
    require(api.file_sha256(source) == CORE_SHA, "original024 selector source differs")
    original = core.reconcile_retirement
    top = compile(source.read_text(), str(source), "exec", dont_inherit=True)
    code = next(c for c in top.co_consts if isinstance(c, types.CodeType) and c.co_name == original.__name__)
    require(original.__code__ == code and original.__globals__ is core.__dict__, "original024 metadata consumer callable differs")
    tree = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == original.__name__)
    stop = next(i for i,n in enumerate(tree.body) if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == 'histories')
    body = tree.body[:stop]
    body = [n for n in body if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
            and ast.unparse(n.value.func) in {'authorization','_released'})]
    class ReadOnly(ast.NodeTransformer):
        checks = 0
        def visit_Expr(self, node):
            if isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == 'api.ShardCache.set_recovery':
                self.checks += 1
                return ast.copy_location(ast.Expr(ast.Call(ast.Name('_pending_replay_gate', ast.Load()), node.value.args[1:], [])), node)
            return self.generic_visit(node)
        def visit_FunctionDef(self, node):
            return node # Preserve nested eligible() boolean semantics verbatim.
        def visit_Return(self, node):
            return ast.copy_location(ast.Return(ast.Tuple([ast.Name('current',ast.Load()), ast.List([],ast.Load())],ast.Load())),node)
        def visit_Call(self,node):
            if (ast.unparse(node.func)=='json.loads' and len(node.args)==1
                    and isinstance(node.args[0],ast.Call) and isinstance(node.args[0].func,ast.Attribute)
                    and node.args[0].func.attr=='read_text'):
                return ast.copy_location(ast.Call(ast.Name('_metadata_json',ast.Load()),[node.args[0].func.value],[]),node)
            return self.generic_visit(node)
    rewrite = ReadOnly(); tree.body = [rewrite.visit(n) for n in body]
    require(rewrite.checks == 1, "one original deferred replay-admission site required")
    tree.body.append(ast.Return(ast.Tuple([ast.Name('current',ast.Load()),ast.Name('interrupted',ast.Load())],ast.Load())))
    pending = []
    def pending_gate(shard, locator):
        # Pin metadata only. Original set_recovery must run after installing view.
        read(api.checked(locator)); pending.append((shard,locator))
    scope = dict(core.__dict__, _metadata_json=read, _pending_replay_gate=pending_gate)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[tree],type_ignores=[])),str(source),'exec'),scope)
    states, deleting = scope[original.__name__](plan,cursor,catalog_locator,
        reservation=reservation,release=None,authorization=None,merge_seals=None,api=api)
    return states, set(tuple(row) for row in deleting), pending


def registered_inputs(plan, registrations, *, run_dir, config, source, runner,
                      validate_registration, api, read, publication, deferred_intents=None):
    require(api.file_sha256(Path(publication.__file__)) == PUBLICATION_SHA,
            "original publication input reader source differs")
    intents=unpublished_preservation_inputs(plan,registrations,run_dir=run_dir,config=config,
        source=source,runner=runner,validate_registration=validate_registration,api=api,publication=publication)
    omitted={row['original_directory'] for row in intents}
    if deferred_intents is not None:deferred_intents.extend(intents)
    run = Path(run_dir).resolve()
    rows = []
    for root, registration in registrations.items():
        pair = Path(root); window = Path(registration['window_root'])
        require(pair.is_absolute() and pair.resolve()==pair and pair.parent==window
                and window.parent.name=='prepared-data' and window.parent.parent.parent==run/'direct-runtime'
                and pair.name.startswith('pair-') and len(pair.name)==37
                and all(c in '0123456789abcdef' for c in pair.name[5:]), 'registered auditor root differs')
        validate_registration(registration)
        initial = read(api.checked(registration['initial_state']))
        state = read(pair/'cache-state.json'); requested=registration['requested_shards']
        require(initial.get('schema')=='block_shard_cache_v1' and initial.get('plan_sha256')==plan.sha256
                and initial.get('owner') is None and initial.get('resident')=={}
                and 0<len(requested)<=2 and len(set(requested))==len(requested) and set(requested)<=set(plan.shards)
                and set(state)==set(initial) and state['schema']==initial['schema']
                and state['plan_sha256']==plan.sha256
                and all(state['seals'].get(k)==v for k,v in initial['seals'].items())
                and set(state['seals'])-set(initial['seals'])==set(state['resident'])
                and set(state['resident'])<=set(requested), 'registered auditor seal/plan transition differs')
        owner=state['owner']
        require(owner is None or (owner.get('phase')=='restore' and owner.get('owner','').startswith('controller-')
                and len(owner['owner'])==43 and all(c in '0123456789abcdef' for c in owner['owner'][11:])),
                'registered auditor owner differs')
        runtime=read(api.checked(registration['runtime'])); operation=Path(runtime['operation'])
        require(window.parent==operation/'prepared-data' and operation.resolve().parent==run/'direct-runtime',
                'registered runtime operation differs')
        for shard in requested:
            target=pair/shard
            require(target.resolve().parent==pair, 'registered target root differs')
            record=state['resident'].get(shard)
            if record is not None and state['seals'][shard].get('original_metadata'):
                metadata=read(api.checked(state['seals'][shard]['original_metadata']))
                require(record['directory']==str(target) and metadata.get('plan_sha256')==plan.sha256
                        and metadata.get('shard')==shard and metadata.get('identity')==state['seals'][shard]['identity']
                        and metadata.get('original_directory')==str(target), 'original registered metadata differs')
            else:
                completion=publication._completed_capture(plan,pair,shard,target,operation,state,
                    config,runner,source,run,api)
                if completion is None:
                    require((not target.exists() and not target.is_symlink()) or str(target) in omitted,
                            'unfinished target lacks exact original preservation proof')
                    continue
                receipt=read(api.checked(completion['publication']))
                record={key:receipt[key] for key in ('manifest','admission','directory')}
            require(Path(record['manifest']['path']).resolve().is_relative_to(target)
                    and Path(record['admission']['path']).resolve().is_relative_to(target),
                    'auditor controls outside original publication')
            read(api.checked(record['manifest']));read(api.checked(record['admission']))
            rows.append(dict(origin='registered:'+shard,manifest=record['manifest'],admission=record['admission']))
    return rows


def completed_retirement_states(plan, cursor, catalog_locator, state, *, canonical_root,
                                 reservation, registrations, core, api, read, expected_producer=None):
    """Existing completed-retirement branch: exact immutable receipt + seal union.

    Returns unadopted metadata solely for auditor input selection. Actual original
    publish/recover/build_catalog consumers still perform all data admission.
    """
    if expected_producer is None:
        # Same-source compatibility only. Cross-version selection must bind the
        # producer through the authenticated prior runtime supplied by caller.
        expected_producer = read(api.checked(catalog_locator))['producer_source']
        require(expected_producer.get('sha256') == CORE_SHA,
                'cross-version completed retirement requires authenticated predecessor')
    catalog = _catalog_metadata(plan, catalog_locator, expected_producer=expected_producer,
                                core=core, api=api, read=read)
    old_roots={str(Path(row['path']).parent) for row in catalog['source_cache_states']}
    primary=Path(canonical_root)/'cache-state.json'
    roots={str(canonical_root),*state.get('prepared_cache_roots',[])}
    require(set(registrations)==roots-old_roots-{str(canonical_root)}
            and set(registrations)<=set(state.get('prepared_owned_pair_roots',[])),
            'unfinished new PREP roots lack exact registered ownership')
    matches=[row for row in state.get('prepared_window_retirements',[])
             if row.get('original_catalog')==catalog_locator and row.get('cursor')==cursor]
    require(bool(matches),'new registered PREP lacks completed original retirement')
    retired=matches[-1]
    require(retired.get('schema')=='prepared_block_window_retirement_v1'
            and set(retired['owned_pair_roots'])==set(catalog['owned_pair_roots'])
            and {str(Path(row['path']).parent) for row in retired['owned_cache_states']}==set(catalog['owned_pair_roots']),
            'completed retirement owned inventory differs')
    archives=retired['source_state_archives']
    require([row['original'] for row in archives]==catalog['source_cache_states'],
            'completed retirement source inventory differs')
    directory=Path(archives[0]['snapshot']['path']).parent
    require(directory.parent==Path(catalog_locator['path']).parent and directory.name.startswith('retirement-')
            and read(directory/'receipt.json')==retired,'completed retirement immutable receipt differs')
    context=read(directory/'context.json')
    require(context['catalog']==catalog_locator and context['cursor']==cursor and context['reservation']==reservation,
            'completed retirement exact cursor/reservation differs')
    snapshots={}
    for row in archives:
        require(row['snapshot']['sha256']==row['original']['sha256']
                and Path(row['snapshot']['path']).parent==directory,'retirement source archive identity/scope differs')
        snapshots[row['original']['path']]=read(api.checked(row['snapshot']))
    require(str(primary) in snapshots,'original canonical retirement archive missing')
    expected=copy.deepcopy(snapshots[str(primary)])
    states={}
    for locator in retired['owned_cache_states']:
        value=read(api.checked(locator));path=Path(locator['path'])
        require(path.name=='cache-state.json' and value.get('owner') is None
                and value.get('plan_sha256')==plan.sha256,'completed original owned state differs')
        states[str(path)]=(path,value)
    for name,(path,value) in states.items():
        for shard,seal in value['seals'].items():
            old=expected['seals'].get(shard)
            require(old is None or core._same_original_seal(old,seal),'retired/new immutable canonical seal differs')
            expected['seals'][shard]=copy.deepcopy(seal)
    baseline_seals=set(expected['seals'])
    for root in registrations:
        path=Path(root)/'cache-state.json';value=read(path)
        states[str(path)]=(path,value)
        for shard,seal in value['seals'].items():
            old=expected['seals'].get(shard)
            require(old is None or core._same_original_seal(old,seal),'retired/new immutable canonical seal differs')
            expected['seals'][shard]=copy.deepcopy(seal)
    current=read(primary)
    # Original restore publishes each shard before the pair-level canonical
    # merge. A stop during its partner can leave new registered seals pending.
    # Select auditor inputs only: retain every retired baseline seal, refuse
    # unknown/changed canonical seals, and never synthesize the missing merge.
    # Released original recover_registered_preparation still fully admits the
    # publications and merges them before runtime's strict complete-union check.
    require({key:value for key,value in current.items() if key!='seals'}
            =={key:value for key,value in expected.items() if key!='seals'}
            and baseline_seals<=set(current['seals'])<=set(expected['seals'])
            and all(seal==expected['seals'][shard] for shard,seal in current['seals'].items()),
            'canonical history differs from original retirement and registered publications')
    states[str(primary)]=(primary,current)
    # Every old source role must be accounted for, not silently read from a new pin.
    require(set(snapshots)=={str(primary)}|{row['path'] for row in retired['owned_cache_states']},
            'unaccounted original source role in completed retirement')
    return states


def unpublished_preservation_inputs(plan, registrations, *, run_dir, config, source,
                                    runner, validate_registration, api, publication):
    """Original05c metadata prefix only; no custody, release or mutation authority."""
    import os
    import stat
    import hashlib
    original=publication.preserve_unpublished_registered_targets
    source_path=_preservation_source(publication,api)
    tree=next(n for n in ast.parse(source_path.read_text()).body
              if isinstance(n,ast.FunctionDef) and n.name==original.__name__)
    stop=next(i for i,n in enumerate(tree.body) if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=='completed')
    prefix=tree.body[:stop];removed=[]
    def custody(node):
        if isinstance(node,ast.Expr) and isinstance(node.value,ast.Call):
            name=ast.unparse(node.value.func)
            if name in {'authorization','original._released'}:removed.append(name);return True
        return False
    tree.body=[n for n in prefix if not custody(n)]
    require(removed==['authorization','original._released','authorization','original._released'],
            'exact original read-only custody boundary differs')
    for node in ast.walk(ast.Module(body=tree.body,type_ignores=[])):
        if isinstance(node,ast.Call):
            require(not (isinstance(node.func,ast.Attribute) and node.func.attr in
                {'rename','unlink','mkdir','publish','evict','acquire','release','atomic_json'})
                and ast.unparse(node.func)!='_finish_publication','mutation in preservation input prefix')
    tree.body.append(ast.Return(ast.Tuple([ast.Name('actions',ast.Load()),ast.Name('observations',ast.Load())],ast.Load())))
    reads={};reader=publication._read
    def observe(path, **limits):
        value,pin,identity=reader(path, **limits);reads[pin['path']]={'locator':pin,'identity':list(identity),
            'max_bytes':limits.get('max_bytes',8 * 1024**2)}
        return value,pin,identity
    scope=dict(publication.__dict__,_read=observe)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[tree],type_ignores=[])),str(source_path),'exec'),scope)
    actions,observations=scope[original.__name__](plan,registrations,run_dir=run_dir,shard_config=config,
        runner=runner,source=source,release=None,authorization=None,validate_registration=validate_registration,api=api)
    for pin,identity in observations:
        reads.setdefault(pin['path'],{'locator':pin,'identity':list(identity),'max_bytes':8 * 1024**2})
    intents=[dict(original_directory=str(target),directory_identity=list(fingerprint),
                  operation=str(operation),registration=registration,producer=producer,
                  metadata_observations=list(reads.values()),
                  authority='auditor-input-omission-only; original released preservation still required')
             for target,fingerprint,operation,registration,producer in actions]
    recheck_preservation_inputs(intents,publication=publication,api=api)
    return intents


def recheck_preservation_inputs(intents, *, publication, api):
    _preservation_source(publication,api)
    for intent in intents:
        require(list(publication._identity(Path(intent['original_directory']).lstat()))==intent['directory_identity'],
                'deferred preservation target changed')
        for expected in intent['metadata_observations']:
            _,locator,identity=publication._read(Path(expected['locator']['path']), max_bytes=expected.get('max_bytes',8 * 1024**2))
            require(locator==expected['locator'] and list(identity)==expected['identity'],
                    'deferred preservation metadata changed')


def _preservation_source(publication,api):
    import os
    import stat
    import hashlib
    source_path=Path(publication.__file__)
    require(api.file_sha256(source_path)==PUBLICATION_SHA,'original preservation source differs')
    compiled=compile(source_path.read_text(),str(source_path),'exec',dont_inherit=True)
    codes={c.co_name:c for c in compiled.co_consts if isinstance(c,types.CodeType)}
    for name in ('preserve_unpublished_registered_targets','_read','_identity','_require','_completed_capture'):
        function=getattr(publication,name)
        require(isinstance(function,types.FunctionType) and function.__code__==codes[name]
                and function.__globals__ is publication.__dict__,'original preservation callable/alias differs: '+name)
    for name,value in {'Path':Path,'os':os,'stat':stat,'json':json,'hashlib':hashlib}.items():
        require(publication.__dict__[name] is value,'original preservation global differs: '+name)
    return source_path

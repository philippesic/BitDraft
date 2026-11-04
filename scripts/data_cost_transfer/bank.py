"""Frozen public TRAIN bank; reuse audited singleton exclusion and native producer."""
from pathlib import Path
import argparse, copy, difflib, inspect, json, sys
P=Path('/home/philip/binary-eagle-decoding')
sys.path.insert(0,str(P/'scripts/w1a8_generalization'))
import modern_natural_confirmation as natural
import modern_expand_capture as capture

SCHEMA='native_public_train_bank_chunk_v1'
def pin(p): return capture.lifecycle.pin(Path(p))
def read(p): return json.loads(Path(p).read_text())
def require(ok,msg): capture.require(ok,msg)
def write(p,v): capture.lifecycle.atomic(Path(p),v)

def select(scope_path):
    scope=read(scope_path); audit=natural.Metadata()
    evidence=natural.exclusion_evidence(scope,audit)
    consumed=audit.read(scope['latest_consumed']); marker=audit.read(scope['latest_marker'])
    require(consumed['schema']=='modern_natural_confirmation_inputs_v1' and len(consumed['groups'])==48,'complete latest consumed48 required')
    require(marker['evaluated'] and marker['input_manifest']==scope['latest_consumed'],'latest consumed marker differs')
    for row in consumed['groups']:
        evidence['hashes'].update(row['identities'].values())
        if 'original_prompt_id' in row: evidence['forbidden_ids'].add(row['original_prompt_id'])
        if 'original_group' in row: evidence['groups'].add(row['original_group'])
        evidence['topics'].update(row.get('original_topics',[]))
    # Reuse complete audited component construction; request all eligible rows,
    # then apply a declared input-only geometry restriction before deterministic selection.
    source=inspect.getsource(natural.select_natural)
    require(source.count('pool[:count]')==1,'selector source changed')
    derived=source.replace('pool[:count]','pool')
    namespace=dict(vars(natural));exec(compile(derived,'<all-eligible-train-components>','exec'),namespace)
    candidates,selection=namespace['select_natural'](scope,pin(scope_path),audit,evidence)
    pools={d:[r for r in candidates if r['domain']==d and len(natural.user_text(r['messages']).encode())<=280] for d in natural.DOMAIN_COUNTS}
    count=min(64,min(len(p) for p in pools.values())//8*8)
    require(count>=8,'not enough eligible short inputs')
    rows=[]
    for d,pool in pools.items():
        for i,row in enumerate(pool[:count]):
            row=copy.deepcopy(row);row.update(id=f'train-bank-v1-{d}-{i:03d}',role='fit',phase='training',historical_source_split='TRAIN',prompt_id=row['original_prompt_id'])
            row['source_ancestry']['source_split']='TRAIN'
            row['message_identity_sha256']=capture.identity(row['messages'])
            row['normalized_message_identity_sha256']=capture.identity(row['messages'],True)
            rows.append(row)
    require(len({r['group_id'] for r in rows})==len(rows),'duplicate group')
    return rows,dict(selection=selection,eligible_short_counts={d:len(p) for d,p in pools.items()},per_domain=count,metadata_pins=list(audit.pins.values()),partition_indices=len(evidence['partitions']),latest_consumed_excluded=48,all_historical_train_excluded=512),source,derived

def prepare(scope,out):
    require(not out.exists(),'bank exists');rows,evidence,source,derived=select(scope)
    out.mkdir(parents=True)
    (out/'selector.patch').write_text(''.join(difflib.unified_diff(source.splitlines(True),derived.splitlines(True))))
    refs=evidence['metadata_pins']+[pin(scope),pin(__file__),pin(natural.__file__),pin(capture.__file__),pin(natural.fresh.__file__)]
    rendering=read(capture.ROSTER)['rendering'];refs.append(pin(capture.ROSTER))
    chunks=[]
    for n in range(evidence['per_domain']//8):
        root=out/f'chunk{n:02d}';root.mkdir()
        selected=[copy.deepcopy(r) for d in natural.DOMAIN_COUNTS for r in [x for x in rows if x['domain']==d][n*8:(n+1)*8]]
        for r in selected:
            path=root/(r['id']+'.txt');path.write_text(capture.render(r['messages'],rendering));r.update(rendered_text_path=str(path.resolve()),rendered_prompt_sha256=pin(path)['sha256'])
        roster=dict(schema=SCHEMA,status='FROZEN_INPUTS_NO_LABELS',rows=selected,source_pins=refs,rendering=rendering,protocol=capture.protocol(),parent_sha256=capture.PARENT_SHA256,target_sha256=capture.TARGET_SHA256,selection=dict(scope=pin(scope),chunk=n,bank_role='TRAIN',outcome_selection=False,geometry='UTF8 user bytes<=280; native prompt+80<=384; no replacement'))
        validate(roster);write(root/'roster.json',roster);chunks.append(pin(root/'roster.json'))
    write(out/'bank.json',dict(schema='native_public_train_bank_v1',status='FROZEN_TRAIN_INPUTS',groups=len(rows),chunks=chunks,evidence=evidence,scope=pin(scope),selector_patch=pin(out/'selector.patch'),permanently_excluded_from_future_evaluation=True))
    print(json.dumps(dict(groups=len(rows),short_counts=evidence['eligible_short_counts'],chunks=len(chunks))))

original_validate=capture.validate_roster

def validate(roster):
    require(roster['schema']==SCHEMA,'bank schema differs')
    require(all(r['phase']=='training' and r['id'].startswith('train-bank-v1-') for r in roster['rows']),'not TRAIN bank')
    compat=copy.deepcopy(roster);compat['schema']=capture.EXPANSION_SCHEMA
    for i,r in enumerate(compat['rows']):r['id']=f"fit-expand-v1-{r['domain']}-{i:02d}"
    original_validate(compat)

def collect(roster,out):
    value=read(roster);validate(value)
    # The full deterministic selector is repeated before any labels are created.
    rows,_,_,_=select(Path(value['selection']['scope']['path']))
    expected={r['id']:r for r in rows}
    for row in value['rows']:
        require({k:v for k,v in row.items() if k not in ('rendered_text_path','rendered_prompt_sha256')}==expected[row['id']],'frozen bank selection differs')
    capture.validate_roster=validate
    capture.select_rows=lambda *args:([{k:v for k,v in r.items() if k not in ('rendered_text_path','rendered_prompt_sha256')} for r in value['rows']],value['source_pins'])
    args=argparse.Namespace(allow_cuda=True,output=out,roster=roster,build=capture.TRACE_BUILD,planned_bytes=capture.PLANNED_BYTES,future_state_bytes=capture.FUTURE_STATE_BYTES,current_roster=capture.ROSTER,split=capture.SPLIT,bindings=capture.BINDINGS,fresh=capture.FRESH)
    capture.collect(args)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','collect']);p.add_argument('--scope',type=Path);p.add_argument('--roster',type=Path);p.add_argument('--out',type=Path,required=True);p.add_argument('--allow-cuda',action='store_true');a=p.parse_args()
    if a.action=='prepare':prepare(a.scope,a.out)
    else:
        require(a.allow_cuda,'explicit CUDA required');collect(a.roster,a.out)

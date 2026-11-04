"""Audit captured chunks, capture missing native embeddings, admit bounded TRAIN shards."""
from pathlib import Path
import argparse, json, os, signal, subprocess, sys, time
P=Path('/home/philip/binary-eagle-decoding')
sys.path[:0]=[str(P/'scripts/contender_pipeline'),str(P/'scripts/w1a8_generalization')]
from runner import pin,read,atomic,require
import modern_actual_data as actual
import native_log_text
native_log_text.install()
import modern_expand_capture as capture
import modern_actual_union as union
from sharded_rows import pack,ShardRows
from snapshot_audit import snapshot_audit


def prepare(cap,out,old_embeddings):
    require(not out.exists(),'new preparation required');out.mkdir(parents=True)
    summary=read(cap/'summary.json');freeze=read(cap/'freeze.json');roster=read(freeze['roster']['path'])
    require(pin(freeze['roster']['path'])==freeze['roster'],'roster changed')
    require(len(summary['groups'])==24 and {g['id'] for g in summary['groups']}=={g['id'] for g in roster['rows']},'complete chunk required')
    releases=[cap/'target-only/release.json']
    for g in summary['groups']:
        require(g['status']=='ACTUAL_PARENT_TRACE_AND_PROTECTION_AUDIT_PASS' and g['role']=='fit','group failed')
        root=cap/g['id'];ref=read(root/'target-reference.json')
        for k in ('request','response'):require(pin(ref[k]['path'])==ref[k],'reference changed')
        require(read(root/'request.json')==read(ref['request']['path']),'request differs')
        require(capture.response_signature(read(root/'response.json'))==capture.response_signature(read(ref['response']['path'])),'parity differs')
        releases.append(root/'release.json')
    for path in releases:
        r=read(path);require(r['owned_process_reaped'] and r['owned_process_group_absent'],'unreleased group')
    begin=time.monotonic();pre=actual.prepare(cap,out/'preliminary',embedding_results=old_embeddings)
    missing=pre['missing_embedding_ids'];embeddings=list(old_embeddings)
    if missing:
        require(len(missing)<=3000,'embedding chunk bound exceeded')
        root=out/'embedding';root.mkdir();tokens=root/'tokens.ids';tokens.write_text('\n'.join(map(str,missing))+'\n')
        build=read(P/'runs/w1a8-modern-capture-build-20261103-09/build.json')
        for ref in [build['source'],build['builder'],build['executable'],build['cuda_overlay']]+build['runtime_libraries']+[freeze['target']]:require(pin(ref['path'])==ref,'native embedding dependency changed')
        clean={k:v for k,v in os.environ.items() if not k.startswith(('GGML_','DSPARK_','DFLASH_','EAGLE_','W1AX_','W1A8_','LLAMA_','LD_'))}
        env={**clean,**build['environment'],'CUDA_VISIBLE_DEVICES':'0','CUDA_CACHE_PATH':str(capture.CACHE)}
        argv=[build['executable']['path'],'--model',freeze['target']['path'],'--tokens',str(tokens),'--out',str(root/'capture'),*build['required_backend_argv'],'--taps','0','--logit-positions',str(len(missing)-1),'--prefill-chunk','512','--decode-chunk','1','--threads','4','--output-cap','1','--max-output-mib','32']
        atomic(root/'freeze.json',dict(source=pin(__file__),tokens=pin(tokens),target=freeze['target'],build=pin(P/'runs/w1a8-modern-capture-build-20261103-09/build.json'),argv=argv,scope='Native target layer0 token embeddings only'))
        with (root/'stdout.log').open('x') as log:subprocess.run(argv,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
        m=read(root/'capture/capture.json');require(m['feature_shape']==[len(missing),2560] and m['taps']==[0] and m['input_tokens']==missing,'embedding geometry differs')
        atomic(root/'result.json',dict(status='NATIVE_EMBEDDING_ROWS_CAPTURED',capture=pin(root/'capture/capture.json'),layer0=pin(root/'capture/layer-0.f32'),unique_ids=len(missing)))
        embeddings.append(root)
    manifest=actual.prepare(cap,out/'final',embedding_results=embeddings);require(not manifest['missing_embedding_ids'],'missing native embeddings')
    require(all(g['role']=='fit' for g in manifest['groups']),'heldout data present')
    # Member audit retains all provenance/row/finite checks without constructing
    # an oversized multi-member union or changing the existing union limit.
    audit=snapshot_audit(union.Audit)()
    union._validate_member(manifest,audit)
    specs=manifest['raw_files']+manifest['embedding_files']
    shards=pack(specs,out/'shards')
    rows=ShardRows(out/'shards/manifest.json');data=actual.ActualDataset(out/'final/manifest.json');data._rows=rows
    # Every window checks ancestry, shape, finite rows and exact row hashes.
    for w in data.windows:data.materialize(w,'cpu')
    for token in manifest['required_embedding_ids']:data.embedding_lookup(token,'cpu')
    atomic(out/'admission.json',dict(status='TRAIN_CHUNK_NATIVE_MATERIALIZATION_PASS',manifest=pin(out/'final/manifest.json'),shards=pin(out/'shards/manifest.json'),capture=pin(cap/'summary.json'),source=pin(__file__),groups=24,windows=len(data.windows),embeddings=len(manifest['required_embedding_ids']),new_embeddings=len(missing),released_groups=len(releases),wall_seconds=time.monotonic()-begin,shard_report=rows.report(),scope='Original fixed target/initializer native captures; all chunk windows verified; no throughput claim'))
    print(json.dumps(dict(status='TRAIN_CHUNK_NATIVE_MATERIALIZATION_PASS',windows=len(data.windows),new_embeddings=len(missing))),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--capture',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--embedding-result',type=Path,action='append',default=[]);p.add_argument('--allow-cuda',action='store_true');a=p.parse_args();require(a.allow_cuda,'explicit GPU required');prepare(a.capture,a.out,a.embedding_result)

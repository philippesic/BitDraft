"""Diagnostic complete host-span panel, kept separate from uninstrumented speed tests."""
from pathlib import Path
import argparse,copy,json,signal,sys,time
P=Path('/home/philip/binary-eagle-decoding')
sys.path[:0]=[str(P/'scripts/design_readiness'),str(P/'scripts/w1a8_generalization'),str(P/'scripts/contender_pipeline')]
from native_panel import Panel
from runner import atomic,pin,read,require

class Profile(Panel):
    def cell(self,rep,arm):
        original=copy.deepcopy(arm)
        trace=self.output/f"rep-{rep:02d}-{arm['id']}"/'cost-trace.jsonl'
        arm['runtime']['environment'].update(LD_PRELOAD=str(self.library),DECODE_COST_TRACE=str(trace))
        arm['runtime']['libraries'].append(pin(self.library))
        try:super().cell(rep,arm)
        finally:arm.clear();arm.update(original)
        spans=[json.loads(line) for line in trace.read_text().splitlines()]
        require(spans and all(s['end_ns']>=s['start_ns'] and s['depth']>=0 for s in spans),'invalid span trace')
        summaries=[]
        for row in [r for r in self.records if r['arm']==arm['id'] and r['repetition']==rep]:
            bounds=read(Path(row['response']['path']).with_suffix('.bounds.json'))
            selected=[s for s in spans if bounds['start_ns']<=s['start_ns'] and s['end_ns']<=bounds['end_ns']]
            roots=sorted([s for s in selected if s['depth']==0],key=lambda s:s['start_ns'])
            require(roots and all(a['end_ns']<=b['start_ns'] for a,b in zip(roots,roots[1:])),'overlapping/missing root spans')
            begins=[s['start_ns'] for s in roots if s['name']=='generation_begin']
            if arm['kind']!='target':
                require(len(begins)==1,'exactly one generation begin expected');boundary=begins[0]
                require(sum(s['name']=='draft_catchup' for s in roots)>0,'catchup hook absent')
            else:
                syncs=[s for s in roots if s['name']=='target_sync'];require(syncs,'target sync hook absent');boundary=syncs[0]['end_ns']
            costs={'prefill':{},'generation':{}}
            for s in roots:
                phase='prefill' if s['start_ns']<boundary else 'generation'
                costs[phase][s['name']]=costs[phase].get(s['name'],0)+(s['end_ns']-s['start_ns'])/1e6
            total=sum(sum(v.values()) for v in costs.values());http_ms=row['full_http_wall_seconds']*1000
            require(total<=http_ms+1,'spans exceed response wall')
            nested={}
            for s in selected:
                if s['depth']>0:
                    key=s['phase']+'/'+s['name'];nested[key]=nested.get(key,0)+(s['end_ns']-s['start_ns'])/1e6
            summaries.append(dict(id=row['id'],warmup=row['warmup'],arm=row['arm'],repetition=rep,root_ms=costs,nested_ms_not_additive=nested,root_spans=len(roots),http_ms=http_ms,unattributed_http_ms=http_ms-total,verification_rounds=row['verification_rounds'],draft_accepted=row['draft_accepted'],draft_proposed=row['draft_proposed'],completion_tokens=row['completion_tokens']))
        atomic(trace.with_suffix('.summary.json'),dict(trace=pin(trace),rows=summaries))
    def completion(self,port,item,cell,index,warmup,arm,rep):
        begin=time.monotonic_ns();row=super().completion(port,item,cell,index,warmup,arm,rep);end=time.monotonic_ns()
        atomic(Path(row['response']['path']).with_suffix('.bounds.json'),dict(start_ns=begin,end_ns=end))
        return row

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--models',type=Path,required=True);p.add_argument('--inputs',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--library',type=Path,required=True);p.add_argument('--repetitions',type=int,default=5);p.add_argument('--allow-cuda',action='store_true');a=p.parse_args();require(a.allow_cuda,'explicit GPU required')
 models=read(a.models);models['arms']=[x for x in models['arms'] if x['id'] in ('target_only','dflash_Q4KM','w1a8_initial','paid5760')];require(len(models['arms'])==4,'four fixed controls required')
 model_path=a.out.with_suffix('.models.json');atomic(model_path,models)
 panel=Profile(argparse.Namespace(models=model_path,inputs=a.inputs,output=a.out,repetitions=5,allow_cuda=True));panel.library=a.library.resolve();panel.records=[];panel.stopping=False
 signal.signal(signal.SIGTERM,panel.request_stop);signal.signal(signal.SIGINT,panel.request_stop)
 panel.authenticate()
 atomic(a.out/'diagnostic-freeze.json',dict(source=pin(__file__),library=pin(a.library),original_models=pin(a.models),executed_repetitions=a.repetitions,framework_admission_repetitions=5,method='Additional host-only LD_PRELOAD spans, no forced synchronization. Root costs disjoint; nested costs overlap. Instrumented timings are diagnostic only.'))
 for rep in range(a.repetitions):
    for arm in (panel.arms if rep%2==0 else list(reversed(panel.arms))):panel.cell(rep,arm)
 atomic(a.out/'result.json',dict(status='DIAGNOSTIC_NATIVE_COST_PANEL_PASS',records=panel.records,repetitions=a.repetitions,timing_claim='instrumented diagnostic only; requires paired uninstrumented overhead and counter comparison'))

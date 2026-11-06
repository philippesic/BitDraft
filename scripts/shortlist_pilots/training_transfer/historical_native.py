"""All retained checkpoints on one frozen historical-exposure state matrix."""
from pathlib import Path
import sys,json,subprocess,os,time
ROOT=Path('/home/philip/binary-eagle-decoding');RUN=ROOT/'runs/training-transfer-diagnosis-20261027';OLD=ROOT/'runs/overnight-quality-20261027'
sys.path.insert(0,str(ROOT/'scripts'))
from shortlist_pilots.training_transfer.metrics import score
from shortlist_pilots.objectives.runtime import pin

def main():
 import numpy as np
 out=RUN/'historical-native';out.mkdir(exist_ok=False);windows=json.loads((RUN/'historical-matrix.json').read_text())['windows'];anchors=json.loads((RUN/'historical-anchors-qualified.json').read_text());parent=json.loads((OLD/'train-control/start.json').read_text())['spec']['parent'];eal=json.loads((OLD/'train-control/finish.json').read_text());q4=json.loads((ROOT/'runs/w1a8-algorithms-20261020/openitems-markov-benchmark06-root/bindings-development.json').read_text())['models']['q4_k_m'];helper=json.loads((RUN/'native/ready.json').read_text())['binary'];assert pin(helper['path'])==helper;results={};started=time.monotonic()
 tasks=[(name,q['model']) for name,q in anchors.items() if 'model' in q]+[('parent3504',parent['model']),('EAL2160',eal['model']),('q4_k_m',q4)]
 for arm,model in tasks:
  assert pin(model['path'])['sha256']==model['sha256'];dest=out/arm;dest.mkdir();rows=[]
  for begin in range(0,len(windows),12):
   cases=[{**w,'case_index':i} for i,w in enumerate(windows[begin:begin+12])];plan=dest/f'plan-{begin:03d}.json';plan.write_text(json.dumps({'schema':'group_transfer12_v1','cases':cases}));cell=dest/f'cell-{begin:03d}'
   with (dest/f'cell-{begin:03d}.log').open('wb') as f:subprocess.run([helper['path'],model['path'],str(plan),str(cell),'on'],env=os.environ.copy(),stdout=f,stderr=subprocess.STDOUT,check=True,timeout=240)
   rawrows=[json.loads(l) for l in (cell/'rows.jsonl').read_text().splitlines()]
   for i,(w,n) in enumerate(zip(windows[begin:begin+12],rawrows)):
    greedy=np.fromfile(n['raw_path'],dtype='<f4').reshape(7,151936);tfpath=cell/f'teacher-{i}.f32';tf=np.fromfile(tfpath,dtype='<f4').reshape(7,151936);gs=score(greedy,w['labels']);ts=score(tf,w['labels']);common=min(gs['prefix']+1,5);assert gs['predictions'][:common]==ts['predictions'][:common]
    rows.append({'window_index':begin+i,'group_id':w['group_id'],'domain':w['domain'],'role':w['role'],'panel':w['panel'],'checkpoint_seen':w['checkpoint_group_exposure'].get(arm),'greedy':gs,'teacher_forced':ts,'greedy_D7':score(greedy,w['labels'],7),'teacher_forced_D7':score(tf,w['labels'],7),'native':n,'greedy_raw':pin(n['raw_path']),'teacher_raw':pin(tfpath),'same_predecessor_head_max_abs':float(np.abs(greedy[:common]-tf[:common]).max())})
   (out/'progress.json').write_text(json.dumps({'arm':arm,'windows':len(rows),'wall_seconds':time.monotonic()-started}))
  summary={p:{role:{'windows':len(rr:=[r for r in rows if r['panel']==p and r['role']==role]),'prefix5':sum(r['greedy']['prefix'] for r in rr),'prefix7':sum(r['greedy_D7']['prefix'] for r in rr),'TF_CE':float(np.mean([v for r in rr for v in r['teacher_forced']['ce']])),'TF_top1':sum(sum(r['teacher_forced']['top1']) for r in rr)} for role in ('historical_seen','historical_not_seen')} for p in ('error_enriched','ordinary')}
  (dest/'result.json').write_text(json.dumps({'arm':arm,'model':model,'rows':rows,'summary':summary},indent=2));results[arm]=summary;print(json.dumps({'arm':arm,'summary':summary}),flush=True)
 (out/'result.json').write_text(json.dumps({'status':'HISTORICAL_NATIVE_COMPLETE','summary':results,'wall_seconds':time.monotonic()-started,'matrix':pin(RUN/'historical-matrix.json'),'new_training':0},indent=2))
if __name__=='__main__':main()

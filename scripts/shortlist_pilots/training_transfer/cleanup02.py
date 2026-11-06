"""Close this diagnostic only; preserve paid states and useful primary raw arrays."""
from pathlib import Path
import json,os,time,hashlib,subprocess,shutil,gzip
ROOT=Path('/home/philip/binary-eagle-decoding');RUN=ROOT/'runs/training-transfer-diagnosis-20261027';OLD=ROOT/'runs/overnight-quality-20261027'
def pin(p):
 with Path(p).open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
 return {'path':str(p),'sha256':h,'bytes':Path(p).stat().st_size}
def atomic(p,v):
 tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,indent=2));os.replace(tmp,p)
def main():
 assert json.loads((RUN/'analysis-current/summary.json').read_text())['status']=='CURRENT_ANALYSIS_COMPLETE'
 assert json.loads((RUN/'analysis-historical/summary.json').read_text())['status']=='HISTORICAL_ANALYSIS_COMPLETE'
 states={p.parent.name:json.loads(p.read_text()) for p in (RUN/'supervision').glob('*/state.json')};assert all(v['status'] in ('finished','interrupted') for v in states.values())
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
 # Audit every prior paid overnight state and each compared historical/current artifact.
 refs=[]
 for arm in ('control','dpal','vat'):
  finish=json.loads((OLD/('train-'+arm)/'finish.json').read_text());refs.append(finish['checkpoint'])
  for n in (1080,2160):refs.append(json.loads((OLD/('train-'+arm)/f'export-{n}.json').read_text())['output'])
 anchors=json.loads((RUN/'historical-anchors-qualified.json').read_text())
 for q in anchors.values():refs.extend([q['checkpoint'],q['model']])
 spec=json.loads((OLD/'train-control/start.json').read_text())['spec'];refs.extend([spec['parent']['checkpoint'],spec['parent']['model'],spec['source_checkpoint']]);unique={r['path']:r for r in refs};paid=[]
 for path,ref in unique.items():
  actual=pin(path);assert actual['sha256']==ref['sha256'];paid.append(actual)
 # Check source custody, original PC tracked edits and existing historical controls.
 pre=json.loads((RUN/'preflight.json').read_text());assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True)==pre['revision']['out'];assert subprocess.check_output(['git','diff','--binary'],cwd=ROOT,text=True)==pre['dirty']['out']
 source_pins=json.loads((RUN/'source-pins.json').read_text());assert all(pin(p)['sha256']==h for p,h in source_pins.items())
 control=json.loads((Path.home()/'.config/binary-eagle-decoding/gpu-control.json').read_text());assert control==pre['controls'] and control['rtx2080ti']['pause_requested']
 # All removed data is owned and reproducible from retained source, input chunks, paid weights and runtime.
 candidates=[];compressed=[]
 for base in ('current-native','historical-native'):
  for p in (RUN/base).glob('*/*.log'):
   raw=p.read_bytes();dest=p.with_suffix(p.suffix+'.gz');assert not dest.exists();dest.write_bytes(gzip.compress(raw,mtime=0));assert gzip.decompress(dest.read_bytes())==raw;compressed.append({'original':str(p),'gzip':pin(dest)});candidates.append(p)
 for base in ('current-native','historical-native'):
  for p in (RUN/base).glob('*/cell-*/hidden-case-*/*'):
   if p.is_file() and (p.name.startswith('dspark_markov_') or p.name in ('result_output.f32','vocab_base_projection.f32')):candidates.append(p)
 for d in (RUN/'supervision').iterdir():
  if not d.is_dir():continue
  for name in ('cache','tmp','torch','huggingface','cuda'):
   p=d/name
   if p.exists():candidates.extend(x for x in p.rglob('*') if x.is_file() and not x.is_symlink())
 for p in (RUN/'native-model-cache').glob('*.gguf'):
  assert pin(p)['sha256']==anchors[p.stem]['model']['sha256'];candidates.append(p)
 helper=RUN/'native/replay'
 if helper.exists():candidates.append(helper)
 # Journal BEFORE mutation; never guess deleted byte totals after failure.
 journal={'status':'PREPARED','recoverability':'Retained exact source/build command/runtime pins and frozen features/labels/paid model digests reproduce native trace intermediates. Primary greedy and teacher-forced raw arrays, layer states, per-depth records, failed receipts, source and all paid states remain.','compressed_log_recovery':compressed,'files':[pin(p) for p in sorted(set(candidates))]};atomic(RUN/'cleanup-journal.json',journal)
 removed=0
 for r in journal['files']:
  p=Path(r['path']);assert p.is_relative_to(RUN) and not p.is_symlink() and pin(p)==r;p.unlink();removed+=r['bytes']
 journal.update(status='REMOVALS_COMPLETE',removed_bytes=removed);atomic(RUN/'cleanup-journal.json',journal)
 k=json.loads((RUN/'keeper/keeper.json').read_text());assert k['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip();assert int(Path(f"/proc/{k['pid']}/stat").read_text().rsplit(')',1)[1].split()[19])==k['birth_ticks'];(RUN/'keeper/STOP').touch()
 for _ in range(30):
  try:current=json.loads((RUN/'keeper/keeper.json').read_text())
  except json.JSONDecodeError:time.sleep(.1);continue
  if current['status']=='stopped':break
  time.sleep(.2)
 assert current['status']=='stopped'
 try:alive=Path(f"/proc/{k['pid']}/stat").read_text().rsplit(')',1)[1].split()[0] not in ('Z','X')
 except FileNotFoundError:alive=False
 assert not alive,'owned keeper remains alive'
 processes=subprocess.check_output(['ps','-eo','pid,ppid,args'],text=True);owned=[line for line in processes.splitlines() if str(RUN) in line and 'cleanup.py' not in line and 'python3 -' not in line];assert not owned,owned
 contexts=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True);assert not contexts.strip()
 retained={}
 for d in RUN.iterdir():
  if d.is_dir():retained[d.name]=sum(p.stat().st_size for p in d.rglob('*') if p.is_file() and not p.is_symlink())
 result={'status':'DIAGNOSTIC_CLOSED_GPU_RELEASED','unix':time.time(),'removed_bytes':removed,'removal_journal':pin(RUN/'cleanup-journal.json'),'paid_artifacts_rehashed':paid,'all_paid_artifacts_preserved':True,'prior_PC_source_diff_preserved':True,'source_files_checked':len(source_pins),'controls_unchanged':control,'GPU_contexts_after':contexts,'owned_processes_after':owned,'keeper':current,'closed_supervisors':list(states),'retained_bytes_by_directory':retained,'retained_bytes_total':sum(retained.values())+sum(p.stat().st_size for p in RUN.iterdir() if p.is_file()),'disk':subprocess.check_output(['df','-B1','/','/mnt/c'],text=True),'new_TRAIN_seconds':0,'new_updates':0,'limitations':'Primary raw arrays and canonical new serving capture retained for reproducibility. No unique paid state removed.'};atomic(RUN/'cleanup-result.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ('paid_artifacts_rehashed','keeper')}),flush=True)
if __name__=='__main__':main()

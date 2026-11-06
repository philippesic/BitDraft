from pathlib import Path
import subprocess,json,os,time,hashlib
P=Path('/home/philip/binary-eagle-decoding');W=Path('/home/philip/.codex/worktrees/w1a8-actual-training-20261103');R=P/'runs/w1a8-modern-actual-training-20261103-01';started=time.monotonic();results=[]
ref=json.loads((R/'reference02/freeze.json').read_text())
for key in ['trainer','view_source']:
 p=Path(ref[key]['path'])
 with p.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==ref[key]['sha256']
for name,coefficient in [('ce01',0),('retention01',.1)]:
 control=json.loads((Path.home()/'.config/binary-eagle-decoding/gpu-control.json').read_text())['rtx5080'];assert not control.get('pause_requested') and not (R/'STOP').exists()
 argv=[str(P/'.venv/bin/python'),str(W/'scripts/w1a8_generalization/modern_train.py'),'--manifest',str(P/'runs/w1a8-modern-actual-training-data-20261103-01/final01/manifest.json'),'--out',str(R/name),'--phase','train','--reference',str(R/'reference02/reference.json'),'--steps','480','--retention',str(coefficient),'--allow-cuda']
 (R/(name+'-command.json')).write_text(json.dumps({'argv':argv,'scope':'Matchedfirst480updatecheckpoint;horizon19200;notconvergencelimitorvalidatedscreen'},indent=2)+'\n')
 process=subprocess.Popen(argv);q=Path('/proc')/str(process.pid);(R/(name+'-identity.json')).write_text(json.dumps({'pid':process.pid,'pgid':os.getpgid(process.pid),'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'start_ticks':int((q/'stat').read_text().rsplit(')',1)[1].split()[19]),'cmdline':(q/'cmdline').read_bytes().replace(b'\0',b' ').decode()},indent=2)+'\n');code=process.wait();assert code==0,(name,code);r=json.loads((R/name/'result.json').read_text());results.append(r);print(json.dumps({'arm':name,'status':r['status'],'updates':r['updates'],'trainer_seconds':r['trainer_seconds']}),flush=True)
(R/'pair-result.json').write_text(json.dumps({'status':'MATCHED480_PAIR_TRAINED_EXPORTED_NATIVE_PENDING','results':results,'wall_seconds':time.monotonic()-started},indent=2)+'\n')

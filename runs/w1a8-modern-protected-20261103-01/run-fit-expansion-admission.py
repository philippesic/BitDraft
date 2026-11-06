from pathlib import Path
import subprocess,json,time,hashlib
P=Path('/home/philip/binary-eagle-decoding');R=P/'runs/w1a8-modern-protected-20261103-01';start=time.monotonic();stages=[]
def pin(p):
 p=Path(p).resolve()
 with p.open('rb') as f:s=hashlib.file_digest(f,'sha256').hexdigest()
 return {'path':str(p),'bytes':p.stat().st_size,'sha256':s}
plan={'scope':'new24FIT labels frozen-target crosscheck, native embeddings and sourceindex union; no training/throughputclaim','source':pin(__file__),'stages':[pin(R/x) for x in ['qualify-expanded-target.py','prepare-fit-expansion.py']]};(R/'fit-expansion-admission-plan.json').write_text(json.dumps(plan,indent=2)+'\n')
for name in ['qualify-expanded-target.py','prepare-fit-expansion.py']:
 t=time.monotonic();code=subprocess.run([str(P/'.venv/bin/python'),str(R/name)]).returncode;assert code==0,(name,code);stages.append({'stage':name,'wall_seconds':time.monotonic()-t,'exit_code':code});(R/'fit-expansion-admission-progress.json').write_text(json.dumps({'stages':stages,'wall_seconds':time.monotonic()-start},indent=2)+'\n')
(R/'fit-expansion-admission-result.json').write_text(json.dumps({'status':'NEW24FIT_DATA_ADMISSION_COMPLETE_NOT_TRAINED','stages':stages,'wall_seconds':time.monotonic()-start},indent=2)+'\n')

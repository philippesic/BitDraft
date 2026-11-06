from pathlib import Path
import json,subprocess,hashlib,os,time,datetime
P=Path('/home/philip/binary-eagle-decoding');W=Path('/home/philip/.codex/worktrees/w1a8-actual-training-20261103');R=P/'runs/w1a8-modern-protected-20261103-01';C=Path('/mnt/c/Users/philip/binary-eagle-decoding-artifacts');out=C/'w1a8-selected-head-real-qualification-20261103-01'
def pin(p):
 p=Path(p).resolve()
 with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
 return dict(path=str(p),bytes=p.stat().st_size,sha256=h)
pins={}
for name,sha in [('modern_train_selected_head.py','54ed1dd8bb927148abbdc1bdd23d08ebf2bb3b7b23aeb14bfb1aa3ccbc39b961'),('modern_selected_head_qualify.py','b9cb00f47c0e30c2c1c6456b3f0397111c511b11cd581ab8374a9c35c6bf34b6'),('modern_selected_head.py','34b24bd17cf45834b48fb7fa1044644d65a1f6632e843ed8bc7c22d24a9e5739')]:
 p=W/'scripts/w1a8_generalization'/name;pins[name]=pin(p);assert pins[name]['sha256']==sha
parent=C/'w1a8-expanded-ce-resident-new24-3840-01/checkpoint-00003840.pt';assert pin(parent)['sha256']=='f4a8d2d31b1016816827d38884f61bbbf881cf029a1a44cc48f94af71e838475';assert not out.exists()
args=[str(P/'.venv/bin/python'),str(W/'scripts/w1a8_generalization/modern_selected_head_qualify.py'),'--manifest',str(P/'runs/w1a8-modern-actual-training-data-20261103-01/final01/manifest.json'),'--union-manifest',str(C/'w1a8-modern-fit-expansion-20261103-01/union01/manifest.json'),'--resident-proof',str(R/'resident-row-real-qualification.json'),'--resident-capacity-mode','error','--reference',str(P/'runs/w1a8-modern-actual-training-20261103-01/reference02/reference.json'),'--parent',str(parent),'--learn-signs','true','--expanded-ce','true','--extra-fit-readout','--final-updates','3842','--out',str(out),'--allow-cuda']
(R/'head-real-qualification-plan.json').write_text(json.dumps(dict(source_pins=pins,parent=pin(parent),argv=args,scope='two unique FIT-only updates, zero-parent export, actual512row finite gradient and exact Adam/RNG/counter replay; not capacity gain'),indent=2)+'\n')
t=time.monotonic();proc=subprocess.Popen(args,cwd=W);q=Path('/proc')/str(proc.pid);identity=dict(pid=proc.pid,pgid=os.getpgid(proc.pid),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),start_ticks=int((q/'stat').read_text().rsplit(')',1)[1].split()[19]),cmdline=(q/'cmdline').read_bytes().replace(b'\0',b' ').decode(),at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),root_owner=True);(R/'head-real-qualification-child-identity.json').write_text(json.dumps(identity,indent=2)+'\n');code=proc.wait();(R/'head-real-qualification-job-result.json').write_text(json.dumps(dict(exit_code=code,wall_seconds=time.monotonic()-t,out=str(out)),indent=2)+'\n');raise SystemExit(code)

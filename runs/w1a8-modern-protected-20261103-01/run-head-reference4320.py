from pathlib import Path
import json, subprocess, hashlib, os, time, datetime
P=Path('/home/philip/binary-eagle-decoding'); W=Path('/home/philip/.codex/worktrees/w1a8-actual-training-20261103');R=P/'runs/w1a8-modern-protected-20261103-01';C=Path('/mnt/c/Users/philip/binary-eagle-decoding-artifacts');out=C/'w1a8-head-resident-reference4320-01'
def pin(p):
 p=Path(p).resolve()
 with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
 return dict(path=str(p),bytes=p.stat().st_size,sha256=h)
script=W/'scripts/w1a8_generalization/modern_train_expanded_ce_resident.py';parent=C/'w1a8-expanded-ce-resident-new24-3840-01/checkpoint-00003840.pt'
assert pin(script)['sha256']=='ab410b5fd5196eef40257c87255ecdff0f8b2fdf83e8a9cd8facf5000e6e9263'
assert pin(parent)['sha256']=='f4a8d2d31b1016816827d38884f61bbbf881cf029a1a44cc48f94af71e838475'
assert not out.exists()
args=[str(P/'.venv/bin/python'),str(script),'--manifest',str(P/'runs/w1a8-modern-actual-training-data-20261103-01/final01/manifest.json'),'--union-manifest',str(C/'w1a8-modern-fit-expansion-20261103-01/union01/manifest.json'),'--resident-proof',str(R/'resident-row-real-qualification.json'),'--resident-capacity-mode','error','--reference',str(P/'runs/w1a8-modern-actual-training-20261103-01/reference02/reference.json'),'--resume',str(parent),'--expanded-ce','true','--extra-fit-readout','--final-updates','4320','--out',str(out),'--allow-cuda']
(R/'head-reference4320-plan.json').write_text(json.dumps(dict(source=pin(script),parent=pin(parent),argv=args,scope='strict original resident 3840 to4320 reference for selected-head migration; no new capacity'),indent=2)+'\n')
t=time.monotonic();proc=subprocess.Popen(args,cwd=W);q=Path('/proc')/str(proc.pid);identity=dict(pid=proc.pid,pgid=os.getpgid(proc.pid),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),start_ticks=int((q/'stat').read_text().rsplit(')',1)[1].split()[19]),cmdline=(q/'cmdline').read_bytes().replace(b'\0',b' ').decode(),at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),root_owner=True);(R/'head-reference4320-child-identity.json').write_text(json.dumps(identity,indent=2)+'\n');code=proc.wait();(R/'head-reference4320-result.json').write_text(json.dumps(dict(exit_code=code,wall_seconds=time.monotonic()-t,out=str(out),scope='full original source strict continuation reference, not selected head'),indent=2)+'\n');raise SystemExit(code)

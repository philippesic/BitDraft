from pathlib import Path
import json,hashlib,time,sys
P=Path('/home/philip/binary-eagle-decoding');W=Path('/home/philip/.codex/worktrees/w1a8-actual-training-20261103');R=P/'runs/w1a8-modern-protected-20261103-01';U=Path('/mnt/c/Users/philip/binary-eagle-decoding-artifacts/w1a8-modern-fit-expansion-20261103-01/union01/manifest.json');t=time.monotonic()
def pin(p):
 p=Path(p).resolve()
 with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
 return dict(path=str(p),bytes=p.stat().st_size,sha256=h)
script=W/'scripts/w1a8_generalization/modern_train_added_replay.py';assert pin(script)['sha256']=='8eab3c21429df19d3c3f95befd6d7a0e01664dac27a2b6bdb114c35fd12aa4c5';sys.path.insert(0,str(script.parent));import modern_train_added_replay as added
manifest=json.loads(U.read_text());assert pin(U)['sha256']=='86eaad87ee644e663fdc4176a4c5bff0cc52bfbeff4c1ceb72425e65ffade5ae';ids=set(manifest['members'][1]['window_ids']);new_fit=[w for w in manifest['windows'] if w['id'] in ids];assert len(new_fit)==1465 and all(w['role']=='fit' for w in new_fit)
packet=json.loads((R/'data-backoff-old_quarter-cases01.json').read_text());mapref=packet['draft_map'];assert pin(mapref['path'])==mapref;mapping=[int(x) for x in Path(mapref['path']).read_text().split()];assert len(mapping)==len(set(mapping))==50000
schedule,refs,report=added.load_added_references(R/'data-native-comparison-plan.json',U,manifest,new_fit,set(mapping),lambda:None);assert report['tokens']==407 and report['groups']==24 and report['windows']==372 and len(refs)==372
assert len({schedule.at(i)[0] for i in range(24)})==24 and all(schedule.at(i)[0]==schedule.at(i+24)[0] for i in range(24))
full=R/'added-replay-real-admission-full.json';full.write_text(json.dumps(report,indent=2)+'\n');summary=dict(status='REAL_ADDED407_FIT_NATIVE_REFERENCE_ADMITTED',source=pin(script),manifest=pin(U),report=pin(full),tokens=407,groups=24,windows=372,metadata_source_files=len(report['sources']),reference_sha256=report['reference_sha256'],wall_seconds=time.monotonic()-t,scope='nativeJSON/metadata/draftmap only; no model/PT/F32 arrays or GPU; no false480 equivalence or treatment result yet');(R/'added-replay-real-admission.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))

from pathlib import Path
import json,hashlib,os,torch,uuid,shutil
P=Path('/home/philip/binary-eagle-decoding');R=P/'runs/w1a8-modern-protected-20261103-01';C=Path('/mnt/c/Users/philip/binary-eagle-decoding-artifacts');Q=C/'w1a8-selected-head-real-qualification-20261103-01';v=json.loads((Q/'result.json').read_text());s=json.loads((P/'runs/local-w1a8-head-real-qualification-20261103-01/state.json').read_text());assert s['status']=='finished' and s['exit_code']==0
assert v['status']=='SELECTED_HEAD_REAL_GRADIENT_RESUME_EXPORT_PASS' and v['native_admission_passed'] is False
assert v['actual_gradient']['shape']==[512,2560] and v['actual_gradient']['finite'] and v['actual_gradient']['nonzero_rows']==512 and v['actual_gradient']['nonzero_elements']==512*2560
assert v['resume']['physical_updates']==3 and v['resume']['unique_updates']==2 and v['resume']['exact_compact_Adam_RNG_resume'] and v['resume']['second']['loss']==v['resume']['repeated_second']['loss']
def pin(p):
 p=Path(p)
 with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
 return dict(path=str(p),bytes=p.stat().st_size,sha256=h)
for r in v['resume']['checkpoints']+[v['learned_export'],v['parent_checkpoint'],v['trainer'],v['qualifier']]:assert pin(r['path'])==r
parent=C/'w1a8-expanded-ce-resident-new24-3840-01/model.gguf';zero=Q/'zero-parent.gguf';a=pin(parent);b=pin(zero);assert a['sha256']==b['sha256']==v['zero_export']['model']['sha256'] and a['bytes']==b['bytes']==40802176
selection=json.loads((Q/'selection.json').read_text());assert selection['count']==512 and selection['FIT_only'] is True and selection['unique_FIT_groups']==48
torch.set_num_threads(1)
for cursor in [3840,3841,3842]:
 saved=torch.load(Q/f'checkpoint-{cursor:08d}.pt',map_location='cpu',weights_only=False);assert saved['cursor']==cursor and saved['head_stage']['stage_updates']==cursor-3840
 assert saved['selected_state']['selected_latent'].shape==(512,2560) and torch.isfinite(saved['selected_state']['selected_latent']).all()
 assert saved['exposure']['ce_windows']==2*cursor and saved['exposure']['replay_windows']==cursor
members=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  if int((p/'stat').read_text().rsplit(')',1)[1].split()[2])==s['pgid']:members.append(int(p.name))
 except (OSError,ValueError,IndexError):pass
assert not members
before=shutil.disk_usage(C).free;temporary=zero.with_name('.zero-dedup-'+uuid.uuid4().hex);os.link(parent,temporary);assert pin(temporary)['sha256']==b['sha256'] and pin(zero)==b;os.replace(temporary,zero);assert pin(zero)==b and zero.stat().st_ino==parent.stat().st_ino
report=dict(status='ROOT_REAL512_HEAD_GRADIENT_RESUME_ZERO_EXPORT_PASS',result=pin(Q/'result.json'),actual_gradient=v['actual_gradient'],physical_updates=3,unique_updates=2,learned_hard_flips=v['selected_metrics']['hard_sign_flips'],selected_FIT_groups=48,selection_supported_label_coverage=selection['supported_label_coverage'],peak_cuda_allocated_bytes=v['peak_cuda_allocated_bytes'],whole_job_seconds=json.loads((R/'head-real-qualification-job-result.json').read_text())['wall_seconds'],head_to_total_gradient_norm_ratio_first=v['actual_gradient']['norm']/v['resume']['first']['gradient_norm'],global_clip_package_is_part_of_treatment=True,group_absent=True,zero_export_exact_duplicate_hardlinked=True,observed_Cfree_delta=shutil.disk_usage(C).free-before,scope='actual gradient/exact unique-step resume/export plumbing only; no native/head capacity/freshspeed claim')
(R/'head-real-qualification-root-audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))

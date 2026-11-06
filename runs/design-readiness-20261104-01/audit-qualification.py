from pathlib import Path
import sys,json,hashlib,os
P=Path('/home/philip/binary-eagle-decoding');R=P/'runs/design-readiness-20261104-01';sys.path.insert(0,str(P/'scripts/design_readiness'));from audit_panel import read,pin,signature
report=[]
for stage,count in [(1,84),(2,42)]:
 root=R/f'qualification-stage{stage}';result=read(root/'result.json');inputs=read(P/'runs/w1a8-modern-actual-training-20261103-01/inputs480.json');gold={r['id']:signature(read(r['reference']['path'])) for r in inputs['rows']+inputs['warmups']};seen=set();groups=set();images={}
 for r in result['records']:
  key=(r['arm'],r['id']);assert key not in seen;seen.add(key)
  assert pin(r['request']['path'])==r['request'] and pin(r['response']['path'])==r['response'];response=read(r['response']['path']);assert signature(response)==gold[r['id']]
  metrics=read(r['native_metrics']['path']);assert pin(r['native_metrics']['path'])==r['native_metrics'];assert metrics['delta']==r['native_metrics_delta'];assert metrics['delta']['spec_decode_num_drafts_total']==r['verification_rounds']>0;assert metrics['delta']['spec_decode_num_accepted_tokens_total']==r['draft_accepted'];assert metrics['delta']['spec_decode_num_draft_tokens_total']==r['draft_proposed']
 for cell in root.glob('rep-*'):
  release=read(cell/'release.json');assert release['owned_process_reaped'] and release['owned_process_group_absent'] and release['returncode']==0;groups.add(release['identity']['pgid'])
  for tag in ['warm','final']:
   for image in read(cell/f'loaded-images-{tag}.json')['images']:images[image['path']]=image
 for image in images.values():assert pin(image['path'])==image
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:group=int((p/'stat').read_text().rsplit(')',1)[1].split()[2])
  except (OSError,ValueError,IndexError):continue
  assert group not in groups
 assert len(seen)==count
 report.append(dict(stage=stage,responses=count,token_text_stop_parity=True,released_groups=len(groups),rehashed_images=len(images)))
(R/'qualification-analysis.json').write_text(json.dumps(dict(status='BOTH_VERBOSE_QUALIFICATIONS_RAW_AUDIT_PASS',stages=report),indent=2)+'\n');print(report)

from pathlib import Path
import hashlib,json,collections,os
P=Path('/home/philip/binary-eagle-decoding');R=P/'runs/w1a8-modern-protected-20261103-01';D=Path('/mnt/c/Users/philip/binary-eagle-decoding-artifacts/w1a8-head-pair-20261103-01/http-qualification01')
def read(p):return json.loads(Path(p).read_text())
def checked(v):
 p=Path(v['path']);b=p.read_bytes();assert len(b)==v['bytes'] and hashlib.sha256(b).hexdigest()==v['sha256'];return json.loads(b)
def sig(v):return {k:v[k] for k in ['tokens','content','stop_type','stopping_word','truncated']}
v=read(D/'result.json');inputs=read(D/'inputs.json');expected={q['id']:q for q in inputs['rows']+inputs['warmups']};counts=collections.Counter();acc=collections.defaultdict(lambda:[0,0]);seen=set()
for row in v['records']:
 assert (row['arm'],row['id']) not in seen;seen.add((row['arm'],row['id']))
 req=checked(row['request']);rsp=checked(row['response']);gold=checked(expected[row['id']]['reference']);origreq=checked(expected[row['id']]['request'])
 assert req==origreq and req['cache_prompt'] is False and req['temperature']==0 and req['seed']==1234 and req['return_tokens'] is True
 assert sig(rsp)==sig(gold) and len(rsp['tokens'])==row['completion_tokens']==rsp['tokens_predicted']
 assert rsp['timings']['cache_n']==0 and rsp['timings']['draft_n_accepted']==row['draft_accepted'] and rsp['timings']['draft_n']==row['draft_proposed']
 assert row['full_http_wall_seconds']>0
 counts[row['arm']]+=1
 if not row['warmup']: acc[row['arm']][0]+=row['draft_accepted'];acc[row['arm']][1]+=row['draft_proposed']
assert sorted(counts.values())==[14,14] and len(seen)==28
cells=[]
for cell in D.glob('rep-*'):
 rel=read(cell/'release.json');assert rel['owned_process_reaped'] and rel['owned_process_group_absent'] and rel['returncode']==0
 assert not (Path('/proc')/str(rel['identity']['pid'])).exists()
 log=(cell/'server.log').read_text();assert 'complete nine-linear fixed row W1A8' in log and log.count('W1A8 packed ')==9
 assert len(list(cell.glob('*response.json')))==14 and len(list(cell.glob('*request.json')))==14
 cells.append(dict(cell=str(cell),packed_linears=9,released=True))
assert len(cells)==2
report=dict(status='ROOT_ALL28_HEAD_HTTP_RAW_PARITY_AND_REQUEST_CONTROLS_PASS',responses=len(seen),per_arm=dict(counts),development_acceptance={k:dict(accepted=a,proposed=b) for k,(a,b) in acc.items()},cells=cells,scope='singlepass correctness/acceptance only; no repeated timing or fresh transfer claim; loaded images checked separately during current native admission')
(R/'head-http-root-audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))

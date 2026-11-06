import sys,copy
from pathlib import Path
P=Path('/home/philip/binary-eagle-decoding');R=P/'runs/contender-infra-20261104-01';sys.path.insert(0,str(R/'source-final'))
from runner import read,atomic,pin
from comparison import summarize
panel=read(R/'common-panel.json');panel['baseline']='candidate_0';result=summarize(panel)
row=next(r for r in result['results'] if r['arm']=='candidate_1')
assert panel['model_pins']['candidate_0']['sha256']==panel['model_pins']['candidate_1']['sha256']
native=read(R/'native-retry/serving/result.json');tables={name:{(r['id'],r['repetition']):r for r in native['records'] if r['arm']==name} for name in ['candidate_0','candidate_1']}
assert tables['candidate_0'].keys()==tables['candidate_1'].keys()
fields=['completion_tokens','draft_proposed','draft_accepted','verification_rounds'];diffs=[]
for key,a in tables['candidate_0'].items():
 b=tables['candidate_1'][key]
 for field in fields:
  if a[field]!=b[field]:diffs.append(dict(key=key,field=field,left=a[field],right=b[field]))
assert not diffs,diffs
atomic(R/'repeatability.json',dict(status='IDENTICAL_EXPORT_NATIVE_COUNTS_MATCH',paired_requests=len(tables['candidate_0']),timing=row,counts_equal=True,scope='Identical continuous/resumed exported model; timing comparison is a repeatability check, not a feature or training benefit.'))
print(row)

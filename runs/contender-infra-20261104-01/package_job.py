import sys
from pathlib import Path
R=Path('/home/philip/binary-eagle-decoding/runs/contender-infra-20261104-01');sys.path.insert(0,str(R/'source-final'))
from bundle import create,verify
from runner import pin,read,atomic
create(R/'bundle-selection.json',R/'bundle-final');verify(R/'bundle-final')
source=Path('/mnt/c/Users/philip/binary-eagle-decoding-artifacts/contender-infra-20261104-01/continuous/result.json');result=read(source);copied=pin(R/'bundle-final/draft.gguf');assert copied['sha256']==result['model']['sha256'] and copied['bytes']==result['model']['bytes']
result.update(model=copied,deployment_copy=dict(source=pin(source),bundle=pin(R/'bundle-final/bundle.json'),scope='Same trained checkpoint/export bytes copied to deployment bundle; no additional training.'))
atomic(R/'packaged-result.json',result)

"""Freeze the small Python execution closure before launching a long arm."""
import os,shutil,subprocess,tarfile,time
from .common import ROOT,R,pin,save,identity

def freeze(label):
 directory=R/'python-sources'/(label+'-'+str(time.time_ns()));directory.mkdir(parents=True,exist_ok=False)
 paths=[*list((ROOT/'src').rglob('*.py')),*list((ROOT/'scripts').glob('*.py'))]
 for name in ('component_campaign','dflash_baseline','execution_campaign'):paths+=list((ROOT/'scripts'/name).glob('*.py'))
 paths+=list((ROOT/'scripts/component_campaign').glob('*.cpp'))
 paths.append(ROOT/'experiments/w1a8-standing-20261102/baseline.json')
 for source in paths:
  dest=directory/source.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest)
 save(directory/'source-manifest.json',dict(owner=identity(),workspace=str(ROOT),project_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),files={str(p.relative_to(directory)):pin(p) for p in directory.rglob('*') if p.is_file()}))
 return directory

def finalize():
 if os.environ.get('COMPONENT_FROZEN_SOURCE')!=str(ROOT):return
 manifest=ROOT/'source-manifest.json'
 if not manifest.exists() or ROOT.parent!=R/'python-sources':raise ValueError('unexpected source closure ownership')
 archive=ROOT.with_suffix('.tar.gz')
 with tarfile.open(archive,'w:gz',compresslevel=3) as out:out.add(ROOT,arcname=ROOT.name)
 with tarfile.open(archive,'r:gz') as source:
  for member in source.getmembers():
   if member.isfile():
    stream=source.extractfile(member)
    for block in iter(lambda:stream.read(1<<20),b''):pass
 save(ROOT.with_suffix('.retention.json'),dict(archive=pin(archive),original_manifest=__import__('json').load(open(manifest)),retention='Exact immutable Python source closure for this arm; loose copies removed after terminal training result.'))
 shutil.rmtree(ROOT)

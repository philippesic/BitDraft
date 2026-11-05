from pathlib import Path
import difflib,json,hashlib
P=Path('/home/philip/binary-eagle-decoding');F=P/'runs/precision-native-20261104-04';R=P/'runs/execution-campaign-20261105-01';S=P/'runs/w1a8-modern-packed-runtime-20261103-01/source';ROOT=Path(__file__).resolve().parents[2];E=ROOT/'experiments/execution-campaign-20261105/runtime';patch=[];pins=[]
files={'ggml/src/ggml-cuda/w1a1.cu':R/'runtime-combined-last02/w1a1.cu','ggml/src/ggml-cuda/ggml-cuda.cu':R/'runtime-combined-last02/ggml/src/ggml-cuda/ggml-cuda.cu','src/models/dflash.cpp':R/'runtime-combined-last02/src/models/dflash.cpp','common/speculative.cpp':R/'runtime-head02/speculative.cpp'}
for name,new in files.items():
 old=(S if name.startswith('common/') else F)/name;before=old.read_text();after=new.read_text();patch.extend(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='a/'+name,tofile='b/'+name))
 pins.append(dict(relative=name,base=dict(path=str(old),sha256=hashlib.file_digest(old.open('rb'),'sha256').hexdigest()),optimized=dict(path=str(new),sha256=hashlib.file_digest(new.open('rb'),'sha256').hexdigest())))
(E/'optimized.patch').write_text(''.join(patch));(E/'source-pins.json').write_text(json.dumps(dict(base_runtime=str(F),files=pins,submodule_gitlink_changed=False),indent=2)+'\n')

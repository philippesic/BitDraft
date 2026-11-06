import json,pathlib,subprocess,sys
root=pathlib.Path.home()/"binary-eagle-decoding"
dest=root/"runs/checkouts/nine-model-qat-5080-b998f4f"
parent="658beadc632fec8cc41eab5c3889e58022b92aa6"
native="50ca2676fb6fbc6a8455e7bce66a614f84e7016c"
def run(args,cwd=dest):
 p=subprocess.run(args,cwd=cwd,text=True,capture_output=True)
 if p.returncode:
  print(json.dumps({"status":"FAILED","argv":args,"rc":p.returncode,"stdout":p.stdout[-4000:],"stderr":p.stderr[-4000:]}))
  raise SystemExit(p.returncode)
 return p.stdout.strip()
run(["git","fetch","origin","main"],cwd=dest)
origin=run(["git","rev-parse","origin/main"],cwd=dest)
if not subprocess.run(["git","merge-base","--is-ancestor",parent,"origin/main"],cwd=dest).returncode==0:
 print(json.dumps({"status":"FREEZE_NOT_ON_PUBLISHED_MAIN","origin_main":origin,"expected":parent}))
 raise SystemExit(8)
run(["git","checkout","--detach",parent],cwd=dest)
run(["git","submodule","update","--init","--recursive","third_party/llama.cpp"],cwd=dest)
actual=run(["git","-C","third_party/llama.cpp","rev-parse","HEAD"],cwd=dest)
if actual!=native:
 print(json.dumps({"status":"NATIVE_GITLINK_MISMATCH","expected":native,"actual":actual}))
 raise SystemExit(9)
dirty=run(["git","status","--porcelain"],cwd=dest)
print(json.dumps({"status":"SOURCE_FREEZE_READY","origin_main":origin,"parent":run(["git","rev-parse","HEAD"],cwd=dest),"native":actual,"clean":not bool(dirty),"path":str(dest.relative_to(root))},indent=2))

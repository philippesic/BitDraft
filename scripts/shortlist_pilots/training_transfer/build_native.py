"""Add a read-only native teacher-forced Markov head to authentic chunk replay."""
from pathlib import Path
import json,subprocess,hashlib
ROOT=Path('/home/philip/binary-eagle-decoding');RUN=ROOT/'runs/training-transfer-diagnosis-20261027';OLD=ROOT/'runs/shortlist-throughput-20261027/native'
def pin(p):
 with Path(p).open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
 return {'path':str(p),'sha256':h}
def main():
 out=RUN/'native';out.mkdir(exist_ok=False);ready=json.loads((OLD/'ready.json').read_text());source=ROOT/'runs/overnight-quality-20261027/native-gate/native-replay.cpp';text=source.read_text()
 needle='name=="vocab_base_projection" ||';assert text.count(needle)==1;text=text.replace(needle,'name=="result_output" || '+needle)
 needle='int main(int argc,char**argv)';assert text.count(needle)==1;text=text.replace(needle,(RUN/'source/teacher_head.h').read_text()+'\n'+needle)
 needle='hidden.complete();resources(dev,resource_log,"after-noise",ci);';assert text.count(needle)==1
 text=text.replace(needle,needle+'\n need(hidden.trace.contains("result_output"),"base head capture required"); teacher_head(model,dev,hidden.trace["result_output"]["path"].get<std::string>(),labels,last,out/("teacher-"+std::to_string(ci)+".f32"));')
 p=out/'replay.cpp';p.write_text(text);binary=out/'replay';inc=['-I'+str(Path(ready['source_root'])/x) for x in ('src','include','ggml/include','common','vendor','vendor/nlohmann')]
 cmd=['g++','-std=c++17','-O2',*inc,str(p),'-L'+str(OLD/'build/bin'),'-Wl,--disable-new-dtags,-rpath,'+str(OLD/'build/bin'),'-lllama','-lggml','-lggml-base','-o',str(binary)]
 v=subprocess.run(cmd,capture_output=True,text=True,timeout=120);(out/'compile.log').write_text(v.stdout+v.stderr);assert v.returncode==0,v.stderr[-6000:]
 (out/'ready.json').write_text(json.dumps({'binary':pin(binary),'source':pin(p),'original':pin(source),'runtime':pin(OLD/'ready.json'),'command':cmd},indent=2));print('NATIVE_HELPER_BUILT',flush=True)
if __name__=='__main__':main()

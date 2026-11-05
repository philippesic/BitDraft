"""Native vocabulary-only context admission, qualified on consumed DEV first."""
import argparse,json,os,select,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.dflash_baseline.precision_benchmark import pin,save,pause
P=Path('/home/philip/binary-eagle-decoding');R=P/'runs/precision-campaign-20261104-01';OUT=R/'tokenizer';runtime=P/'runs/precision-native-20261104-04';base=P/'runs/w1a8-modern-packed-runtime-20261103-01/source'
def build():
 if not (R/'confirmation-dev12/result.json').exists():raise ValueError('respect formal timing quiet window')
 OUT.mkdir(exist_ok=False)
 source=ROOT/'scripts/dflash_baseline/tokenize.cpp';binary=OUT/'tokenize';argv=['c++','-std=c++17','-O2',str(source),'-I'+str(base/'include'),'-I'+str(runtime/'ggml/include'),'-I'+str(base/'ggml/include'),'-I'+str(base/'vendor'),'-L'+str(runtime/'bin'),'-Wl,-rpath,'+str(runtime/'bin'),'-Wl,-rpath-link,'+str(runtime/'bin'),'-l:libllama.so.0.8.0','-o',str(binary)]
 subprocess.run(argv,check=True,timeout=120)
 save(OUT/'build.json',dict(command=argv,source=pin(source),binary=pin(binary),headers=dict(llama=pin(base/'include/llama.h'),ggml=pin(runtime/'ggml/include/ggml.h'),json=pin(base/'vendor/nlohmann/json.hpp')),libraries={p.name:pin(p) for p in (runtime/'bin').glob('*.so*') if p.is_file()},compiler=subprocess.check_output(['c++','--version'],text=True),scope='Vocabulary only; same libllama/target as measured HTTP; no runtime replacement or model generation.'))
class Tokenizer:
 def __init__(self,out):self.out=Path(out)
 def readline(self,timeout):
  if not select.select([self.process.stdout],[],[],timeout)[0]:raise TimeoutError('native tokenizer response timeout')
  return self.process.stdout.readline()
 def __enter__(self):
  pause();self.out.mkdir(exist_ok=False);self.log=(self.out/'stderr.log').open('w')
  try:self.process=subprocess.Popen([str(OUT/'tokenize'),str(P/'runs/spec-shortlist-20261030/models/Qwen3.5-4B-Q4_K_M.gguf')],env=dict(os.environ,LD_LIBRARY_PATH=str(runtime/'bin')),stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,text=True)
  except BaseException:self.log.close();raise
  try:
   proc=Path('/proc')/str(self.process.pid);save(self.out/'ownership.json',dict(pid=self.process.pid,stat=(proc/'stat').read_text(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text(),executable=pin(OUT/'tokenize'),target=pin(P/'runs/spec-shortlist-20261030/models/Qwen3.5-4B-Q4_K_M.gguf'),supervised_group=os.getpgrp()))
   ready=self.readline(120)
   if not ready or json.loads(ready)!={'ready':True,'vocab_only':True}:raise RuntimeError('native tokenizer startup failed')
   (self.out/'library-maps.txt').write_text((proc/'maps').read_text());return self
  except BaseException:self.__exit__(None,None,None);raise
 def tokens(self,text):
  pause();self.process.stdin.write(json.dumps(dict(text=text))+'\n');self.process.stdin.flush();line=self.readline(60)
  if not line:raise RuntimeError('native tokenizer exited')
  row=json.loads(line)
  if row['n_tokens']!=len(row['tokens']):raise ValueError('native tokenizer count mismatch')
  return row['tokens']
 def __exit__(self,*_):
  try:self.process.stdin.close()
  except BrokenPipeError:pass
  try:self.process.wait(timeout=10)
  except subprocess.TimeoutExpired:
   self.process.terminate()
   try:self.process.wait(timeout=10)
   except subprocess.TimeoutExpired:self.process.kill();self.process.wait()
  self.process.stdout.close();self.log.close();save(self.out/'release.json',dict(reaped=True,returncode=self.process.returncode))
def qualify():
 from scripts.dflash_baseline.precision_benchmark import cohort
 geometry=json.load(open(R/'confirmation-dev12/00-target/geometry.json'));expected={x['id']:x['prompt_tokens'] for x in geometry};records=[]
 with Tokenizer(OUT/'qualification') as tokenizer:
  for row in cohort('natural12'):
   tokens=tokenizer.tokens(row['payload']['prompt'])
   if len(tokens)!=expected[row['id']]:raise ValueError('CPU native tokenization differs from measured HTTP geometry')
   records.append(dict(id=row['id'],native_tokens=len(tokens),measured_http_tokens=expected[row['id']],exact_count=True))
 save(OUT/'qualification.json',dict(status='NATIVE_VOCAB_ADMISSION_QUALIFIED',records=records,geometry=pin(R/'confirmation-dev12/00-target/geometry.json'),build=pin(OUT/'build.json'),scope='All12 consumed DEV requests match actual measured HTTP token counts; exact same vocabulary API and add_special/parse_special flags. No heldout payload read.'))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['build','qualify','prepare']);a=ap.parse_args()
 if a.mode in ('build','prepare'):build()
 if a.mode in ('qualify','prepare'):qualify()

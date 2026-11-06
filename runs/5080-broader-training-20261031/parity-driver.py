from pathlib import Path
import os,sys,json,subprocess,fcntl,time
P=Path('/home/philip/binary-eagle-decoding');R=P/'runs/5080-broader-training-20261031';O=P/'runs/spec-parity-gpu-20261031';S=P/'runs/spec-shortlist-20261030/runtime'
f=(Path.home()/'.config/binary-eagle-decoding/rtx5080-campaign.lock').open('a+');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
assert (R/'cpu-done.json').exists()
assert not json.loads((Path.home()/'.config/binary-eagle-decoding/gpu-control.json').read_text())['rtx5080']['pause_requested']
cmd=json.loads((O/'compile-command.json').read_text());src=(O/'cases-rs.cpp').read_text();results=[]
for label,case in [('equal',src.replace('o.n_rs_seq=0; auto serial','o.n_rs_seq=16; auto serial')),('original',src)]:
 c=R/(label+'-cases.cpp');c.write_text(case);cc=[str(c) if x==str(O/'cases-rs.cpp') else str(R/('test-'+label)) if x==str(O/'test-parity') else x for x in cmd];subprocess.run(cc,check=True)
for label,fusion,fa in [('equal',False,'on'),('equal',True,'on'),('original',True,'on'),('equal',True,'off')]:
 name=label+('-nofusion' if fusion else '-default')+'-fa'+fa;log=R/(name+'.log');env=os.environ.copy()
 if fusion:env['GGML_CUDA_DISABLE_FUSION']='1'
 argv=[str(R/('test-'+label)),'-m',str(O/'models/Qwen3.5-4B-Q4_K_M.gguf'),'-ngl','99','-t','4','-fa',fa,'--test','spec-parity','--logs',str(R/name)]
 with log.open('x') as out:
  t=time.time();r=subprocess.run(argv,env=env,stdout=out,stderr=subprocess.STDOUT,timeout=300)
 row={'name':name,'command':argv,'env':{'GGML_CUDA_DISABLE_FUSION':env.get('GGML_CUDA_DISABLE_FUSION')},'rc':r.returncode,'seconds':time.time()-t,'tail':log.read_text()[-4500:]};results.append(row);(R/'parity-results.json').write_text(json.dumps(results,indent=2));print(json.dumps(row),flush=True)

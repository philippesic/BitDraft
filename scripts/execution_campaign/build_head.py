"""Honor requested output rows in DFlash, and optionally crop after final attention."""
from pathlib import Path
import argparse,sys,subprocess,shlex,json
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.execution_campaign.build import build,P,F,B,pin
from scripts.execution_campaign.extend import relink
ap=argparse.ArgumentParser();ap.add_argument('out',type=Path);ap.add_argument('--last-ffn',action='store_true');a=ap.parse_args();r=build(a.out,copy=True,bound=True,threads=256)
cpp=(r/'dflash.cpp').read_text();start=cpp.index('        if (layer.ffn_up_w1a8.packed) {');last=cpp.rfind('    for (int il = 0;',0,start);cpp=cpp[:last]+'    bool exec_cropped=false;\n'+cpp[last:]
guard='!model.dspark_markov_w1 && !model.dflash_selector_hidden && !model.lilicorr_pass_hidden_proj'
if a.last_ffn:
 needle='        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpL);'
 code='''        if (il==n_layer-1 && n_outputs<n_tokens && !cparams.embeddings && !cparams.embeddings_nextn && '''+guard+''') {
            auto* ids=build_inp_out_ids();cur=ggml_get_rows(ctx0,cur,ids);inpL=ggml_get_rows(ctx0,inpL,ids);exec_cropped=true;
        }
'''
 assert cpp.count(needle)==2;cpp=cpp.replace(needle,code+needle,1)
needle='    cur = build_lora_mm(output, cur, output_s);'
code='    if (!exec_cropped && n_outputs<n_tokens && '+guard+') cur=ggml_get_rows(ctx0,cur,build_inp_out_ids());\n'
assert cpp.count(needle)>=1;cpp=cpp.replace(needle,code+needle,1)
relink(r,{'src/models/dflash.cpp':cpp})
S=P/'runs/w1a8-modern-packed-runtime-20261103-01/source';src=S/'common/speculative.cpp';text=src.read_text();needle='batch.add(i == 0 ? dp.id_last : mask_token_id, n + i, seq_id, !is_dflash2 && !is_lilicorr)';assert text.count(needle)==1;text=text.replace(needle,'batch.add(i == 0 ? dp.id_last : mask_token_id, n + i, seq_id, !is_dflash2 && !is_lilicorr && (i!=0 || is_dspark))')
f=r/'speculative.cpp';f.write_text(text)
def command(target):return shlex.split(subprocess.check_output(['ninja','-C',str(B),'-t','commands',target],text=True).splitlines()[-1])
obj=r/'speculative.cpp.o';cmd=command('common/CMakeFiles/llama-common.dir/speculative.cpp.o');cmd[cmd.index('-c')+1]=str(f);cmd[cmd.index('-o')+1]=str(obj);cmd+=['-I'+str(src.parent)]
if '-MF' in cmd:cmd[cmd.index('-MF')+1]=str(obj)+'.d'
with (r/'head-build.log').open('w') as log:subprocess.run(cmd,cwd=B,stdout=log,stderr=subprocess.STDOUT,check=True)
link=command('bin/libllama-common.so.0.8.0')[2:-2];link=[str(obj) if x=='common/CMakeFiles/llama-common.dir/speculative.cpp.o' else x for x in link if not x.startswith('-Wl,--dependency-file=')];link[link.index('-o')+1]=str(r/'bin/libllama-common.so.0.8.0');(r/'bin/libllama-common.so.0.8.0').unlink()
with (r/'head-build.log').open('a') as log:subprocess.run(link,cwd=B,stdout=log,stderr=subprocess.STDOUT,check=True)
for name in ['libllama-common.so','libllama-common.so.0']:
 f=r/'bin'/name;f.unlink();f.symlink_to(r/'bin/libllama-common.so.0.8.0')
d=json.load(open(r/'build.json'));d.update(head_commands=[cmd,link],head_sources=[pin(f) for f in [r/'src/models/dflash.cpp',r/'speculative.cpp']],libraries=[pin(f) for f in (r/'bin').iterdir() if not f.is_symlink()]);(r/'build.json').write_text(json.dumps(d,indent=2)+'\n')

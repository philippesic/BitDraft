"""Build only new diagnostic helpers against the retained qualified runtime."""
import subprocess
from scripts.component_campaign.common import P, ROOT, pin, save
from scripts.dflash2_smoke.train import R

RUNTIME=P/'runs/component-campaign-20261105-01/runtime04'
SOURCE=P/'runs/w1a8-modern-packed-runtime-20261103-01/source'


def main():
    commands=[]
    includes=['-I'+str(SOURCE/n) for n in ('include','ggml/include','vendor/nlohmann')]
    links=['-L'+str(RUNTIME/'bin'),'-Wl,-rpath,'+str(RUNTIME/'bin')]
    for name,flags,libraries in [('quantize',[],['-lllama']),('selector_audit',['-fPIC','-shared'],['-ldl']),
                                  ('replay',['-std=c++17'],['-lllama','-lggml-base','-lggml','-ldl'])]:
        source=ROOT/'scripts/dflash2_smoke'/f'{name}.cpp'
        output=R/('libselector-audit.so' if name=='selector_audit' else name)
        command=['c++','-O2',*flags,*includes,str(source),*links,*libraries,'-o',str(output)]
        with (R/f'{name}-build.log').open('w') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        commands.append(dict(command=command,source=pin(source),output=pin(output)))
    save(R/'diagnostic-build.json',dict(commands=commands,runtime=pin(RUNTIME/'build.json'),
         headers=[pin(SOURCE/n) for n in ('include/llama.h','src/llama-ext.h')]))


if __name__=='__main__': main()

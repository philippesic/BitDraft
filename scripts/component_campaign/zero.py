"""Qualify the finite initial native formats before exposing a broad training queue."""
from .common import *
from .export import rewrite
from .archive import archive

def specs():
 return [dict(name='H0'),dict(name='H6',head='Q6_K'),dict(name='H4',head='Q4_K'),
  dict(name='E6',embedding='Q6_K'),dict(name='E4',embedding='Q4_K'),dict(name='HE4',head='Q4_K',embedding='Q4_K',tied=True),
  *[dict(name=n,kv=kv) for n,kv in [('K8F',['q8_0','f16']),('KF8',['f16','q8_0']),('K88',['q8_0','q8_0']),('K48',['q4_0','q8_0']),('K84',['q8_0','q4_0']),('K44',['q4_0','q4_0'])]],
  *[dict(name='S'+g,scalar={g:True}) for g in ('scale','midpoint','gain','clip','norm')],
  dict(name='Sall',scalar={g:True for g in ('scale','midpoint','gain','clip','norm')})]

def main():
 bases=R/'bases';bases.mkdir(exist_ok=True)
 source=R/'arms/H0-pilot/model-0.gguf'
 if not (bases/'shared.gguf').exists():(bases/'shared.gguf').hardlink_to(source)
 dest=R/'zero';dest.mkdir(exist_ok=True)
 for spec in specs():
  save(R/'specs'/(spec['name']+'.json'),spec)
  output=dest/(spec['name']+'.gguf')
  if output.exists() or output.with_suffix('.gguf.native.zst').exists():continue
  rewrite(source,output,spec)
  family=spec.get('head') or spec.get('embedding') or 'shared';basis=bases/(family+'.gguf')
  if not basis.exists():basis.hardlink_to(output)
  else:archive(output,basis)
  print('ZERO_EXPORT',spec['name'],flush=True)
 save(R/'initial-specs.json',specs())

if __name__=='__main__':main()

"""Read actual checkpoint GGUF tensor types; no inference from variant names."""
from pathlib import Path
import json,sys,collections
ROOT=Path('/home/philip/binary-eagle-decoding');RUN=ROOT/'runs/training-transfer-diagnosis-20261027';OLD=ROOT/'runs/overnight-quality-20261027'
def main():
 sys.path.insert(0,str(ROOT/'third_party/llama.cpp/gguf-py'))
 import gguf
 anchors=json.loads((RUN/'historical-anchors-qualified.json').read_text());parent=json.loads((OLD/'train-control/start.json').read_text())['spec']['parent'];eal=json.loads((OLD/'train-control/finish.json').read_text());q4=json.loads((ROOT/'runs/w1a8-algorithms-20261020/openitems-markov-benchmark06-root/bindings-development.json').read_text())['models']['q4_k_m'];models={n:q['model'] for n,q in anchors.items()};models.update(parent3504=parent['model'],EAL2160=eal['model'],q4_k_m=q4);results={}
 for name,ref in models.items():
  reader=gguf.GGUFReader(ref['path']);rows=[{'name':t.name,'type':t.tensor_type.name,'shape':list(map(int,t.shape))} for t in reader.tensors];results[name]={'model':ref,'tensor_type_counts':dict(collections.Counter(r['type'] for r in rows)),'binary_projection_count':sum('.w1a1_packed' in r['name'] for r in rows),'tensors':rows,'binary_activation_bits':reader.fields['dflash.w1ax.activation_bits'].contents() if 'dflash.w1ax.activation_bits' in reader.fields else None}
 (RUN/'precision-inventory.json').write_text(json.dumps(results,indent=2));print(json.dumps({n:{k:v for k,v in q.items() if k not in ('model','tensors')} for n,q in results.items()}))
if __name__=='__main__':main()

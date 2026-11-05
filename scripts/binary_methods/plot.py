from pathlib import Path
import json,numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
C=Path(__file__).resolve().parent;E=C.parents[1]/'experiments/binary-methods-20261104';D=json.loads((E/'analysis.json').read_text())
names={'A8_tensor':'A8 Tensor Core default','A8_dp4a':'A8 DP4A','A8_planes1':'A8 bit-serial separate','A8_planes7':'A8 bit-serial shared','A8_lut4':'A8 global LUT4','A8_sharedlut':'A8 shared LUT exact','A1_xor':'A1 XOR/popcount','A1_bmma':'A1 binary MMA → INT8','A1_tensor':'A1 Tensor Core adapter','A1_sharedlut':'A1 shared LUT exact'}
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(1,2,figsize=(15,6),layout='constrained');rs=D['native'];yy=np.arange(len(rs));left=np.zeros(len(rs))
for key,label,color in [('quantize_pack','Activation quantize/pack','#397ac6'),('representation_prep','Extra representation/layout','#ce9230'),('dot_scale','Dot/output scale','#319780')]:
 vals=np.array([r['binary_stages_us'][key]/r['speculative']['rounds'] for r in rs]);ax[0].barh(yy,vals,left=left,label=label,color=color);left+=vals
ax[0].set_yticks(yy,[names[r['arm']] for r in rs]);ax[0].invert_yaxis();ax[0].set_xlabel('CUDA kernel µs per round — lower is better');ax[0].set_title('Native binary projection pipeline');ax[0].legend(fontsize=8,loc='lower right')
for i,v in enumerate(left):ax[0].text(v+5,i,f'{v:.1f}',va='center',fontsize=9)
rs=D['throughput'];v=np.array([r['decode_tps'] for r in rs]);lo=v-np.array([min(r['decode_tps_per_repetition']) for r in rs]);hi=np.array([max(r['decode_tps_per_repetition']) for r in rs])-v
ax[1].barh(yy,v,xerr=[lo,hi],color=['#397ac6' if r['arm'].startswith('A8') else '#9367b0' for r in rs],capsize=3);ax[1].set_yticks(yy,[names[r['arm']] for r in rs]);ax[1].invert_yaxis();ax[1].set_xlabel('Decode tokens/s — higher is better');ax[1].set_title('Native generation throughput');ax[1].set_xlim(0,260)
for i,value in enumerate(v):ax[1].text(value+4,i,f'{value:.1f}',va='center',fontsize=9)
fig.suptitle('RTX 5080 • Qwen3-4B F16 + DSpark, D5\nA1 is an untrained activation-precision override; 2 × 3 DEV prompts, max128 tokens\nGPU stages: one warmed code request. Error bars: two repetition rates, not confidence intervals.',fontsize=12)
fig.savefig(E/'native-comparison.png',dpi=170);plt.close(fig)
fig,axs=plt.subplots(2,2,figsize=(15,10),layout='constrained')
for ax,(m,k,n) in zip(axs.flat,[(9728,2560,1),(9728,2560,7),(2560,9728,1),(2560,9728,7)]):
 rs=[r for r in D['kernels'] if (r['m'],r['k'],r['n'])==(m,k,n)];yy=np.arange(len(rs));v=[r['full_us'] for r in rs];ax.barh(yy,v,color=['#397ac6' if r['mode'].startswith('A8') else '#9367b0' for r in rs]);ax.set_yticks(yy,[names[r['mode']] for r in rs]);ax.invert_yaxis();ax.set_title(f'{m:,} × {k:,} • {n} position(s)');ax.set_xlabel('Full preparation + dot/output µs');ax.set_xlim(0,max(v)*1.13)
 for i,value in enumerate(v):ax.text(value+.3,i,f'{value:.1f}',va='center',fontsize=9)
fig.suptitle('Isolated actual-shape projections • resident synthetic inputs\nThree graph repetitions of32 operations; Tensor adapters specialize in7 positions.\nShared-LUT adapter includes weight transposition on every call.',fontsize=13);fig.savefig(E/'projection-comparison.png',dpi=170)

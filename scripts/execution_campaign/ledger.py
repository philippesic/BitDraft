"""Finite execution inventory; pending entries prevent campaign completion."""
from pathlib import Path
import json
P=Path('/home/philip/binary-eagle-decoding');R=P/'runs/execution-campaign-20261105-01'
items=[
('trace','Unchanged W1/Q4 CUDA/API dispatch and actual shapes','trace-analysis.json','tested','N3; consistent grouped Q4 and per-position DP4A W1; observational traces complete'),
('replay','Model-specific cache-restoring replay and changing context','replay-tranche4/','tested','Same external features/history; each model rebuilds its own cache; every repeat clears/reconstructs cache'),
('copies','Conditional pack contiguity fallback and copy-only ablation','tranche1-correctness.json','qualified','12 graph CONT candidates alias in production;6 visible named callback copies; no12-copy CUDA saving'),
('known_bound','Single-pass fixed/learned bounds, finite checks and scaled code sum','edge-checks02/result.json','qualified','256-thread bound path preserves codes/scales/sums and native caches'),
('bit_modes','A1/A2/A4/A8 and dynamic/fixed/learned specialization','runtime-both32/ops-N3.jsonl','qualified','A1 uses gained rounded value sign including negative-zero underflow; dynamic max retained'),
('warp_block','Warp/block preparation at actual N','micro2/result.json','rejected','32/128-thread mappings slower than256'),
('tiled_prep','Multi-block preparation plus workspace/completion sum','replay-tranche4/tiled-clean.json','qualified','Actual2-launch tiled quantization, maximum432-byte N3 partial-sum workspace per down op'),
('cross_dp4a','Packed sign reuse across positions','replay-tranche2/cross256-clean.json','rejected','Full gate/up plus down variant loses; source-level read reuse is not measured DRAM reduction'),
('row_position_tiling','Bounded row/position tiling, activation reuse and registers','micro2/result.json','tested','N1/N3;four rows/warp tile versus64-row signed MMA and cross-position accumulation'),
('signed_mma','Signed INT8 MMA with inactive columns and affine epilogue','runtime-mma/ops-N8.jsonl','rejected','Actual IMMA.16832.S8.S8; full-forward slower, especially down; no integer recovery from F32 sums'),
('layout_prepack','Load-time binary layout prepacking','mechanisms/summary.json','evaluated','Packed row-major accesses already coalesced at word level; unpack work is register-local; tested expanded-resident limit quantifies bounded possible benefit'),
('resident_int8','Resident-expanded INT8 sign weights','micro2/result.json','tested','405MiB signs versus50.625MiB packed; current test retains both. Replacement delta354.375MiB, additive residency405MiB. Gate/up modest gain/down loss'),
('silu_pack','SiLU/up/gain/down quantization fusion','replay-tranche4/silu03-clean.json','pending','First graph-fusion attempt did not dispatch; dependency/lifetime investigation continuing'),
('residual','Down affine epilogue plus residual','replay-tranche4/residual03-clean.json','qualified','Exact F32 add order;custom W1-only dispatch before global fusion restriction; no clear incremental speed'),
('joint_gate_up','Joint gate/up matrix execution','replay-tranche4/joint-clean.json','qualified','Shared prepared codes and two exact dot/affine outputs in one kernel'),
('prenorm','Prenorm/norm gain/channel gain/quantizer fusion','replay-tranche4/prenorm-clean.json','tested','Actual fused kernel, copied native RMS reduction; approximate lane because FTZ contract differs; native observed logits/caches exact on bounded replay'),
('head_anchor','Prune unused anchor vocabulary projection and sampler work','replay-tranche4/head02-clean.json','qualified','Ordinary sampler reads masks; requested-output flags and standard output-index gather; embeddings retain full rows'),
('last_anchor_ffn','Dead final-layer anchor FFN after final attention','replay-tranche4/lastffn02-clean.json','qualified','Only when outputs cropped and no embedding/nextn requests; all attention and cache writes remain; required logits/cache exact'),
('generic_fusion','Isolated drafter-only existing backend fusions','replay-tranche2/generic-audit.json','rejected','Target/verifier remain consistent32; some pack codes/caches change, small timing benefit; no exact promotion'),
('packed_a4','Packed A4 activation storage','mechanisms/summary.json','evaluated','Byte codes already direct; N3 codes are7.5KiB gate/up or27KiB down. DownA8 cannot be nibble-packed without changing grid; no extra byte-upcast exists to remove'),
('native_int4','Native INT4 machine lowering on SM120','mechanisms/summary.json','blocked','Tested m16n8k64 s4 PTX lowers to two signed INT8 IMMA sites; no native INT4 in this qualified compiler path'),
('fp16_codes','Exact-code FP16 operands/F32 accumulation','micro2/result.json','tested','Existing integer codes/signs exactly representable; all raw sums below2^24; conversion charged. Resident weight memory810MiB if all18 matrices; down loses'),
('a1_xor','A1 XOR/popcount reserve','../binary-methods-20261104-01/kernels.json','inapplicable','Primary fixedG4/D8 codes are not two-level signs. A1 would change quantization; retain historical separate-precision evidence'),
('a2_special','A2 specialized arithmetic reserve','runtime-both32/ops-N3.jsonl','inapplicable','Primary has no A2 projection; substituting ternary arithmetic changes its grid. A2 quantizer contract remains independently qualified'),
('bitplanes','Exact activation bitplanes and reuse','micro2/result.json','pending','Modern N1/N3 conversion-inclusive screen of historical exact-code algorithms'),
('lut','Exact-code LUT4 reuse','micro2/result.json','pending','Modern N1/N3 conversion-inclusive screen, affine midpoint completion charged'),
('persistent','Broad persistent-kernel reserve','trace-analysis.json','not_triggered','Native draft is already one CUDA graph replay; no per-kernel host launch gap inside it. Remaining dominant head/protected-weight work, no measured persistent-launch defect'),
('graphs','Graph recapture/replay and changing-context behavior','trace-analysis.json','evaluated','Request has graph updates as context changes; replay confirms one warm graph per forward. No forced artificial reuse or consistent32 removal'),
('workspace','Scratch/workspace reuse','replay-tranche4/tiled-clean.json','qualified','Backend pool owns432-byte partial sums, reused across operators/graphs; no persistent allocation per repeat'),
('allocations','Per-request allocation audit','trace-analysis.json','evaluated','Warm request API inventory has no repeated cudaMalloc/cudaFree; not an evidenced allocation defect'),
('readback_sync','CPU readbacks/synchronization and launch gaps','trace-analysis.json','evaluated','API spans diagnostic; target feature/readback/sampler semantics require sync; no subtraction-based Amdahl claim'),
('output_transfer','Head output and backend sampler transfer','trace-analysis.json','qualified','Live server graph includes backend top-k; replay without backend sampler reads full logits. Requested-row pruning reduces live sampler work; do not equate replay readback with live transfer'),
('attention','Drafter attention dispatch','trace-analysis.json','evaluated','Actual F16 flash-attention and stream-k fixup; anchor required by noncausal masks and cache. No unsupported replacement or target dispatch changes'),
('features_cache','Target-feature projection/catch-up/cache update ancestry','replay-tranche4/','evaluated','Source consumes newly verified features and injects model-specific projections. No duplicate exact rows proven reusable; sharing W1/Q4 projected caches invalid'),
('nvfp4','NVFP4 activation conversion','docs/EXECUTION_OPTIMIZATION.md','out_of_scope','Changes code grid/precision hypothesis; execution-only authorization excludes it'),
('precision_topology','Head/embed/KV/target/topology changes and training','docs/EXECUTION_CAMPAIGN.md','out_of_scope','Frozen checkpoint/quantization/architecture and no training'),
('final_native','Matched alternating actual-request latency/throughput confirmation','native-final/','pending','2same-process warmups and>=5alternating paired reps on consumed DEV12;original W1,retained W1,Q4 and fair optimizedQ4'),
('publish','JSON/CSV,latency/stage/ablation charts,static/interactive artifacts','charts/','pending','Not complete until actual data and uncertainty published'),
]
rows=[dict(id=i,opportunity=title,evidence=str(R/evidence) if not evidence.startswith('docs/') else evidence,status=status,reason=reason,correctness=None,isolated_speed=None,full_speed=None,memory=None) for i,title,evidence,status,reason in items]
(R/'ledger-start.json').write_text(json.dumps(dict(schema='dflash.execution_inventory.v1',items=rows),indent=2)+'\n')
print('Inventory items',len(rows))

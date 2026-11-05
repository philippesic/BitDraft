from pathlib import Path
import json,statistics
P=Path('/home/philip/binary-eagle-decoding');R=P/'runs/execution-campaign-20261105-01';ROOT=Path(__file__).resolve().parents[2];E=ROOT/'experiments/execution-campaign-20261105'
def load(p):return json.load(open(p))
ledger=load(R/'ledger-start.json');data=load(R/'charts-final/comparison.json');items={x['id']:x for x in ledger['items']}
retained=['copies','known_bound','bit_modes','tiled_prep','joint_gate_up','head_anchor','last_anchor_ffn','workspace','output_transfer']
for k in retained:items[k]['status']='retained'
for k in ['cross_dp4a','row_position_tiling','signed_mma','resident_int8','fp16_codes','bitplanes','lut','silu_pack','residual','prenorm','generic_fusion']:items[k]['status']='rejected'
for k in ['trace','replay','final_native','publish']:items[k]['status']='completed'
for k in ['layout_prepack','packed_a4','persistent','graphs','allocations','readback_sync','attention','features_cache']:items[k]['status']='evaluated_no_additional_change'
items['silu_pack'].update(reason='Actual12CUDA fusion fixtures change9241packed-stream bytes across all-bit/subnormal cases. Primary replay happened to match, but arbitrary-code exactness fails; not retained.',evidence=str(R/'fusion-fixtures/result.json'))
items['prenorm']['reason']='Actual fused prenorm/gain/quant kernel executes. Bounded primary replay exact; floating reduction/FTZ general contract differs and generic norm fusion changes some pack codes. Small measured benefit does not justify approximate promotion.'
items['residual']['reason']='Actual epilogue+residual fusion is native-exact, but no independent full-forward timing gain resolved. Keep the measured simpler retained combination.'
items['joint_gate_up']['reason']='One kernel shares codes across exact gate/up dots; all bit/mode tests at N1/3/8/9 dispatch joint36times per fixture set and pass independent integer/affine checks. Combined with tiled preparation.'
items['known_bound']['correctness']=str(R/'edge-checks03/result.json');items['tiled_prep']['correctness']=str(R/'joint-overflow-checks/result.json')
items['bitplanes'].update(reason='Eight-plane modern N1/N3 conversion-inclusive screen loses. A4-specific four-plane implementation passes independent all-mode native operations and required-logit/cache checks.5alternating pairs across3changing captureDEV contexts give planes4/joint0.9938[0.9772,1.0116], unresolved gain; retain simpler qualified joint runtime.',evidence=str(R/'replay-methods/result.json'),full_speed=load(R/'replay-methods/result.json')['planes4_over_joint'])
items['lut'].update(reason='Modern LUT4 improves isolated N3 gate/up, loses down. Selective native LUT4 full-forward1.301ms loses to joint+tiled retained combination; not promoted.',evidence=str(R/'replay-tranche6/lut02-clean.json'))
items['resident_int8']['reason']='Actual-code projection benchmark measured expanded residency: N3 gate/up improves, down loses. One405MiB full-model sign expansion would add405MiB while packed data remains; replacing signs trades+354.375MiB. Joint packed gate/up plus tiled preparation wins full-forward without this residency; no full-model expanded-residency deployment was built or claimed.'
items['layout_prepack']['reason']='Bounded feasibility: row-major packed sign words already load contiguously. Stronger expanded-residency projection limit measured; no additional coalescing defect or competitive full-pipeline rationale established. Arbitrary binary-layout search not performed.'
items['packed_a4']['reason']='Direct INT8 codes are7.5KiB at primary gate/up N3 and27KiB at downA8. No packed-nibble/upcast detour exists. Tested native-INT4 interface lowers to two INT8 MMA sites, requiring extra conversion. No native smaller-grid path to retain under fixed precision.'
items['final_native'].update(evidence=str(R/'native-final/result.json'),correctness=str(R/'charts-final/proposal-audit.json'),reason='All plain/traced requests target-identical; original/optimized greedy proposal tokens, accepted/proposed/round/prefix counts match.5alternating reps with2same-process warmups on consumed DEV12.')
items['replay'].update(evidence=str(R/'replay-final/result.json'),correctness='540actual forwards including warmups;180measured regions;30per-domain/repetition arm pairs; all required-logit/cache bytes exact.',reason='Same external native features/history and3changing contexts; each model reconstructs its own cache every iteration. All measured regions have one warm CUDA graph. Host outlier retained.')
items['publish'].update(evidence=str(R/'charts-final/comparison.html'),reason='Exact JSON/CSV, standalone interactive HTML and five PNG/SVG pairs generated from final data.')
# Fill comparable performance fields without inventing isolated effects for audits/dead-work items.
variant={'copies':'copy','known_bound':'bound256','tiled_prep':'tiled','cross_dp4a':'cross','signed_mma':'mma','silu_pack':'silu04','residual':'residual03','joint_gate_up':'joint','prenorm':'prenorm','head_anchor':'head02','last_anchor_ffn':'lastffn02','lut':'lut02'}
for key,name in variant.items():
 row=next(x for x in data['ablations'] if x['variant']==name);items[key]['full_speed']=dict(screen_gpu_median_ms=row['gpu_median_ms'],scope=row['scope'],evidence=row['evidence'])
for key in ['warp_block','row_position_tiling','resident_int8','fp16_codes','bitplanes','lut']:
 items[key]['isolated_speed']=dict(evidence=str(R/'micro2/result.json'),scope='7CUDA-event repetitions of32graph calls;N1/N3;complete prep+dot and separate stages;conversion costs included;resident weight initialization excluded and memory disclosed')
for k,x in items.items():
 x['source']=['docs/EXECUTION_OPTIMIZATION.md','scripts/execution_campaign/']
 x['attempts']=[x['evidence']]
 if x['correctness'] is None:x['correctness']='Native integer/affine, required-output/cache and proposal gates for retained exact changes; no qualification/promotion claimed for unimplemented or rejected paths.' if x['status']=='retained' else 'See actual attempt evidence; audit/scope-only items have no new arithmetic to qualify.'
 if x['memory'] is None:x['memory']='No model-format/storage change. Stage/workspace memory included in native sampled peaks; no separate exact per-stage peak claimed.'
 if x['full_speed'] is None:x['full_speed']='Not independently measurable/applicable for this audit or scope disposition; final combined runtime and fair Q4 comparison are separately measured.'
 if x['isolated_speed'] is None:x['isolated_speed']='Not separately isolated; actual native screen or concrete feasibility/scope evidence linked.'
items['resident_int8']['memory']='405MiB expanded signs vs50.625MiB packed;actual microbench retains both, potential replacement delta354.375MiB.'
items['fp16_codes']['mechanism']=str(R/'post-checks/fp16-dispatch.json')
items['fp16_codes']['memory']='810MiB expanded FP16 signs across18matrices;projection experiment only. Activation conversion charged each call.'
items['tiled_prep']['memory']='Maximum432-byte N3 per-down partial-sum workspace, backend-pool reused; plus existing codes/scales/sums.'
ledger.update(status='INVENTORY_DISPOSITION_COMPLETE',items=list(items.values()),combined_final_comparisons=data['comparisons'],native_correctness=str(R/'charts-final/proposal-audit.json'),replay_correctness=str(R/'replay-final/result.json'))
assert not any(x['status'] in ('pending','tested','qualified') for x in ledger['items']);(R/'ledger.json').write_text(json.dumps(ledger,indent=2)+'\n');(E/'ledger.json').write_text(json.dumps(ledger,indent=2)+'\n')
text='# Execution inventory dispositions\n\nAll38agreed opportunities have an explicit disposition. “Evaluated” means a concrete trace/source/conditional-reserve audit found no additional change to retain; it is not a claim every possible kernel design was implemented. Final comparison and evidence are in [report.md](report.md).\n\n| Opportunity | Disposition | Evidence and reason |\n|---|---|---|\n'
for x in ledger['items']:text+='| '+x['opportunity']+' | '+x['status']+' | '+x['reason'].replace('|','/')+' [Evidence]('+x['evidence']+') |\n'
(E/'ledger.md').write_text(text)

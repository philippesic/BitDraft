#!/usr/bin/env python3
"""Reconcile retained historical segment/feature inventories without GPU or data reads.

Reads compact historical receipts and stats referenced payloads. Does not open
held-out prompts, model weights, teacher arrays, or change historical catalogs.
"""
from __future__ import annotations

import collections
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'runs/unattended-training-preparation-20261028/gpu-resume-20261028'
CATALOG = BASE / 'catalog-source08.json'
RECEIPTS = BASE / 'source20-linux-authentic-fixture'
OUT = ROOT / 'experiments/design-readiness-20261104'

# These are review/planning estimates, never measurements or automatic allocations.
PROFILES = {
    'row_scale_continuation': ('scale_fit', 'current_scale_path', 'Exact inherited scales/Adam/RNG/cursor, same native targets and fixed sign buffers.', 'Reuses current representation; old DSpark result does not estimate modern EAGLE benefit.', 'Matched CE continuation from one saved modern scale-only parent with fixed LR horizon; native checkpoint at1920 then longer only if informative.', '10–30min GPU including repeated consumed-DEV serving; existing captures assumed.'),
    'joint_sign_scale': ('sign_optimization', 'current_representation_sign_branch', 'F32 latent signs, clipping/STE, moment state, finite gradients and deployed-sign export; Reuse the already qualified/mature selected-O branch; all-nine-sign state/export remains unqualified.', 'Same weight format can support trained signs; full-sign transfer is unmeasured on current successful lineage.', 'Reuse paid selected-O zero-export,570/727/4981 deployed-flip, mature-resume and native-admission receipts. For a new optimizer question, use that existing branch against its matched control; full-sign expansion needs separate qualification.', 'Paid selected-O qualification can be reused; new bounded probe20–60minGPU if informative. Full-sign memory/optimizer projection remains design-dependent.'),
    'joint48_initialization': ('sign_optimization', 'current_representation_sign_branch', 'Historical48 trainable tensors and BF16/C0 initializer contracts are DSpark-specific; modern nine-projection shapes differ.', 'Initialization/optimizer concepts transfer; historical trainable scope and rates do not.', 'Reuse current selected-O paid source/checkpoint/native evidence and preserved sign/norm guards. Extend an initializer question only if existing comparisons leave it material; do not rerun successful plumbing.', 'CPU evidence reuse<1h; conditional20–60min current-branch probe. No released-BF16 architecture migration under current scope.'),
    'norm_only': ('normalization', 'current_representation_trainability_adapter_needed', 'Actual deployed norm ownership/offset form and frozen target feature taps; modern scale-only trainer freezes norms.', 'No weight-format change; norm placement and optimizer trainability depend on architecture.', 'Audit norm gradients/export ownership before one current-model norm-only versus scale-only matched probe.', '1–3h source/admission work then10–30min GPU, estimate.'),
    'first_error_margin': ('reachable_objectives', 'current_scale_path_loss_adapter', 'Native reachable labels/first error, fixed-history masks and original denominator; unsupported labels explicit.', 'Concept is architecture-general; coefficients and reachable masks need recalibration.', 'Current scales CE versus CE+bounded native-reachable margin at1920 updates, then a longer losing sample if learning curve remains active.', '15–45min GPU incl five-repeat DEV; loss adapter CPU work first.'),
    'hidden_kd': ('teacher_distillation', 'current_scale_path_capture_adapter', 'Exact current target hidden taps and current actor output semantics; historical approximate hidden teacher must not be silently reused.', 'Loss concept general; tensor meaning/taps and normalization are architecture-specific.', 'Check native hidden alignment, then CE versus one norm-calibrated hidden loss on identical captured TRAIN examples.', '1–3h adapter/capture audit;20–60min GPU if current tensors sufficient.'),
    'grouped128_r512': ('group_scale_representation', 'reserved_representation_change', 'Within-row group128 scales, fixedR512 rotation, sign layout, exporter metadata and native consumer; distinct from current per-output-row vectors.', 'Historical grouped-own-initializer gain exists; current-candidate group-scale effect untested and reserved.', 'Preparation only: count groups/bytes/optimizer state and map exporter/kernel changes; after user choice, zero-export and one matched native admission.', '1–3engineering days plus qualification; current group sweep prohibited.'),
    'ce_depth': ('reachable_objectives', 'current_scale_path_loss_adapter', 'Current actor step shift, actual proposed depth, unreachable suffix and unsupported-label denominators.', 'Depth-loss concept general; seven computed historical rows versus D5 is not modern D2.', 'Current CE versus reached/prefix-depth weighting with identical updates/data and native DEV full requests.', '15–45min GPU; existing current depth2 serving retained.'),
    'data_coverage': ('data_replay_coverage', 'current_scale_path_data_adapter', 'Original group split, source ancestry, native committed histories, same label exposures and capture settings.', 'Broader data is a hypothesis; historical and recent equal-update comparisons do not establish diversity benefit.', 'One matched existing-data replay/coverage arm versus unchanged control; log distinct groups/rows and use consumed DEV only.', '20–60min GPU; new capture cost separate and avoid fresh confirmation during selection.'),
    'released_restart': ('initialization', 'historical_initializer_only', 'Authentic released checkpoint, original scale fitting, inherited/reset optimizer policy; modern architecture held fixed.', 'Restart concept general, concrete historical initializer is incompatible with present candidate.', 'Audit preserved initializer ancestry and exact zero-update export; do not switch current architecture/precision.', 'CPU/source audit<1h; any future restart qualification budget must include capture and training.'),
    'input_gains': ('input_representation', 'reserved_representation_change', 'Post-rotation gains require graph/export support; positive scaling beforeA1 may not change sign bits.', 'Potential representation change; current packedA8 row operator does not implement this historical branch.', 'Preparation only: identify whether gain can fold without changing A8 codes; exact arithmetic and native graph contract after user decision.', '1–3engineering days; GPU cost unknown until implementation.'),
    'eal_prefix': ('reachable_objectives', 'current_scale_path_loss_adapter', 'Prefix survival/first-rejection mask, preserved history ancestry, same native verifier and learning-rate horizon.', 'Surrogate concept general; aggregate proxy improvement does not predict verifier-round benefit.', 'Current CE versus native-prefix weighted objective at meaningful matched exposure, plus full-request DEV checks.', '15–45min GPU; no universal early-pruning rule.'),
    'serving_conditioning': ('data_replay_coverage', 'current_scale_path_data_adapter', 'Actual chat template/thinking mode/committed serving histories and original source groups.', 'Conditioning mismatch general; historical answer-vs-reasoning rates cannot be mixed with current modern cohorts.', 'Check current capture/request settings and one matched replay sampler probe on existing TRAIN states.', '15–45min GPU if captures retained; additional capture accounted separately.'),
    'sign_lr_scope': ('sign_optimization', 'current_representation_sign_branch', 'Original latent-sign reference, full savedAdam state, selected parameter families and sign-flip diagnostic.', 'Optimizer/LR scope concept useful across models; current scale-only success has no full-sign validation.', 'Reuse already measured selected-O570/727/4981 flips, exact mature resume and native549/HTTP qualification; its1e-3/3e-3 LR evidence already exists. Test only a remaining distinct optimizer question; full-sign scope stays unqualified.', 'Existing selected-O LR evidence reuse<1hCPU; conditional new bounded probe20–60minGPU. Full-sign long-QAT reserved.'),
    'parent_kl': ('retention', 'current_scale_path_capture_adapter', 'Frozen parent logits on identical histories/support, original model pin, temperature/tail accounting and selection-only DEV.', 'Retention concept general; parent distribution is architecture/precision-specific and cannot be recycled blindly.', 'Current CE versus one capped parent-KL term with current parent reference and native decision retention audit.', '20–60min GPU; cached-reference capture/setup separate.'),
    'processed_f16': ('teacher_distillation', 'historical_teacher_requires_current_rebind', 'Historical processedT1 FP16 teacher is not current Q4 target; soft-label processing must retain actual target/verifier semantics.', 'Distillation may generalize as an idea; changing current teacher precision is not authorized.', 'Audit raw-versus-processed informational mass; only current native target soft-label adapter may be tested now.', 'CPU audit<1h; current-teacher experiment20–60min GPU after data support.'),
    'residual': ('floating_corrections', 'reserved_architecture_change', 'Rank8/16 factors, F16 rounding/zero-add semantics, additional operators and export/native consumer.', 'Historical all-output fit improves native anchors yet request benefit inconsistent; modern effect unknown.', 'Preparation only: projection sensitivity/storage/launch estimate and exact zero-correction integration plan; no residual install now.', '1–3engineering days; historical30min training is not present runtime estimate.'),
    'raw_f16_t2': ('teacher_distillation', 'historical_teacher_requires_current_rebind', 'RawT2 top64+complete tail, informative alternatives, exact native target/model/capture pins.', 'General distillation concept; historical FP16 teacher and Qwen3 labels do not validate modern Q4 teacher change.', 'Audit retained mass and align current target logits; one current-teacher CE+soft objective only if compatible capture exists.', '1–3h CPU/capture setup plus20–60min GPU; historical teacher raw regeneration may be needed.'),
    'reached_ce': ('reachable_objectives', 'current_scale_path_loss_adapter', 'Reachable-prefix mask from actual native proposals; original per-step normalization and label support.', 'General objective idea; measured historical local gain did not establish modern throughput.', 'CE versus reached-only CE on current native-captured rows at fixed exposure with five-repeat consumed DEV.', '15–45min GPU.'),
    'prefix_protection': ('retention', 'current_scale_path_loss_adapter', 'Frozen parent native-success positions, capped margin floors, current same-history labels and supported vocabulary.', 'Current replay/protection effect must be separated from extra updates/data; no zero-loss gate.', 'Current CE+prefix loss versus matched CE, log recovered/introduced losses and native full requests.', '15–45min GPU plus bounded reference pass.'),
    'reached_ce_protection': ('retention_combination', 'current_scale_path_loss_adapter', 'Reached CE and native protection references on same original groups; coefficients prefrozen.', 'Combination may interact; historical fixed mix failed domain tradeoff, not all combinations.', 'At most one compatible combination against best singleton and unchanged control; extend a loser only with informative gradients.', '20–60min GPU including checkpoint serving; use existing DEV.'),
    'rotations': ('input_representation', 'reserved_representation_change', 'Learned pair-angle vectors, rotary ordering, A8 quantization, exporter/graph and identityC0.', 'Basis optimization useful to discuss; historical native+1/180 is weak and modern implementation differs.', 'Preparation only: identify graph positions and cache/export/memory costs; no angle training now.', '1–3engineering days then bounded native qualification after user selection.'),
    'retention': ('retention', 'current_scale_path_loss_adapter', 'SavedAdam direction under current trainable scope, parent-success masks, correct sign/scale gradients.', 'Historical local direction certificate did not predict native gain; current scale-only dynamics differ.', 'Gradient/direction audit plus one matched current CE+retention run; no certificate-only readiness claim.', '20–60min GPU plus CPU test.'),
    'plane_amplitude': ('extra_binary_planes', 'reserved_representation_change', 'Additional binary planes and bounded row amplitudes, native bit-plane op/export plus deployment precision accounting.', 'More capacity hypothesis; historical native plateau does not prove global rejection.', 'Preparation only: count payload/operators and derive zero-amplitude parity/gradient test for later user-selected design.', '1–3engineering days; no extra-plane experiment now.'),
    'gradient_control': ('sign_optimization', 'current_representation_sign_branch', 'Elastic/trust/selective surrogate needs trainable latent signs; scale-only current success has no sign gradient. Current selected-O paid branch already has meaningful gradients, deployed flips, mature resume/export/native admission.', 'Training-rule concept general but sign training prerequisite not established for all current projections.', 'Reuse qualified mature selected-O branch and its paid controls for at most one loss-equivalent gradient intervention at meaningful exposure. Do not repeat sign/export/resume plumbing; all-nine-sign support remains separate.', 'Evidence reuse<1hCPU; conditional20–60minGPU gradient-control probe, larger full-sign projection memory still needs measurement.'),
    'calibration': ('scale_fit', 'current_scale_path_calibration_adapter', 'Authentic current A8 operands/target projection values, row signs fixed and train/eval groups separated.', 'Fixed-sign row-scale fitting can be compatible; historical weight-MSE/A8 pilots are not native generalization proof.', 'One train-only row-scale calibration against unchanged current model; exact export/native DEV, no group granularity change.', '1–3h CPU implementation/projection audit plus15–45min GPU.'),
    'mos': ('multi_binary_capacity', 'reserved_architecture_change', 'Rank32 mixture/binary factorization, extra operators, transition optimizer and original decodedF32 consumer.', 'Historical actor-only tiny pilot; no packedW1/native request validation at catalog stage.', 'Preparation only: map factor sizes/sign packing/operator ancestry and current architecture dependencies.', '2–5engineering days; current adoption/comparison prohibited.'),
    'double_binary': ('multi_binary_capacity', 'reserved_representation_change', 'Two-stage binary factorization/rank1984 variant and native intermediate scaling; new export/graph.', 'Historical initializer changed actor score before72updates; cannot attribute to training or transfer.', 'Preparation only: count shape/MAC/payload/optimizer costs and zero-path semantics for future decision.', '2–5engineering days; model-backed native qualification later.'),
    'ternary': ('ternary_representation', 'reserved_representation_change', 'Zero state/thresholds, ternary packing/kernel, scale/STE and exporter format; incompatible current W1 format.', 'Historical tiny actor pilot unchanged; no native packed ternary deployment result.', 'Preparation only: map supported kernels/exceptions and exact ternary decoder fixtures.', '2–5engineering days; format experiment reserved.'),
    'eal': ('reachable_objectives', 'current_scale_path_loss_adapter', 'Native emitted/accepted positions and current actor shift; originalD5 objective not copied blindly toD2.', 'Loss mechanism general; historical narrow/longer fit gains did not predict HTTP rounds.', 'Current plainCE versus one survival-weighted CE control at1920 and informative later checkpoint; reused DEV native requests.', '15–45min GPU.'),
    'vat': ('value_auxiliary_objective', 'current_training_auxiliary_adapter_needed', 'Survival labels, optional authentic teacher distributions and value-head gradients; training-only head changes checkpoint scope.', 'Verifier-aware concept general; source adaptation is greedy recorded-prefix surrogate, not paper-faithful stochastic result.', 'CPU verify supervision/no label leakage and train-only head export exclusion; after current implementation exists, one meaningful probe versusCE.', '1–3h adapter then20–60min GPU; no inference architecture change intended.'),
    'dpal': ('distribution_objective', 'current_scale_path_capture_adapter', 'Full current target distributions, detached overlap weights, Renyi1/2 and same vocabulary/support.', 'Stochastic-overlap motivation is not proof for greedy exact-match native verifier; current Q4 soft capture needed.', 'Check current soft target availability, then one detached overlap loss versusCE with fixed coefficient and native checkpoint serving.', '1–3h adapter/capture audit plus20–60min GPU.'),
    'edr': ('round_cost_objective', 'current_scale_path_replay_adapter', 'Dense absolute round starts, six detached continuation values, parent/horizon pins and Bellman semi-gradient semantics.', 'Recorded token-confidence/independence surrogate is not exact native stochastic acceptance; runtime savings unmeasured modern.', 'Build current consumedTRAIN continuation cache from actual native rounds; compare one semi-gradient objective againstCE.', '2–4h adapter/cache audit plus20–60min GPU; no new verifier policy.'),
    'factor_head': ('output_head_architecture', 'reserved_architecture_change', 'Rank320 head factors, vocabulary mapping, candidate set/logit confidence semantics and fused native indexed consumer.', 'Historical cold factor head severely regressed native rates; trained alternatives remain an architecture question.', 'Preparation only: map current50000-row packed head cost, factor state/storage and exact candidate/recurrent admission gates.', '2–5engineering days; no head swap now.'),
}

# Supplemental F IDs include fixed fits, kernels, policies, and fidelity work that
# never produced a training-segment ID. Preserve them rather than inventing rows.
FEATURE_MAP = {
    'F01':'teacher_distillation','F02':'reachable_objectives','F03':'retention','F04':'retention_combination',
    'F05':'floating_corrections','F06':'floating_corrections','F07':'floating_corrections','F08':'floating_corrections','F09':'floating_corrections',
    'F10':'exact_binary_kernel','F11':'sparse_corrections','F12':'sparse_corrections',
    'F13':'extra_binary_planes','F14':'extra_binary_planes','F15':'extra_binary_planes','F16':'extra_binary_planes','F17':'extra_binary_planes',
    'F18':'affine_bias','F19':'output_shortlist','F20':'output_shortlist','F21':'output_shortlist',
    'F22':'draft_policy','F23':'draft_policy','F24':'draft_policy','F25':'draft_policy',
    'F26':'runtime_transfer','F27':'capture_fidelity','F28':'input_representation','F29':'retention','F30':'floating_corrections','F31':'teacher_distillation','F32':'tree_verifier',
}
EXTRA_PROFILES = {
    'exact_binary_kernel': ('current_kernel_diagnosis_only','Current fixedA8 DP4A kernel differs from historicalSM75 binaryMMA; preserve exact arithmetic and native dispatch.','Audit operation shapes/packing/head attribution before new kernel work; no binary-format switch.','CPU/source<1h; prospective measured kernel pilot only if bottleneck warrants.'),
    'sparse_corrections': ('reserved_architecture_change','Sparse floating payload, layout/indexing and actual GPU launches; historical synthetic cost gate not modern acceptance.','Preparation only: map sparse payload/zero-correction/export/operator contracts and cheapest component test.','1–3engineering days plus future native model qualification.'),
    'affine_bias': ('reserved_representation_change','Bias/center adds input-sum or residual arithmetic; current row-scaled symmetric format lacks that field.','Preparation only: distinguish folded bias from affine binary centers and map metadata/consumer cost.','1–3engineering days plus future native qualification.'),
    'output_shortlist': ('reserved_architecture_change','Current50000-row support, stable absolute-token mapping, top-k/confidence and fallback semantics; head routing changes drafter graph.','Preparation only: cost/coverage map on TRAIN metadata, no current head modification.','2–5engineering days; historical static/dynamic recall is not current evidence.'),
    'draft_policy': ('current_runtime_policy_not_implemented_here','Same target/verifier and exact committed histories; policy changes proposal counts and trajectories; needs matched total latency.','Audit old policy failure and current depth2 costs; only propose a bounded source-specific native policy probe, no generic framework.','1–3engineering days if selected; existing policy6prompt smoke insufficient.'),
    'runtime_transfer': ('current_runtime_audit','Resolved backend candidate transfer path, top10 stable sampling and graph outputs; old source may already remove transfer.','Inspect actual current loaded graph/logit outputs; avoid credit for already-present optimization.','CPU/source<1h; native trace10–20min if material unknown.'),
    'capture_fidelity': ('current_native_capture_path','Native capture/token/history shift, exact target/helper labels, KV positions/cache rollback and numerically admitted supported labels.','Rehash current data/model source; one native feature/label capture and bounded actor/native decision replay; do not chase harmless floating drift.','15–45min actual capture/replay plus source audit.'),
    'tree_verifier': ('reserved_verifier_architecture_change','Ancestor mask, physicalKV retention/remap and selected-branch actual features; changes verifier infrastructure and target batch geometry.','Preparation only: retain31,250CPU verifier cases and enumerate missing model-backed logits/taps/physicalKV native gates.','3–7engineering days estimated before reliable current-runtime confirmation.'),
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def normalize(value: str) -> str:
    if '/binary-eagle-decoding/' in value:
        return value.split('/binary-eagle-decoding/', 1)[1]
    return value


def resolve(value: str) -> Path | None:
    rel = normalize(value)
    possibilities = [ROOT / rel, RECEIPTS / rel]
    if value.startswith('/'):
        possibilities.insert(0, Path(value))
    # Source receipts contain migrated original-PC paths. Never guess a different
    # filename or treat a nearby endpoint as the missing file.
    for p in possibilities:
        if p.is_file():
            return p
    return None


def walk(value, pointer=''):
    yield pointer, value
    if isinstance(value, dict):
        for k,v in value.items():
            yield from walk(v, f'{pointer}/{k}')
    elif isinstance(value, list):
        for k,v in enumerate(value):
            yield from walk(v, f'{pointer}/{k}')


def receipt(src: dict) -> dict:
    p=resolve(src['path'])
    return {'original_path':src['path'], 'expected_sha256':src.get('sha256'),
            'local_path':relative(p) if p else None,
            'status':('hash_verified' if sha(p)==src.get('sha256') else 'hash_mismatch') if p else 'unavailable_at_resolved_paths',
            'bytes':p.stat().st_size if p else None}


def native_summary(s: dict) -> str:
    if s['role']=='control':
        return 'matched_control_not_independent_feature_effect'
    if s.get('native_result')=='unmeasured':
        return 'actor_or_proxy_only_no_native_effect_at_catalog_stage'
    if s.get('request_result')=='unmeasured':
        return 'native_or_proxy_local_only_full_request_effect_unmeasured'
    if 'repeated' in s.get('request_result',''):
        return 'historical_repeated_within_panel_effect_only'
    return 'historical_local_or_onepass_effect_inconclusive_for_modern_transfer'


def main():
    cat=json.loads(CATALOG.read_text())
    assert len(cat['segments'])==len({s['id'] for s in cat['segments']})==98
    assert set(PROFILES)=={s['recipe'] for s in cat['segments']}
    receipts=[receipt(s) for s in cat['authenticated_sources']]
    assert len(receipts)==51 and all(s['status']=='hash_verified' for s in receipts)
    byoriginal={r['original_path']:r for r in receipts}
    index=receipt(cat['source_index']); inventory=receipt(cat['inventory']); supplement=receipt(cat['exposure_supplement'])
    assert index['status']==inventory['status']=='hash_verified'
    old_inventory=json.loads((ROOT/inventory['local_path']).read_text())
    purge=json.loads((ROOT/'runs/storage-purge-20261104-01/manifest.json').read_text())
    purged={str(Path(v['path'])) for v in purge['files']}
    purged.update(str(Path(v['path'])) for v in purge['aliases'])
    source_objects={r['original_path']:json.loads((ROOT/r['local_path']).read_text()) for r in receipts if r['local_path'].endswith('.json')}
    families=[]
    for name,profile in PROFILES.items():
        mechanism,boundary,deps,transfer,test,cost=profile
        members=[s for s in cat['segments'] if s['recipe']==name]
        families.append({'id':name,'mechanism_family':mechanism,'segment_ids':[s['id'] for s in members],
                         'count':len(members),'current_candidate_boundary':boundary,'dependencies':deps,
                         'cross_design_usefulness':transfer,'cheapest_decisive_test':test,
                         'prospective_cost_estimate':cost,'cost_estimate_is_not_measurement':True,
                         'historical_TRAIN_seconds':sum(s['new_train_seconds'] for s in members),
                         'historical_updates':sum(s['new_updates'] for s in members)})
    segments=[]
    source_asset_audits={}
    for s in cat['segments']:
        r=byoriginal[s['source_pin']['path']]
        src=source_objects.get(s['source_pin']['path'])
        matches=[]
        if src is not None:
            matches=[ptr for ptr,v in walk(src) if isinstance(v,(int,float)) and not isinstance(v,bool) and abs(v-s['new_train_seconds'])<1e-7]
        hostspec=BASE/'source20-host-admission01/specs'/f"{s['id']}.json"
        binding=dict(s['executable_binding'])
        refs=[]
        if src is not None:
            seen=set()
            for ptr,v in walk(src):
                if not isinstance(v,str) or not v.startswith(('/home/philip/','/mnt/c/','/Users/pippo/','runs/','experiments/')) or '\n' in v or len(v)>600:
                    continue
                if v in seen or not re.search(r'\.(f32|f16|bf16|pt|npz|npy|gguf|json|jsonl|md|bin)$',v):
                    continue
                seen.add(v); path=resolve(v)
                # Payload status checks are stat-only; no held-out/raw/model read.
                ispurged=str(ROOT/normalize(v)) in purged or v in purged
                refs.append({'original':v,'pointer':ptr,'local':relative(path) if path else None,
                             'status':'retained_path_stat_only' if path else 'explicit_purge_manifest_match' if ispurged else 'unavailable_unresolved_not_assumed_purged'})
        asset_key=r['expected_sha256']
        source_asset_audits.setdefault(asset_key, {
            'source_receipt_path':r['local_path'],
            'audit_scope':'All source strings matching recognized path/extension patterns, stat-only; not a transitive payload/reproduction guarantee.',
            'reference_count':len(refs),
            'counts':dict(collections.Counter(v['status'] for v in refs)),
            'unresolved_examples':[v for v in refs if v['local'] is None][:8],
            'explicit_purge_matches':[v for v in refs if v['status']=='explicit_purge_manifest_match'],
            'retained_payload_examples':[v for v in refs if v['local'] and re.search(r'\.(f32|f16|bf16|pt|npz|npy|gguf|bin)$',v['original'])][:5],
        })
        family=next(f for f in families if f['id']==s['recipe'])
        segments.append({'id':s['id'],'name':s['name'],'recipe_family':s['recipe'], 'mechanism_family':family['mechanism_family'],
                         'historical_campaign_label':s['family'],'role':s['role'],
                         'evidence':{'catalog_path':relative(CATALOG),'catalog_segment_index':cat['segments'].index(s),
                                     'authenticated_receipt':r,'actual_TRAIN_numeric_source_pointers':matches,
                                     'TRAIN_check':'exact_numeric_receipt_match' if matches else 'catalog_and_hashed_receipt_context_only_no_numeric_pointer_match',
                                     'implementation_only_spec':{'path':relative(hostspec),'sha256':sha(hostspec),'warning':'Later qualification spec; NOT settings of original measured historical run.'} if hostspec.is_file() else None,
                                     'referenced_asset_audit_key':asset_key},
                         'actual_test':{'new_updates':s['new_updates'],'new_TRAIN_seconds':s['new_train_seconds'],
                                        'exposure':s['exposure_accounting'],'initializer':s['parent'],
                                        'trainable_scope':s['trainable_scope'],'historical_binding':binding,'settings':s['knobs']},
                         'outcome':s['outcome'],'native_result':s['native_result'],'request_result':s['request_result'],
                         'interpretation':native_summary(s),'current_candidate_effect':'not_demonstrated_by_this_historical_segment',
                         'dependencies':family['dependencies'],'implementation_readiness':family['current_candidate_boundary'],
                         'cross_design_usefulness':family['cross_design_usefulness'],
                         'cheapest_decisive_test':family['cheapest_decisive_test'],'prospective_cost_estimate':family['prospective_cost_estimate'],
                         'raw_replay_limit':'Stat-only reference audit does not prove all transitive training payloads, original states or raw arrays retained. Historical authorized purge removed old tensors/intermediates; exact historical replay may require regeneration and is not promised.',
                         'scientifically_eliminated':False,'original_catalog_record':s})
    supplemental_search={
        'scope':'runs/unattended-training-preparation-20261028, docs, experiments and existing checkouts; recognized compactJSON/Python/Markdown only, max2MiB/file; models/runtime/build/Git/venv/node_modules excluded. No weights,raw arrays,held-out inputs or full-disk search.',
        'query':'exact exposure-supplement01 filename or94978ba461a937406c8ae6cabc9777ef7a2ca5b414b05982ef8a704667c17010 pin',
        'result':'Only original catalog and derived inventory reference the pin/name. No original or renamed compact supplement source found in bounded search. Six catalog IDs retain supplementary provenance pointers; missing metadata is not attributed to raw-data purge.',
        'six_referencing_segment_ids':[s['id'] for s in cat['segments'] if s['exposure_accounting'].get('supplement')],
        'direct_feature_path_check':'Stat original/migrated authentic-fixture paths plus exact campaign/root→campaign-root transformation. No additional existing direct links found by that transformation; all32 remain backed by authenticated feature-results compact receipt. Alternative nearby files are not treated as the missing original.',
    }
    feature_source=next(r for r in receipts if r['original_path'].endswith('/w1a8-openitems-closeout-20261023/feature-results.json'))
    feature_data=source_objects[feature_source['original_path']]
    assert len(feature_data['features'])==32
    features=[]
    for f in feature_data['features']:
        mech=FEATURE_MAP[f['id']]
        representative=next((v for v in families if v['mechanism_family']==mech),None)
        if representative:
            boundary,deps,test,cost=(representative[k] for k in ['current_candidate_boundary','dependencies','cheapest_decisive_test','prospective_cost_estimate'])
        else:
            boundary,deps,test,cost=EXTRA_PROFILES[mech]
        aliases=[s['id'] for s in cat['segments'] if s['id'].startswith(f['id']+'_')]
        if f['id']=='F16': aliases=['amplitude900','amplitude3600']
        if f['id']=='F17': aliases=['amplitude_broad_control','amplitude_broad_treatment']
        direct=[]
        for ev in f.get('evidence',[]):
            p=resolve(ev)
            direct.append({'original':ev,'local_path':relative(p) if p else None,'status':'retained_path_stat_only' if p else 'unresolved_compact_receipt_retained'})
        features.append({'id':f['id'],'name':f['feature'],'mechanism_family':mech,'training_segment_aliases':aliases,
                         'evidence':{'authenticated_feature_receipt':feature_source,'feature_array_index':int(f['id'][1:])-1,'direct_links':direct,'direct_link_count':len(direct),'direct_links_retained':sum(v['local_path'] is not None for v in direct),'compact_closeout_receipt_available':True},
                         'actual_test_and_settings':f['exploration_completed'],'historical_training_cost':f['training'],
                         'outcome':f['acceptance_or_proxy'],'full_request':f['full_request'],
                         'historical_disposition':f['disposition'],'scope_limitation':f['scope'],
                         'dependencies':deps,'implementation_readiness':boundary,'cross_design_usefulness':'Mechanism-level hypothesis only for current modern candidate; historical source/precision/target/cohort differ.',
                         'cheapest_decisive_test':test,'prospective_cost_estimate':cost,'original_feature_record':f})
    totals={'canonical_segment_ids':98,'recipe_families':len(families),'broad_campaign_labels':len({s['family'] for s in cat['segments']}),
            'controls':sum(s['role']=='control' for s in cat['segments']),'treatments_or_continuations':sum(s['role']!='control' for s in cat['segments']),
            'original_source_receipts':len(receipts),'rehash_verified_source_receipts':sum(r['status']=='hash_verified' for r in receipts),
            'TRAIN_numeric_source_matches':sum(bool(s['evidence']['actual_TRAIN_numeric_source_pointers']) for s in segments),
            'historical_TRAIN_seconds':sum(s['new_train_seconds'] for s in cat['segments']),
            'historical_updates':sum(s['new_updates'] for s in cat['segments']),
            'historical_under_60_TRAIN_seconds':sum(s['new_train_seconds']<60 for s in cat['segments']),
            'historical_under_300_TRAIN_seconds':sum(s['new_train_seconds']<300 for s in cat['segments']),
            'unknown_unique_groups':sum(s['unique_groups'] is None for s in cat['segments']),
            'unknown_unique_windows':sum(s['unique_windows'] is None for s in cat['segments']),
            'supplemental_feature_ids':32,'features_with_training_segment_aliases':sum(bool(f['training_segment_aliases']) for f in features),
            'features_without_training_segment_aliases':sum(not f['training_segment_aliases'] for f in features),
            'scientifically_eliminated_by_this_audit':0,'new_GPU_seconds_this_audit':0}
    assert abs(totals['historical_TRAIN_seconds']-old_inventory['summary']['candidate_training']['seconds_sum'])<1e-7
    result={'schema':'design_readiness_backlog_reconciliation_v1','scope':'Historical evidence audit; no architecture/precision selection, no new model experiment.',
            'coverage':totals,'canonical_catalog':{'path':relative(CATALOG),'sha256':sha(CATALOG),'bytes':CATALOG.stat().st_size},
            'source_custody':{'retained_authenticated_receipts_root':relative(RECEIPTS),'receipts':receipts,'inventory':inventory,'source_index':index,'optional_exposure_supplement':supplement,'supplement_search':supplemental_search},
            'interpretation':['98 training segments are not98 mechanisms or98 feature IDs; controls and continuations are preserved.',
                              '34 is recipe-family count. The segment family field has13 broad campaign/date labels; these are not state contracts.',
                              '32-feature closeout includes analytic fits/kernels/policies absent from training-segment inventory. Its training aliases overlap98, so costs must not be added.',
                              'All51 original compact source receipts rehashed exactly in migrated authentic fixture. Missing original Mac paths are resolved custody aliases, not lost receipts.',
                              'Optional exposure supplement source file remains unresolved; six catalog records reference its pin. Supplementary exposure provenance is not independently rechecked; measured TRAIN totals independently match retained inventory.',
                              'Historical target/precision is Qwen3F16/DSpark mixedW1A8 with recorded exceptions; never compare its rates directly with modern Qwen3.5Q4_K_M/RTX5080 cohorts.',
                              'Authorized storage purge removed historical raw tensors/intermediates while retaining endpoints/compact evidence; raw exact replay is not claimed.',
                              'Later qualification specs and catalog implemented labels are source-preparation evidence, not proof of replay/input/export/native qualification.',
                              'Current successful modern lineage is nine per-output-row scale vectors with fixed binary signs; not nine scalar parameters, not grouped scales, not validated full-sign QAT.',
                              'No early rejection rule or universal ranking follows from these small reused-data pilots.'],
            'purge_evidence':'experiments/storage-purge-20261104.json',
            'current_selected_sign_evidence':{'path':'experiments/design-readiness-20261104/qat-source-audit.json','branch':'selected_attention_output_signs_plus_nine_row_scales','meaning':'Paid selected-O branch already has570deployed flips after1920updates and mature480update resumes yielding727/4981flips plus native549/HTTP qualification. Reuse this evidence; all-nine-sign QAT remains unqualified. Current successful scale-only lineage remains fixed signs.'},
            'current_effect_evidence':['experiments/w1a8-modern-native-20261103/natural-confirmation.json','experiments/w1a8-modern-native-20261103/duration-predictor.json'],
            'source_asset_audits':source_asset_audits,'family_profiles':families,'segments':segments,'supplemental_features':features}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'backlog-inventory.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    lines=['# Backlog evidence reconciliation', '',
           'All **98 canonical training-segment IDs** are retained, including18 controls and80 treatments/continuations. They cover34 recipe families, not98 independent mechanisms. The original `family` field is13 broad campaign/date labels; it is not an executable state contract. This audit also preserves every **F01–F32 feature ID**, including22 features that never produced a canonical training-segment ID. The two ledgers overlap; do not add their costs.', '',
           'The canonical content is [catalog-source08.json](../../'+relative(CATALOG)+'). Its51 compact source receipts, inventory and source index rehash exactly in the migrated [authentic evidence bundle](../../'+relative(RECEIPTS)+'). The catalog reproduces the inventory total: **35,611.652 measured TRAIN seconds and119,006 updates**. These are historical optimizer-training costs only: lineage initialization, capture, source work, exports/builds/evaluation and earlier pretraining are separate.80/98 segments ran under300TRAIN seconds;32 under60. Unique group counts are unknown for16 segments and window counts unknown for20; null is not zero.', '',
           'The [machine ledger](backlog-inventory.json) contains complete original per-ID records, exact original source pins, locally resolved links, actual update/seconds/exposure settings, numeric source pointers when found, raw-asset stat audits, dependencies, implementation boundaries, possible cross-design usefulness and prospective cheapest tests/costs. This is an evidence audit, not a new scientific screen. No item is permanently eliminated.', '',
           'The optional exposure-supplement source file is unresolved at its recorded and migrated paths. Six catalog entries reference its pin; supplementary exposure provenance is not independently rechecked. All98 TRAIN totals remain backed by retained catalog/inventory and51 hashed receipts. A bounded compact-source search over preparation runs/docs/experiments/existing checkouts found only catalog and derived-inventory references to the exact supplement filename/hash; models/runtime/build/Git/venv paths and files over2MiB were excluded. No original/renamed supplement source was found. Missing supplementary metadata is not attributed to the raw-data purge. A missing original Mac path alone does not mean evidence was purged. The [authorized cleanup](../storage-purge-20261104.json) removed old raw teacher/capture tensors and intermediate states while preserving final/best endpoints and compact reports. Some exact historical replay requires regeneration; nearby retained endpoints are not substitutes for deleted state. EachF01–F32 row retains its direct-link total/available count in the machine ledger. Direct originals remain unavailable where marked; exact campaign-root path transformation found no additional links. All32 have the authenticated complete compact closeout as retained evidence.', '',
           'Current modern evidence must be read separately: [fresh24 confirmation](../w1a8-modern-native-20261103/natural-confirmation.json) demonstrates+23.74% versus its initializer within that cohort. [Old/expanded3840/5760 diagnostic](../w1a8-modern-native-20261103/duration-predictor.json) leaves the recipe/duration rankings unresolved. Neither establishes a general causal repair or calibrated early-pruning rule. Current source enables nine **per-output-row scale tensors** with fixed signs; the nine tensors contain76,880 row scales, not nine scalar parameters. Historical grouped128+R512 is a different representation and remains reserved. The [current source/readiness audit](qat-source-audit.json) separately establishes paid selected-O sign training:570deployed flips, mature resumes with727/4981flips, native549 andHTTP qualification. Reuse those receipts and controls; no redundant sign/export/resume plumbing is a readiness requirement. All-nine-sign training/export/state support remains unqualified.', '',
           '## Selection-relevant findings', '',
           '- Native prefix/acceptance improvements repeatedly failed to improve full requests. Historical longerDPAL/VAT2160-update runs improve anchor counts yet increase HTTP verifier rounds (DPAL763→784final;VAT→804). Reused native full-request checkpoints belong in each informative current probe.',
           '- Grouped128+R512 improved its own initializer on two old panels, but remained below row parent andQ4. That does not compare group scales to the current modern nine-row-vector implementation. Research/storage/kernel/optimizer mapping is useful now; switching scale granularity requires the user.',
           '- Historical raw/processed teacher, retention and reached-prefix losses have local positives, nulls and domain tradeoffs. The72-update objective pilots used16.52–16.71TRAIN seconds; their nominal120-second allowance is not a training duration or calibrated rejection budget. Test a small informative compatible subset, one combination, and a meaningful longer apparent loser.',
           '- Fused rank8 historical correction has five-repeat evidence but only a small one-panel benefit; generalization and modern effect remain open. Tree verification(F32) is directional six-prompt/instrumented evidence with physicalKV/native qualification missing. Corrections, extra planes, shortlists and tree changes are future design work, not ready drop-ins.',
           '- F26 transfer removal already existed historically. Verify the actual current resolved graph before crediting a new optimization. F27 exposes real feature/history/label risks, but harmless float drift alone is not a blocker when native captured labels and bounded native trajectories pass.', '',
           '## Recipe families and cheapest next test', '',
           'Budgets below are planning estimates, not measured current-model costs, allocations or pruning deadlines. `reserved` means prepare only until the user decides architecture/precision. Historical source implementations do not prove current live input/export/native readiness.', '',
           '| Recipe family | IDs | Boundary | Cheapest decisive test | Estimated cost |',
           '| --- | ---: | --- | --- | --- |']
    for f in families:
        lines.append('| '+ ' | '.join([f['id'],str(f['count']),f['current_candidate_boundary'],f['cheapest_decisive_test'],f['prospective_cost_estimate']])+' |')
    lines += ['', '## Every canonical training-segment ID', '',
              'Settings/source pins and source-specific initializer/optimizer/representation contracts are complete in the machine ledger. Historical numbers below are kept within their original comparisons; they are not modern-cohort rates. Native/proxy negatives remain bounded observations.', '',
              '| ID | Recipe | Actual TRAIN s / updates | Historical outcome |', '| --- | --- | ---: | --- |']
    for s in cat['segments']:
        lines.append(f"| {s['id']} | {s['recipe']} | {s['new_train_seconds']:.3f} / {s['new_updates']} | {s['outcome'].replace('|','/')} |")
    lines += ['', '## Every original feature ID', '',
              'F IDs are a different unit from training segments. Training aliases and full original closeout records are preserved in the machine ledger. No alias means the feature was a fixed fit/runtime/policy/diagnostic with no corresponding canonical training segment.', '',
              '| Feature ID | Feature | Historical scope/result |', '| --- | --- | --- |']
    for f in features:
        lines.append(f"| {f['id']} | {f['name']} | {f['outcome'].replace('|','/')} |")
    lines += ['', '## Recompute / verification', '',
              'Run `python3 scripts/design_readiness/inventory_reconcile.py` from the repository. It rehashes the51 compact source receipts plus inventory/index, asserts unique98IDs and exact34recipe coverage, reconciles historical TRAIN total, resolves direct feature links, and stats referenced assets without opening model tensors, raw arrays or held-out prompts. All original records remain immutable. Later `source20-host-admission01/specs` files are explicitly labeled implementation-only; their short qualification horizons are never presented as original historical run settings.', '',
              f"Numeric TRAIN values have an exact JSON source match for{totals['TRAIN_numeric_source_matches']}/98 IDs; the remaining IDs use retained catalog plus hashed source context (including Markdown evidence), with that limitation explicit per row. This audit ran0GPU seconds. Source engineering and bounded CPU audit cost are not reconstructed as optimizer time.", '']
    (OUT/'backlog-audit.md').write_text('\n'.join(lines))
    print(json.dumps(totals,indent=2))

if __name__=='__main__':
    main()

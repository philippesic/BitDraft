"""Bounded CPU qualification of packed-only actor, compact state and scale-only export."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import resource
import sys
import time

assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
import numpy as np
import torch

P = Path('/home/philip/binary-eagle-decoding')
W = Path('/home/philip/.codex/worktrees/w1a8-modern-replay-20261103')
R = P / 'runs/w1a8-modern-compact-qualification-20261103-01'
sys.path[:0] = [str(W / 'src'), str(P / 'third_party/llama.cpp/gguf-py')]
spec = importlib.util.spec_from_file_location('modern_actor', R / 'modern_actor.py')
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
from gguf import GGUFReader
from w1a1_eagle.recurrent_qat import shared_round_hard_signs

torch.set_num_threads(2)
torch.set_num_interop_threads(1)
started = time.perf_counter()
phases = {}
references = {}


def pin(path):
    value = module.file_pin(path)
    references[value['path']] = value
    return value


def read(path):
    pin(path)
    return json.loads(Path(path).read_text())


def checked(ref):
    actual = pin(ref['path'])
    assert actual['sha256'] == ref['sha256']
    if 'bytes' in ref:
        assert actual['bytes'] == ref['bytes']
    return Path(ref['path'])


def write(name, data):
    with (R / name).open('x') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write('\n')


source_pins = [pin(R / f) for f in ('qualify.py', 'modern_actor.py', 'test_w1a8_modern_actor.py')]
source_pins += [pin(W / 'src/w1a1_eagle' / f) for f in ('native_step.py', 'recurrent_qat.py', 'recurrent_binary.py')]
reference = read(P / 'runs/w1a8-modern-recurrent-actor-20261103-01/report.json')
audit = read(P / 'runs/w1a8-modern-recurrent-actor-20261103-01/independent-audit.json')
checked(audit['producer_report'])
assert audit['status'] == 'SAVED_OUTPUT_AND_RECURRENT_STATE_AUDIT_PASS'
first_receipt = read(checked(reference['cases']['2']['admission']))
empty_lookup = module.ExternalNativeF32Embedding(
    lookup=lambda _: (_ for _ in ()).throw(KeyError('qualification embedding not yet bound')),
    vocab_size=248320, hidden_size=2560, target_identity=first_receipt['target_id'],
    target_sha256=first_receipt['evidence']['target']['sha256'],
    capture_sha256=first_receipt['evidence']['capture']['sha256'])
initializer = P / 'runs/w1a8-modern-packed-init-20261103-01/model.gguf'
config = P / 'runs/spec-shortlist-20261030/models/hf-eagle-4B/config.source.json'
stage = time.perf_counter()
actor = module.build_modern_actor(initializer, config, embedding_lookup=empty_lookup, scale_only=True)
phases['construct_seconds'] = time.perf_counter() - stage
assert len(actor.scale_parameters()) == 9
assert sum(x.numel() for x in actor.scale_parameters()) == 76880
assert not hasattr(actor.drafter, 'embed_tokens')
assert actor.adapter.head_dim == 160 and actor.adapter.rotary_dim == 40
assert actor.adapter.heads == 16 if hasattr(actor.adapter, 'heads') else True
cases = {}
for pid in ('2', '4'):
    case = reference['cases'][pid]
    receipt = read(checked(case['admission']))
    packet = receipt['evidence']
    meta = read(checked(packet['capture']))
    shape = tuple(meta['feature_shape'])
    arrays = {name: np.fromfile(checked(ref), dtype='<f4').reshape(shape)
              for name, ref in packet['arrays'].items() if name.startswith('layer-')}
    features = torch.from_numpy(np.concatenate([arrays['layer-%d.f32' % tap] for tap in (2,16,29)], axis=1))
    rows = {}
    for i, token in enumerate(meta['input_tokens']):
        value = torch.from_numpy(arrays['layer-0.f32'][i].copy())
        assert token not in rows or torch.equal(rows[token], value)
        rows[token] = value
    lookup = module.ExternalNativeF32Embedding(lookup=rows.__getitem__, vocab_size=248320, hidden_size=2560,
        target_identity=receipt['target_id'], target_sha256=packet['target']['sha256'],
        capture_sha256=packet['capture']['sha256'])
    adapter = module.NativeStepAdapter(actor.drafter, embedding_lookup=lookup, embedding_precision='external_native_f32')
    stage = time.perf_counter()
    teacher_logits = np.fromfile(checked(case['teacher_files']['logits']['file']), dtype='<f4').reshape(8,50000)
    teacher_state = np.fromfile(checked(case['teacher_files']['pre_norm']['file']), dtype='<f4').reshape(8,2560)
    teacher_encoded = np.fromfile(checked(case['teacher_files']['encoded']['file']), dtype='<f4').reshape(8,2560)
    recurrent_logits = np.fromfile(checked(case['recurrent_files']['logits']['file']), dtype='<f4').reshape(-1,50000)
    recurrent_state = np.fromfile(checked(case['recurrent_files']['pre_norm']['file']), dtype='<f4').reshape(-1,2560)
    with torch.inference_mode(), shared_round_hard_signs(actor.linears):
        anchor = meta['prompt_tokens'] - 1
        cache = adapter.build_context_cache(meta['input_tokens'][1:anchor+1], features[:anchor], chunk_size=32)
        for i, step in enumerate(case['teacher_forced_steps']):
            state = adapter.encode_feature(features[step['decoder_position']])
            output = adapter.decode_step(step['input_token_id'], state, step['decoder_position'], cache)
            assert np.array_equal(state.numpy(), teacher_encoded[i]), ('encoded', pid, i)
            assert np.array_equal(output.logits.numpy(), teacher_logits[i]), ('teacher_logits', pid, i)
            assert np.array_equal(output.pre_norm.numpy(), teacher_state[i]), ('teacher_state', pid, i)
            cache = output.cache
        for window in case['recurrent']:
            anchor = window['anchor_position']
            cache = adapter.build_context_cache(meta['input_tokens'][1:anchor+1], features[:anchor], chunk_size=32)
            state = adapter.encode_feature(features[anchor])
            for step in window['steps']:
                output = adapter.decode_step(step['input_token_id'], state, step['decoder_position'], cache)
                row = step['logits_row_index']
                assert np.array_equal(output.logits.numpy(), recurrent_logits[row]), ('recurrent_logits',pid,row)
                assert np.array_equal(output.pre_norm.numpy(), recurrent_state[row]), ('recurrent_state',pid,row)
                state, cache = output.pre_norm, output.cache
    cases[pid] = {'teacher_logits_prenorm_encoded_byteexact': True,
                  'recurrent_logits_prenorm_byteexact': True,
                  'teacher_rows':8, 'recurrent_rows':len(recurrent_logits),
                  'seconds':time.perf_counter()-stage}
    del adapter, cache, state, output, rows, lookup, arrays, features
phases['reference_equivalence_seconds'] = sum(x['seconds'] for x in cases.values())

stage = time.perf_counter()
zero = module.export_scale_only_gguf(actor, R / 'zero.gguf')
assert zero['output']['sha256'] == module.INITIALIZER_SHA256
write('zero-export.json', zero)
phases['zero_export_seconds'] = time.perf_counter() - stage

# Outer recipe demonstration: actual 76,880-element scale tensors, synthetic
# scalar objective only. These are persistence tests, NOT model training gains.
stage = time.perf_counter()
optimizer = torch.optim.Adam(actor.scale_parameters(), lr=1e-5)
torch.manual_seed(187)
cursor = 0
with torch.no_grad():
    for module_index, parameter in enumerate(actor.scale_parameters()):
        parameter.copy_(torch.arange(parameter.numel(), dtype=torch.float32) * (module_index+1) * 1e-8)
    actor.linears['fc'].scale_offset[0] = -2 * actor.linears['fc'].initial_scale[0]


def update():
    global cursor
    optimizer.zero_grad()
    randoms = torch.rand(9)
    loss = sum((p.square().mean() + r*p.mean()) for p,r in zip(actor.scale_parameters(), randoms, strict=True))
    loss.backward()
    optimizer.step()
    cursor += 1
    return float(loss.detach())


update()
saved = {'schema':'outer_compact_resume_qualification_v1', 'scale_state':module.collect_scale_state(actor),
         'optimizer':copy.deepcopy(optimizer.state_dict()), 'torch_rng':torch.get_rng_state().clone(),
         'numpy_rng':np.random.get_state(), 'cursor':cursor,
         'objective':'synthetic scale-state persistence test, not model training'}
with (R/'resume.pt').open('xb') as stream:
    torch.save(saved,stream)
expected_loss = update()
expected = module.collect_scale_state(actor)
expected_optimizer = copy.deepcopy(optimizer.state_dict())
expected_rng = torch.get_rng_state().clone()
restored = torch.load(R/'resume.pt',weights_only=False,map_location='cpu')
module.load_scale_state(actor,restored['scale_state'])
optimizer.load_state_dict(restored['optimizer'])
torch.set_rng_state(restored['torch_rng'])
np.random.set_state(restored['numpy_rng'])
cursor=restored['cursor']
actual_loss=update()
assert actual_loss==expected_loss and cursor==2 and torch.equal(torch.get_rng_state(),expected_rng)
actual = module.collect_scale_state(actor)
for base,value in expected['scale_offsets'].items():
    assert torch.equal(value,actual['scale_offsets'][base])
for key, value in expected_optimizer['state'].items():
    for name,tensor in value.items():
        assert torch.equal(tensor,optimizer.state_dict()['state'][key][name])
changed = module.export_scale_only_gguf(actor,R/'changed.gguf')
assert changed['output']['sha256'] != zero['output']['sha256']
write('changed-export.json',changed)
reader = GGUFReader(str(R/'changed.gguf'))
tensors={t.name:t for t in reader.tensors}
for base,path in module.CANDIDATE_D_BASE_TO_PATH.items():
    values=actor.linears[path].effective_scales().detach().numpy()
    if base in actor.qk_layout:
        values=values[module.qk_row_indices(**actor.qk_layout[base])]
    assert np.array_equal(values,tensors[base+'.w1a1_scale'].data), base
# No overwrite and no disposable publication files left behind.
try:
    module.export_scale_only_gguf(actor,R/'changed.gguf')
except ValueError:
    pass
else:
    raise AssertionError('existing export overwritten')
assert not list(R.glob('.changed.gguf.*'))
phases['real_scale_resume_and_nonzero_export_seconds']=time.perf_counter()-stage

peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
assert peak<6*1024**3,peak
size=sum(p.stat().st_size for p in R.rglob('*') if p.is_file())
assert size<150*1024**2,size
report={'status':'PACKED_ONLY_COMPACT_ACTOR_CPU_QUALIFICATION_PASS',
        'source_pins':source_pins,'initializer':actor.initializer,'config':actor.config,'contract':actor.contract,
        'cases':cases,'scale_optimizer_tensors':len(actor.scale_parameters()),'scale_elements':76880,
        'zero_export_byteidentical_initializer':True,'changed_export_only_nine_scale_payloads':True,
        'qk_native_scale_order_independently_readback_verified':True,
        'resume_offsets_optimizer_rng_cursor_exact':True,'resume_checkpoint':pin(R/'resume.pt'),
        'zero_export':pin(R/'zero.gguf'),'changed_export':pin(R/'changed.gguf'),
        'input_pins':list(references.values()),'phases':phases,
        'wall_seconds':time.perf_counter()-started,'peak_rss_bytes':peak,'artifact_bytes_before_report':size,
        'failed_attempts':[{'action':'invoke .venv/bin/python relative to isolated worktree',
                            'outcome':'shell exit127; corrected to absolute shared interpreter; no model allocated'},
                           {'action':'initial lint before formatter', 'outcome':'formatting issues fixed; final lint passes'}],
        'hf_source_opened':False,'target_model_opened_or_hashed':False,'gpu_seconds':0,
        'model_training_updates':0,'synthetic_scale_persistence_updates_physical':3,
        'scope':'CPU actor construction/export/persistence qualification only; no native replay of modified scales, model training gain, quality, generalization or throughput proof'}
write('report.json',report)
print(json.dumps({k:report[k] for k in ('status','cases','wall_seconds','peak_rss_bytes','artifact_bytes_before_report')},indent=2),flush=True)

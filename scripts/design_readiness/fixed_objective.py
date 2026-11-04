"""Bounded loss ablations for the retained modern row-scale candidate only.

This is an explicit checkpoint migration, not a general architecture trainer.
The paid5760 parent, native histories, two replay banks and arithmetic stay fixed.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import random
import signal
import sys
import time

PROJECT = Path(__file__).resolve().parents[2]
# Existing union manifests authenticate both content and these source paths.
# Reuse their admitted checkout; do not silently weaken the source-identity gate.
PINNED_SOURCE = Path('/home/philip/.codex/worktrees/w1a8-actual-training-20261103')
sys.path[:0] = [str(PINNED_SOURCE / "scripts/w1a8_generalization"), str(PINNED_SOURCE / "src"),
                str(PROJECT / "third_party/llama.cpp/gguf-py")]
import torch
import modern_train_added_replay as legacy
import modern_train_expanded_ce_resident as resident
from modern_actor import (build_modern_actor, collect_scale_state, export_scale_only_gguf,
                          file_pin, load_scale_state)
from modern_actual_data import ActualDataset
from modern_actual_union import Audit, UnionActualDataset, build_union
from modern_train_protected import (cpu, equal, require, optimizer_scope,
                                    assert_excluded_unchanged, normalized_replay_loss)
from w1a1_eagle.native_step import ExternalNativeF32Embedding
from w1a1_eagle.recurrent_qat import shared_round_hard_signs

PARENT = Path('/mnt/c/Users/philip/binary-eagle-decoding-artifacts/w1a8-added-replay-pair-20261103-01/true5760/checkpoint-00005760.pt')
SCHEMA = 'modern_fixed_objective_checkpoint_v1'
RECIPES = {
    'control': {'depth_weights': [1., 1.], 'label_smoothing': 0.},
    'depth': {'depth_weights': [1.5, .5], 'label_smoothing': 0.},
    'smooth': {'depth_weights': [1., 1.], 'label_smoothing': .05},
    'combined': {'depth_weights': [1.5, .5], 'label_smoothing': .05},
}


def ordinary_ce(values, classes, reachable, recipe):
    """Static native reachability; unsupported labels remain untrained; fixed4."""
    require(len(values) <= 2, 'only the fixed depth2 actor is supported')
    total = values.sum() * 0.0
    count = 0
    for d, cls in enumerate(classes.tolist()):
        if cls >= 0 and reachable[d]:
            loss = torch.nn.functional.cross_entropy(
                values[d:d+1], classes[d:d+1],
                label_smoothing=recipe['label_smoothing'])
            # Avoid even an extra multiply in the disabled/noop recipe.
            if recipe['depth_weights'][d] != 1.:
                loss = loss * recipe['depth_weights'][d]
            total = total + loss
            count += 1
    return total / 4., count


def validate_local_state(saved, contract, shapes, expected_exposure):
    """Reject semantically incompatible or corrupt resumes before GPU updates."""
    require(saved.get('schema') == SCHEMA and saved.get('contract') == contract,
            'resume contract differs')
    require(type(saved['cursor']) is int and 5760 <= saved['cursor'] <= 7680,
            'resume cursor invalid')
    legacy.validate_scale_adam(saved)
    for index, shape in enumerate(shapes):
        row = saved['optimizer']['state'][index]
        for name in ('exp_avg', 'exp_avg_sq'):
            value = row[name]
            require(tuple(value.shape) == shape and value.dtype == torch.float32
                    and bool(torch.isfinite(value).all()), 'invalid Adam moment')
        require(bool((row['exp_avg_sq'] >= 0).all()), 'negative Adam second moment')
    require(saved.get('local_exposure') == expected_exposure, 'local exposure differs')
    seconds = saved.get('local_trainer_seconds')
    require(type(seconds) in (int, float) and math.isfinite(seconds) and seconds >= 0,
            'invalid local training cost')


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--recipe', choices=RECIPES, required=True)
    p.add_argument('--final-updates', type=int, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--resume', type=Path)
    p.add_argument('--freeze', type=Path, required=True)
    p.add_argument('--allow-cuda', action='store_true')
    a = p.parse_args()
    require(a.allow_cuda and 5760 < a.final_updates <= 7680, 'bounded GPU authorization required')
    freeze = json.loads(a.freeze.read_text())
    require(freeze['parent'] == file_pin(PARENT), 'paid parent changed')
    for pin in freeze['sources']:
        require(file_pin(pin['path']) == pin, 'frozen source changed: '+pin['path'])
    require(torch.cuda.get_device_name(0) == 'NVIDIA GeForce RTX 5080', 'wrong hardware')
    require(a.out.resolve().is_relative_to(resident.ARTIFACTS), 'use bounded artifact root')
    guard = resident.CadencedGuard(PROJECT, a.out)
    signal.signal(signal.SIGTERM, guard.request_stop)
    signal.signal(signal.SIGINT, guard.request_stop)
    guard.before_write(64*1024)
    a.out.mkdir(exist_ok=False, parents=True)
    started = time.perf_counter()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    parent = torch.load(PARENT, map_location='cpu', weights_only=False)
    c = parent['contract']
    require(parent['schema'] == legacy.SCHEMA and parent['cursor'] == 5760
            and c['added_replay'] is True and c['expanded_ce'] is True,
            'exact selected5760 lineage required')
    legacy.validate_scale_adam(parent)
    device = torch.device('cuda:0')
    original = ActualDataset(c['manifest']['path'])
    data = UnionActualDataset(c['union_manifest']['path'])
    audit = Audit(guard)
    rebuilt = build_union([m['manifest']['path'] for m in data.manifest['members']], audit=audit)
    require(rebuilt == data.manifest, 'actual source union changed')
    fit, dev, added = resident.controlled_views(original.manifest, data.manifest, c['manifest'])
    require(file_pin(c['manifest']['path']) == c['manifest'] and
            file_pin(c['union_manifest']['path']) == c['union_manifest'], 'data pins changed')
    proof_pin, proof = resident.validate_resident_proof(
        Path(c['resident']['proof']['path']), Path(c['union_manifest']['path']), data.manifest)
    cache = resident.bind_resident(data, c['resident']['config'])
    embeddings = resident.load_embeddings(data, device, guard)
    lookup = ExternalNativeF32Embedding(lookup=embeddings.__getitem__, vocab_size=248320,
        hidden_size=2560, target_identity='frozen native Qwen3.5-4B Q4_K_M actual parent',
        target_sha256=data.manifest['target']['sha256'], capture_sha256=c['union_manifest']['sha256'])
    actor = build_modern_actor(PROJECT/'runs/w1a8-modern-packed-init-20261103-01/model.gguf',
        PROJECT/'runs/spec-shortlist-20261030/models/hf-eagle-4B/config.source.json',
        embedding_lookup=lookup, device=device, scale_only=True)
    require(actor.contract == c['actor'], 'actor arithmetic/representation changed')
    absolute = actor.drafter.d2t + torch.arange(50000, device=device)
    reverse = torch.full((248320,), -1, device=device, dtype=torch.long)
    reverse[absolute] = torch.arange(50000, device=device)
    supported_ids = set(absolute.cpu().tolist())
    refs = json.loads(Path(c['reference']['path']).read_text())
    require(file_pin(c['reference']['path']) == c['reference'], 'native old reference changed')
    # reference02 is already pinned and admitted; retain its exact ancestry.
    refs = refs['rows'] if isinstance(refs, dict) and 'rows' in refs else refs
    if isinstance(refs, list):
        refs = {r['id']: r for r in refs}
    native_refs = resident.native_fit_references(fit, refs, supported_ids)
    schedule = resident.native_schedule(fit, native_refs)
    require(schedule.contract == c['protected_schedule'], 'old replay schedule changed')
    added_schedule, added_refs, report = legacy.load_added_references(
        Path(c['added_reference']['plan']['path']), Path(c['union_manifest']['path']),
        data.manifest, added, supported_ids, guard)
    require(report == c['added_reference'], 'added407 native references changed')
    ce_fit = fit + added
    order = resident.ce_order(ce_fit)
    require([ce_fit[i]['id'] for i in order] == c['ce_schedule']['order'], 'CE order changed')
    paths, params, excluded = optimizer_scope(actor, 'all')
    require(paths == c['optimized_scale_paths'], 'optimizer paths changed')
    contract = dict(schema=SCHEMA, parent=file_pin(PARENT), recipe=RECIPES[a.recipe],
                    recipe_name=a.recipe, freeze=file_pin(a.freeze), actor=actor.contract,
                    optimizer_paths=paths, replay='old356 + added407 independent lambda10',
                    ce='2 windows/fixed4; static native reachable mask; cursor order unchanged',
                    parent_horizon=19200, max_bounded_cursor=7680)
    saved = torch.load(a.resume, map_location='cpu', weights_only=False) if a.resume else parent

    def expected_exposure(cursor):
        expected = dict(ce_windows=0, labels=0, old_replay_tokens=0, added_replay_tokens=0)
        for i in range(5760, cursor):
            for w in resident.ce_at(ce_fit, order, i):
                expected['ce_windows'] += 1
                expected['labels'] += sum(label in supported_ids and reachable for label, reachable
                    in zip(w['labels'], w['native_supervision_reachable_mask']))
            _, w = schedule.at(i)
            expected['old_replay_tokens'] += sum(native_refs[w['id']]['protected'])
            _, w = added_schedule.at(i-3840)
            expected['added_replay_tokens'] += sum(added_refs[w['id']]['protected'])
        return expected

    if a.resume:
        validate_local_state(saved, contract, [tuple(p.shape) for p in params],
                             expected_exposure(saved['cursor']))
    load_scale_state(actor, saved['scales'])
    optimizer = torch.optim.Adam(params, lr=1e-5, betas=(.9,.999), eps=1e-8)
    optimizer.load_state_dict(resident.named_optimizer_state(
        saved['optimizer'], saved['contract'].get('optimizer_paths', paths), paths))
    cursor = saved['cursor']
    require(5760 <= cursor < a.final_updates, 'invalid resume cursor')
    exposure = copy.deepcopy(saved.get('local_exposure', dict(ce_windows=0, labels=0,
        old_replay_tokens=0, added_replay_tokens=0)))
    seconds = saved.get('local_trainer_seconds', 0.)
    initial = cpu(saved['scales']['scale_offsets'])
    torch.set_rng_state(saved['rng'])
    torch.cuda.set_rng_state(saved['cuda_rng'])
    random.setstate(saved['python_rng'])

    def forward(window):
        batch = data.materialize(window, device)
        adapter = actor.adapter
        cache = adapter.build_context_cache(batch.prefix_tokens, batch.prefix_rawfeatures, chunk_size=64)
        state = adapter.encode_feature(batch.seed_rawfeature)
        token = window['seed_token']
        values = []
        for d in range(window['max_depth']):
            step = adapter.decode_step(token, state, window['anchor'] + d, cache)
            cache = step.cache
            values.append(step.logits)
            state = step.pre_norm
            token = window['labels'][d]
        return torch.stack(values)

    def write(path, value, recovery=False):
        resident.guarded_write(path, value, guard, recovery=recovery)

    def save(status):
        assert_excluded_unchanged(actor, excluded)
        state = dict(schema=SCHEMA, contract=contract, cursor=cursor,
            scales=collect_scale_state(actor), optimizer=cpu(optimizer.state_dict()),
            rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(), python_rng=random.getstate(),
            local_exposure=exposure, local_trainer_seconds=seconds, status=status)
        validate_local_state(state, contract, [tuple(p.shape) for p in params],
                             expected_exposure(cursor))
        path = a.out/f'checkpoint-{cursor:08d}.pt'
        if not path.exists():
            resident.guarded_checkpoint(path, state, guard, recovery=status=='INTERRUPTED')
        return path

    write(a.out/'freeze.json', contract)
    try:
        with (a.out/'updates.jsonl').open('x') as log:
            while cursor < a.final_updates:
                guard()
                actor.drafter.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                t = time.perf_counter()
                loss = 0.
                labels = 0
                with shared_round_hard_signs(actor.linears):
                    for w in resident.ce_at(ce_fit, order, cursor):
                        values = forward(w)
                        classes = reverse[torch.tensor(w['labels'], device=device)]
                        ce, n = ordinary_ce(values, classes, w['native_supervision_reachable_mask'], RECIPES[a.recipe])
                        loss = loss + ce
                        labels += n
                    _, w = schedule.at(cursor)
                    values = forward(w)
                    classes = reverse[torch.tensor(w['labels'], device=device)]
                    replay, old_tokens = normalized_replay_loss(values, classes, native_refs[w['id']], cap=2.)
                    loss = loss + 10.*replay
                    _, w = added_schedule.at(cursor-3840)
                    values = forward(w)
                    classes = reverse[torch.tensor(w['labels'], device=device)]
                    replay, new_tokens = normalized_replay_loss(values, classes, added_refs[w['id']], cap=2.)
                    loss = loss + 10.*replay
                    require(bool(torch.isfinite(loss)), 'nonfinite loss')
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(params, 1.)
                    require(bool(torch.isfinite(norm)), 'nonfinite gradient')
                    optimizer.step()
                    for path in paths:
                        actor.linears[path].project_scales_()
                torch.cuda.synchronize()
                elapsed = time.perf_counter()-t
                seconds += elapsed
                cursor += 1
                exposure['ce_windows'] += 2
                exposure['labels'] += labels
                exposure['old_replay_tokens'] += old_tokens
                exposure['added_replay_tokens'] += new_tokens
                row = dict(cursor=cursor, loss=float(loss.detach()), gradient_norm=float(norm),
                           update_seconds=elapsed, local_trainer_seconds=seconds, labels=labels)
                log.write(json.dumps(row)+'\n')
                log.flush()
                if cursor % 48 == 0:
                    print(json.dumps(row), flush=True)
                if cursor % 480 == 0:
                    save('RECOVERY')
        checkpoint = save('COMPLETE')
    except BaseException:
        save('INTERRUPTED')
        raise
    # Consumed DEV, deterministic panel; actual native serving remains the selector.
    rows = []
    with torch.no_grad(), shared_round_hard_signs(actor.linears):
        for w in dev:
            guard()
            values = forward(w)
            predictions = absolute[values.argmax(-1)].cpu().tolist()
            prefix = 0
            for pred, label in zip(predictions, w['labels']):
                if pred != label:
                    break
                prefix += 1
            rows.append(dict(id=w['id'], prefix=prefix, predictions=predictions, labels=w['labels']))
    write(a.out/'development.json', rows)
    guard.before_write(actor.initializer['bytes'])
    export = export_scale_only_gguf(actor, a.out/'model.gguf')
    final = collect_scale_state(actor)['scale_offsets']
    changed = {k: int((final[k].cpu()!=initial[k]).sum()) for k in final}
    write(a.out/'result.json', dict(status='TRAINED_EXPORTED_NOT_NATIVE_ADMITTED',
        contract=contract, checkpoint=file_pin(checkpoint), model=export['output'],
        cursor=cursor, changed_scale_elements=changed, local_exposure=exposure,
        local_trainer_seconds=seconds, wall_seconds=time.perf_counter()-started,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(), resident=cache.report(),
        dev_windows=len(rows), dev_prefix_tokens=sum(r['prefix'] for r in rows),
        scope='Consumed DEV only; no generalization or throughput claim'))


if __name__ == '__main__':
    main()

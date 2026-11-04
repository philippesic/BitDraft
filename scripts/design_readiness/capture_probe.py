"""Re-exercise the exact retained native capture producer on two consumed FITs.

The only producer change selects two already-audited groups after its complete
36-group roster gate. It does not alter any target, model, feature or teacher rule.
Derived source and its diff remain in the owned run directory.
"""
from pathlib import Path
import argparse
import difflib
import hashlib
import json
import os
import subprocess
import sys
import time
import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
SOURCE = PROJECT/'runs/w1a8-modern-actual-parent-capture-20261103-01/collect.py'
SOURCE_SHA = 'd6f652229ddc5f558e49909e07aa163652fb193e5940aca133a2acc5ff17c97b'
OLD_CAPTURE = SOURCE.parent/'capture01'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def candidate_rows(trace):
    rows = {}
    for line in (trace/'calls.jsonl').read_text().splitlines():
        for selected in json.loads(line).get('selected', []):
            if 'candidate_logits' in selected:
                rows[selected['candidate_logits']['file']] = selected
    return rows


def compare_arrays(old, new):
    before, after = candidate_rows(old), candidate_rows(new)
    assert set(before) == set(after), 'diagnostic file roster changed'
    arrays = list(new.glob('*.f32'))
    assert {p.name for p in arrays} == {p.name for p in old.glob('*.f32')}
    reordered = 0
    for path in arrays:
        if path.name not in before:
            assert digest(path) == digest(old/path.name), 'training array differs: '+path.name
            continue
        a, b = before[path.name], after[path.name]
        assert a['logits_mode'] == b['logits_mode']
        assert a['argmax_target_token'] == b['argmax_target_token']
        assert a['backend_sampled_token'] == b['backend_sampled_token']
        left = np.fromfile(old/path.name, dtype='<f4')
        right = np.fromfile(path, dtype='<f4')
        ai, bi = a['candidate_ids'], b['candidate_ids']
        assert len(set(ai)) == len(ai) == len(left)
        assert len(set(bi)) == len(bi) == len(right)
        assert set(ai) == set(bi), 'candidate token set changed'
        av, bv = dict(zip(ai, left)), dict(zip(bi, right))
        assert all(np.isfinite(av[k]) and av[k] == bv[k] for k in av), 'token-aligned logit differs'
        reordered += ai != bi
    return dict(f32_arrays=len(arrays), training_arrays_byte_equal=True,
                token_aligned_candidate_logits_exact=True, reordered_candidate_lists=reordered)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--allow-cuda', action='store_true')
    p.add_argument('--reuse-capture', action='store_true')
    a = p.parse_args()
    if not a.allow_cuda:
        raise ValueError('sole GPU operator authorization required')
    a.output.mkdir(parents=True, exist_ok=a.reuse_capture)
    assert digest(SOURCE) == SOURCE_SHA
    source = SOURCE.read_text()
    needle = "        requests = {}\n"
    assert source.count(needle) == 1
    derived = source.replace(needle,
        "        groups = [row for row in groups if row['id'] in ('fit-code-00', 'fit-reasoning-00')]\n"
        "        require(len(groups) == 2 and all(row['role'] == 'fit' for row in groups), 'two consumed FIT groups required')\n"
        + needle)
    path = a.output/'collect_two_fit.py'
    path.write_text(derived)
    (a.output/'producer.patch').write_text(''.join(difflib.unified_diff(
        source.splitlines(True), derived.splitlines(True), fromfile=str(SOURCE), tofile=str(path))))
    started = time.perf_counter()
    env = {**os.environ, 'CUDA_CACHE_PATH':str(PROJECT/'runs/local-w1a8-modern-serving-trace-20261103-01/cuda'),
           'PYTHONDONTWRITEBYTECODE':'1'}
    if a.reuse_capture:
        assert (a.output/'capture/summary.json').is_file(), 'missing completed capture'
    else:
        code = subprocess.call([sys.executable, str(path), '--output', str(a.output/'capture'),
                                '--allow-cuda'], env=env, cwd=PROJECT)
        assert code == 0, code
    summary = json.loads((a.output/'capture/summary.json').read_text())
    comparisons = []
    for group in summary['groups']:
        assert group['status'] == 'ACTUAL_PARENT_TRACE_AND_PROTECTION_AUDIT_PASS'
        old = OLD_CAPTURE/group['id']
        new = a.output/'capture'/group['id']
        before = json.loads((old/'response.json').read_text())
        after = json.loads((new/'response.json').read_text())
        keys = ['tokens','content','stop_type','stopping_word','truncated']
        assert all(before[k] == after[k] for k in keys)
        comparison = compare_arrays(old/'trace', new/'trace')
        comparisons.append(dict(group=group['id'], exact_output=True, **comparison))
    sys.path.insert(0, str(PROJECT/'scripts/w1a8_generalization'))
    from modern_actual_data import prepare, ActualDataset
    original = json.loads((PROJECT/'runs/w1a8-modern-actual-training-data-20261103-01/final01/manifest.json').read_text())
    # Exact retained embedding result is a preparation input, not a dense fallback.
    embedding_results = []
    for ref in original['source_pins']:
        q = Path(ref['path'])
        if q.name == 'result.json' and q.is_file():
            data = json.loads(q.read_text())
            if data.get('status') == 'NATIVE_EMBEDDING_ROWS_CAPTURED':
                embedding_results.append(q.parent)
    # prepare also binds the authenticated original conditional embedding rows.
    out = a.output/'prepared'
    prepare(a.output/'capture', out, embedding_results=embedding_results)
    data = ActualDataset(out/'manifest.json')
    count = 0
    for window in data.windows:
        if window['role'] != 'fit':
            raise ValueError('capture probe opened non-FIT group')
        data.materialize(window, 'cpu')
        count += 1
    result = dict(status='ACTUAL_TWO_FIT_RECAPTURE_AND_PREPARATION_PASS', source_sha256=SOURCE_SHA,
        derived_sha256=digest(path), comparisons=comparisons, actual_materialized_fit_windows=count,
        prepared_counts=data.manifest['counts'], missing_embedding_ids=data.manifest['missing_embedding_ids'],
        wall_seconds=time.perf_counter()-started, training_updates=0,
        scope='Existing exact initializer/native capture lineage; no changed model, fresh data, teacher precision or timing claim')
    (a.output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()

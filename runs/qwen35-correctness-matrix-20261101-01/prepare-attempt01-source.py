"""Pin retained models, export missing quantizations and freeze development cells."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from run_matrix import load_module, pin, read, sha, write


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    project, run = args.project.resolve(), args.run.resolve()
    run.mkdir(exist_ok=True)
    models = run / 'models'
    models.mkdir(exist_ok=True)
    original = project / 'runs/spec-shortlist-20261030'
    fix = project / 'runs/upstream-consistency-20261101'
    start = time.perf_counter()
    assert not (run / 'screen-freeze.json').exists()
    repair = read(run / 'eagle-sourcefaithful-repair.json')
    assert repair['completed']
    # Export repair receipt is retained; full native fidelity is a separate admission.
    conversions = []
    environment = dict(os.environ, LD_LIBRARY_PATH=str(original / 'build/bin'), OMP_NUM_THREADS='4')
    for size in ('4B', '9B'):
        source = models / f'Qwen3.5-{size}-eagle3-sourcefaithful-BF16.gguf'
        assert source.is_file()
        for dtype in ('Q4_0', 'Q4_K_M', 'Q8_0'):
            output = models / f'Qwen3.5-{size}-eagle3-sourcefaithful-{dtype}.gguf'
            assert not output.exists(), 'preserve prior exports; choose a new run directory'
            command = [str(original / 'build/bin/llama-quantize'), str(source), str(output), dtype, '4']
            begin = time.perf_counter()
            with (run / f'quantize-{size}-{dtype}.log').open('w') as log:
                result = subprocess.run(command, cwd=project, env=environment, stdout=log, stderr=subprocess.STDOUT)
            conversions.append({'command': command, 'exit_code': result.returncode, 'wall_seconds': time.perf_counter() - begin})
            write(run / 'quantization-costs.json', conversions)
            assert result.returncode == 0
    targets, arms = {}, []
    for size in ('4B', '9B'):
        for target_format in ('Q4_K_M', 'NVFP4'):
            target_id = size + '-' + target_format
            target_file = f'Qwen3.5-{size}-Q4_K_M.gguf' if target_format == 'Q4_K_M' else f'qwen35-{size.lower()}-mtp-nvfp4.gguf'
            targets[target_id] = pin(original / 'models' / target_file)
            families = [('target_only', 'none', 'none', None, [0]), ('mtp', 'draft-mtp', 'embedded', None, [1, 3, 7])]
            for dtype in ('BF16', 'Q4_0', 'Q4_K_M', 'Q8_0'):
                families.append(('eagle3', 'draft-eagle3', dtype, models / f'Qwen3.5-{size}-eagle3-sourcefaithful-{dtype}.gguf', [1, 3, 7]))
            for dtype in ('BF16', 'Q4_K_M', 'Q8_0'):
                dflash_file = f'qwen35-{size.lower()}-dflash-' + ('bf16' if dtype == 'BF16' else dtype) + '.gguf'
                families.append(('dflash', 'draft-dflash', dtype, original / 'models' / dflash_file, [3, 7, 15]))
                if size == '4B':
                    families.append(('dspark', 'draft-dspark', dtype, original / 'models' / f'Qwen3.5-4B-dspark-{dtype}.gguf', [3, 7, 16]))
            for dtype in ('BF16', 'Q4_K_M', 'Q8_0', 'NVFP4'):
                families.append(('simple_0p8b', 'draft-simple', dtype, original / 'models' / f'Qwen3.5-0.8B-{dtype}.gguf', [1, 3, 7]))
            for family, spec_type, dtype, draft, depths in families:
                draft_pin = pin(draft) if draft else None
                group = family + '-' + dtype
                for depth in depths:
                    arms.append({'target_id': target_id, 'id': group + f'-d{depth}', 'group': group,
                                 'family': family, 'dtype': dtype, 'draft': draft_pin,
                                 'spec_type': spec_type, 'depth': depth})
    # Verify we are loading the already validated CUDA library, not rebuilding it.
    old_freeze = read(fix / 'final-freeze.json')
    cuda = pin(fix / 'quant4group-overlay/libggml-cuda.so.0.28.0')
    assert cuda['sha256'] == old_freeze['candidate_library']['sha256']
    for target in targets.values():
        assert target['sha256'] in {t['sha256'] for t in old_freeze['targets'].values()}
    sources = [pin(Path(__file__).parent / name) for name in ('run_matrix.py', 'prepare.py', 'generate_inputs.py', 'repair_eagle_export.py', 'analyze.py')]
    judge_pin = pin(fix / 'quality_judge.py')
    pins = sources + [judge_pin, pin(project / 'scripts/benchmark_native_eagle.py'), cuda,
                      pin(original / 'build/bin/llama-server'), pin(original / 'build/bin/llama-quantize'),
                      pin(run / 'eagle-sourcefaithful-repair.json')] + list(targets.values())
    pins += [arm['draft'] for arm in arms if arm['draft']]
    pins += [pin(file) for file in (original / 'build/bin').glob('lib*.so*') if 'libggml-cuda' not in file.name]
    pins = list({item['path']: item for item in pins}.values())
    sys.path.insert(0, str(project / 'third_party/llama.cpp/gguf-py'))
    from gguf import GGUFReader
    inventories = []
    for path in dict.fromkeys([t['path'] for t in targets.values()] + [arm['draft']['path'] for arm in arms if arm['draft']]):
        reader = GGUFReader(path)
        inventories.append({'path': path,
                            'metadata': {k: v.contents() for k, v in reader.fields.items() if any(s in k for s in ('architecture', 'target_layers', 'block_size', 'dimension_count', 'file_type'))},
                            'tensor_types': dict(Counter(t.tensor_type.name for t in reader.tensors))})
    write(run / 'model-inventory.json', inventories)
    manifest = {'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
                'scope': 'development selection only; fresh final inputs generated after final freeze',
                'runtime': 'retained quant4group correctness CUDA overlay, GGML_CUDA_DECODE_CONSISTENT=32',
                'server': pin(original / 'build/bin/llama-server'), 'cuda': cuda,
                'library_path': str(fix / 'quant4group-overlay') + ':' + str(original / 'build/bin'),
                'targets': targets, 'arms': arms, 'sources': sources, 'pins': pins, 'judge': judge_pin,
                'repetitions': 1,
                'settings': {'context': 4096, 'batch': 512, 'ubatch': 512, 'kv': 'F16 target and draft',
                             'temperature': 0, 'seed': 1234, 'output_limit': 128, 'cache_prompt': False,
                             'concurrency': 1, 'warmups_per_cell': 2, 'flash_attention': 'on'},
                'selection_rule': 'Fastest exact-output active configuration per target/family/draft-format on development only. Preserve all failures. Refine a boundary optimum before final selection. Primary Q4_0 EAGLE retained regardless of competitive rank.',
                'unavailable': ['9B DSpark: no matching checkpoint identified', 'NVFP4 EAGLE/DSpark/DFlash sidecars: no admitted artifact or supported retained quantizer export'],
                'training_seconds': 0}
    write(run / 'screen-freeze.json', manifest)
    inputs = run / 'screen-inputs'
    inputs.mkdir(exist_ok=False)
    write(inputs / 'requests.json', read(original / 'prompts.json'))
    judge = load_module('preparation_quality_judge', judge_pin['path'])
    cases = judge.cases()
    controls = []
    for case in cases:
        if case['kind'] == 'json':
            good, bad = json.dumps(case['expected']), '{}'
        elif case['kind'] == 'python':
            if case['function'] == 'first_unique':
                good = 'def first_unique(s):\n    return next((x for x in s if s.count(x)==1),None)\n'
            else:
                good = 'def merge_intervals(intervals):\n    out=[]\n    for a,b in sorted(intervals):\n        if out and a<=out[-1][1]: out[-1][1]=max(out[-1][1],b)\n        else: out.append([a,b])\n    return out\n'
            bad = f"def {case['function']}(x):\n    return None\n"
        else:
            good = "SELECT customer_id,SUM(total) FROM orders WHERE status='paid' GROUP BY customer_id HAVING COUNT(*)>=2 AND SUM(total)>=30 ORDER BY customer_id"
            bad = 'SELECT customer_id,total FROM orders'
        assert judge.judge(case, good)['passed'], case['id']
        assert not judge.judge(case, bad)['passed'], case['id']
        controls.append({'case_id': case['id'], 'correct_control_passed': True, 'known_wrong_control_rejected': True})
    write(inputs / 'quality-cases.json', cases)
    write(inputs / 'quality-controls.json', controls)
    write(inputs / 'receipt.json', {'created_at_utc': datetime.now(timezone.utc).isoformat(),
                                  'manifest_sha256': sha(run / 'screen-freeze.json'),
                                  'files': {f.name: sha(f) for f in inputs.glob('*.json')}})
    write(run / 'preparation-receipt.json', {'completed': True, 'wall_seconds': time.perf_counter() - start,
                                           'quantizations': conversions, 'sources': sources, 'training_seconds': 0,
                                           'note': 'repair export CPU time separately recorded; not GPU validation'})
    print(json.dumps({'screen_cells': len(arms), 'targets': list(targets), 'wall_seconds': time.perf_counter() - start}), flush=True)


if __name__ == '__main__':
    main()

"""Finite, restartable supplemental capture. Original artifacts remain immutable."""
import collections
import fcntl
import importlib.util
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time
import traceback
import urllib.request

P = pathlib.Path('/home/philip/he4-capture-2080ti')
S = P/'supplemental'
spec = importlib.util.spec_from_file_location('overnight', P/'project/scripts/he4_capture/run_resume.py')
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)
COUNTS = ('sequences', 'windows', 'unique_label_positions', 'unique_label_positions_in_windows')
WEBHOOK = 'https://api2.cursor.sh/automations/webhook/4115c6ba-94e3-59b0-b7c6-628f693830cd'
read = lambda p: json.loads(p.read_text())


def aggregate(records):
    totals = dict.fromkeys(COUNTS, 0)
    sources = {}
    ids = set()
    for row in records:
        assert row['id'] not in ids, row['id']
        ids.add(row['id'])
        receipt = pathlib.Path(row['receipt']['path'])
        assert R.sha(receipt) == row['receipt']['sha256']
        proof = read(receipt)
        assert proof['greedy_vs_full_logits_argmax'] and proof['no_post_eog_labels']
        assert proof['taps'] == R.TAPS and not pathlib.Path(proof['full_logits']['path']).exists()
        group = sources.setdefault(row['pool'], dict.fromkeys(COUNTS, 0))
        for key in COUNTS:
            totals[key] += proof[key]
            group[key] += proof[key]
    return totals, sources


def run():
    lock = (S/'supervisor.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    pool = read(S/'prompt-pool.json')
    rows = pool['records']
    base = read(S/'base-manifest.json')
    base_sha = R.sha(P/'manifest.json')
    assert read(P/'manifest.json') == base
    state = dict(status='preflight', started=R.now(), expected_sequences=len(rows), sequences_added=0,
                 windows_added=0, unique_labels_added=0, blocker=None)
    manifest = dict(schema=base['schema'], status='running', records=[], failures=[],
                    target_sha256=R.PIN, taps=R.TAPS, contract=base['contract'],
                    feature_semantics=base['feature_semantics'], hardware=base['hardware'], training_executed=False,
                    bug_fix=pool['bug_fix'], fixed_historical_admitted=pool['fixed_historical_admitted'],
                    original_historical_admitted=pool['original_historical_admitted'],
                    prompt_pool=R.pin(S/'prompt-pool.json'), base_manifest=R.pin(S/'base-manifest.json'))
    if (S/'manifest.json').exists():
        manifest = read(S/'manifest.json')
    completed = {r['id'] for r in manifest['records']}
    failed = {r['id'] for r in manifest['failures']}
    counts, sources = aggregate(manifest['records'])
    manifest.update(counts)

    def checkpoint(status=None, combine=False):
        if status:
            state['status'] = status
        manifest.update(status=state['status'], updated_at=R.now(), blocker=state['blocker'])
        R.write(S/'manifest.json', manifest)
        (S/'manifest.sha256').write_text(R.sha(S/'manifest.json')+'  manifest.json\n')
        state.update(updated_at=R.now(), sequences_added=manifest['sequences'], windows_added=manifest['windows'],
                     unique_labels_added=manifest['unique_label_positions'], failed_length_or_tokenizer=len(manifest['failures']),
                     free_disk_bytes=shutil.disk_usage(P).free)
        R.write(S/'status.json', state)
        if combine:
            combined = dict(base)
            combined.update(records=base['records']+manifest['records'], failures=base['failures']+manifest['failures'],
                            status=state['status'], updated_at=R.now(), bug_fix=pool['bug_fix'],
                            supplemental_manifest=R.pin(S/'manifest.json'), deadline_utc=None)
            combined_counts, breakdown = aggregate(combined['records'])
            combined.update(combined_counts, source_breakdown=breakdown)
            combined['domain_mix'] = dict(collections.Counter(r['domain'] for r in combined['records']))
            combined['trainer'] = dict(base['trainer'], index=str(S/'combined-manifest.json'))
            R.write(S/'combined-manifest.json', combined)
            (S/'combined-manifest.sha256').write_text(R.sha(S/'combined-manifest.json')+'  combined-manifest.json\n')
            return combined_counts, breakdown

    try:
        checkpoint()
        assert R.sha(R.MODEL) == R.PIN
        assert subprocess.check_output(['nvidia-smi','--query-gpu=name','--format=csv,noheader'], text=True).strip() == 'NVIDIA GeForce RTX 2080 Ti'
        assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'], text=True).strip(), 'GPU already occupied'
        for i, row in enumerate(rows):
            if row['id'] in completed | failed:
                continue
            entry = S/'captures'/f'shard-{i//100:04d}'/f'{i:06d}'
            out = entry/'capture'
            # Adopt durable verification left by a crash between verify and checkpoint.
            if (out/'verification.json').exists():
                proof = read(out/'verification.json')
                assert proof['id'] == row['id']
                if (out/'logits.f32').exists():
                    assert R.sha(out/'logits.f32') == proof['full_logits']['sha256']
                    (out/'logits.f32').unlink()
                for f in proof['files']:
                    assert R.sha(pathlib.Path(f['path'])) == f['sha256']
                receipt = R.pin(out/'verification.json')
            else:
                # Retry only this job's incomplete artifacts; no finished sequences are removed.
                if entry.exists():
                    shutil.rmtree(entry)
                entry.mkdir(parents=True)
                R.write(entry/'source.json', row)
                (entry/'prompt.txt').write_text(row['rendered_text'])
                state.update(current_index=i, current_id=row['id'], attempt=0)
                proof = None
                for attempt in range(3):
                    state['attempt'] = attempt+1
                    checkpoint('tokenizing')
                    try:
                        R.command([R.BIN,'--tokenize','--model',R.MODEL,'--text-file',entry/'prompt.txt','--out',entry/'tokenization'], entry/'tokenization.log', 120)
                    except subprocess.CalledProcessError as e:
                        if attempt < 2:
                            continue
                        manifest['failures'].append(dict(id=row['id'],reason='tokenizer rejected after 3 attempts',log=str(entry/'tokenization.log'),error=str(e)))
                        failed.add(row['id'])
                        break
                    tokens = (entry/'tokenization/tokens.ids').read_text().split()
                    if len(tokens)+128 > 4096:
                        manifest['failures'].append(dict(id=row['id'],reason='over context capacity; no cropping',prompt_tokens=len(tokens)))
                        failed.add(row['id'])
                        break
                    needed = (len(tokens)+128)*81920+129*248320*4
                    if shutil.disk_usage(P).free < needed+30*1024**3:
                        raise RuntimeError('Disk reserve reached; verified captures preserved')
                    argv = [R.BIN,'--model',R.MODEL,'--tokens',entry/'tokenization/tokens.ids','--out',out,'--cpu-library',R.LIB/'libggml-cpu.so.0.28.0','--cuda-library',R.CUDA,'--greedy-steps','128','--taps',','.join(map(str,R.TAPS)),'--prefill-chunk','32','--threads','6','--output-cap','32','--max-output-mib','1024']
                    R.write(entry/'launch.json',dict(argv=list(map(str,argv)),environment={k:R.ENV[k] for k in ('CUDA_VISIBLE_DEVICES','GGML_CUDA_DECODE_CONSISTENT','LD_LIBRARY_PATH')}))
                    checkpoint('capturing')
                    try:
                        if out.exists():
                            shutil.rmtree(out)
                        R.command(argv, entry/'capture.log')
                        receipt, proof = R.verify(out,row)
                        break
                    except Exception:
                        (entry/f'attempt-{attempt+1}-failure.txt').write_text(traceback.format_exc())
                        if attempt == 2:
                            raise
                        checkpoint('retrying_capture')
                if proof is None:
                    checkpoint('length_or_tokenizer_exclusion')
                    continue
            manifest['records'].append(dict(id=row['id'], receipt=receipt, pool=row['pool'], domain=row['domain']))
            completed.add(row['id'])
            for key in COUNTS:
                manifest[key] += proof[key]
            checkpoint('capturing')
        assert len(completed)+len(failed) == len(rows)
        assert R.sha(P/'manifest.json') == base_sha
        checkpoint('auditing')
        counts, sources = aggregate(manifest['records'])
        for key in COUNTS:
            assert counts[key] == manifest[key]
        # Full sample tensor audit is shared with the overnight capture.
        R.command([sys.executable,P/'project/scripts/he4_capture/audit.py',S/'manifest.json'],S/'audit.json')
        # Delete scratch from attempts that failed verification; all are supplemental-owned.
        scratch = list((S/'captures').rglob('logits.f32'))
        for f in scratch:
            f.unlink()
        assert not list((S/'captures').rglob('logits.f32'))
        totals, breakdown = checkpoint('complete',combine=True)
        R.write(S/'final-audit.json',dict(status='PASS',counts=counts,source_breakdown=breakdown,
                                        combined_totals=totals,prompt_ids_unique=True,
                                        baseline_manifest_unchanged=True,full_logit_scratch_remaining=0,
                                        length_or_tokenizer_failures=manifest['failures']))
    except BaseException as e:
        state['blocker'] = str(e)
        (S/'failure.txt').write_text(traceback.format_exc())
        for f in (S/'captures').rglob('logits.f32'):
            f.unlink()
        totals, breakdown = checkpoint('failed',combine=True)
    body = dict(status=state['status'], sequences_added=manifest['sequences'], windows_added=manifest['windows'],
                unique_labels_added=manifest['unique_label_positions'], new_totals=totals,
                source_breakdown=breakdown, length_or_tokenizer_failures=len(manifest['failures']),
                output_paths=[str(S/'captures'),str(S/'manifest.json'),str(S/'combined-manifest.json'),str(S/'status.json')],blocker=state['blocker'])
    R.write(S/'completion.json',body)
    for attempt in range(6):
        try:
            req = urllib.request.Request(WEBHOOK,data=json.dumps(body).encode(),headers={'Content-Type':'application/json'},method='POST')
            with urllib.request.urlopen(req,timeout=30) as response:
                R.write(S/'webhook-receipt.json',dict(status=response.status,time=R.now(),response=response.read(4096).decode(errors='replace'),body=body))
            break
        except Exception as e:
            R.write(S/'webhook-error.json',dict(attempt=attempt+1,error=str(e)))
            time.sleep(10)
    return 0 if state['status']=='complete' else 1


if __name__ == '__main__':
    sys.exit(run())

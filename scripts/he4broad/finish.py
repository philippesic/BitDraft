"""One-shot CPU completion callback after the supervised GPU workflow exits."""
import json
import subprocess
from pathlib import Path
from .common import P, R, OUT, REPORT, DEST, POINTS, BUDGET, pin, save


def command(*args, **kwargs):
    return subprocess.check_output(args,cwd=P,text=True,**kwargs).strip()


def section(document):
    marker = '## HE4broad source-initialized QAT'
    start = document.index(marker)
    end = document.find('\n## ',start+len(marker))
    return document[:start]+(document[end+1:] if end >= 0 else '')


def main():
    receipt = dict(status='STARTED', source=pin(__file__))
    save(R/'completion-callback.json',receipt)
    supervisor = json.load(open(P/'runs/he4broad-supervisor-20261108-02/state.json'))
    assert supervisor['status'] != 'running'
    owner = json.load(open(R/'gpu-owner.json'))['owner']
    process = Path(f'/proc/{owner["pid"]}/stat')
    if process.exists():
        assert process.read_text().rsplit(')',1)[1].split()[19] != owner['stat'].rsplit(')',1)[1].split()[19]
    assert json.load(open(R/'gpu-release.json'))['lock_released']
    from .publish import main as publish
    publish()
    result = json.load(open(REPORT/'results.json'))
    complete = result['status'] == 'COMPLETE'
    if complete:
        audit = result['audit']
        assert audit['actual_update_seconds'] >= BUDGET
        assert audit['uncheckpointed_completed_updates'] == 0
        assert len(result['training']['readouts']) == len(POINTS)
        assert result['final_recovery']['status'] == 'FULL_STATE_GPU_RESTORE_AND_NATIVE_BYTE_IDENTITY_PASS'
    inventory = command(str(P/'.venv/bin/python'),str(P/'scripts/milestone_inventory.py'),str(R))
    save(R/'milestone-inventory.json',json.loads(inventory))
    retained = []
    for row in result.get('training',{}).get('readouts',[]):
        rec = row['recovery']
        assert pin(rec['archive'])['sha256'] == rec['archive_sha256']
        assert pin(row['model']['path']) == row['model']
        retained.append(dict(checkpoint=rec,export=row['model']))
    assert not list(DEST.glob('reserve-*'))
    save(R/'milestone-retention.json',dict(status='PASS',retained=retained,
        sources=pin(R/'source/source-manifest.json'), recipe=pin(R/'recipe.json'),
        data=result['recipe']['config']['data'],source=result['recipe']['config']['source'],
        target=result['recipe']['config']['evaluation']['target'],
        q4_matrix=json.load(open(R/'q4-matrix-proof.json')),
        logs=[pin(OUT/'updates.jsonl'),pin(REPORT/'audit.json')],
        policy='All HE4broad checkpoints, exports and update/native evidence retained; all older frontier states preserved. Only unused reservation files and reproducible export temporaries removed.'))
    # Stage only this owned report and its status section. Do not commit a user's
    # concurrent staged changes or unrelated edits in the shared STATUS file.
    own = ['docs/STATUS.md',*[str(p.relative_to(P)) for p in REPORT.iterdir() if p.is_file()]]
    assert command('git','branch','--show-current') == 'main'
    assert command('git','remote','get-url','--push','origin') == 'git@github.com:philippesic/binary-eagle-decoding.git'
    assert not command('git','diff','--cached','--name-only'), 'Concurrent staged work; final report remains local'
    base = command('git','show','HEAD:docs/STATUS.md')+'\n'
    current = (P/'docs/STATUS.md').read_text()
    assert section(base) == section(current), 'Concurrent STATUS edit outside HE4broad; final report remains local'
    subprocess.run(['git','add','--',*own],cwd=P,check=True)
    if command('git','diff','--cached','--name-only'):
        subprocess.run(['git','commit','-m','Complete HE4broad QAT run' if complete else 'Record stopped HE4broad run'],cwd=P,check=True)
    head = command('git','rev-parse','HEAD')
    try:
        subprocess.run(['git','push','origin','main'],cwd=P,check=True)
        divergence = command('git','rev-list','--left-right','--count','HEAD...origin/main')
        assert divergence.split() == ['0','0']
        receipt.update(status='COMPLETE' if complete else 'STOPPED_RUN_RECORDED',head=head,pushed=True,divergence=divergence)
    except BaseException as error:
        receipt.update(status='PUSH_BLOCKED_LOCAL_REPORT_RETAINED',head=head,pushed=False,error=repr(error))
        raise
    finally:
        save(R/'completion-callback.json',receipt)
    (R/'git-status-final.txt').write_text(command('git','status','--short')+'\n')
    (R/'git-status-ignored-final.txt').write_text(command('git','status','--ignored','--short')+'\n')
    print(receipt['status'],flush=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        path = R/'completion-callback.json'
        receipt = json.load(open(path)) if path.exists() else {}
        receipt.update(status='FINALIZATION_BLOCKED',error=repr(error),local_evidence_retained=True)
        save(path,receipt)
        raise

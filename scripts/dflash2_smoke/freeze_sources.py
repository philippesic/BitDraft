"""Retain exact source bytes for every attempt without creating a checkout."""
import hashlib
import json
import subprocess
from pathlib import Path
from scripts.component_campaign.common import ROOT, pin, save
from scripts.dflash2_smoke.train import R


def main():
    store=R/'source-bytes';store.mkdir(exist_ok=True);records={}
    for resolved in sorted(R.glob('candidate*/resolved.json')):
        attempt={}
        for relative,expected in json.load(open(resolved))['source_pins'].items():
            digest=expected['sha256'];destination=store/(digest+'.py')
            origin='existing authenticated source store'
            if not destination.exists():
                current=ROOT/relative;body=current.read_bytes() if current.is_file() else b''
                origin='current source'
                if hashlib.sha256(body).hexdigest()!=digest:
                    commits=subprocess.check_output(['git','log','--all','--format=%H','--',relative],cwd=ROOT,text=True).splitlines()
                    for commit in commits:
                        candidate=subprocess.run(['git','show',commit+':'+relative],cwd=ROOT,capture_output=True)
                        if candidate.returncode==0 and hashlib.sha256(candidate.stdout).hexdigest()==digest:
                            body=candidate.stdout;origin='Git '+commit;break
                    else:raise ValueError('Cannot recover pinned source '+str(resolved)+':'+relative)
                destination.write_bytes(body);destination.chmod(0o444)
            actual=pin(destination)
            if actual['sha256']!=digest or actual['bytes']!=expected['bytes']:
                raise ValueError('Frozen source bytes differ')
            attempt[relative]=dict(original=expected,recovery=actual,origin=origin)
        records[resolved.parent.name]=attempt
    save(R/'source-byte-retention.json',dict(status='ALL_ATTEMPT_SOURCE_BYTES_AUTHENTICATED',
        attempts=records,script=pin(__file__),policy='Exact digest-addressed source copies; no checkout or branch created. Git remains the maintained source.'))
    print('ATTEMPT_SOURCE_BYTES_PRESERVED',len(records),flush=True)


if __name__=='__main__':main()

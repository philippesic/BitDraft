"""Explicit November 10 retention policy for superseded model artifacts.

Keep source, data, measurements, selected recoverable candidates and baselines.
Do not infer that every ignored file or every old training state is disposable.
"""
import argparse
import collections
import json
import os
import stat
import time
from pathlib import Path,PureWindowsPath

from .common import P,R,REPORT,save,pin

WORK=R/'pause-fix-20261110'
CATALOG=P/'experiments/storage-cleanup-20261110'
PROTECTED_RUNS={
    'spec-shortlist-20261030','component-campaign-20261105-01',
    'precision-campaign-20261104-01','dflash2-smoke-20261106-01',
    'he4broad-20261110-01','he4broad-20261108-01',
    'dflash-modern-train-20261104-14','dflash-modern-resume-20261104-12',
    'dspark-q4km-campaign-20261018','w1a8-generalization-20261102',
}


def registry_paths():
    paths=set()
    def walk(value):
        if isinstance(value,dict):
            for item in value.values():walk(item)
        elif isinstance(value,list):
            for item in value:walk(item)
        elif isinstance(value,str) and value.startswith(('/home/philip/','/mnt/c/')):paths.add(value)
    for path in (P/'experiments/dflash-w1a8-baseline-20261104/manifest.json',
                 P/'experiments/w1a8-standing-20261102/baseline.json',
                 R/'q4-matrix-proof.json',P/'scripts/dspark_screen/released_inventory.json'):
        walk(json.load(open(path)))
    return paths


REGISTRY_PATHS=registry_paths()


def candidate(path,root):
    if str(path) in REGISTRY_PATHS:return False
    parts=path.relative_to(root).parts;name=path.name.lower()
    if not parts:return False
    if any(part in ('source','src','scripts','checkouts','vendor','third_party','.git') or part.startswith('source-') for part in parts):return False
    if any(part.startswith(('he4-localize-','he4fresh-supervisor-')) for part in parts):return False
    if parts[0] in PROTECTED_RUNS:return False
    if parts[0]=='he4-continuation-20261107-01':
        if '173193' in name or name=='model-21600.gguf':return False
        return name.endswith(('.pt.bsz','.gguf')) and ('checkpoints' in parts or 'candidate' in parts)
    # Explicit original/released weights and recovery dictionaries remain.
    if any(part.lower() in ('bases','basis','dictionaries','released','external-models','pretrained') for part in parts):return False
    if name.endswith('.gguf'):
        if any(token in name for token in ('qwen','dspark','dflash','bf16','f16-target','target-','baseline')):return False
        return True
    if name.endswith(('.pt','.pth','.pt.bsz','.pt.zst','.pt.gz','.ckpt')):
        if any(('capture' in part.lower() or part.lower() in ('data','prepared-data','datasets','features')) for part in parts):return False
        if name.endswith(('.pt.bsz','.pt.zst','.pt.gz')):return True
        return ('checkpoint' in name or name.startswith(('step-','model-')) or
                any('checkpoint' in part.lower() for part in parts[:-1]))
    return False


def plan():
    rows=[];evidence=collections.defaultdict(list);by_campaign=collections.defaultdict(lambda:collections.Counter())
    journal=WORK/'checkpoint-retirement.jsonl'
    retired=[json.loads(line) for line in journal.read_text().splitlines()] if journal.exists() else []
    for row in retired:
        by_campaign[row['campaign']]['files']+=1;by_campaign[row['campaign']]['logical_bytes']+=row['bytes']
    roots=[P/'runs',Path('/mnt/c/Users/philip/binary-eagle-decoding-artifacts')]
    for root in roots:
        if root==P/'runs':
            paths=[P/line for line in (WORK/'linux-model-inventory.txt').read_text().splitlines()]
        else:
            raw=json.loads((WORK/'windows-model-inventory.json').read_text(encoding='utf-8-sig'))
            paths=[Path('/mnt/c').joinpath(*PureWindowsPath(row['FullName']).parts[1:]) for row in raw]
        retired_paths={Path(row['path']) for row in retired if row['root']==str(root)}
        paths=list(dict.fromkeys([*paths,*retired_paths]))
        neighbors=set()
        for path in paths:
            if not candidate(path,root) and path not in retired_paths:continue
            for parent in list(path.parents)[:5]:
                if not parent.is_relative_to(root):break
                for name in ('result.json','results.json','summary.json','report.md','manifest.json','accounting.json'):
                    neighbors.add(parent/name)
        paths+=sorted(neighbors)
        # Native inventories avoid one WSL/NTFS metadata round trip per unrelated
        # source/cache file. Missing/inaccessible entries remain protected.
        for path in paths:
                base=path.parent;name=path.name
                path=Path(base,name);relative=path.relative_to(root);campaign=relative.parts[0]
                selected=candidate(path,root)
                small_evidence=name in ('result.json','results.json','summary.json','report.md','manifest.json','accounting.json')
                if not selected and not small_evidence:continue
                try:info=path.lstat()
                except OSError:continue
                if not stat.S_ISREG(info.st_mode):continue
                if selected and info.st_size>32*2**20:
                    row=dict(path=str(path),root=str(root),campaign=campaign,bytes=info.st_size,
                        allocated_bytes=info.st_blocks*512,device=info.st_dev,inode=info.st_ino,
                        mtime_ns=info.st_mtime_ns,links=info.st_nlink)
                    rows.append(row);by_campaign[campaign]['files']+=1;by_campaign[campaign]['logical_bytes']+=info.st_size
                elif small_evidence and info.st_size<256*1024:
                    evidence[campaign].append(path)
    # Windows HE4broad predecessor: preserve zero and final; retire intermediate
    # optimizer states only, with every original result/readout retained.
    win=roots[1]/'he4broad-20261108-01'
    for path in win.glob('HE4broad-*.pt.bsz'):
        if path.name in ('HE4broad-0-0.pt.bsz','HE4broad-28800-226853.pt.bsz'):continue
        s=path.stat();row=dict(path=str(path),root=str(roots[1]),campaign=win.name,bytes=s.st_size,
            allocated_bytes=s.st_blocks*512,device=s.st_dev,inode=s.st_ino,mtime_ns=s.st_mtime_ns,links=s.st_nlink)
        rows.append(row);by_campaign[win.name]['files']+=1;by_campaign[win.name]['logical_bytes']+=s.st_size
    save(WORK/'checkpoint-retirement-plan.json',dict(checked_unix=time.time(),files=rows,
        policy='Philip November 10: retire obsolete one-off checkpoints and generated model exports; preserve directly used HE4, baselines, selected fallback candidates, source/data/evidence.'))
    summaries=[]
    for campaign,counts in sorted(by_campaign.items()):
        samples=[]
        for path in sorted(evidence[campaign],key=lambda p:(p.name!='result.json',len(p.parts),str(p)))[:12]:
            try:
                value=json.load(open(path)) if path.suffix=='.json' else path.read_text()
                if isinstance(value,dict):
                    # Bound the Git-visible snapshot; original raw evidence stays.
                    selected={k:v for k,v in value.items() if k.lower() not in ('rows','records','files','parameters','optimizer','source_files','commands','input_tokens','labels')}
                    encoded=json.dumps(selected)
                    if len(encoded)>24000:selected={k:v for k,v in selected.items() if not isinstance(v,(list,dict))}
                else:selected=value[:12000] if isinstance(value,str) else dict(items=len(value))
                samples.append(dict(source=pin(path),summary=selected))
            except (OSError,ValueError,TypeError):continue
        summaries.append(dict(campaign=campaign,**counts,evidence=samples,
            evidence_policy='All original non-model raw result/config/chart/log files retained in place; selected compact summaries preserved here.'))
    save(CATALOG/'retired-campaign-evidence.json',dict(checked_unix=time.time(),campaigns=summaries))
    summary=dict(status='PLANNED',files=len(rows),already_retired_files=len(retired),logical_bytes=sum(x['bytes'] for x in rows),
        allocated_bytes_before_hardlink_dedup=sum(x['allocated_bytes'] for x in rows),
        campaigns=[dict(campaign=k,**v) for k,v in sorted(by_campaign.items())],
        keep_policy=sorted(PROTECTED_RUNS),raw_plan=str(WORK/'checkpoint-retirement-plan.json'))
    save(CATALOG/'retirement-summary.json',summary)
    print(json.dumps(summary,indent=2))


def execute(plan_path=None):
    import subprocess
    assert not subprocess.check_output(['/usr/lib/wsl/lib/nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip()
    plan_path=plan_path or WORK/'checkpoint-retirement-plan.json'
    rows=json.load(open(plan_path))['files']
    journal=WORK/'checkpoint-retirement.jsonl';done=set()
    if journal.exists():done={json.loads(line)['path'] for line in journal.read_text().splitlines()}
    opened=set()
    for proc in Path('/proc').iterdir():
        if not proc.name.isdecimal():continue
        try:
            for fd in (proc/'fd').iterdir():
                try:opened.add(os.readlink(fd))
                except OSError:pass
        except OSError:pass
    for row in rows:
        if row['path'] in done:continue
        path=Path(row['path']);assert candidate(path,Path(row['root'])) or (
            path.parent.name=='he4broad-20261108-01' and path.name not in ('HE4broad-0-0.pt.bsz','HE4broad-28800-226853.pt.bsz'))
        s=path.lstat();assert stat.S_ISREG(s.st_mode)
        assert (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)==(row['device'],row['inode'],row['bytes'],row['mtime_ns']),path
        assert str(path) not in opened,'Active consumer: '+str(path)
        path.unlink()
        with journal.open('a') as out:
            out.write(json.dumps(dict(**row,deleted_unix=time.time()))+'\n');out.flush();os.fsync(out.fileno())
    summary=json.load(open(CATALOG/'retirement-summary.json')) if (CATALOG/'retirement-summary.json').exists() else {}
    summary.update(status='EXECUTED',completed_unix=time.time(),
        deleted_files=len(rows),note='Logical totals can overstate physically freed hard links; measured filesystem space is reported separately.')
    save(CATALOG/'retirement-summary.json',summary)
    print('RETIRED',len(rows),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--execute',action='store_true');ap.add_argument('--plan-file',type=Path);args=ap.parse_args()
    execute(args.plan_file) if args.execute else plan()

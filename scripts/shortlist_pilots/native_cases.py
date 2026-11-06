"""Capture fixed, group-separated diagnostic blocks using the current native helper."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import subprocess


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    c = json.loads(args.config.read_text())
    plan = json.loads(Path(c['case_plan']).read_text())
    domains = defaultdict(dict)
    for case in plan['cases']:
        domains[case['domain']].setdefault(case['group_id'], case)
    selected = []
    for domain, groups in sorted(domains.items()):
        if len(groups) < 4:
            raise ValueError('four original diagnostic groups per domain required')
        for case in list(groups.values())[:4]:
            selected.append({**case, 'case_index':len(selected)})
    if len(selected) != 12:
        raise ValueError('fixed three-domain twelve-group screen required')
    for case in selected:
        if digest(case['features']['path']) != case['features']['sha256']:
            raise ValueError('feature pin changed')
    args.output.mkdir(parents=True, exist_ok=False)
    cases = args.output/'cases.json'
    cases.write_text(json.dumps({'schema':'group_transfer12_v1','cases':selected}, indent=2)+'\n')
    results = {}
    for arm, model in c['models'].items():
        if digest(model['path']) != model['sha256']:
            raise ValueError('model pin changed')
        dest = args.output/arm
        with (args.output/(arm+'.log')).open('wb') as log:
            subprocess.run([c['helper']['path'], model['path'], str(cases), str(dest), 'on'],
                           stdout=log, stderr=subprocess.STDOUT, timeout=300, check=True)
        rows = [json.loads(v) for v in (dest/'rows.jsonl').read_text().splitlines()]
        if len(rows) != 12:
            raise ValueError('native capture coverage differs')
        depths = []
        for row in rows:
            first = next((i for i,v in enumerate(row['rows'][:5]) if v['argmax'] != v['label']), 5)
            depths.append(first)
        results[arm] = {'depths5':depths,'sum_depth5':sum(depths),
                        'rows':str(dest/'rows.jsonl'),'cases':str(cases),
                        'model':model,'groups':12,'computed_rows':7}
    result = {'status':'NATIVE_CAPTURE_COMPLETE','results':results,
              'scope':'reused development diagnostic groups; no request throughput or fresh holdout claim'}
    (args.output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()

"""Execute one explicitly prepared native head/tree arm under the root guard."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--matrix', type=Path, required=True)
    parser.add_argument('--arm', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    matrix = json.loads(args.matrix.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    records = []
    for i, stage in enumerate(matrix['commands'][args.arm]):
        environment = {**os.environ, **matrix['required_native_environment'],
                       **stage.get('environment', {})}
        start = time.monotonic()
        with (args.output/(str(i)+'-'+stage['stage']+'.log')).open('wb') as log:
            result = subprocess.run(stage['argv'], env=environment,
                                    stdout=log, stderr=subprocess.STDOUT,
                                    timeout=900)
        records.append({'stage':stage['stage'], 'argv':stage['argv'],
                        'returncode':result.returncode,
                        'wall_seconds':time.monotonic()-start})
        (args.output/'stages.json').write_text(json.dumps(records, indent=2)+'\n')
        print(json.dumps(records[-1]), flush=True)
        if result.returncode:
            raise SystemExit(result.returncode)
    (args.output/'finish.json').write_text(json.dumps({
        'status':'ALL_MATRIX_STAGES_COMPLETE', 'arm':args.arm,
        'initialization':matrix['initialization'], 'stages':records,
        'scope':'head native quality and PCTree target/cache component; request timing separate'
    }, indent=2)+'\n')


if __name__ == '__main__':
    main()

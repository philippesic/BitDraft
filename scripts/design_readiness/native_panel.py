"""Current-runtime native panels with exact round metrics outside HTTP timing."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
import signal
import sys
import time
import urllib.request

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT/'scripts/w1a8_generalization'))
from modern_requests import Requests, atomic, pin, read, require

KEYS = ('spec_decode_num_drafts_total', 'spec_decode_num_draft_tokens_total',
        'spec_decode_num_accepted_tokens_total')


def parse_metrics(text):
    values = {}
    for line in text.splitlines():
        if line.startswith('llamacpp:'):
            key, number = line.rsplit(' ', 1)
            if key.removeprefix('llamacpp:') in KEYS or 'accepted_tokens_per_pos_total' in key:
                value = float(number)
                require(value.is_integer() and value >= 0, 'invalid native counter')
                values[key.removeprefix('llamacpp:')] = int(value)
    require(all(key in values for key in KEYS), 'native round counters missing')
    return values


class Panel(Requests):
    def metrics(self, port):
        with urllib.request.urlopen(f'http://127.0.0.1:{port}/metrics', timeout=15) as reply:
            return parse_metrics(reply.read().decode())

    def completion(self, port, item, cell, index, warmup, arm, rep):
        before = self.metrics(port)
        row = super().completion(port, item, cell, index, warmup, arm, rep)
        after = self.metrics(port)
        delta = {k: after.get(k, 0)-before.get(k, 0) for k in set(before)|set(after)}
        require(all(v >= 0 for v in delta.values()), 'counter decreased within server')
        require(delta[KEYS[1]] == row['draft_proposed'] and
                delta[KEYS[2]] == row['draft_accepted'], 'metrics/request count disagreement')
        if arm['kind'] != 'target':
            require(delta[KEYS[0]] > 0, 'no native verification rounds')
        prefix = f"{index:02d}-{item['id']}"
        path = cell/(prefix+'-metrics.json')
        atomic(path, dict(before=before, after=after, delta=delta,
            scope='GET metrics snapshots outside full_http_wall_seconds timer'))
        row['native_metrics'] = pin(path)
        row['verification_rounds'] = delta[KEYS[0]]
        row['native_metrics_delta'] = delta
        atomic(cell/(prefix+'-receipt.json'), row, replace=True)
        return row


def qualify(args):
    x = Panel(args)
    x.records = []
    x.stopping = False
    signal.signal(signal.SIGTERM, x.request_stop)
    signal.signal(signal.SIGINT, x.request_stop)
    x.authenticate()
    for arm in x.arms:
        if arm['kind'] == 'w1a8' and arm['runtime'].get('log_verbosity') == 4:
            x.cell(0, arm)
    result = dict(status='NEW_ENDPOINT_LOADER_AND_METRICS_ADMITTED', records=x.records,
                  wall_seconds=time.perf_counter()-x.started, timing_claim=False)
    atomic(args.output/'result.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--models', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repetitions', type=int, default=5)
    parser.add_argument('--qualify', action='store_true')
    parser.add_argument('--allow-cuda', action='store_true')
    args = parser.parse_args()
    if args.qualify:
        qualify(args)
    else:
        Panel(args).run()


if __name__ == '__main__':
    main()

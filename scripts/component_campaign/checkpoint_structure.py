"""Inspect a retained state in RAM; no native export, GPU or disk hydration."""
import argparse
import json
from pathlib import Path

import torch

from .common import R, identity, pin, save
from .tensor_checkpoint import load, tensors
from scripts.dflash_baseline.prototype import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('arm')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    directory = R / 'arms' / args.arm
    result = json.loads((directory / 'result.json').read_text())
    receipt = json.loads((directory / 'recovery.json').read_text())
    archive = pin(receipt['archive'])
    if archive['sha256'] != receipt['archive_sha256'] or archive['bytes'] != receipt['archive_bytes']:
        raise ValueError('Retained checkpoint differs from receipt')
    state = load(receipt['archive'])
    progress = {k: v for k, v in state['progress'].items() if k != 'order'}
    if not state['resume_capable'] or progress != result['progress']:
        raise ValueError('Saved cursor/exposure differs from completed arm')
    if state['config_hash'] != digest(state['config']) or state['component_spec'] != result['spec']:
        raise ValueError('Saved config/spec differs')
    steps = {int(v['step']) for v in state['optimizer']['state'].values()}
    if steps != {progress['updates']} or len(state['rng']) != 4 or state['rng'][3] is None:
        raise ValueError('Incomplete Adam/RNG history')
    groups = {}
    for group in ('parameters', 'optimizer', 'deployment_exps'):
        values = tensors(state.get(group, {}))
        if any(not bool(torch.isfinite(v).all()) for v in values.values()):
            raise ValueError('Nonfinite retained ' + group)
        groups[group] = dict(tensors=len(values), bytes=sum(v.numel()*v.element_size() for v in values.values()))
    order = state['progress']['order']
    if sorted(order) != list(range(len(order))):
        raise ValueError('Saved training order is not a complete permutation')
    save(args.out, dict(status='CPU_MASTER_ADAM_RNG_CURSOR_STRUCTURE_PASS', owner=identity(),
        arm=args.arm, checkpoint=archive, receipt=pin(directory / 'recovery.json'),
        arm_result=pin(directory / 'result.json'), source=pin(__file__), progress=progress,
        optimizer_steps=sorted(steps), groups=groups, order_windows=len(order),
        measured_native=result['readouts'][-1]['model'],
        scope='Actual lossless state loaded entirely in RAM. Masters, Adam, RNG, cursor, exposure and saved deployment values checked. Native re-export equality and an actual resumed optimizer update remain separate qualifications; no finalist promotion or heldout exposure.'))
    print('CPU_RETAINED_STATE_STRUCTURE_PASS', args.arm, flush=True)


if __name__ == '__main__':
    main()

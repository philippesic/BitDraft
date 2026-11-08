"""Freeze only configuration, existing authenticated inputs and source; no old states."""
import copy
import json
import shutil
import subprocess
from .common import *


def main():
    assert not (R/'gpu-owner.json').exists(), 'Cannot replace a launched recipe'
    R.mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True)
    he4_path = COMPONENT/'arms/HE4/resolved.json'
    he4 = json.load(open(he4_path))
    retained = json.load(open(P/'runs/he4-continuation-20261107-01/candidate/resolved.json'))
    assert all(he4['config'][k] == retained['config'][k] for k in
               ('coverage','features','quantizer','components','source','evaluation'))
    broad = [json.load(open(COMPONENT/f'arms/{arm}/spec.json'))['training_data']
             for arm in ('Csmallbroad','H0broad')]
    assert broad[0] == broad[1]
    assert broad[0]['sha256'] == '022cfe8ececc2f69198448dcebd7db216c39197c49eccc819c77b03b2fd11ca2'
    config = copy.deepcopy(he4['config'])
    config.update(parent=config['source'], data=broad[0], budget_seconds=BUDGET)
    assert len(config['coverage']) == 18
    assert all(v == dict(weight='W1',bits=8 if k.endswith('down') else 4)
               for k,v in config['coverage'].items())
    assert config['components'] == dict(fusion='Q4_0',attention_qk='Q4_0',attention_vo='Q4_0')
    assert config['schedule'] == 'constant' and config['quantizer']['mode'] == 'learned'
    spec = dict(name='HE4broad',head='Q4_K',embedding='Q4_K',tied=True,training_data=broad[0])
    save(R/'recipe.json',dict(config=config,spec=spec,architecture_reference=pin(he4_path),
        authorization='Philip Pesic 2026-11-08 ~10:01 America/Los_Angeles; 28800 actual update seconds',
        readout_update_seconds=POINTS,smoke_points=[0,14400,28800],formal_points=[0,28800]))
    # Authenticate the cached private Q4_K matrix and its target ancestry.
    matrix = json.load(open(COMPONENT/'matrices/Q4_K.json'))
    authenticate(matrix['source'])
    from scripts.component_campaign.quant import matrix_data
    import hashlib
    encoded = matrix_data('Q4_K')
    assert hashlib.sha256(encoded).hexdigest() == matrix['matrix']['sha256']
    assert encoded.nbytes == matrix['matrix']['bytes']
    save(R/'q4-matrix-proof.json',dict(status='PASS',original=matrix,
        basis=pin(COMPONENT/'bases/Q4_K.gguf'),matrix_sha256=hashlib.sha256(encoded).hexdigest()))
    assert matrix['source']['sha256'] == config['evaluation']['target']['sha256']
    for ref in (config['source'],config['data'],config['evaluation']['target']): authenticate(ref)
    paths = list((ROOT/'src').rglob('*.py')) + list((ROOT/'scripts').glob('*.py'))
    for name in ('component_campaign','dflash_baseline','execution_campaign','he4broad'):
        paths += list((ROOT/'scripts'/name).glob('*.py'))
    paths.append(ROOT/'experiments/w1a8-standing-20261102/baseline.json')
    for path in paths:
        dest = R/'source'/path.relative_to(ROOT)
        dest.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(path,dest)
    save(R/'source/source-manifest.json',dict(head=subprocess.check_output(
        ['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),files=[pin(R/'source'/p.relative_to(ROOT)) for p in paths]))
    save(REPORT/'manifest.json',dict(status='PREPARED',run=str(R),recipe=pin(R/'recipe.json'),
        frozen_source=pin(R/'source/source-manifest.json'),checkpoint_directory=str(DEST),
        policy='Retain every hourly full state/export; no retained frontier deletion; existing supervised GPU lane.'))
    print('PREPARED',R,flush=True)


if __name__ == '__main__': main()

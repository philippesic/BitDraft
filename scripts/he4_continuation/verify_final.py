"""Recover the completed continuation and qualify its actual next GPU update."""
import json
from scripts.component_campaign.common import pin, save
from scripts.component_campaign.tensor_checkpoint import load
from scripts.component_campaign.export import rewrite
from scripts.component_campaign.gpu_resume import qualify
from scripts.dflash_baseline.export_precision import export
from scripts.he4_continuation.train import R, OUT, COMPONENT


def main():
    import torch
    torch.set_num_threads(2)
    result = json.load(open(OUT/'result.json'))
    assert result['status'] == 'COMPLETE' and result['actual_update_seconds'] >= 21600
    assert result['one_update_overshoot_seconds'] < result['last_update_seconds']
    receipt = result['recovery']
    assert pin(receipt['archive'])['sha256'] == receipt['archive_sha256']
    state = load(receipt['archive'])
    assert state['resume_capable'] and len(state['parameters']) == 104
    assert {k:v for k,v in state['progress'].items() if k!='order'} == result['final_progress']
    steps = {int(v['step']) for v in state['optimizer']['state'].values()}
    assert steps == {state['progress']['updates']}
    dest = OUT/'final-recovery'
    dest.mkdir(exist_ok=False)
    base, native = dest/'base.gguf', dest/'native.gguf'
    export(None,base,state=state,checkpoint_pin=pin(receipt['archive']))
    rewrite(base,native,state['component_spec'],deployment_exps=state['deployment_exps'])
    expected = result['readouts'][-1]['model']
    assert pin(native)['sha256'] == expected['sha256']
    gpu = qualify(state,'HE4',dest,COMPONENT/'arms/HE4')
    save(dest/'result.json',dict(status='FULL_FINAL_STATE_NATIVE_BYTE_IDENTITY_AND_GPU_RESUME_PASS',
        recovery=receipt,native=expected,optimizer_steps=sorted(steps),master_count=104,
        optimizer_groups=[dict({k:v for k,v in group.items() if k!='params'},
                               parameter_count=len(group['params'])) for group in state['optimizer']['param_groups']],
        gpu_resume=gpu,source=pin(__file__),qualification_charged_to_budget=False))
    for p in (base,base.with_suffix('.json'),native,native.with_suffix('.component.json')):
        p.unlink()
    print('FINAL_STATE_FULL_RECOVERY_PASS',flush=True)


if __name__=='__main__': main()

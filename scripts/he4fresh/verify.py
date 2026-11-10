"""Recover the final own-state checkpoint and reproduce its native bytes."""
import gc
import json
import torch
from .common import *
from .clock import UPDATE_BUDGET, POLICY, charged_seconds
from .train import source_model
from scripts.component_campaign.tensor_checkpoint import memory_load
from scripts.component_campaign.gpu_resume import isolated_optimizer_state
from scripts.component_campaign.export import rewrite
from scripts.dflash_baseline.export_precision import export
from scripts.dflash_baseline.pipeline import load_parameters, rng, restore_rng
from scripts.dflash_baseline.prototype_model import optimizer
from scripts.component_campaign.qualify_tensor_checkpoint import fingerprint


def main():
    torch.set_num_threads(2)
    result = json.load(open(OUT/'result.json'))
    receipt = result['recovery']
    assert pin(receipt['archive'])['sha256'] == receipt['archive_sha256']
    state = memory_load(receipt['archive'])
    assert state['progress']['updates'] == result['updates']
    assert state['training_policy']==POLICY and result['training_policy']==POLICY
    assert charged_seconds(state['progress'])>=UPDATE_BUDGET
    from .audit import audit
    exposure=audit()
    assert exposure['completed_updates']==result['updates'] and exposure['uncheckpointed_completed_updates']==0
    assert abs(exposure['actual_update_seconds']-state['progress']['update_seconds'])<1e-5
    assert {int(v['step']) for v in state['optimizer']['state'].values()} == {result['updates']}
    model = source_model(state['config'],state['component_spec']).to('cuda')
    opt = optimizer(model)
    load_parameters(model,state['parameters'])
    opt.load_state_dict(isolated_optimizer_state(state['optimizer']))
    assert all(torch.equal(p.detach().cpu(),state['parameters'][n]) for n,p in model.named_parameters())
    restored = opt.state_dict()
    assert restored['param_groups'] == state['optimizer']['param_groups']
    for key, fields in state['optimizer']['state'].items():
        for field, expected in fields.items():
            actual = restored['state'][key][field]
            assert torch.equal(actual.cpu(),expected.cpu()) if torch.is_tensor(expected) else actual == expected
    restore_rng(state['rng']); assert fingerprint(rng()) == fingerprint(state['rng'])
    del model,opt,restored; gc.collect(); torch.cuda.empty_cache()
    base = OUT/'recovery-base.gguf'; native = OUT/'recovered-final.gguf'
    export(None,base,state=state,checkpoint_pin=pin(receipt['archive']))
    rewrite(base,native,state['component_spec'],deployment_exps=state['deployment_exps'])
    assert pin(native)['sha256'] == result['readouts'][-1]['model']['sha256']
    records = []
    for point in result['readouts']:
        rec = point['recovery']; assert pin(rec['archive'])['sha256'] == rec['archive_sha256']
        assert pin(point['model']['path']) == point['model']
        records.append(dict(checkpoint=rec,model=point['model']))
    save(R/'final-recovery.json',dict(status='FULL_STATE_GPU_RESTORE_AND_NATIVE_BYTE_IDENTITY_PASS',
        recovery=receipt, native=pin(native), retained=records, parameters_equal=True,
        adam_moments_groups_steps_equal=True,rng_equal=True,
        extra_optimizer_updates=0, note='Hourly resumes execute the next real budgeted update. Final restore adds no update beyond the authorized budget.'))
    # The verified duplicate is reproducible from the retained full state/export.
    for path in (base,base.with_suffix('.json'),native,native.with_suffix('.component.json')): path.unlink()
    print('FINAL_RECOVERY_PASS',flush=True)


if __name__ == '__main__': main()

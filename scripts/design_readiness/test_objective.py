"""Independent loss/mask checks for the fixed depth2 ablation."""
import unittest
import copy
import torch
from fixed_objective import ordinary_ce, RECIPES, SCHEMA, validate_local_state


class ObjectiveTests(unittest.TestCase):
    def test_noop_value_and_gradient(self):
        x = torch.tensor([[1., 3., -2.], [4., -1., 2.]], requires_grad=True)
        y = torch.tensor([1, 2])
        actual, n = ordinary_ce(x, y, [True, True], RECIPES['control'])
        expected = -(x.log_softmax(-1)[0, 1] + x.log_softmax(-1)[1, 2])/4
        self.assertEqual(n, 2)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(torch.autograd.grad(actual, x, retain_graph=True)[0],
                                   torch.autograd.grad(expected, x)[0], rtol=0, atol=0)

    def test_static_reachable_and_unsupported_mask(self):
        x = torch.randn(2, 7, requires_grad=True)
        loss, count = ordinary_ce(x, torch.tensor([-1, 2]), [True, False], RECIPES['combined'])
        self.assertEqual(count, 0)
        self.assertEqual(loss.item(), 0.)
        self.assertTrue(torch.equal(torch.autograd.grad(loss, x)[0], torch.zeros_like(x)))

    def test_smoothing_and_weight_reference(self):
        x = torch.tensor([[1., 0., 2.], [2., -2., 1.]], requires_grad=True)
        y = torch.tensor([0, 2])
        logp = x.log_softmax(-1)
        per = -.95*logp[torch.arange(2), y]-.05*logp.mean(-1)
        expected = (per*torch.tensor([1.5, .5])).sum()/4
        loss, count = ordinary_ce(x, y, [True, True], RECIPES['combined'])
        self.assertEqual(count, 2)
        torch.testing.assert_close(loss, expected)

    def test_depth_guard(self):
        with self.assertRaises(ValueError):
            ordinary_ce(torch.zeros(3, 4), torch.zeros(3, dtype=torch.long), [True]*3,
                        RECIPES['control'])

    def test_resume_rejects_corruption(self):
        parameters = [torch.nn.Parameter(torch.zeros(3)) for _ in range(9)]
        optimizer = torch.optim.Adam(parameters, lr=1e-5)
        sum(p.sum() for p in parameters).backward()
        optimizer.step()
        state = optimizer.state_dict()
        for row in state['state'].values():
            row['step'].fill_(5808)
        exposure = dict(ce_windows=96, labels=123, old_replay_tokens=50, added_replay_tokens=56)
        saved = dict(schema=SCHEMA, contract={'recipe': 'control'}, cursor=5808,
                     optimizer=state, local_exposure=exposure, local_trainer_seconds=5.)
        validate_local_state(saved, saved['contract'], [(3,)]*9, exposure)
        for mutate in [
            lambda s: s['optimizer']['param_groups'][0].update(lr=.1),
            lambda s: s['optimizer']['state'][0]['step'].fill_(2),
            lambda s: s['optimizer']['state'][0]['exp_avg'].fill_(float('nan')),
            lambda s: s['optimizer']['state'][0]['exp_avg_sq'].fill_(-1),
            lambda s: s['local_exposure'].update(labels=124),
            lambda s: s.update(local_trainer_seconds=float('inf')),
        ]:
            changed = copy.deepcopy(saved)
            mutate(changed)
            with self.assertRaises(ValueError):
                validate_local_state(changed, saved['contract'], [(3,)]*9, exposure)


if __name__ == '__main__':
    unittest.main()

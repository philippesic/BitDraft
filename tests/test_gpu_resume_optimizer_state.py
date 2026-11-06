"""Scratch Adam qualification must not advance the retained CPU step counter."""
import unittest
import torch
from scripts.component_campaign.gpu_resume import isolated_optimizer_state


class ScratchAdamStateTest(unittest.TestCase):
    def test_cpu_step_survives_scratch_optimizer_update(self):
        parameter = torch.nn.Parameter(torch.tensor([1.]))
        original = torch.optim.AdamW([parameter])
        parameter.grad = torch.ones_like(parameter)
        original.step()
        saved = original.state_dict()
        scratch_parameter = torch.nn.Parameter(torch.tensor([1.]))
        scratch = torch.optim.AdamW([scratch_parameter])
        scratch.load_state_dict(isolated_optimizer_state(saved))
        scratch_parameter.grad = torch.ones_like(scratch_parameter)
        scratch.step()
        self.assertEqual(int(saved['state'][0]['step']), 1)
        self.assertEqual(int(scratch.state_dict()['state'][0]['step']), 2)


if __name__ == '__main__': unittest.main()

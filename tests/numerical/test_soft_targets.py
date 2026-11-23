import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from w1a1_eagle.contracts import Record
from w1a1_eagle.data import Artifact
from w1a1_eagle.numerical.captures import soft_targets
from w1a1_eagle.numerical.worker import soft_loss


class TestSoftTargets(unittest.TestCase):
    def test_loss_matches_cross_entropy_with_a_residual_bucket(self) -> None:
        generator = torch.Generator().manual_seed(3)
        logits = torch.randn(4, 10, generator=generator, requires_grad=True)
        target = torch.softmax(torch.randn(4, 10, generator=generator), -1)
        ids = target.topk(3, -1).indices
        logprobs = target.gather(-1, ids).log()
        loss = soft_loss(logits, ids, logprobs)
        predicted = torch.log_softmax(logits, -1)
        inside = predicted.gather(-1, ids)
        rest = 1 - predicted.exp().gather(-1, ids).sum(-1)
        residual = 1 - target.gather(-1, ids).sum(-1)
        expected = -(target.gather(-1, ids) * inside).sum(-1) - residual * rest.log()
        self.assertTrue(torch.allclose(loss, expected, atol=1e-5))
        loss.sum().backward()
        assert logits.grad is not None
        self.assertTrue(bool(torch.isfinite(logits.grad).all()))
        full = target.topk(10, -1).indices
        complete = soft_loss(logits.detach(), full, target.gather(-1, full).log())
        cross = -(target * torch.log_softmax(logits.detach(), -1)).sum(-1)
        self.assertTrue(torch.allclose(complete, cross, atol=1e-5))

    def test_loader_maps_positions_and_refuses_bad_geometry(self) -> None:
        with tempfile.TemporaryDirectory(prefix="soft-targets-") as directory:
            root = Path(directory)
            ids = np.array([[5, 1], [2, 7]], dtype=np.int32)
            logprobs = np.array([[-0.1, -2.5], [-0.3, -1.4]], dtype=np.float32)
            rows = b"".join(ids[i].tobytes() + logprobs[i].tobytes() for i in range(2))
            (root / "soft.bin").write_bytes(rows)
            ref = Artifact.capture(root / "soft.bin")
            metadata: Record = {
                "logit_positions": [9, 10],
                "soft_targets": {"file": "soft.bin", "k": 2, "positions": 2},
            }
            loaded = soft_targets(metadata, {"soft.bin": ref})
            assert loaded is not None
            self.assertEqual(loaded.rows, {9: 0, 10: 1})
            self.assertEqual(loaded.ids.tolist(), [[5, 1], [2, 7]])
            self.assertTrue(torch.allclose(loaded.logprobs, torch.tensor(logprobs)))
            self.assertIsNone(soft_targets({"logit_positions": [9, 10]}, {}))
            with self.assertRaisesRegex(ValueError, "geometry"):
                soft_targets(
                    metadata | {"soft_targets": {"file": "soft.bin", "k": 3, "positions": 2}},
                    {"soft.bin": ref},
                )

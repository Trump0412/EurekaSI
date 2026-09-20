import math
import unittest
import torch
from spatial_intelligence.geometry_rft_objective import response_mask, group_standardized_advantages, gspo_loss


class ObjectiveTests(unittest.TestCase):
    def test_eos_and_padding(self):
        ids = torch.tensor([[4, 2, 2, 2], [4, 5, 6, 2]])
        self.assertEqual(response_mask(ids, 2, 2).tolist(), [[True, True, False, False], [True]*4])
        self.assertEqual(response_mask(torch.tensor([[4, 2, 0]]), 2, 0).tolist(), [[True, True, False]])

    def test_group_standardization(self):
        r = torch.tensor([[1., 3.], [100., 100.]], requires_grad=True)
        a = group_standardized_advantages(r)
        self.assertFalse(a.requires_grad)
        self.assertTrue(torch.allclose(a[0], torch.tensor([-1., 1.]), atol=2e-6))
        self.assertEqual(a[1].tolist(), [0., 0.])

    def test_hand_value_and_clipped_gradient(self):
        new = torch.tensor([[math.log(1.2)]*2, [math.log(.8)]*2], dtype=torch.float64, requires_grad=True)
        zeros = torch.zeros_like(new)
        loss, metrics = gspo_loss(new, zeros, zeros, torch.tensor([1., -1.]), torch.ones_like(new), beta=0)
        self.assertAlmostEqual(float(loss), (-1.0004 + .9997)/2, places=12)
        loss.backward()
        self.assertTrue(torch.equal(new.grad, zeros))
        self.assertEqual(metrics["clip_fraction"], 1)

    def test_masked_nan_no_gradient_and_detached_reference(self):
        new = torch.tensor([[0., float("nan")]], dtype=torch.float64, requires_grad=True)
        old = torch.tensor([[0., float("nan")]], dtype=torch.float64, requires_grad=True)
        ref = torch.tensor([[0., float("nan")]], dtype=torch.float64, requires_grad=True)
        a = torch.tensor([1.], requires_grad=True)
        loss, _ = gspo_loss(new, old, ref, a, torch.tensor([[1, 0]]), beta=.02)
        loss.backward()
        self.assertEqual(new.grad.tolist(), [[-1., 0.]])
        self.assertIsNone(old.grad)
        self.assertIsNone(ref.grad)
        self.assertIsNone(a.grad)

    def test_length_normalization(self):
        for length in (1, 5):
            new = torch.zeros((1, length), dtype=torch.float64, requires_grad=True)
            loss, _ = gspo_loss(new, new.detach(), new.detach(), torch.ones(1), torch.ones_like(new), beta=0)
            loss.backward()
            self.assertEqual(float(loss), -1)
            self.assertAlmostEqual(float(new.grad.sum()), -1)

    def test_zero_variance_and_kl(self):
        new = torch.full((2, 2), .1, dtype=torch.float64, requires_grad=True)
        zeros = torch.zeros_like(new)
        a = group_standardized_advantages(torch.ones(2))
        loss, _ = gspo_loss(new, zeros, zeros, a, torch.ones_like(new), beta=0)
        loss.backward()
        self.assertEqual(float(new.grad.abs().sum()), 0)
        new.grad = None
        loss, _ = gspo_loss(new, zeros, zeros, a, torch.ones_like(new), beta=.02)
        self.assertAlmostEqual(float(loss), .02 * (math.exp(-.1) + .1 - 1))
        loss.backward()
        self.assertGreater(float(new.grad.sum()), 0)

    def test_empty_response_rejected(self):
        with self.assertRaises(ValueError):
            gspo_loss(torch.zeros(1,2), torch.zeros(1,2), torch.zeros(1,2), torch.ones(1), torch.zeros(1,2))


if __name__ == "__main__":
    unittest.main()

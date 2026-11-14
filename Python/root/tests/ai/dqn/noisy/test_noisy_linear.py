
import unittest

import torch
from torch import nn

from core.ai.dqn.noisy.dqn_noisy_linear import NoisyLinear


class TestNoisyLinear(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        self.layer = NoisyLinear(512, 1032)
        self.x = torch.randn(4, 512)

    def test_output_shape(self):
        output = self.layer(self.x)
        self.assertEqual(output.shape, (4, 1032))

    def test_deterministic_output(self):
        self.layer.disable_noise()

        output1 = self.layer(self.x)
        output2 = self.layer(self.x)

        self.assertTrue(torch.equal(output1, output2))

    def test_noise_changes_output(self):
        self.layer.enable_noise()

        self.layer.reset_noise()
        output1 = self.layer(self.x)

        self.layer.reset_noise()
        output2 = self.layer(self.x)

        self.assertFalse(torch.equal(output1, output2))

    def test_noise_remains_fixed_without_reset(self):
        self.layer.enable_noise()
        self.layer.reset_noise()

        output1 = self.layer(self.x)
        output2 = self.layer(self.x)

        self.assertTrue(torch.equal(output1, output2))

    def test_noise_disabled_matches_mean_parameters(self):
        self.layer.disable_noise()

        output = self.layer(self.x)

        expected = nn.functional.linear(
            self.x,
            self.layer.weight_mu,
            self.layer.bias_mu,
        )

        self.assertTrue(torch.allclose(output, expected))

    def test_noise_parameters_receive_gradients(self):
        self.layer.enable_noise()
        self.layer.reset_noise()

        self.layer.zero_grad()
        output = self.layer(self.x)
        output.sum().backward()

        self.assertIsNotNone(self.layer.weight_sigma.grad)
        self.assertIsNotNone(self.layer.bias_sigma.grad)

        self.assertGreater(
            self.layer.weight_sigma.grad.abs().sum().item(),
            0.0,
        )

        self.assertGreater(
            self.layer.bias_sigma.grad.abs().sum().item(),
            0.0,
        )

    def test_parameters_update(self):
        self.layer.enable_noise()

        optimizer = torch.optim.Adam(
            self.layer.parameters(),
            lr=0.001,
        )

        original_mu = self.layer.weight_mu.detach().clone()
        original_sigma = self.layer.weight_sigma.detach().clone()

        optimizer.zero_grad()
        output = self.layer(self.x)
        loss = output.square().mean()
        loss.backward()
        optimizer.step()

        self.assertFalse(
            torch.equal(original_mu, self.layer.weight_mu)
        )

        self.assertFalse(
            torch.equal(original_sigma, self.layer.weight_sigma)
        )

    def test_noise_buffers_not_trainable(self):
        parameters = dict(self.layer.named_parameters())

        self.assertNotIn("weight_epsilon", parameters)
        self.assertNotIn("bias_epsilon", parameters)

    def test_no_nan_or_infinite_values(self):
        self.layer.enable_noise()
        self.layer.reset_noise()

        output = self.layer(self.x)

        self.assertTrue(torch.isfinite(output).all().item())


if __name__ == "__main__":
    unittest.main(verbosity=2)

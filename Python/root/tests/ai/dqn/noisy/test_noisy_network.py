
import unittest
from unittest.mock import patch

import torch
from torch import nn

from core.ai.dqn.dqn_network import DQNNetwork
from core.ai.dqn.noisy.dqn_noisy_network import DQNNoisyNetwork
from core.ai.dqn.noisy.dqn_noisy_linear import NoisyLinear


class TestNoisyNetwork(unittest.TestCase):

    def setUp(self):
        self.params = {
            "hidden_layers": [512, 512, 512],
            "noisy_sigma0": 0.5,
        }

    def create_noisy_network(self):
        with patch(
            "core.ai.dqn.noisy.dqn_noisy_network.load_hyperparameters",
            return_value=self.params,
        ):
            return DQNNoisyNetwork(967, 1032)

    def test_noisy_layer_count(self):
        network = self.create_noisy_network()

        noisy_layers = [
            module
            for module in network.modules()
            if isinstance(module, NoisyLinear)
        ]

        self.assertEqual(len(noisy_layers), 2)

    def test_standard_layer_count(self):
        network = self.create_noisy_network()

        linear_layers = [
            module
            for module in network.modules()
            if isinstance(module, nn.Linear)
        ]

        self.assertEqual(len(linear_layers), 2)

    def test_output_shape(self):
        network = self.create_noisy_network()

        x = torch.randn(4, 967)
        output = network(x)

        self.assertEqual(output.shape, (4, 1032))

    def test_deterministic_mode(self):
        network = self.create_noisy_network()

        x = torch.randn(4, 967)

        network.disable_noise()

        output1 = network(x)
        output2 = network(x)

        self.assertTrue(torch.equal(output1, output2))

    def test_noise_resampling(self):
        network = self.create_noisy_network()

        x = torch.randn(4, 967)

        network.enable_noise()

        network.reset_noise()
        output1 = network(x)

        network.reset_noise()
        output2 = network(x)

        self.assertFalse(torch.equal(output1, output2))

    def test_existing_network_unchanged(self):
        with patch(
            "core.ai.dqn.noisy.dqn_noisy_network.load_hyperparameters",
            return_value=self.params,
        ):
            original = DQNNetwork(967, 1032)

        noisy_layers = [
            module
            for module in original.modules()
            if isinstance(module, NoisyLinear)
        ]

        self.assertEqual(len(noisy_layers), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

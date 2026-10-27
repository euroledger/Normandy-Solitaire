
import unittest
import numpy as np
import torch

from core.ai.dqn.dqn_agent import DQNAgent


class TestDQNAgent(unittest.TestCase):

    def setUp(self):
        # Small network with predictable Q-values
        self.network = torch.nn.Linear(4, 5)

        with torch.no_grad():
            self.network.weight.zero_()
            self.network.bias.copy_(
                torch.tensor([10.0, 2.0, 8.0, 5.0, 1.0])
            )

        self.agent = DQNAgent(self.network, torch.device("cpu"))
        self.observation = np.zeros(4, dtype=np.float32)
        self.mask = np.array([0, 1, 0, 1, 0], dtype=np.int8)

    def test_hyperparameters(self):
        # Check parameters loaded from YAML
        self.assertEqual(self.agent.epsilon, 1.0)
        self.assertEqual(self.agent.mini_batch_size, 64)
        self.assertEqual(self.agent.discount_factor, 0.99)

    def test_exploitation(self):
        # Action 0 has the highest Q-value but is illegal
        self.agent.epsilon = 0.0

        action = self.agent.select_action(
            self.observation, self.mask
        )

        self.assertEqual(action, 3)

    def test_exploration(self):
        # Random selection must never choose an illegal action
        self.agent.epsilon = 1.0

        actions = {
            self.agent.select_action(self.observation, self.mask)
            for _ in range(1000)
        }

        self.assertEqual(actions, {1, 3})

    def test_no_legal_actions(self):
        # An empty mask must raise an error
        with self.assertRaises(ValueError):
            self.agent.select_action(
                self.observation,
                np.zeros(5, dtype=np.int8)
            )


if __name__ == "__main__":
    unittest.main()

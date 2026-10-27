
import unittest
import numpy as np
import torch

from core.ai.dqn.dqn_network import DQNNetwork
from core.ai.dqn.dqn_agent import DQNAgent
from core.ai.dqn.dqn_game_state_encoder import OBSERVATION_SIZE
from core.ai.headless_action_masks import get_total_actions


class TestDQNNetwork(unittest.TestCase):

    def setUp(self):
        self.action_size, _ = get_total_actions()
        self.network = DQNNetwork(
            OBSERVATION_SIZE,
            self.action_size,
        )

    def test_network_output_shape(self):
        observation = torch.zeros(OBSERVATION_SIZE)

        q_values = self.network(observation)

        self.assertEqual(q_values.shape, (self.action_size,))

    def test_agent_selects_legal_action(self):
        agent = DQNAgent(self.network, torch.device("cpu"))
        agent.epsilon = 0.0

        observation = np.zeros(OBSERVATION_SIZE, dtype=np.float32)

        mask = np.zeros(self.action_size, dtype=np.float32)
        mask[10] = 1.0
        mask[20] = 1.0

        action_id = agent.select_action(observation, mask)

        self.assertIn(action_id, [10, 20])


if __name__ == "__main__":
    unittest.main()

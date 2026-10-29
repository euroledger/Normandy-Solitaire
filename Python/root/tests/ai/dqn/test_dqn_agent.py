
import unittest
from unittest.mock import patch
import numpy as np
import torch

from core.ai.dqn.dqn_agent import DQNAgent
from core.ai.dqn.dqn_replay_buffer import DQNReplayBuffer


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
        self.assertEqual(self.agent.epsilon, 0.9)
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


    def test_replay_buffer_initialization(self):
        # Check replay buffer is created with the configured capacity
        self.assertIsInstance(
            self.agent.replay_buffer,
            DQNReplayBuffer,
        )

        self.assertEqual(len(self.agent.replay_buffer), 0)

        self.assertEqual(
            self.agent.replay_buffer.buffer.maxlen,
            self.agent.replay_memory_size,
        )
        

    def test_select_actions_batched_greedy(self):

        observations = np.zeros(
            (4, self.agent.policy_net.in_features),
            dtype=np.float32,
        )

        masks = np.zeros(
            (4, self.agent.policy_net.out_features),
            dtype=np.float32,
        )

        # Give each worker exactly one legal action.
        masks[0, 1] = 1.0
        masks[1, 2] = 1.0
        masks[2, 3] = 1.0
        masks[3, 4] = 1.0

        self.agent.epsilon = 0.0

        with patch.object(
            self.agent.policy_net,
            "forward",
            wraps=self.agent.policy_net.forward,
        ) as mock_forward:

            actions = self.agent.select_actions(
                observations,
                masks,
            )

        np.testing.assert_array_equal(
            actions,
            [1, 2, 3, 4],
        )

        self.assertEqual(
            mock_forward.call_count,
            1,
        )
        

    def test_select_actions_skips_reset_workers(self):

        observations = np.zeros(
            (4, self.agent.policy_net.in_features),
            dtype=np.float32,
        )

        masks = np.zeros(
            (4, self.agent.policy_net.out_features),
            dtype=np.float32,
        )

        masks[0, 1] = 1.0
        masks[2, 3] = 1.0

        # Workers 1 and 3 are awaiting automatic reset.
        needs_reset = np.array(
            [False, True, False, True],
            dtype=bool,
        )

        self.agent.epsilon = 0.0

        actions = self.agent.select_actions(
            observations,
            masks,
            needs_reset,
        )

        np.testing.assert_array_equal(
            actions,
            [1, 0, 3, 0],
        )
if __name__ == "__main__":
    unittest.main()

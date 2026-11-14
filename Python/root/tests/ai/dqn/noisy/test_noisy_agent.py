
import unittest
from unittest.mock import patch

import numpy as np
import torch

from core.ai.dqn.noisy.dqn_noisy_network import DQNNoisyNetwork
from core.ai.dqn.noisy.dqn_noisy_agent import NoisyDQNAgent


class TestNoisyDQNAgent(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)

        self.observation_size = 12
        self.action_size = 8
        self.device = torch.device("cpu")

        self.params = {
            "hidden_layers": [32, 32, 32],
            "noisy_sigma0": 0.5,
            "gradient_clip_norm": 10.0,
            "replay_memory_size": 100,
            "mini_batch_size": 4,
            "learning_rate": 0.001,
            "discount_factor": 0.99,
            "epsilon_init": 0.2,
            "epsilon_decay": 0.999,
            "epsilon_min": 0.02,
            "n_step": 6,
            "network_sync_rate": 2,
            "enable_double_dqn": True,
            "enable_dueling_dqn": False,
            "replay_buffer_type": "uniform",
            "train_frequency": 1,
        }

        with patch(
            "core.ai.dqn.noisy.dqn_noisy_network.load_hyperparameters",
            return_value=self.params,
        ):
            network = DQNNoisyNetwork(
                self.observation_size,
                self.action_size,
            )

        with patch(
            "core.ai.dqn.dqn_agent.load_hyperparameters",
            return_value=self.params,
        ):
            self.agent = NoisyDQNAgent(
                network,
                self.device,
            )

    def make_experience(self):
        observation = np.random.randn(
            self.observation_size
        ).astype(np.float32)

        next_observation = np.random.randn(
            self.observation_size
        ).astype(np.float32)

        next_mask = np.ones(
            self.action_size,
            dtype=bool,
        )

        return (
            observation,
            2,
            1.0,
            next_observation,
            False,
            next_mask,
            6,
        )

    def populate_replay(self, count=8):
        for _ in range(count):
            self.agent.replay_buffer.add(
                *self.make_experience()
            )

    def test_initialisation(self):
        self.assertEqual(self.agent.epsilon, 0.0)
        self.assertEqual(self.agent.epsilon_min, 0.0)
        self.assertEqual(self.agent.epsilon_decay, 1.0)

        self.assertTrue(self.agent.enable_double_dqn)
        self.assertEqual(self.agent.n_step, 6)
        self.assertEqual(
            self.agent.replay_buffer_type,
            "uniform",
        )

    def test_single_action_respects_mask(self):
        observation = np.random.randn(
            self.observation_size
        ).astype(np.float32)

        mask = np.zeros(self.action_size, dtype=bool)
        mask[3] = True

        for _ in range(20):
            action = self.agent.select_action(
                observation,
                mask,
            )
            self.assertEqual(action, 3)

    def test_single_action_rejects_empty_mask(self):
        observation = np.zeros(
            self.observation_size,
            dtype=np.float32,
        )
        mask = np.zeros(
            self.action_size,
            dtype=bool,
        )

        with self.assertRaises(ValueError):
            self.agent.select_action(
                observation,
                mask,
            )

    def test_parallel_actions_respect_masks(self):
        observations = np.random.randn(
            4,
            self.observation_size,
        ).astype(np.float32)

        masks = np.zeros(
            (4, self.action_size),
            dtype=bool,
        )

        masks[0, 1] = True
        masks[1, 3] = True
        masks[2, 5] = True
        masks[3, 7] = True

        actions = self.agent.select_actions(
            observations,
            masks,
        )

        np.testing.assert_array_equal(
            actions,
            np.array([1, 3, 5, 7]),
        )

    def test_parallel_worker_reset(self):
        observations = np.random.randn(
            3,
            self.observation_size,
        ).astype(np.float32)

        masks = np.zeros(
            (3, self.action_size),
            dtype=bool,
        )

        masks[0, 2] = True
        masks[2, 6] = True

        actions = self.agent.select_actions(
            observations,
            masks,
            needs_reset=np.array(
                [False, True, False]
            ),
        )

        self.assertEqual(actions[0], 2)
        self.assertEqual(actions[2], 6)

    def test_parallel_rejects_empty_active_mask(self):
        observations = np.zeros(
            (2, self.observation_size),
            dtype=np.float32,
        )

        masks = np.zeros(
            (2, self.action_size),
            dtype=bool,
        )

        masks[0, 1] = True

        with self.assertRaises(ValueError):
            self.agent.select_actions(
                observations,
                masks,
            )

    def test_noise_resampled_for_single_action(self):
        observation = np.zeros(
            self.observation_size,
            dtype=np.float32,
        )

        mask = np.ones(
            self.action_size,
            dtype=bool,
        )

        with patch.object(
            self.agent.policy_net,
            "reset_noise",
            wraps=self.agent.policy_net.reset_noise,
        ) as reset:
            self.agent.select_action(
                observation,
                mask,
            )
            self.assertEqual(reset.call_count, 1)

    def test_noise_resampled_for_parallel_batch(self):
        observations = np.zeros(
            (4, self.observation_size),
            dtype=np.float32,
        )

        masks = np.ones(
            (4, self.action_size),
            dtype=bool,
        )

        with patch.object(
            self.agent.policy_net,
            "reset_noise",
            wraps=self.agent.policy_net.reset_noise,
        ) as reset:
            self.agent.select_actions(
                observations,
                masks,
            )
            self.assertEqual(reset.call_count, 1)

    def test_deterministic_mode(self):
        observation = np.random.randn(
            self.observation_size
        ).astype(np.float32)

        mask = np.ones(
            self.action_size,
            dtype=bool,
        )

        self.agent.set_deterministic()

        actions = [
            self.agent.select_action(
                observation,
                mask,
            )
            for _ in range(20)
        ]

        self.assertEqual(len(set(actions)), 1)

        self.agent.set_noisy()

    def test_training_updates_mean_and_sigma(self):
        self.populate_replay()

        layer = self.agent.policy_net.network[6]

        original_mu = (
            layer.weight_mu.detach().clone()
        )
        original_sigma = (
            layer.weight_sigma.detach().clone()
        )

        loss = self.agent.train_step()

        self.assertIsNotNone(loss)
        self.assertTrue(np.isfinite(loss))

        self.assertFalse(
            torch.equal(
                original_mu,
                layer.weight_mu,
            )
        )

        self.assertFalse(
            torch.equal(
                original_sigma,
                layer.weight_sigma,
            )
        )

    def test_training_resamples_both_networks(self):
        self.populate_replay()

        with (
            patch.object(
                self.agent.policy_net,
                "reset_noise",
                wraps=self.agent.policy_net.reset_noise,
            ) as policy_reset,
            patch.object(
                self.agent.target_net,
                "reset_noise",
                wraps=self.agent.target_net.reset_noise,
            ) as target_reset,
        ):
            self.agent.train_step()

            self.assertEqual(
                policy_reset.call_count, 1
            )
            self.assertEqual(
                target_reset.call_count, 1
            )

    def test_target_network_sync(self):
        self.populate_replay()

        self.agent.train_step()
        self.agent.train_step()

        policy_state = (
            self.agent.policy_net.state_dict()
        )
        target_state = (
            self.agent.target_net.state_dict()
        )

        for key in policy_state:
            self.assertTrue(
                torch.equal(
                    policy_state[key],
                    target_state[key],
                ),
                msg=f"Target mismatch: {key}",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)

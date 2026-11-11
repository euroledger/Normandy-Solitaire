import random
import unittest

import numpy as np

from core.ai.dqn.dqn_priority_replay_buffer import DQNPriorityReplayBuffer
from core.ai.dqn.dqn_replay_buffer import DQNReplayBuffer
from core.ai.dqn.dqn_agent import DQNAgent
from core.ai.dqn.dqn_n_step import NStepTransitionCollector


class TestDQNPriorityReplayBuffer(unittest.TestCase):

    def setUp(self):
        random.seed(12345)
        self.buffer = DQNPriorityReplayBuffer(capacity=10)

        self.observation = np.zeros(967, dtype=np.float32)
        self.next_observation = np.ones(967, dtype=np.float32)

        self.next_action_mask = np.zeros(1032, dtype=np.float32)
        self.next_action_mask[5] = 1.0

    def add_experience(self, action_id=5, reward=1.0, terminated=False, buffer=None):
        target = self.buffer if buffer is None else buffer
        target.add(
            observation=self.observation,
            action_id=action_id,
            reward=reward,
            next_observation=self.next_observation,
            terminated=terminated,
            next_action_mask=self.next_action_mask,
        )

    def test_add_experience(self):
        # A transition can be stored and sampled with its original values.
        self.assertEqual(len(self.buffer), 0)
        self.add_experience()
        self.assertEqual(len(self.buffer), 1)

        batch = self.buffer.sample(1)
        self.assertEqual(batch[1][0], 5)
        self.assertEqual(batch[2][0], 1.0)
        self.assertFalse(batch[4][0])
        self.assertEqual(batch[6][0], 0)
        self.assertAlmostEqual(float(batch[7][0]), 1.0)
        self.assertEqual(batch[8][0], 1)

    def test_n_step_experience(self):
        self.buffer.add(
            observation=self.observation,
            action_id=5,
            reward=2.0,
            next_observation=self.next_observation,
            terminated=False,
            next_action_mask=self.next_action_mask,
            n_steps=3,
        )
        batch = self.buffer.sample(1)
        self.assertEqual(len(batch), 9)
        self.assertEqual(batch[8][0], 3)
        self.assertEqual(self.buffer.buffer[0][6], 3)

    def test_legacy_experience_compatibility(self):
        self.add_experience()
        # Simulate a six-field transition loaded from an old checkpoint.
        self.buffer.buffer[0] = self.buffer.buffer[0][:6]
        batch = self.buffer.sample(1)
        self.assertEqual(len(batch), 9)
        self.assertEqual(batch[8][0], 1)

    def test_capacity_limit_and_fifo_replacement(self):
        # After capacity is reached, new transitions replace the oldest ones.
        for i in range(15):
            self.add_experience(action_id=i)

        self.assertEqual(len(self.buffer), 10)
        stored_actions = {experience[1] for experience in self.buffer.buffer}
        self.assertEqual(stored_actions, set(range(5, 15)))

    def test_sample_shapes_and_types(self):
        # PER returns six transition arrays, indices, weights, and n-step counts.
        for i in range(5):
            self.add_experience(action_id=i)

        batch = self.buffer.sample(3)
        self.assertEqual(len(batch), 9)
        observations, actions, rewards, next_observations, terminated, next_masks, indices, weights, n_steps = batch

        self.assertEqual(observations.shape, (3, 967))
        self.assertEqual(actions.shape, (3,))
        self.assertEqual(rewards.shape, (3,))
        self.assertEqual(next_observations.shape, (3, 967))
        self.assertEqual(terminated.shape, (3,))
        self.assertEqual(next_masks.shape, (3, 1032))
        self.assertEqual(indices.shape, (3,))
        self.assertEqual(weights.shape, (3,))
        self.assertEqual(n_steps.shape, (3,))

        self.assertEqual(observations.dtype, np.float32)
        self.assertEqual(actions.dtype, np.int64)
        self.assertEqual(rewards.dtype, np.float32)
        self.assertEqual(next_observations.dtype, np.float32)
        self.assertEqual(terminated.dtype, np.bool_)
        self.assertEqual(next_masks.dtype, np.float32)
        self.assertEqual(indices.dtype, np.int64)
        self.assertEqual(weights.dtype, np.float32)
        self.assertEqual(n_steps.dtype, np.int64)
        np.testing.assert_array_equal(n_steps, [1, 1, 1])
        self.assertTrue(np.all(weights > 0))
        self.assertTrue(np.all(weights <= 1))
        self.assertTrue(np.all(indices < len(self.buffer)))

    def test_sample_too_many(self):
        # Sampling requires at least batch_size stored transitions.
        self.add_experience()
        with self.assertRaises(ValueError):
            self.buffer.sample(2)

    def test_observation_copies(self):
        # Changes to the caller's arrays must not affect stored transitions.
        self.add_experience()
        self.observation[0] = 99.0
        self.next_observation[0] = 99.0
        self.next_action_mask[5] = 0.0

        batch = self.buffer.sample(1)
        self.assertEqual(batch[0][0, 0], 0.0)
        self.assertEqual(batch[3][0, 0], 1.0)
        self.assertEqual(batch[5][0, 5], 1.0)

    def test_priority_updates_change_sampling_probabilities(self):
        # With alpha=1, a much larger priority should dominate samples.
        buffer = DQNPriorityReplayBuffer(capacity=4, alpha=1.0)
        for i in range(4):
            self.add_experience(action_id=i, buffer=buffer)

        buffer.update_priorities([0, 1, 2, 3], [1.0, 1.0, 1.0, 100.0])
        counts = np.zeros(4, dtype=np.int64)
        for _ in range(500):
            indices = buffer.sample(1)[6]
            counts[indices[0]] += 1

        self.assertGreater(counts[3], 400)
        self.assertGreater(counts[3], counts[:3].sum())

    def test_alpha_zero_gives_equal_priorities(self):
        # alpha=0 disables prioritization regardless of TD errors.
        buffer = DQNPriorityReplayBuffer(capacity=4, alpha=0.0)
        for i in range(4):
            self.add_experience(action_id=i, buffer=buffer)

        buffer.update_priorities([0, 1, 2, 3], [0.01, 1.0, 10.0, 100.0])
        leaves = buffer.sum_tree[buffer.tree_capacity:buffer.tree_capacity + 4]
        np.testing.assert_allclose(leaves, np.ones(4))

    def test_importance_weights(self):
        # A lower-probability transition should receive a higher correction weight.
        buffer = DQNPriorityReplayBuffer(capacity=2, alpha=1.0)
        self.add_experience(action_id=0, buffer=buffer)
        self.add_experience(action_id=1, buffer=buffer)
        buffer.update_priorities([0, 1], [1.0, 9.0])

        # Sampling is with replacement, so collect both indices across draws.
        by_index = {}
        for _ in range(100):
            indices, weights = buffer.sample(2, beta=1.0)[6:8]
            by_index.update(zip(indices.tolist(), weights.tolist()))
            if len(by_index) == 2:
                break
        self.assertEqual(set(by_index), {0, 1})
        self.assertAlmostEqual(by_index[0], 1.0, places=5)
        self.assertAlmostEqual(by_index[1], (1.0 + buffer.priority_epsilon) / (9.0 + buffer.priority_epsilon), places=5)

    def test_beta_zero_returns_unit_weights(self):
        # beta=0 disables importance-sampling correction.
        for i in range(4):
            self.add_experience(action_id=i)
        self.buffer.update_priorities([0, 1, 2, 3], [1.0, 2.0, 3.0, 4.0])
        weights = self.buffer.sample(4, beta=0.0)[7]
        np.testing.assert_allclose(weights, np.ones(4))

    def test_new_experience_uses_neutral_priority(self):
        # New transitions start with priority 1.0, not the historical maximum.
        self.add_experience(action_id=0)
        self.buffer.update_priorities([0], [25.0])
        self.add_experience(action_id=1)

        first = self.buffer.sum_tree[self.buffer.tree_capacity]
        second = self.buffer.sum_tree[self.buffer.tree_capacity + 1]

        self.assertGreater(first, second)
        self.assertAlmostEqual(second, 1.0)

    def test_import_from_uniform_buffer(self):
        # Existing checkpoint experiences can be copied without changing the source.
        uniform = DQNReplayBuffer(capacity=10)
        for i in range(7):
            self.add_experience(action_id=i, reward=float(i), buffer=uniform)

        self.buffer.load_from_uniform(uniform)
        self.assertEqual(len(self.buffer), 7)
        self.assertEqual(len(uniform), 7)
        self.assertEqual([item[1] for item in self.buffer.buffer[:7]], list(range(7)))

        leaves = self.buffer.sum_tree[self.buffer.tree_capacity:self.buffer.tree_capacity + 7]
        np.testing.assert_allclose(leaves, np.ones(7))

        # Verify imported arrays are independent of the original buffer.
        uniform.buffer[0][0][0] = 123.0
        self.assertEqual(self.buffer.buffer[0][0][0], 0.0)

    def test_invalid_configuration_and_sampling_parameters(self):
        # Invalid capacity, alpha, epsilon, batch size and beta are rejected.
        with self.assertRaises(ValueError):
            DQNPriorityReplayBuffer(capacity=0)
        with self.assertRaises(ValueError):
            DQNPriorityReplayBuffer(capacity=10, alpha=1.1)
        with self.assertRaises(ValueError):
            DQNPriorityReplayBuffer(capacity=10, priority_epsilon=0)

        self.add_experience()
        with self.assertRaises(ValueError):
            self.buffer.sample(0)
        with self.assertRaises(ValueError):
            self.buffer.sample(1, beta=-0.1)
        with self.assertRaises(ValueError):
            self.buffer.sample(1, beta=1.1)

    def test_invalid_priority_update(self):
        # Out-of-range indices and non-finite TD errors are rejected.
        self.add_experience()
        with self.assertRaises(IndexError):
            self.buffer.update_priorities([1], [1.0])
        with self.assertRaises(ValueError):
            self.buffer.update_priorities([0], [np.nan])
        with self.assertRaises(ValueError):
            self.buffer.update_priorities([0], [np.inf])

    def test_state_dict_restores_priorities(self):
        # Saving and restoring must preserve all PER sampling information.
        for i in range(7):
            self.add_experience(action_id=i, reward=float(i))

        self.buffer.update_priorities(
            [0, 1, 2, 3, 4, 5, 6],
            [1.0, 2.0, 5.0, 10.0, 20.0, 3.0, 7.0],
        )

        saved = self.buffer.state_dict()

        restored = DQNPriorityReplayBuffer(capacity=10)
        restored.load_state_dict(saved)

        self.assertEqual(len(restored), len(self.buffer))
        self.assertEqual(restored.position, self.buffer.position)
        self.assertEqual(restored.size, self.buffer.size)
        self.assertEqual(restored.max_priority, self.buffer.max_priority)

        np.testing.assert_array_equal(
            restored.sum_tree, self.buffer.sum_tree
        )
        np.testing.assert_array_equal(
            restored.min_tree, self.buffer.min_tree
        )

        for i in range(self.buffer.size):
            original = self.buffer.buffer[i]
            recovered = restored.buffer[i]

            np.testing.assert_array_equal(recovered[0], original[0])
            self.assertEqual(recovered[1], original[1])
            self.assertEqual(recovered[2], original[2])
            np.testing.assert_array_equal(recovered[3], original[3])
            self.assertEqual(recovered[4], original[4])
            np.testing.assert_array_equal(recovered[5], original[5])

    def test_state_dict_restores_circular_buffer_position(self):
        # Restoring a full buffer must preserve the next replacement position.
        for i in range(13):
            self.add_experience(action_id=i)

        saved = self.buffer.state_dict()

        restored = DQNPriorityReplayBuffer(capacity=10)
        restored.load_state_dict(saved)

        self.assertEqual(restored.position, 3)
        self.assertEqual(len(restored), 10)

        restored.add(
            observation=self.observation,
            action_id=99,
            reward=1.0,
            next_observation=self.next_observation,
            terminated=False,
            next_action_mask=self.next_action_mask,
        )

        self.assertEqual(restored.buffer[3][1], 99)
        self.assertEqual(restored.position, 4)
        self.assertEqual(len(restored), 10)

    def test_state_dict_rejects_incompatible_capacity(self):
        # A saved priority tree cannot be loaded into an incompatible tree.
        for i in range(5):
            self.add_experience(action_id=i)

        saved = self.buffer.state_dict()

        restored = DQNPriorityReplayBuffer(capacity=20)

        with self.assertRaises(ValueError):
            restored.load_state_dict(saved)


class TestDQNAgentPERBeta(unittest.TestCase):
    # Test the beta schedule without constructing a neural network or environment.
    def setUp(self):
        self.agent = DQNAgent.__new__(DQNAgent)
        self.agent.per_beta_initial = 0.4
        self.agent.per_beta_final = 1.0
        self.agent.per_beta_anneal_steps = 500000
        self.agent.per_beta_steps = 0

    def test_per_beta_initial_value(self):
        self.assertAlmostEqual(
            self.agent.current_per_beta(),
            self.agent.per_beta_initial,
        )

    def test_per_beta_midpoint(self):
        self.agent.per_beta_steps = self.agent.per_beta_anneal_steps // 2
        self.assertAlmostEqual(self.agent.current_per_beta(), 0.7)

    def test_per_beta_reaches_one(self):
        self.agent.per_beta_steps = self.agent.per_beta_anneal_steps
        self.assertAlmostEqual(
            self.agent.current_per_beta(),
            self.agent.per_beta_final,
        )
        self.agent.per_beta_steps *= 2
        self.assertAlmostEqual(
            self.agent.current_per_beta(),
            self.agent.per_beta_final,
        )
        


if __name__ == "__main__":
    unittest.main()

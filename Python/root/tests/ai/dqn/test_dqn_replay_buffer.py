import unittest

import numpy as np

from core.ai.dqn.dqn_replay_buffer import DQNReplayBuffer


class TestDQNReplayBuffer(unittest.TestCase):

    def setUp(self):
        self.buffer = DQNReplayBuffer(capacity=10)

        self.observation = np.zeros(967, dtype=np.float32)
        self.next_observation = np.ones(967, dtype=np.float32)

        self.next_action_mask = np.zeros(1032, dtype=np.float32)
        self.next_action_mask[5] = 1.0

    def add_experience(self, action_id=5, reward=1.0):
        self.buffer.add(
            observation=self.observation,
            action_id=action_id,
            reward=reward,
            next_observation=self.next_observation,
            terminated=False,
            next_action_mask=self.next_action_mask,
        )

    def test_add_experience(self):
        # Check that an experience is stored
        self.assertEqual(len(self.buffer), 0)

        self.add_experience()

        self.assertEqual(len(self.buffer), 1)

        batch = self.buffer.sample(1)

        self.assertEqual(batch[1][0], 5)
        self.assertEqual(batch[2][0], 1.0)
        self.assertFalse(batch[4][0])

    def test_capacity_limit(self):
        # Oldest experiences should be discarded
        for i in range(15):
            self.add_experience(action_id=i)

        self.assertEqual(len(self.buffer), 10)

        batch = self.buffer.sample(10)
        stored_actions = set(batch[1])

        self.assertEqual(stored_actions, set(range(5, 15)))

    def test_sample_shapes_and_types(self):
        # Check dimensions and data types
        for i in range(5):
            self.add_experience(action_id=i)

        batch = self.buffer.sample(3)

        observations, actions, rewards, next_observations, terminated, next_masks = batch

        self.assertEqual(observations.shape, (3, 967))
        self.assertEqual(actions.shape, (3,))
        self.assertEqual(rewards.shape, (3,))
        self.assertEqual(next_observations.shape, (3, 967))
        self.assertEqual(terminated.shape, (3,))
        self.assertEqual(next_masks.shape, (3, 1032))

        self.assertEqual(observations.dtype, np.float32)
        self.assertEqual(actions.dtype, np.int64)
        self.assertEqual(rewards.dtype, np.float32)
        self.assertEqual(next_observations.dtype, np.float32)
        self.assertEqual(terminated.dtype, np.bool_)
        self.assertEqual(next_masks.dtype, np.float32)

    def test_sample_too_many(self):
        # Cannot sample more experiences than available
        self.add_experience()

        with self.assertRaises(ValueError):
            self.buffer.sample(2)

    def test_observation_copies(self):
        # Changing original arrays must not change stored data
        self.add_experience()

        self.observation[0] = 99.0
        self.next_observation[0] = 99.0
        self.next_action_mask[5] = 0.0

        batch = self.buffer.sample(1)

        self.assertEqual(batch[0][0, 0], 0.0)
        self.assertEqual(batch[3][0, 0], 1.0)
        self.assertEqual(batch[5][0, 5], 1.0)


if __name__ == "__main__":
    unittest.main()

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


        observations, actions, rewards, next_observations, terminated, next_masks, n_steps = batch
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
        
        self.assertEqual(n_steps.shape, (3,))

        self.assertEqual(n_steps.dtype, np.int64)
        np.testing.assert_array_equal(n_steps, [1, 1, 1])

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

    def test_n_step_experience(self):
        # A multi-step experience should store its actual step count
        self.buffer.add(
            observation=self.observation,
            action_id=5,
            reward=2.97,
            next_observation=self.next_observation,
            terminated=False,
            next_action_mask=self.next_action_mask,
            n_steps=3,
        )

        self.assertEqual(len(self.buffer), 1)

        experience = self.buffer.buffer[0]

        self.assertEqual(len(experience), 7)
        self.assertAlmostEqual(experience[2], 2.97, places=5)
        self.assertEqual(experience[6], 3)

        # Sampling must still return the original six arrays
        batch = self.buffer.sample(1)
        self.assertEqual(len(batch), 7)
        self.assertEqual(batch[6][0], 3)
        self.assertAlmostEqual(batch[2][0], 2.97, places=5)

    def test_default_one_step_experience(self):
        # Existing calls without n_steps must default to one step
        self.add_experience()

        experience = self.buffer.buffer[0]

        self.assertEqual(len(experience), 7)
        self.assertEqual(experience[6], 1)

    def test_legacy_experience_compatibility(self):
        # Simulate an experience loaded from an old checkpoint
        legacy_experience = (
            self.observation.copy(),
            5,
            1.0,
            self.next_observation.copy(),
            False,
            self.next_action_mask.copy(),
        )

        self.buffer.buffer.append(legacy_experience)

        # Old six-field experiences must remain sampleable
        batch = self.buffer.sample(1)

        self.assertEqual(len(batch), 7)
        self.assertEqual(batch[6][0], 1)
        self.assertEqual(batch[1][0], 5)
        self.assertEqual(batch[2][0], 1.0)



if __name__ == "__main__":
    unittest.main()

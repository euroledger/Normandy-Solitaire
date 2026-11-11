import unittest

import numpy as np

from core.ai.dqn.dqn_n_step import NStepTransitionCollector


class TestNStepTransitionCollector(unittest.TestCase):
    def make_transition(self, index, reward, terminated=False):
        return (
            f"state_{index}",
            index,
            reward,
            f"state_{index + 1}",
            terminated,
            f"mask_{index + 1}",
        )

    def add_transition(self, collector, index, reward, terminated=False):
        return collector.add(*self.make_transition(index, reward, terminated))

    def assert_transition(self, actual, state, action, reward, next_state,
                          terminated, next_mask, steps):
        self.assertEqual(len(actual), 7)
        self.assertEqual(actual[0], state)
        self.assertEqual(actual[1], action)
        self.assertAlmostEqual(actual[2], reward)
        self.assertEqual(actual[3], next_state)
        self.assertEqual(actual[4], terminated)
        self.assertEqual(actual[5], next_mask)
        self.assertEqual(actual[6], steps)

    def test_one_step_emits_immediately(self):
        collector = NStepTransitionCollector(n_step=1, gamma=0.9)
        result = self.add_transition(collector, 0, 5.0)
        self.assertEqual(len(result), 1)
        self.assert_transition(result[0], "state_0", 0, 5.0,
                               "state_1", False, "mask_1", 1)
        self.assertEqual(len(collector.pending), 0)

    def test_three_step_discounted_return(self):
        collector = NStepTransitionCollector(n_step=3, gamma=0.9)
        self.assertEqual(self.add_transition(collector, 0, 1.0), [])
        self.assertEqual(self.add_transition(collector, 1, 2.0), [])
        result = self.add_transition(collector, 2, 3.0)
        self.assertEqual(len(result), 1)
        self.assert_transition(result[0], "state_0", 0,
                               1.0 + 0.9 * 2.0 + 0.9 ** 2 * 3.0,
                               "state_3", False, "mask_3", 3)

    def test_rolling_window(self):
        collector = NStepTransitionCollector(n_step=3, gamma=0.5)
        for index in range(2):
            self.assertEqual(self.add_transition(collector, index, index + 1), [])
        first = self.add_transition(collector, 2, 3.0)
        second = self.add_transition(collector, 3, 4.0)
        self.assertEqual(len(first), 1)
        self.assertEqual(len(second), 1)
        self.assert_transition(first[0], "state_0", 0, 1 + 0.5 * 2 + 0.25 * 3,
                               "state_3", False, "mask_3", 3)
        self.assert_transition(second[0], "state_1", 1, 2 + 0.5 * 3 + 0.25 * 4,
                               "state_4", False, "mask_4", 3)

    def test_terminal_flushes_remaining_transitions(self):
        collector = NStepTransitionCollector(n_step=3, gamma=0.9)
        self.add_transition(collector, 0, 1.0)
        self.add_transition(collector, 1, 2.0)
        result = self.add_transition(collector, 2, 3.0, terminated=True)
        self.assertEqual(len(result), 3)
        self.assert_transition(result[0], "state_0", 0, 1 + 0.9 * 2 + 0.81 * 3,
                               "state_3", True, "mask_3", 3)
        self.assert_transition(result[1], "state_1", 1, 2 + 0.9 * 3,
                               "state_3", True, "mask_3", 2)
        self.assert_transition(result[2], "state_2", 2, 3,
                               "state_3", True, "mask_3", 1)
        self.assertEqual(len(collector.pending), 0)

    def test_terminal_before_window_fills(self):
        collector = NStepTransitionCollector(n_step=3, gamma=0.9)
        self.add_transition(collector, 0, 4.0)
        result = self.add_transition(collector, 1, 6.0, terminated=True)
        self.assertEqual(len(result), 2)
        self.assert_transition(result[0], "state_0", 0, 4 + 0.9 * 6,
                               "state_2", True, "mask_2", 2)
        self.assert_transition(result[1], "state_1", 1, 6,
                               "state_2", True, "mask_2", 1)

    def test_reset_discards_pending_transitions(self):
        collector = NStepTransitionCollector(n_step=3, gamma=0.9)
        self.add_transition(collector, 0, 99.0)
        collector.reset()
        self.assertEqual(len(collector.pending), 0)
        self.assertEqual(self.add_transition(collector, 10, 1.0), [])
        self.assertEqual(self.add_transition(collector, 11, 2.0), [])
        result = self.add_transition(collector, 12, 3.0)
        self.assertEqual(len(result), 1)
        self.assert_transition(result[0], "state_10", 10, 1 + 0.9 * 2 + 0.81 * 3,
                               "state_13", False, "mask_13", 3)

    def test_invalid_n_step(self):
        for n_step in (0, -1):
            with self.subTest(n_step=n_step), self.assertRaises(ValueError):
                NStepTransitionCollector(n_step=n_step)

    def test_separate_collectors_do_not_mix_workers(self):
        worker_a = NStepTransitionCollector(n_step=3, gamma=1.0)
        worker_b = NStepTransitionCollector(n_step=3, gamma=1.0)
        for index in range(2):
            self.add_transition(worker_a, index, 1.0)
            self.add_transition(worker_b, index + 100, 10.0)
        a = self.add_transition(worker_a, 2, 1.0)
        b = self.add_transition(worker_b, 102, 10.0)
        self.assertEqual(a[0][2], 3.0)
        self.assertEqual(b[0][2], 30.0)
        self.assertEqual(a[0][0], "state_0")
        self.assertEqual(b[0][0], "state_100")


    def test_three_step_discounted_reward(self):
        collector = NStepTransitionCollector(
            n_step=3,
            gamma=0.99,
        )

        completed = []

        for i in range(3):
            completed.extend(
                collector.add(
                    observation=np.array([float(i)], dtype=np.float32),
                    action_id=i,
                    reward=10.0,
                    next_observation=np.array([float(i + 1)], dtype=np.float32),
                    terminated=False,
                    next_action_mask=np.array([1.0], dtype=np.float32),
                )
            )

        self.assertEqual(len(completed), 1)

        experience = completed[0]

        expected_reward = 10.0 + 0.99 * 10.0 + 0.99**2 * 10.0

        self.assertAlmostEqual(experience[2], expected_reward)
        self.assertEqual(experience[6], 3)
        self.assertFalse(experience[4])
        np.testing.assert_array_equal(experience[3], [3.0])

if __name__ == "__main__":
    unittest.main()

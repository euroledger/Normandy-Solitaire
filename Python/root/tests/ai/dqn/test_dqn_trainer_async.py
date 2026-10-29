import unittest
from unittest.mock import Mock, patch
from gymnasium.vector import AsyncVectorEnv, AutoresetMode
import numpy as np

from core.ai.dqn.dqn_replay_buffer import DQNReplayBuffer
from core.ai.dqn.dqn_trainer import DQNTrainer
from gymnasium import spaces
import csv
import tempfile
import torch

from pathlib import Path
from unittest.mock import MagicMock
from core.ai.dqn.dqn_trainer import run_greedy_evaluation


class FakeNormandyEnv:
    # Deterministic environment with two decisions per game

    def __init__(self):
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32,
        )

        self.action_space = spaces.Discrete(5)

        self.reset_count = 0
        self.step_count = 0
        self.actions_received = []

    def reset(self):
        self.reset_count += 1
        self.step_count = 0

        observation = np.array(
            [1, 0, 0, 0],
            dtype=np.float32,
        )

        info = {
            "action_mask": np.array(
                [0, 1, 0, 0, 0],
                dtype=np.float32,
            )
        }

        return observation, info

    def step(self, action_id):
        self.actions_received.append(action_id)
        self.step_count += 1

        if self.step_count == 1:
            # First decision: game continues
            observation = np.array(
                [0, 1, 0, 0],
                dtype=np.float32,
            )

            reward = 1.0
            terminated = False

            info = {
                "action_mask": np.array(
                    [0, 0, 1, 0, 0],
                    dtype=np.float32,
                )
            }

        else:
            # Second decision: game ends
            observation = np.array(
                [0, 0, 1, 0],
                dtype=np.float32,
            )

            reward = 2.0
            terminated = True

            info = {
                "result": "lost",
                "turn_reached": self.step_count,
                "action_mask": np.zeros(
                    5,
                    dtype=np.float32,
                ),
            }

        return (
            observation,
            reward,
            terminated,
            False,
            info,
        )

    def close(self):
        pass

class FakeAsyncVectorEnv:

    def __init__(self, env_fns, autoreset_mode=None):

        self.num_envs = len(env_fns)

        self.steps = np.zeros(self.num_envs, dtype=int)
        self.needs_reset = np.zeros(self.num_envs, dtype=bool)

        self.closed = False

    def reset(self):

        self.steps[:] = 0
        self.needs_reset[:] = False

        observations = np.tile(
            np.array([1, 0, 0, 0], dtype=np.float32),
            (self.num_envs, 1),
        )

        masks = np.tile(
            np.array([0, 1, 0, 0, 0], dtype=np.float32),
            (self.num_envs, 1),
        )

        return observations, {"action_mask": masks}

    def step(self, actions):

        observations = np.zeros(
            (self.num_envs, 4), dtype=np.float32
        )

        rewards = np.zeros(self.num_envs, dtype=float)
        terminated = np.zeros(self.num_envs, dtype=bool)
        truncated = np.zeros(self.num_envs, dtype=bool)

        masks = np.zeros(
            (self.num_envs, 5), dtype=np.float32
        )

        results = np.empty(self.num_envs, dtype=object)
        results[:] = None
        turns_reached = np.zeros(self.num_envs, dtype=int)

        for i in range(self.num_envs):

            if self.needs_reset[i]:

                self.steps[i] = 0
                self.needs_reset[i] = False

                observations[i] = [1, 0, 0, 0]
                masks[i] = [0, 1, 0, 0, 0]

                continue

            self.steps[i] += 1

            if self.steps[i] == 1:

                assert actions[i] == 1

                observations[i] = [0, 1, 0, 0]
                masks[i] = [0, 0, 1, 0, 0]

                rewards[i] = 1.0

            else:

                assert actions[i] == 2

                observations[i] = [0, 0, 1, 0]

                rewards[i] = 2.0
                terminated[i] = True

                results[i] = "lost"
                turns_reached[i] = self.steps[i]
                self.needs_reset[i] = True

        return (
            observations,
            rewards,
            terminated,
            truncated,
            {
                "action_mask": masks,
                "result": results,
                "turn_reached": turns_reached,
            },
        )

    def close(self):
        self.closed = True


class FakeVariableLengthVectorEnv(FakeAsyncVectorEnv):

    def __init__(self, env_fns, autoreset_mode=None):

        super().__init__(env_fns, autoreset_mode)

        # Workers require 2, 3, 4 and 5 decisions.
        self.game_lengths = np.arange(
            2, self.num_envs + 2
        )



    def step(self, actions):

        observations = np.zeros(
            (self.num_envs, 4), dtype=np.float32
        )

        rewards = np.zeros(self.num_envs, dtype=float)

        terminated = np.zeros(
            self.num_envs, dtype=bool
        )

        truncated = np.zeros(
            self.num_envs, dtype=bool
        )

        masks = np.zeros(
            (self.num_envs, 5), dtype=np.float32
        )

        results = np.empty(self.num_envs, dtype=object)
        results[:] = None
        turns_reached = np.zeros(self.num_envs, dtype=int)

        for i in range(self.num_envs):

            if self.needs_reset[i]:

                self.steps[i] = 0
                self.needs_reset[i] = False

                observations[i] = [1, 0, 0, 0]
                masks[i] = [0, 1, 0, 0, 0]

                continue

            self.steps[i] += 1

            if self.steps[i] < self.game_lengths[i]:

                # Game continues.
                observations[i] = [0, 1, 0, 0]
                masks[i] = [0, 1, 0, 0, 0]

                rewards[i] = 1.0

            else:

                # Game ends.
                observations[i] = [0, 0, 1, 0]

                rewards[i] = 2.0
                terminated[i] = True

                results[i] = "lost"
                turns_reached[i] = self.steps[i]
                self.needs_reset[i] = True

        return (
            observations,
            rewards,
            terminated,
            truncated,
            {
                "action_mask": masks,
                "result": results,
                "turn_reached": turns_reached,
            },
        )

class TestDQNTrainer(unittest.TestCase):

    def setUp(self):

        def mock_select_actions(observations, masks, needs_reset=None):

            actions = np.zeros(
                len(observations),
                dtype=np.int64,
            )

            for i in range(len(observations)):

                if needs_reset is not None and needs_reset[i]:
                    continue

                actions[i] = int(
                    np.flatnonzero(masks[i])[0]
                )

            return actions


        self.env = FakeNormandyEnv()
        self.agent = Mock()
        self.agent.select_actions.side_effect = mock_select_actions
        self.agent.replay_buffer = DQNReplayBuffer(
            capacity=100
        )

        self.agent.epsilon = 1.0
        self.agent.epsilon_min = 0.1
        self.agent.epsilon_decay = 0.5
        self.agent.train_frequency = 1

        self.agent.select_action.side_effect = (
            lambda observation, mask:
            int(np.flatnonzero(mask)[0])
        )

        self.agent.train_step.return_value = 0.25

        self.trainer = DQNTrainer(
            env=self.env,
            agent=self.agent,
        )
    def test_four_parallel_workers(self):

        with patch(
            "core.ai.dqn.dqn_trainer.AsyncVectorEnv",
            FakeAsyncVectorEnv,
        ):

            results = self.trainer.train(
                num_games=4,
                num_workers=4,
            )

        self.assertEqual(len(results), 4)

        self.assertEqual(
            self.trainer.games_completed, 4
        )

        self.assertEqual(
            self.trainer.total_decisions, 8
        )

        self.assertEqual(
            len(self.agent.replay_buffer), 8
        )

        self.assertEqual(
            self.trainer.total_training_updates, 8
        )

        for result in results:

            self.assertEqual(result["reward"], 3.0)
            self.assertEqual(result["decisions"], 2)
            self.assertEqual(result["result"], "lost")

    def test_autoreset_does_not_add_experiences(self):

        with patch(
            "core.ai.dqn.dqn_trainer.AsyncVectorEnv",
            FakeAsyncVectorEnv,
        ):

            results = self.trainer.train(
                num_games=8,
                num_workers=4,
            )

        self.assertEqual(len(results), 8)

        # Eight games, two decisions each.
        self.assertEqual(
            self.trainer.total_decisions, 16
        )

        # Automatic resets must not enter replay memory.
        self.assertEqual(
            len(self.agent.replay_buffer), 16
        )

        self.assertEqual(
            self.agent.select_actions.call_count, 5
        )
        
    def test_parallel_replay_buffer_contents(self):

        with patch(
            "core.ai.dqn.dqn_trainer.AsyncVectorEnv",
            FakeAsyncVectorEnv,
        ):

            self.trainer.train(
                num_games=4,
                num_workers=4,
            )

        experiences = list(
            self.agent.replay_buffer.buffer
        )

        self.assertEqual(len(experiences), 8)

        # Four workers execute their first decision,
        # followed by four second decisions.

        for experience in experiences[:4]:

            observation, action, reward, next_observation, done, next_mask = experience

            np.testing.assert_array_equal(
                observation,
                [1, 0, 0, 0],
            )

            self.assertEqual(action, 1)
            self.assertEqual(reward, 1.0)
            self.assertFalse(done)

            np.testing.assert_array_equal(
                next_observation,
                [0, 1, 0, 0],
            )

            np.testing.assert_array_equal(
                next_mask,
                [0, 0, 1, 0, 0],
            )

        for experience in experiences[4:]:

            observation, action, reward, next_observation, done, next_mask = experience

            np.testing.assert_array_equal(
                observation,
                [0, 1, 0, 0],
            )

            self.assertEqual(action, 2)
            self.assertEqual(reward, 2.0)
            self.assertTrue(done)

            np.testing.assert_array_equal(
                next_observation,
                [0, 0, 1, 0],
            )

            np.testing.assert_array_equal(
                next_mask,
                [0, 0, 0, 0, 0],
            )
            
    def test_workers_finish_at_different_times(self):

        with patch(
            "core.ai.dqn.dqn_trainer.AsyncVectorEnv",
            FakeVariableLengthVectorEnv,
        ):

            results = self.trainer.train(
                num_games=4,
                num_workers=4,
            )

        self.assertEqual(len(results), 4)

        self.assertEqual(
            self.trainer.games_completed,
            4,
        )

        # Each worker finishes one game.
        # Game lengths are 2, 3, 4 and 5 decisions.
        self.assertEqual(
            sorted(result["decisions"] for result in results),
            [2, 2, 3, 4],
        )

        # Each nonterminal decision earns 1 point.
        # The terminal decision earns 2 points.
        self.assertEqual(
            sorted(result["reward"] for result in results),
            [3.0, 3.0, 4.0, 5.0],
        )

        for result in results:
            self.assertEqual(result["result"], "lost")
            
    def test_training_frequency_with_variable_length_games(self):

        self.agent.train_frequency = 4

        with patch(
            "core.ai.dqn.dqn_trainer.AsyncVectorEnv",
            FakeVariableLengthVectorEnv,
        ):

            results = self.trainer.train(
                num_games=4,
                num_workers=4,
            )

        self.assertEqual(len(results), 4)

        decisions = self.trainer.total_decisions

        expected_updates = decisions // 4

        self.assertEqual(
            self.agent.train_step.call_count,
            expected_updates,
        )

        self.assertEqual(
            self.trainer.total_training_updates,
            expected_updates,
        )

        self.assertEqual(
            len(self.agent.replay_buffer),
            decisions,
        )
        
            
    def test_greedy_evaluation_selects_legal_actions(self):

        class FakeNetwork(torch.nn.Module):

            def __init__(self, observation_size, action_size):
                super().__init__()

            def load_state_dict(self, state):
                pass

            def forward(self, observation):
                # Action 0 has highest Q but is illegal.
                # Legal actions are 1, then 2.
                return torch.tensor(
                    [[100.0, 10.0, 20.0, 0.0, 0.0]],
                    dtype=torch.float32,
                )

        class FakeEvaluationEnv(FakeNormandyEnv):

            def reset(self):
                observation, info = super().reset()

                info["action_mask"] = np.array(
                    [0, 1, 1, 0, 0],
                    dtype=np.float32,
                )

                return observation, info

            def step(self, action_id):
                self.actions_received.append(action_id)

                return (
                    np.zeros(4, dtype=np.float32),
                    0.0,
                    True,
                    False,
                    {
                        "result": "lost",
                        "turn_reached": 10,
                    },
                )

        fake_env = FakeEvaluationEnv()

        with patch(
            "core.ai.dqn.dqn_trainer.DQNNetwork",
            FakeNetwork,
        ), patch(
            "core.ai.dqn.dqn_trainer.NormandyEnv",
            return_value=fake_env,
        ):

            result = run_greedy_evaluation(
                policy_state={},
                observation_size=4,
                action_size=5,
            )

        self.assertEqual(fake_env.actions_received, [2])
        self.assertEqual(result["turn_reached"], 10)
        self.assertEqual(result["result"], "lost")
            
                
    def test_evaluation_does_not_modify_training_state(self):

        self.agent.epsilon = 0.5856
        self.agent.training_steps = 100

        self.trainer.games_completed = 14400
        self.trainer.total_decisions = 475243
        self.trainer.total_training_updates = 118795

        epsilon_before = self.agent.epsilon
        steps_before = self.agent.training_steps
        replay_before = len(self.agent.replay_buffer)

        games_before = self.trainer.games_completed
        decisions_before = self.trainer.total_decisions
        updates_before = self.trainer.total_training_updates

        with patch.object(
            self.trainer,
            "evaluate_policy",
            return_value={
                "turn_reached": 20,
                "result": "lost",
            },
        ):

            result = self.trainer.evaluate_policy()

        self.assertEqual(result["turn_reached"], 20)

        self.assertEqual(self.agent.epsilon, epsilon_before)
        self.assertEqual(self.agent.training_steps, steps_before)
        self.assertEqual(len(self.agent.replay_buffer), replay_before)

        self.assertEqual(self.trainer.games_completed, games_before)
        self.assertEqual(self.trainer.total_decisions, decisions_before)
        self.assertEqual(
            self.trainer.total_training_updates,
            updates_before,
        )


    def test_evaluation_runs_at_correct_intervals(self):

        evaluation = {
            "turn_reached": 15,
            "result": "lost",
        }

        with patch(
            "core.ai.dqn.dqn_trainer.AsyncVectorEnv",
            FakeAsyncVectorEnv,
        ), patch.object(
            self.trainer,
            "evaluate_policy",
            return_value=evaluation,
        ) as mock_evaluate, patch.object(
            self.trainer,
            "log_evaluation",
        ) as mock_log:

            results = self.trainer.train(
                num_games=20,
                num_workers=4,
                evaluation_interval=5,
                evaluation_games=1,
            )

        self.assertEqual(len(results), 20)

        # Four evaluations during 20 training games.
        self.assertEqual(mock_evaluate.call_count, 4)
        self.assertEqual(mock_log.call_count, 4)

        # Evaluation must not add games to training counters.
        self.assertEqual(self.trainer.games_completed, 20)

        # Two decisions per training game.
        self.assertEqual(self.trainer.total_decisions, 40)


    def test_evaluation_results_are_logged(self):

        with tempfile.TemporaryDirectory() as temp_dir:

            evaluation_path = (
                Path(temp_dir) / "evaluation_history.csv"
            )

            training_path = (
                Path(temp_dir) / "training_history.csv"
            )

            # Existing training history must remain unchanged.
            training_path.write_text(
                "game,reward\n1,10.0\n",
                encoding="utf-8",
            )

            original_training = training_path.read_text(
                encoding="utf-8",
            )

            self.trainer.log_evaluation(
                training_games=14900,
                result={
                    "turn_reached": 21,
                    "result": "lost",
                },
                path=evaluation_path,
            )

            self.trainer.log_evaluation(
                training_games=15400,
                result={
                    "turn_reached": 25,
                    "result": "won",
                },
                path=evaluation_path,
            )

            with evaluation_path.open(
                newline="",
                encoding="utf-8",
            ) as file:

                rows = list(csv.DictReader(file))

            self.assertEqual(len(rows), 2)

            self.assertEqual(rows[0]["training_games"], "14900")
            self.assertEqual(rows[0]["epsilon"], "0.0")
            self.assertEqual(rows[0]["turn_reached"], "21")
            self.assertEqual(rows[0]["result"], "lost")

            self.assertEqual(rows[1]["training_games"], "15400")
            self.assertEqual(rows[1]["turn_reached"], "25")
            self.assertEqual(rows[1]["result"], "won")

            self.assertEqual(
                training_path.read_text(encoding="utf-8"),
                original_training,
            )

class TestDQNTrainerAsyncIntegration(unittest.TestCase):

    def test_four_real_normandy_games(self):

        import contextlib
        import os
        import time

        import numpy as np

        from core.ai.dqn.dqn_agent import DQNAgent
        from core.ai.dqn.dqn_network import DQNNetwork
        from core.ai.dqn.dqn_trainer import DQNTrainer
        from core.ai.dqn.normandy_env import NormandyEnv

        # Reproducibility
        torch.manual_seed(42)
        np.random.seed(42)

        # Obtain observation and action dimensions
        reference_env = NormandyEnv()

        try:
            observation_size = (
                reference_env.observation_space.shape[0]
            )

            action_size = (
                reference_env.action_space.n
            )

        finally:
            reference_env.close()

        # Create actual DQN network
        network = DQNNetwork(
            observation_size,
            action_size,
        )

        # Use CPU for this integration test
        agent = DQNAgent(
            network,
            torch.device("cpu"),
        )

        # Small batch so actual learning occurs
        agent.mini_batch_size = 2

        # One training update per decision
        agent.train_frequency = 1

        trainer = DQNTrainer(
            agent=agent,
        )

        start = time.perf_counter()

        # Suppress output from the game engine
        with open(
            os.devnull,
            "w",
            encoding="utf-8",
        ) as devnull, contextlib.redirect_stdout(devnull):

            results = trainer.train(
                num_games=4,
                num_workers=4,
            )

        elapsed = time.perf_counter() - start

        # -----------------------------------------
        # Verify completed games
        # -----------------------------------------

        self.assertEqual(len(results), 4)

        self.assertEqual(
            trainer.games_completed,
            4,
        )

        for result in results:

            self.assertIn(
                result["result"],
                ("won", "lost"),
            )

            self.assertGreater(
                result["decisions"],
                0,
            )

            self.assertTrue(
                np.isfinite(result["reward"])
            )

        # -----------------------------------------
        # Verify replay buffer
        # -----------------------------------------

        self.assertGreater(
            len(agent.replay_buffer),
            0,
        )

        self.assertEqual(
            len(agent.replay_buffer),
            trainer.total_decisions,
        )

        # -----------------------------------------
        # Verify actual learning
        # -----------------------------------------

        self.assertGreater(
            trainer.total_training_updates,
            0,
        )

        self.assertEqual(
            agent.training_steps,
            trainer.total_training_updates,
        )

        # -----------------------------------------
        # Verify numerical stability
        # -----------------------------------------

        for parameter in agent.policy_net.parameters():

            self.assertTrue(
                torch.isfinite(parameter).all().item()
            )

        # -----------------------------------------
        # Performance report
        # -----------------------------------------

        print()
        print("REAL ASYNC DQN INTEGRATION TEST")
        print("--------------------------------")
        print("Workers:           4")
        print(f"Games completed:   {len(results)}")
        print(f"Decisions:         {trainer.total_decisions}")
        print(f"Training updates:  {trainer.total_training_updates}")
        print(f"Elapsed:           {elapsed:.2f}s")
        print(f"Games/second:      {len(results) / elapsed:.2f}")

if __name__ == "__main__":
    unittest.main()

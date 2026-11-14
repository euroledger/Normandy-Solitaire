import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from core.ai.dqn.noisy import dqn_noisy_trainer as module
from core.ai.dqn.noisy.dqn_noisy_trainer import NoisyDQNTrainer


class FakeAgent:
    def __init__(self):
        self.n_step = 6
        self.train_frequency = 1
        self.replay_buffer_type = "uniform"
        self.replay_buffer = SimpleNamespace(buffer=[], add=self.add_transition)
        self.policy_net = torch.nn.Linear(2, 2)
        self.target_net = torch.nn.Linear(2, 2)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.training_steps = 7
        self.per_beta_steps = 0
        self.epsilon = 0.0
        self.epsilon_min = 0.0
        self.epsilon_decay = 1.0
        self.discount_factor = 0.99
        self.grad_norm_count = 0
        self.actions_seen = []
        self.training_calls = 0
        self.noisy_calls = 0
        self.deterministic_calls = 0

    def add_transition(self, **kwargs):
        self.replay_buffer.buffer.append(kwargs)

    def set_noisy(self):
        self.noisy_calls += 1

    def set_deterministic(self):
        self.deterministic_calls += 1

    def select_actions(self, observations, masks, needs_reset):
        self.actions_seen.append(np.array(needs_reset, copy=True))
        return np.zeros(len(observations), dtype=np.int64)

    def select_action(self, observation, mask):
        return 0

    def train_step(self):
        self.training_calls += 1
        return 0.25


class FakeCollector:
    def __init__(self, n_step, gamma):
        self.n_step = n_step
        self.gamma = gamma

    def add(self, observation, action_id, reward, next_observation,
            terminated, next_action_mask):
        return [(observation, action_id, reward, next_observation,
                 terminated, next_action_mask, 1)]


class FakeVectorEnv:
    # Worker 0 completes after its first decision, worker 1 after two.
    # NEXT_STEP requires a reset-only step after termination.
    instances = []

    def __init__(self, factories, autoreset_mode):
        self.n = len(factories)
        self.steps = np.zeros(self.n, dtype=int)
        self.pending_reset = np.zeros(self.n, dtype=bool)
        self.closed = False
        self.__class__.instances.append(self)

    def reset(self):
        return self._obs(), {"action_mask": self._masks()}

    def _obs(self):
        return np.zeros((self.n, 2), dtype=np.float32)

    def _masks(self):
        return np.ones((self.n, 2), dtype=bool)

    def step(self, actions):
        terminated = np.zeros(self.n, dtype=bool)
        result = np.array([None] * self.n, dtype=object)
        turn_reached = np.ones(self.n, dtype=int) * 10
        for i in range(self.n):
            if self.pending_reset[i]:
                self.pending_reset[i] = False
                self.steps[i] = 0
                continue
            self.steps[i] += 1
            if self.steps[i] >= (1 if i == 0 else 2):
                terminated[i] = True
                self.pending_reset[i] = True
                result[i] = "lost"
                turn_reached[i] = 10 + self.steps[i]
        return (self._obs(), np.ones(self.n, dtype=float), terminated,
                np.zeros(self.n, dtype=bool),
                {"action_mask": self._masks(), "result": result,
                 "turn_reached": turn_reached,
                 "cards_in_drawn_pile": turn_reached.copy()})

    def close(self):
        self.closed = True


class TestNoisyDQNTrainer(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.agent = FakeAgent()
        # Construct without invoking DQNTrainer.__init__, so these unit tests
        # do not start a real game or load a checkpoint.
        self.trainer = object.__new__(NoisyDQNTrainer)
        self.trainer.agent = self.agent
        self.trainer.games_completed = 749004
        self.trainer.total_decisions = 100
        self.trainer.total_training_updates = 50
        self.trainer.CHECKPOINT_DIR = self.root / "checkpoints" / "noisy"
        self.trainer.LOG_DIR = self.root / "logs" / "noisy"
        self.trainer.INITIAL_CHECKPOINT = self.trainer.CHECKPOINT_DIR / "noisy_initial.pt"
        self.trainer.CHECKPOINT_DIR.mkdir(parents=True)
        self.trainer.LOG_DIR.mkdir(parents=True)
        self.trainer.close_evaluation_pool = MagicMock()
        self.trainer.print_checkpoint_performance = MagicMock()
        self.trainer.log_game = MagicMock()
        self.trainer.log_evaluation = MagicMock()

    def test_rejects_checkpoint_outside_noisy_directory(self):
        with self.assertRaises(ValueError):
            self.trainer.load_checkpoint(self.root / "normandy_25.pt")
        with self.assertRaises(ValueError):
            self.trainer.save_checkpoint(self.root / "normandy_25.pt")

    def test_explicit_checkpoint_path_has_no_fallback(self):
        source = self.trainer.CHECKPOINT_DIR / "chosen.pt"
        with patch.object(module, "load_noisy_checkpoint", return_value=source) as load:
            self.assertEqual(self.trainer.load_checkpoint(source), source)
            load.assert_called_once_with(self.trainer, source.resolve())

    def test_latest_numbered_checkpoint_selected(self):
        for name in ("noisy_initial.pt", "noisy_2.pt", "noisy_12.pt"):
            (self.trainer.CHECKPOINT_DIR / name).touch()
        with patch.object(module, "load_noisy_checkpoint") as load:
            self.trainer.load_checkpoint()
        self.assertEqual(load.call_args.args[1].name, "noisy_12.pt")

    def test_initial_checkpoint_selected_when_no_numbered_files(self):
        with patch.object(module, "load_noisy_checkpoint") as load:
            self.trainer.load_checkpoint()
        self.assertEqual(load.call_args.args[1].name, "noisy_initial.pt")

    def test_save_numbering_and_metadata(self):
        (self.trainer.CHECKPOINT_DIR / "noisy_3.pt").touch()
        path = self.trainer.save_checkpoint()
        self.assertEqual(path.name, "noisy_4.pt")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        self.assertEqual(checkpoint["network_architecture"], "noisy")
        self.assertEqual(checkpoint["games_completed"], 749004)
        self.assertEqual(checkpoint["n_step"], 6)
        self.assertEqual(checkpoint["epsilon"], 0.0)
        self.assertEqual(checkpoint["total_decisions"], 100)

    def test_save_does_not_overwrite_existing_file(self):
        existing = self.trainer.CHECKPOINT_DIR / "noisy_1.pt"
        existing.write_bytes(b"ORIGINAL")
        self.trainer.save_checkpoint()
        self.assertEqual(existing.read_bytes(), b"ORIGINAL")
        self.assertTrue((self.trainer.CHECKPOINT_DIR / "noisy_2.pt").exists())

    def test_save_rejects_priority_buffer(self):
        self.agent.replay_buffer_type = "priority"
        with self.assertRaises(ValueError):
            self.trainer.save_checkpoint()

    def test_log_paths_are_isolated(self):
        with patch.object(module.DQNTrainer, "log_game", return_value=None) as log:
            self.trainer.log_game = NoisyDQNTrainer.log_game.__get__(self.trainer)
            self.trainer.log_game({"game": 1}, path=self.root / "original.csv")
            self.assertEqual(log.call_args.kwargs["path"],
                             self.trainer.LOG_DIR / "training_history.csv")
        with patch.object(module.DQNTrainer, "log_evaluation", return_value=None) as log:
            self.trainer.log_evaluation = NoisyDQNTrainer.log_evaluation.__get__(self.trainer)
            self.trainer.log_evaluation(5, {"result": "lost"}, path=self.root / "original.csv")
            self.assertEqual(log.call_args.kwargs["path"],
                             self.trainer.LOG_DIR / "evaluation_history.csv")

    def test_cleanup_keeps_all_noisy_checkpoints(self):
        file = self.trainer.CHECKPOINT_DIR / "noisy_1.pt"
        file.touch()
        self.trainer.cleanup_old_checkpoints()
        self.assertTrue(file.exists())

    def test_train_rejects_invalid_arguments(self):
        for kwargs in ({"num_games": 0}, {"num_workers": 0},
                       {"progress_interval": 0}, {"evaluation_interval": 0},
                       {"evaluation_games": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.trainer.train(**kwargs)

    def test_train_rejects_checkpoint_without_resume(self):
        with self.assertRaises(ValueError):
            self.trainer.train(num_games=1, checkpoint_path="ignored.pt")

    def test_train_rejects_overwrite(self):
        with self.assertRaises(ValueError):
            self.trainer.train(num_games=1, overwrite=True)

    def test_parallel_training_counts_and_resets(self):
        FakeVectorEnv.instances.clear()
        fake_manager = MagicMock()
        fake_manager.Value.return_value = MagicMock()
        fake_manager.Lock.return_value = MagicMock()
        with (patch.object(module, "AsyncVectorEnv", FakeVectorEnv),
              patch.object(module, "NStepTransitionCollector", FakeCollector),
              patch.object(module.multiprocessing, "Manager", return_value=fake_manager),
              patch.object(module, "load_hyperparameters", return_value={}),
              patch.dict(module.MIDGAME_CONFIG, {"enabled": False})):
            results = self.trainer.train(
                num_games=3, num_workers=2, progress_interval=100,
                evaluation_interval=1000, save_checkpoints=False)
        self.assertEqual(len(results), 3)
        self.assertEqual(self.trainer.games_completed, 749007)
        self.assertEqual(self.agent.epsilon, 0.0)
        self.assertEqual(self.agent.training_calls,
                         self.trainer.total_training_updates - 50)
        self.assertGreater(len(self.agent.replay_buffer.buffer), 0)
        self.assertTrue(FakeVectorEnv.instances[-1].closed)
        fake_manager.shutdown.assert_called_once()
        self.assertTrue(any(mask.any() for mask in self.agent.actions_seen))
        self.assertTrue(all(r["result"] == "lost" for r in results))

    def test_evaluation_restores_noisy_mode(self):
        fake_env = MagicMock()
        fake_env.reset.return_value = (np.zeros(2), {"action_mask": np.ones(2, dtype=bool)})
        fake_env.step.return_value = (
            np.zeros(2), 0.0, True, False,
            {"turn_reached": 35, "result": "lost"})
        with patch.object(module, "NormandyEnv", return_value=fake_env):
            result = self.trainer.evaluate_policy()
        self.assertEqual(result, {"turn_reached": 35, "result": "lost"})
        self.assertEqual(self.agent.deterministic_calls, 1)
        self.assertEqual(self.agent.noisy_calls, 1)
        fake_env.close.assert_called_once()

    def test_evaluation_restores_noisy_mode_after_error(self):
        fake_env = MagicMock()
        fake_env.reset.side_effect = RuntimeError("evaluation failed")
        with patch.object(module, "NormandyEnv", return_value=fake_env):
            with self.assertRaises(RuntimeError):
                self.trainer.evaluate_policy()
        self.assertEqual(self.agent.noisy_calls, 1)
        fake_env.close.assert_called_once()


if __name__ == "__main__":
    unittest.main(verbosity=2)

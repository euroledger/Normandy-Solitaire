
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from core.ai.dqn.noisy.dqn_noisy_network import DQNNoisyNetwork
from core.ai.dqn.noisy.dqn_noisy_agent import NoisyDQNAgent
from core.ai.dqn.noisy.dqn_noisy_checkpoint import load_noisy_checkpoint


class DummyTrainer:
    # Minimal trainer interface required by the checkpoint loader.

    def __init__(self, agent):
        self.agent = agent
        self.games_completed = 0
        self.total_decisions = 0
        self.total_training_updates = 0


class TestNoisyCheckpoint(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)

        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)

        self.checkpoint_path = (
            Path(self.temp_dir.name) / "test_noisy.pt"
        )

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
            "network_sync_rate": 100,
            "enable_double_dqn": True,
            "enable_dueling_dqn": False,
            "replay_buffer_type": "uniform",
            "train_frequency": 1,
        }

        with patch(
            "core.ai.dqn.noisy.dqn_noisy_network.load_hyperparameters",
            return_value=self.params,
        ):
            network = DQNNoisyNetwork(12, 8)

        with patch(
            "core.ai.dqn.dqn_agent.load_hyperparameters",
            return_value=self.params,
        ):
            self.agent = NoisyDQNAgent(
                network,
                torch.device("cpu"),
            )

        self.trainer = DummyTrainer(self.agent)

        self.checkpoint = {
            "version": 1,
            "network_architecture": "noisy",
            "policy_net": self.agent.policy_net.state_dict(),
            "target_net": self.agent.target_net.state_dict(),
            "optimizer": None,
            "replay_buffer": [],
            "n_step": 6,
            "train_frequency": 1,
            "epsilon": 0.2,
            "training_steps": 1234,
            "per_beta_steps": 0,
            "games_completed": 749004,
            "total_decisions": 8500000,
            "total_training_updates": 500000,
        }

    def save_checkpoint(self):
        torch.save(
            self.checkpoint,
            self.checkpoint_path,
        )

    def test_successful_restoration(self):
        self.save_checkpoint()

        restored_path = load_noisy_checkpoint(
            self.trainer,
            self.checkpoint_path,
        )

        self.assertEqual(
            restored_path,
            self.checkpoint_path,
        )
        self.assertEqual(
            self.trainer.games_completed,
            749004,
        )
        self.assertEqual(
            self.trainer.total_decisions,
            8500000,
        )
        self.assertEqual(
            self.trainer.total_training_updates,
            500000,
        )
        self.assertEqual(
            self.agent.training_steps,
            1234,
        )
        self.assertEqual(self.agent.epsilon, 0.0)

    def test_policy_weights_restored(self):
        self.save_checkpoint()

        expected = {
            key: value.clone()
            for key, value in self.checkpoint[
                "policy_net"
            ].items()
        }

        with torch.no_grad():
            for parameter in self.agent.policy_net.parameters():
                parameter.add_(1.0)

        load_noisy_checkpoint(
            self.trainer,
            self.checkpoint_path,
        )

        actual = self.agent.policy_net.state_dict()

        for key, value in expected.items():
            self.assertTrue(
                torch.equal(value, actual[key]),
                msg=f"Policy mismatch: {key}",
            )

    def test_target_weights_restored(self):
        self.save_checkpoint()

        expected = {
            key: value.clone()
            for key, value in self.checkpoint[
                "target_net"
            ].items()
        }

        with torch.no_grad():
            for parameter in self.agent.target_net.parameters():
                parameter.add_(2.0)

        load_noisy_checkpoint(
            self.trainer,
            self.checkpoint_path,
        )

        actual = self.agent.target_net.state_dict()

        for key, value in expected.items():
            self.assertTrue(
                torch.equal(value, actual[key]),
                msg=f"Target mismatch: {key}",
            )

    def test_optimizer_reinitialised(self):
        self.save_checkpoint()

        original_optimizer = self.agent.optimizer

        load_noisy_checkpoint(
            self.trainer,
            self.checkpoint_path,
        )

        # Converted checkpoints have optimizer=None.
        # The agent's freshly initialised optimizer is retained.
        self.assertIs(
            self.agent.optimizer,
            original_optimizer,
        )

    def test_replay_buffer_restored(self):
        # Use a valid transition generated by the
        # existing replay-buffer implementation.
        import numpy as np

        experience = (
            np.zeros(12, dtype=np.float32),
            2,
            1.0,
            np.ones(12, dtype=np.float32),
            False,
            np.ones(8, dtype=bool),
            6,
        )

        self.checkpoint["replay_buffer"] = [experience]
        self.save_checkpoint()

        load_noisy_checkpoint(
            self.trainer,
            self.checkpoint_path,
        )

        self.assertEqual(
            len(self.agent.replay_buffer),
            1,
        )

    def test_missing_checkpoint(self):
        with self.assertRaises(FileNotFoundError):
            load_noisy_checkpoint(
                self.trainer,
                self.checkpoint_path,
            )

    def test_rejects_wrong_architecture(self):
        self.checkpoint["network_architecture"] = "standard"
        self.save_checkpoint()

        with self.assertRaises(ValueError):
            load_noisy_checkpoint(
                self.trainer,
                self.checkpoint_path,
            )

    def test_rejects_wrong_version(self):
        self.checkpoint["version"] = 999
        self.save_checkpoint()

        with self.assertRaises(ValueError):
            load_noisy_checkpoint(
                self.trainer,
                self.checkpoint_path,
            )

    def test_rejects_n_step_mismatch(self):
        self.checkpoint["n_step"] = 3
        self.save_checkpoint()

        with self.assertRaises(ValueError):
            load_noisy_checkpoint(
                self.trainer,
                self.checkpoint_path,
            )

    def test_rejects_train_frequency_mismatch(self):
        self.checkpoint["train_frequency"] = 5
        self.save_checkpoint()

        with self.assertRaises(ValueError):
            load_noisy_checkpoint(
                self.trainer,
                self.checkpoint_path,
            )

    def test_rejects_invalid_replay_buffer(self):
        self.checkpoint["replay_buffer"] = {
            "type": "priority"
        }
        self.save_checkpoint()

        with self.assertRaises(ValueError):
            load_noisy_checkpoint(
                self.trainer,
                self.checkpoint_path,
            )

    def test_rejects_oversized_replay_buffer(self):
        self.checkpoint["replay_buffer"] = [
            None
        ] * 101
        self.save_checkpoint()

        with self.assertRaises(ValueError):
            load_noisy_checkpoint(
                self.trainer,
                self.checkpoint_path,
            )

    def test_noise_enabled_after_restoration(self):
        self.save_checkpoint()

        self.agent.set_deterministic()

        load_noisy_checkpoint(
            self.trainer,
            self.checkpoint_path,
        )

        self.assertTrue(
            self.agent.policy_net.network[6].noise_enabled
        )
        self.assertTrue(
            self.agent.target_net.network[6].noise_enabled
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)

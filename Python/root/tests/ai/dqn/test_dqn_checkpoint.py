import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from core.ai.dqn.dqn_agent import DQNAgent
from core.ai.dqn.dqn_trainer import DQNTrainer


class TestDQNCheckpoint(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        # This is a directory/legacy-file hint. Numbered checkpoints
        # are created alongside it as normandy_1.pt, normandy_2.pt, etc.
        self.checkpoint_path = Path(self.temp_dir.name) / "checkpoint.pt"

        env = SimpleNamespace(
            observation_space=SimpleNamespace(shape=(3,)),
            action_space=SimpleNamespace(n=4),
        )

        network = nn.Sequential(
            nn.Linear(3, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
        )

        agent = DQNAgent(network, torch.device("cpu"))
        agent.mini_batch_size = 2

        self.trainer = DQNTrainer(env=env, agent=agent)

    def tearDown(self):
        self.temp_dir.cleanup()

    def add_experiences(self):
        for i in range(3):
            self.trainer.agent.replay_buffer.add(
                observation=np.array([i, 0, 1], dtype=np.float32),
                action_id=i,
                reward=float(i + 1),
                next_observation=np.array([i + 1, 0, 1], dtype=np.float32),
                terminated=False,
                next_action_mask=np.ones(4, dtype=np.float32),
            )

    def test_save_and_load_checkpoint(self):
        agent = self.trainer.agent

        self.add_experiences()

        # Perform an update to populate the optimizer state.
        agent.train_step()

        agent.epsilon = 0.42
        self.trainer.games_completed = 100
        self.trainer.total_decisions = 3500
        self.trainer.total_training_updates = agent.training_steps

        original_policy = {
            key: value.clone()
            for key, value in agent.policy_net.state_dict().items()
        }
        original_target = {
            key: value.clone()
            for key, value in agent.target_net.state_dict().items()
        }
        original_optimizer = agent.optimizer.state_dict()
        original_training_steps = agent.training_steps

        saved_path = self.trainer.save_checkpoint(self.checkpoint_path)
        self.assertTrue(saved_path.is_file())
        self.assertEqual(saved_path.name, "normandy_1.pt")

        # Modify all the state that should be restored.
        with torch.no_grad():
            for parameter in agent.policy_net.parameters():
                parameter.zero_()
            for parameter in agent.target_net.parameters():
                parameter.zero_()

        agent.epsilon = 1.0
        agent.training_steps = 0
        agent.replay_buffer.buffer.clear()
        self.trainer.games_completed = 0
        self.trainer.total_decisions = 0
        self.trainer.total_training_updates = 0

        # The loader discovers the latest numbered checkpoint in this directory.
        self.trainer.load_checkpoint(self.checkpoint_path)

        for key, value in original_policy.items():
            self.assertTrue(
                torch.equal(value, agent.policy_net.state_dict()[key])
            )
        for key, value in original_target.items():
            self.assertTrue(
                torch.equal(value, agent.target_net.state_dict()[key])
            )

        self.assertAlmostEqual(agent.epsilon, 0.42)
        self.assertEqual(agent.training_steps, original_training_steps)
        self.assertEqual(self.trainer.games_completed, 100)
        self.assertEqual(self.trainer.total_decisions, 3500)
        self.assertEqual(
            self.trainer.total_training_updates,
            original_training_steps,
        )
        self.assertEqual(len(agent.replay_buffer), 3)

        restored_optimizer = agent.optimizer.state_dict()
        self.assertEqual(
            restored_optimizer["param_groups"],
            original_optimizer["param_groups"],
        )
        self.assertEqual(
            len(restored_optimizer["state"]),
            len(original_optimizer["state"]),
        )




    def test_incompatible_training_frequency(self):
        saved_path = self.trainer.save_checkpoint(self.checkpoint_path)
        self.assertTrue(saved_path.is_file())

        self.trainer.agent.train_frequency += 1

        with self.assertRaisesRegex(
            RuntimeError,
            "No valid checkpoint could be loaded"
        ):
            self.trainer.load_checkpoint(self.checkpoint_path)



    def test_replay_buffer_capacity_preserved(self):
        self.add_experiences()
        capacity = self.trainer.agent.replay_buffer.buffer.maxlen

        saved_path = self.trainer.save_checkpoint(self.checkpoint_path)
        self.assertTrue(saved_path.is_file())

        self.trainer.agent.replay_buffer.buffer.clear()
        self.trainer.load_checkpoint(self.checkpoint_path)

        self.assertEqual(
            self.trainer.agent.replay_buffer.buffer.maxlen,
            capacity,
        )
        self.assertEqual(len(self.trainer.agent.replay_buffer), 3)

    def test_missing_checkpoint(self):
        with self.assertRaises(FileNotFoundError):
            self.trainer.load_checkpoint(self.checkpoint_path)
            
if __name__ == "__main__":
    unittest.main()

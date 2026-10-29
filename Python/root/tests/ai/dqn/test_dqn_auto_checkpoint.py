import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from core.ai.dqn.dqn_agent import DQNAgent
from core.ai.dqn.dqn_trainer import DQNTrainer
from tests.ai.dqn.test_dqn_trainer_async import (
    FakeAsyncVectorEnv,
    FakeNormandyEnv,
)


class TestDQNAutoCheckpoint(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

        # This is now a directory reference for checkpoint discovery.
        # The actual saved files will be normandy_1.pt, normandy_2.pt, etc.
        self.checkpoint_path = (
            Path(self.temp_dir.name) / "normandy_latest.pt"
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def make_trainer(self):
        env = FakeNormandyEnv()

        network = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 5),
        )

        agent = DQNAgent(network, torch.device("cpu"))
        agent.mini_batch_size = 2

        return DQNTrainer(env=env, agent=agent)

    def numbered_checkpoints(self):
        """Return numbered checkpoint files in numerical order."""

        checkpoints = []

        for file in self.checkpoint_path.parent.glob("normandy_*.pt"):
            suffix = file.stem.removeprefix("normandy_")

            if suffix.isdigit():
                checkpoints.append((int(suffix), file))

        checkpoints.sort()

        return [file for _, file in checkpoints]

    def run_training(self, trainer, **kwargs):
        """Use the deterministic fake vector environment."""

        with patch(
            "core.ai.dqn.dqn_trainer.AsyncVectorEnv",
            FakeAsyncVectorEnv,
        ), contextlib.redirect_stdout(io.StringIO()):

            return trainer.train(
                num_workers=1,
                **kwargs,
            )

    def test_checkpoint_created_after_training(self):
        trainer = self.make_trainer()

        self.run_training(
            trainer,
            num_games=2,
            resume=True,
            checkpoint_path=self.checkpoint_path,
            save_interval=10000,
        )

        checkpoints = self.numbered_checkpoints()

        self.assertEqual(len(checkpoints), 1)
        self.assertEqual(checkpoints[0].name, "normandy_1.pt")
        self.assertEqual(trainer.games_completed, 2)

    def test_resume_restores_training_progress(self):
        first = self.make_trainer()

        self.run_training(
            first,
            num_games=3,
            resume=True,
            checkpoint_path=self.checkpoint_path,
            save_interval=10000,
        )

        saved_decisions = first.total_decisions
        saved_updates = first.total_training_updates
        saved_epsilon = first.agent.epsilon
        saved_replay_size = len(first.agent.replay_buffer)

        second = self.make_trainer()

        results = self.run_training(
            second,
            num_games=2,
            resume=True,
            checkpoint_path=self.checkpoint_path,
            save_interval=10000,
        )

        self.assertEqual(second.games_completed, 5)
        self.assertGreater(second.total_decisions, saved_decisions)
        self.assertGreaterEqual(
            second.total_training_updates,
            saved_updates,
        )
        self.assertLessEqual(second.agent.epsilon, saved_epsilon)
        self.assertLess(second.agent.epsilon, 0.90)
        self.assertGreater(
            len(second.agent.replay_buffer),
            saved_replay_size,
        )
        self.assertEqual(
            [result["game"] for result in results],
            [4, 5],
        )

        # The second run should have produced a new checkpoint.
        # Cleanup should have retained only that checkpoint.
        checkpoints = self.numbered_checkpoints()

        self.assertEqual(len(checkpoints), 1)
        self.assertEqual(checkpoints[0].name, "normandy_2.pt")

    def test_sequential_checkpoint_numbering(self):
        trainer = self.make_trainer()

        # Save directly to inspect numbering before cleanup.
        first = trainer.save_checkpoint(self.checkpoint_path)
        second = trainer.save_checkpoint(self.checkpoint_path)
        third = trainer.save_checkpoint(self.checkpoint_path)

        self.assertEqual(first.name, "normandy_1.pt")
        self.assertEqual(second.name, "normandy_2.pt")
        self.assertEqual(third.name, "normandy_3.pt")

        self.assertTrue(first.is_file())
        self.assertTrue(second.is_file())
        self.assertTrue(third.is_file())

    def test_existing_checkpoint_is_not_overwritten(self):
        trainer = self.make_trainer()

        first_path = trainer.save_checkpoint(self.checkpoint_path)
        original_bytes = first_path.read_bytes()

        second_path = trainer.save_checkpoint(self.checkpoint_path)

        self.assertNotEqual(first_path, second_path)

        self.assertEqual(
            first_path.read_bytes(),
            original_bytes,
        )

        self.assertTrue(second_path.is_file())

    def test_resume_uses_highest_numbered_checkpoint(self):
        trainer = self.make_trainer()

        trainer.games_completed = 3
        trainer.save_checkpoint(self.checkpoint_path)

        trainer.games_completed = 7
        trainer.save_checkpoint(self.checkpoint_path)

        trainer.games_completed = 12
        trainer.save_checkpoint(self.checkpoint_path)

        resumed = self.make_trainer()

        loaded_path = resumed.load_checkpoint(
            self.checkpoint_path
        )

        self.assertEqual(loaded_path.name, "normandy_3.pt")
        self.assertEqual(resumed.games_completed, 12)

    def test_corrupt_latest_checkpoint_falls_back(self):
        trainer = self.make_trainer()

        trainer.games_completed = 5
        trainer.save_checkpoint(self.checkpoint_path)

        trainer.games_completed = 10
        latest = trainer.save_checkpoint(self.checkpoint_path)

        # Simulate an interrupted or corrupted checkpoint.
        latest.write_bytes(b"invalid checkpoint")

        resumed = self.make_trainer()

        with contextlib.redirect_stdout(io.StringIO()):
            loaded_path = resumed.load_checkpoint(
                self.checkpoint_path
            )

        self.assertEqual(loaded_path.name, "normandy_1.pt")
        self.assertEqual(resumed.games_completed, 5)

    def test_periodic_saving_by_game_count(self):
        trainer = self.make_trainer()

        # Disable cleanup temporarily so we can inspect
        # every checkpoint produced during training.
        with patch.object(
            trainer,
            "cleanup_old_checkpoints",
        ), patch.object(
            trainer,
            "save_checkpoint",
            wraps=trainer.save_checkpoint,
        ) as mock_save:

            self.run_training(
                trainer,
                num_games=5,
                resume=True,
                checkpoint_path=self.checkpoint_path,
                save_interval=2,
            )

            # Periodic saves at games 2 and 4,
            # plus a final save at game 5.
            self.assertEqual(mock_save.call_count, 3)

        checkpoints = self.numbered_checkpoints()

        self.assertEqual(
            [file.name for file in checkpoints],
            [
                "normandy_1.pt",
                "normandy_2.pt",
                "normandy_3.pt",
            ],
        )

        saved_games = [
            torch.load(
                file,
                map_location="cpu",
                weights_only=False,
            )["games_completed"]
            for file in checkpoints
        ]

        self.assertEqual(saved_games, [2, 4, 5])

    def test_no_duplicate_final_checkpoint(self):
        trainer = self.make_trainer()

        with patch.object(
            trainer,
            "save_checkpoint",
            wraps=trainer.save_checkpoint,
        ) as mock_save:

            self.run_training(
                trainer,
                num_games=4,
                resume=True,
                checkpoint_path=self.checkpoint_path,
                save_interval=2,
            )

            # Save at games 2 and 4.
            # No third save at completion.
            self.assertEqual(mock_save.call_count, 2)

        # Cleanup retains only the final checkpoint.
        checkpoints = self.numbered_checkpoints()

        self.assertEqual(len(checkpoints), 1)
        self.assertEqual(checkpoints[0].name, "normandy_2.pt")

    def test_cleanup_keeps_only_latest_checkpoint(self):
        trainer = self.make_trainer()

        trainer.save_checkpoint(self.checkpoint_path)
        trainer.save_checkpoint(self.checkpoint_path)
        trainer.save_checkpoint(self.checkpoint_path)

        self.assertEqual(
            len(self.numbered_checkpoints()),
            3,
        )

        trainer.cleanup_old_checkpoints(self.checkpoint_path)

        checkpoints = self.numbered_checkpoints()

        self.assertEqual(len(checkpoints), 1)
        self.assertEqual(checkpoints[0].name, "normandy_3.pt")

    def test_cleanup_preserves_legacy_checkpoint(self):
        trainer = self.make_trainer()

        legacy_path = self.checkpoint_path
        legacy_path.write_bytes(b"legacy checkpoint placeholder")

        trainer.save_checkpoint(self.checkpoint_path)
        trainer.save_checkpoint(self.checkpoint_path)

        trainer.cleanup_old_checkpoints(self.checkpoint_path)

        self.assertTrue(legacy_path.is_file())

        checkpoints = self.numbered_checkpoints()

        self.assertEqual(len(checkpoints), 1)
        self.assertEqual(checkpoints[0].name, "normandy_2.pt")

    def test_failed_save_preserves_previous_checkpoint(self):
        trainer = self.make_trainer()

        first_path = trainer.save_checkpoint(
            self.checkpoint_path
        )

        original_bytes = first_path.read_bytes()

        # Simulate a disk/write failure.
        with patch(
            "core.ai.dqn.dqn_trainer.torch.save",
            side_effect=OSError("Simulated disk failure"),
        ), self.assertRaises(OSError):
            trainer.save_checkpoint(self.checkpoint_path)

        self.assertTrue(first_path.is_file())

        self.assertEqual(
            first_path.read_bytes(),
            original_bytes,
        )

        # Failed checkpoint must not remain as a numbered file.
        self.assertEqual(
            [file.name for file in self.numbered_checkpoints()],
            ["normandy_1.pt"],
        )

    def test_missing_checkpoint_starts_fresh(self):
        trainer = self.make_trainer()

        results = self.run_training(
            trainer,
            num_games=2,
            resume=True,
            checkpoint_path=self.checkpoint_path,
            save_interval=10000,
        )

        self.assertEqual(trainer.games_completed, 2)

        self.assertEqual(
            [result["game"] for result in results],
            [1, 2],
        )

        self.assertEqual(
            len(self.numbered_checkpoints()),
            1,
        )

    def test_training_without_checkpointing(self):
        trainer = self.make_trainer()

        self.run_training(
            trainer,
            num_games=2,
        )

        self.assertEqual(trainer.games_completed, 2)

        self.assertEqual(
            self.numbered_checkpoints(),
            [],
        )

        self.assertFalse(self.checkpoint_path.exists())


if __name__ == "__main__":
    unittest.main()

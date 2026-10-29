import csv
import tempfile
import unittest
from pathlib import Path

from core.ai.dqn.dqn_trainer import DQNTrainer


class TestDQNLogging(unittest.TestCase):

    def test_game_log_created_and_appended(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "logs" / "training_history.csv"

            trainer = DQNTrainer.__new__(DQNTrainer)

            game1 = {
                "game": 1,
                "reward": 5.0,
                "decisions": 35,
                "result": "LOSS",
                "epsilon": 0.99,
                "average_loss": None,
            }

            game2 = {
                "game": 2,
                "reward": 8.0,
                "decisions": 48,
                "result": "LOSS",
                "epsilon": 0.98,
                "average_loss": 0.15,
            }

            trainer.log_game(game1, path)
            trainer.log_game(game2, path)

            self.assertTrue(path.exists())

            with path.open(newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))

            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["game"], "1")
            self.assertEqual(rows[1]["game"], "2")
            self.assertEqual(rows[0]["reward"], "5.0")
            self.assertEqual(rows[1]["reward"], "8.0")
            self.assertEqual(rows[1]["average_loss"], "0.15")


if __name__ == "__main__":
    unittest.main()

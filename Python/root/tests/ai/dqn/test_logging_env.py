"""Unit tests for conditional capture of Normandy's existing print output.

Run with:
    python -m pytest tests/ai/dqn/test_logging_env.py -v

These tests mock game-engine reset/step, exercising the real logging wrapper.
"""

import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from core.ai.dqn.normandy_env import NormandyEnv
from core.ai.dqn.normandy_logging_env import NormandyLoggingEnv
from core.global_game_state import GlobalGameState


class SharedCount:
    """Minimal stand-in for multiprocessing.Manager().Value('i', 0)."""

    def __init__(self, value=0):
        self.value = value


class TestNormandyLoggingEnv(unittest.TestCase):
    def setUp(self):
        self.original_cards_drawn = GlobalGameState.cards_drawn
        self.addCleanup(self._restore_cards_drawn)
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.output_dir = Path(self.temp_dir.name)
        self.counter = SharedCount()
        self.lock = threading.Lock()
        self.config = {
            "enabled": True,
            "result": "won",
            "target_turn": None,
            "max_files": 5,
            "output_dir": str(self.output_dir),
        }
        self.env = None
        self.addCleanup(self._close_env)

    def _restore_cards_drawn(self):
        GlobalGameState.cards_drawn = self.original_cards_drawn

    def _close_env(self):
        if self.env is not None:
            self.env.close()
            self.env = None

    def _make_env(self, **overrides):
        self._close_env()
        self.config = {**self.config, **overrides}
        self.env = NormandyLoggingEnv(self.config, self.counter, self.lock)
        return self.env

    def _files(self):
        return sorted(self.output_dir.glob("*.txt"))

    def _reset(self, env, message="PRINT FROM RESET"):
        def fake_reset(_env, *, seed=None, options=None):
            GlobalGameState.cards_drawn = 0
            print(message)
            return np.zeros(1, dtype=np.float32), {"action_mask": np.ones(1)}

        with patch.object(NormandyEnv, "reset", fake_reset):
            env.reset()

    def _step(self, env, *, message, turn, result=None, terminated=False,
              truncated=False):
        def fake_step(_env, action):
            print(message)
            GlobalGameState.cards_drawn = turn
            return (
                np.zeros(1, dtype=np.float32),
                0.0,
                terminated,
                truncated,
                {"result": result, "turn_reached": turn},
            )

        with patch.object(NormandyEnv, "step", fake_step):
            return env.step(0)

    # ============================================================
    # ERROR LOGGING TESTS
    # ============================================================

    def _make_error_env(
        self,
        *,
        game_enabled=False,
        error_enabled=True,
        **game_overrides,
    ):
        """Create an environment with independent logging settings."""
        self._close_env()

        game_config = {
            **self.config,
            **game_overrides,
            "enabled": game_enabled,
        }

        error_config = {
            "enabled": error_enabled,
            "output_dir": str(self.output_dir / "errors"),
        }

        self.env = NormandyLoggingEnv(
            config=game_config,
            shared_count=self.counter,
            shared_lock=self.lock,
            error_config=error_config,
        )
        return self.env

    def _error_files(self):
        return sorted((self.output_dir / "errors").glob("*.txt"))

    def _raise_on_step(self, env, message="SIMULATED GAME ERROR"):
        """Raise an exception inside the real logging wrapper."""
        def fake_step(_env, action):
            print("EVENT IMMEDIATELY BEFORE ERROR")
            raise RuntimeError(message)

        with patch.object(NormandyEnv, "step", fake_step):
            return env.step(0)

    def test_error_logging_saves_full_game_history(self):
        """Crash log contains reset, earlier turns and traceback."""
        env = self._make_error_env()

        self._reset(env, message="INITIAL GAME STATE")
        self._step(env, message="ALLIED ADVANCE TO RENNES", turn=20)
        self._step(env, message="GERMAN COUNTERATTACK", turn=21)

        with self.assertRaisesRegex(
            RuntimeError, "SIMULATED GAME ERROR"
        ):
            self._raise_on_step(env)

        files = self._error_files()
        self.assertEqual(len(files), 1)

        content = files[0].read_text(encoding="utf-8")

        self.assertIn("INITIAL GAME STATE", content)
        self.assertIn("ALLIED ADVANCE TO RENNES", content)
        self.assertIn("GERMAN COUNTERATTACK", content)
        self.assertIn("EVENT IMMEDIATELY BEFORE ERROR", content)
        self.assertIn("SIMULATED GAME ERROR", content)
        self.assertIn("Traceback", content)

    def test_error_logging_works_when_game_logging_disabled(self):
        """No ordinary game logging is required to capture a crash."""
        env = self._make_error_env(
            game_enabled=False,
            error_enabled=True,
        )

        self._reset(env)
        self.assertIsNotNone(env.log_buffer)

        with self.assertRaises(RuntimeError):
            self._raise_on_step(env)

        self.assertEqual(len(self._error_files()), 1)
        self.assertEqual(self._files(), [])
        self.assertEqual(self.counter.value, 0)

    def test_error_logging_disabled_creates_no_error_file(self):
        """Both logging modes disabled means no captured log."""
        env = self._make_error_env(
            game_enabled=False,
            error_enabled=False,
        )

        self._reset(env)
        self.assertIsNone(env.log_buffer)

        with self.assertRaises(RuntimeError):
            self._raise_on_step(env)

        self.assertEqual(self._error_files(), [])
        self.assertEqual(self._files(), [])

    def test_error_logging_ignores_game_result_and_turn_filters(self):
        """Crashes are saved even when normal game filters exclude them."""
        env = self._make_error_env(
            game_enabled=True,
            error_enabled=True,
            result="won",
            target_turn=[40, 48],
        )

        self._reset(env)
        self._step(env, message="TURN TWO", turn=2)

        with self.assertRaises(RuntimeError):
            self._raise_on_step(env)

        files = self._error_files()
        self.assertEqual(len(files), 1)
        self.assertIn(
            "TURN TWO",
            files[0].read_text(encoding="utf-8"),
        )

    def test_error_logging_ignores_normal_file_limit(self):
        """A full normal-log quota must not disable crash capture."""
        self.counter.value = 5

        env = self._make_error_env(
            game_enabled=True,
            error_enabled=True,
            max_files=5,
        )

        self._reset(env)
        self.assertIsNotNone(env.log_buffer)

        with self.assertRaises(RuntimeError):
            self._raise_on_step(env)

        self.assertEqual(len(self._error_files()), 1)
        self.assertEqual(self.counter.value, 5)

    def test_error_logging_catches_reset_exception(self):
        """Errors during game initialization are also captured."""
        env = self._make_error_env()

        def failing_reset(_env, *, seed=None, options=None):
            print("RESET BEGAN")
            raise ValueError("RESET FAILED")

        with patch.object(NormandyEnv, "reset", failing_reset):
            with self.assertRaisesRegex(ValueError, "RESET FAILED"):
                env.reset()

        files = self._error_files()
        self.assertEqual(len(files), 1)

        content = files[0].read_text(encoding="utf-8")
        self.assertIn("RESET BEGAN", content)
        self.assertIn("RESET FAILED", content)
        self.assertIn("ValueError", content)

    def test_error_logging_preserves_original_exception(self):
        """The wrapper must re-raise the original exception object."""
        env = self._make_error_env()
        self._reset(env)

        original_error = RuntimeError("ORIGINAL GAME ERROR")

        def failing_step(_env, action):
            raise original_error

        with patch.object(NormandyEnv, "step", failing_step), self.assertRaises(RuntimeError) as caught:
            env.step(0)

        self.assertIs(caught.exception, original_error)
        self.assertEqual(len(self._error_files()), 1)

    def test_error_logging_survives_file_write_failure(self):
        """Failure to write a log must not replace the game exception."""
        env = self._make_error_env()
        self._reset(env)

        original_error = RuntimeError("GAME ENGINE FAILURE")

        def failing_step(_env, action):
            raise original_error

        with patch.object(NormandyEnv, "step", failing_step), patch.object(
            env,
            "_save_error",
            side_effect=OSError("DISK FULL"),
        ), self.assertRaises(RuntimeError) as caught:
            env.step(0)

        self.assertIs(caught.exception, original_error)

    def test_error_logging_creates_unique_files(self):
        """Separate crashes must not overwrite earlier crash logs."""
        env = self._make_error_env()

        for game_number in range(2):
            self._reset(
                env,
                message=f"GAME {game_number} RESET",
            )

            with self.assertRaises(RuntimeError):
                self._raise_on_step(
                    env,
                    message=f"GAME {game_number} CRASH",
                )

        files = self._error_files()
        self.assertEqual(len(files), 2)
        self.assertNotEqual(files[0].name, files[1].name)

        contents = [
            path.read_text(encoding="utf-8")
            for path in files
        ]

        self.assertTrue(
            any("GAME 0 CRASH" in text for text in contents)
        )
        self.assertTrue(
            any("GAME 1 CRASH" in text for text in contents)
        )

    def test_error_logging_discards_previous_game_history(self):
        """A reset must start a fresh game history."""
        env = self._make_error_env()

        self._reset(env, message="PREVIOUS GAME RESET")
        self._step(env, message="PREVIOUS GAME EVENT", turn=5)

        self._reset(env, message="CURRENT GAME RESET")

        with self.assertRaises(RuntimeError):
            self._raise_on_step(env)

        content = self._error_files()[0].read_text(
            encoding="utf-8"
        )

        self.assertIn("CURRENT GAME RESET", content)
        self.assertNotIn("PREVIOUS GAME RESET", content)
        self.assertNotIn("PREVIOUS GAME EVENT", content)

    def test_error_logging_captures_after_normal_target_exceeded(self):
        """Normal early-discard optimisation must not lose crash history."""
        env = self._make_error_env(
            game_enabled=True,
            error_enabled=True,
            result="lost",
            target_turn=[5, 6],
        )

        self._reset(env)

        for turn in range(1, 11):
            self._step(
                env,
                message=f"GAME TURN {turn}",
                turn=turn,
            )

        self.assertIsNotNone(env.log_buffer)

        with self.assertRaises(RuntimeError):
            self._raise_on_step(env)

        content = self._error_files()[0].read_text(
            encoding="utf-8"
        )

        self.assertIn("GAME TURN 1", content)
        self.assertIn("GAME TURN 10", content)

    def test_normal_game_does_not_create_error_log(self):
        """Successful completion must not create a crash report."""
        env = self._make_error_env(
            game_enabled=False,
            error_enabled=True,
        )

        self._reset(env)
        self._step(
            env,
            message="NORMAL GAME COMPLETION",
            turn=48,
            result="won",
            terminated=True,
        )

        self.assertEqual(self._error_files(), [])
        self.assertIsNone(env.log_buffer)

    def test_winning_game_writes_existing_print_output(self):
        env = self._make_env(result="won", target_turn=None)
        self._reset(env)
        self._step(env, message="FIRST TURN OUTPUT", turn=1)
        self._step(env, message="FINAL COMBAT OUTPUT", turn=48,
                   result="won", terminated=True)

        files = self._files()
        self.assertEqual(len(files), 1)
        content = files[0].read_text(encoding="utf-8")
        self.assertIn("PRINT FROM RESET", content)
        self.assertIn("FIRST TURN OUTPUT", content)
        self.assertIn("FINAL COMBAT OUTPUT", content)
        self.assertEqual(self.counter.value, 1)

    def test_nonmatching_result_is_not_saved(self):
        env = self._make_env(result="won")
        self._reset(env)
        self._step(env, message="LOSING GAME OUTPUT", turn=5,
                   result="lost", terminated=True)
        self.assertEqual(self._files(), [])
        self.assertEqual(self.counter.value, 0)
        self.assertIsNone(env.log_buffer)

    def test_target_turn_uses_info_turn_reached(self):
        env = self._make_env(result="lost", target_turn=5)
        self._reset(env)
        self._step(env, message="LOSS ON TURN FIVE", turn=5,
                   result="lost", terminated=True)
        files = self._files()
        self.assertEqual(len(files), 1)
        self.assertIn("LOSS ON TURN FIVE", files[0].read_text(encoding="utf-8"))

    def test_nonmatching_turn_is_not_saved(self):
        env = self._make_env(result="lost", target_turn=5)
        self._reset(env)
        self._step(env, message="LOSS ON TURN FOUR", turn=4,
                   result="lost", terminated=True)
        self.assertEqual(self._files(), [])
        self.assertEqual(self.counter.value, 0)

    def test_capture_discarded_once_target_turn_exceeded(self):
        env = self._make_env(result="lost", target_turn=5)
        self._reset(env)
        self._step(env, message="TURN FIVE", turn=5)
        self.assertIsNotNone(env.log_buffer)
        self._step(env, message="TURN SIX", turn=6)
        self.assertIsNone(env.log_buffer)
        self._step(env, message="LATE LOSS", turn=7,
                   result="lost", terminated=True)
        self.assertEqual(self._files(), [])

    def test_any_result_accepts_loss(self):
        env = self._make_env(result="any", target_turn=None)
        self._reset(env)
        self._step(env, message="ANY RESULT OUTPUT", turn=3,
                   result="lost", terminated=True)
        self.assertEqual(len(self._files()), 1)

    def test_global_file_limit_prevents_additional_files(self):
        env = self._make_env(result="won", max_files=1)
        self._reset(env)
        self._step(env, message="FIRST WIN", turn=48,
                   result="won", terminated=True)
        self.assertEqual(len(self._files()), 1)
        self.assertEqual(self.counter.value, 1)

        self._reset(env, message="SECOND GAME RESET")
        self.assertIsNone(env.log_buffer)
        self._step(env, message="SECOND WIN", turn=48,
                   result="won", terminated=True)
        self.assertEqual(len(self._files()), 1)
        self.assertEqual(self.counter.value, 1)

    def test_existing_global_limit_prevents_capture_at_reset(self):
        self.counter.value = 5
        env = self._make_env(max_files=5)
        self._reset(env)
        self.assertIsNone(env.log_buffer)

    def test_reset_discards_previous_unfinished_game(self):
        env = self._make_env()
        self._reset(env, message="FIRST EPISODE")
        self._step(env, message="OLD GAME", turn=2)
        self._reset(env, message="SECOND EPISODE")
        self._step(env, message="SECOND GAME WIN", turn=48,
                   result="won", terminated=True)
        content = self._files()[0].read_text(encoding="utf-8")
        self.assertNotIn("FIRST EPISODE", content)
        self.assertNotIn("OLD GAME", content)
        self.assertIn("SECOND EPISODE", content)
        self.assertIn("SECOND GAME WIN", content)

    def test_capture_does_not_change_step_return_values(self):
        env = self._make_env()
        self._reset(env)
        result = self._step(env, message="A GAME EVENT", turn=1)
        self.assertEqual(len(result), 5)
        self.assertEqual(result[1], 0.0)
        self.assertFalse(result[2])
        self.assertFalse(result[3])
        self.assertEqual(result[4]["turn_reached"], 1)

    def test_multiple_target_turns_capture_only_selected_losses(self):
        env = self._make_env(result="lost", target_turn=[4, 5, 6])
        for turn in [3, 4, 5, 6, 7]:
            with self.subTest(turn=turn):
                self._reset(env)
                self._step(env, message=f"LOSS ON TURN {turn}", turn=turn,
                           result="lost", terminated=True)
                expected = sum(t in (4, 5, 6) for t in range(3, turn + 1))
                self.assertEqual(len(self._files()), expected)
                self.assertEqual(self.counter.value, expected)
                if turn in (4, 5, 6):
                    content = self._files()[-1].read_text(encoding="utf-8")
                    self.assertIn(f"LOSS ON TURN {turn}", content)
                    self.assertIn("PRINT FROM RESET", content)

    def test_multiple_targets_discard_after_last_target(self):
        env = self._make_env(result="lost", target_turn=[4, 5, 6])
        self._reset(env)
        for turn in [4, 5, 6]:
            self._step(env, message=f"TURN {turn}", turn=turn)
            self.assertIsNotNone(env.log_buffer)
        self._step(env, message="TURN 7", turn=7)
        self.assertIsNone(env.log_buffer)
        self._step(env, message="LATE LOSS", turn=8,
                   result="lost", terminated=True)
        self.assertEqual(self._files(), [])
        self.assertEqual(self.counter.value, 0)


if __name__ == "__main__":
    unittest.main()

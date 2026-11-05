"""Integration tests for DQN game-state snapshots.

Run from the project root:
    python -m unittest tests.test_save_game_state -v

Adjust the module path above if this file is not in tests/.
"""
import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import contextlib
import os

from core.ai.dqn.normandy_env import NormandyEnv
from core.ai.dqn.dqn_save_load import (
    VERSION,
    capture_state,
    restore_state,
    save_snapshot,
    load_snapshot,
    save_game,
    load_game,
)
from core.global_game_state import GlobalGameState
from core.ai.dqn.dqn_action_adapter import (
    get_action_phase_decision,
    get_hitler_intervention_decision,
)


class TestSaveGameState(unittest.TestCase):
    """Round-trip tests using real NormandyEnv game positions."""

    TARGET_TURN = 20
    MAX_SOURCE_GAMES = 100
    MAX_DECISIONS_PER_GAME = 10000


    def setUp(self):
        self._devnull = open(os.devnull, "w")
        self._stdout_redirect = contextlib.redirect_stdout(self._devnull)
        self._stderr_redirect = contextlib.redirect_stderr(self._devnull)

        self._stdout_redirect.__enter__()
        self._stderr_redirect.__enter__()

        self.addCleanup(self._devnull.close)
        self.addCleanup(self._stdout_redirect.__exit__, None, None, None)
        self.addCleanup(self._stderr_redirect.__exit__, None, None, None)

        self.env = NormandyEnv()
        self.rng = np.random.default_rng(12345)
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.addCleanup(self.env.close)

    def current_decision(self):
        """Recompute the actual DQN observation AND mask with decision context."""
        if self.env.current_decision_type == "hitler_intervention":
            observation, mask = get_hitler_intervention_decision(
                GlobalGameState.current_card
            )
        elif self.env.current_decision_type == "action_phase":
            observation, mask = get_action_phase_decision()
        else:
            raise AssertionError("Snapshot is not at an active decision")
        return observation, self.env.pad_action_mask(mask)

    def legal_mask(self):
        return self.current_decision()[1]

    def play_to_end(self, observation, info):
        decisions = 0
        while True:
            legal = np.flatnonzero(info["action_mask"])
            self.assertGreater(len(legal), 0, "No legal actions available")
            action = int(self.rng.choice(legal))
            observation, reward, terminated, truncated, info = self.env.step(action)
            decisions += 1
            self.assertLess(
                decisions, self.MAX_DECISIONS_PER_GAME,
                "Game exceeded decision safety limit",
            )
            if terminated or truncated:
                self.assertIn(info["result"], ("won", "lost"))
                return info, decisions

    def reach_turn_20(self):
        """Find a real, nonterminal first decision on turn 20."""
        for game_number in range(1, self.MAX_SOURCE_GAMES + 1):
            observation, info = self.env.reset()
            decisions = 0
            while True:
                if int(GlobalGameState.cards_drawn) == self.TARGET_TURN:
                    self.assertGreater(
                        np.count_nonzero(info["action_mask"]), 0
                    )
                    return observation.copy(), info["action_mask"].copy(), game_number

                legal = np.flatnonzero(info["action_mask"])
                self.assertGreater(len(legal), 0)
                action = int(self.rng.choice(legal))
                observation, _, terminated, truncated, info = self.env.step(action)
                decisions += 1
                self.assertLess(decisions, self.MAX_DECISIONS_PER_GAME)
                if terminated or truncated:
                    break
        self.fail(
            f"No game reached turn {self.TARGET_TURN} in "
            f"{self.MAX_SOURCE_GAMES} attempts"
        )

    def test_capture_is_independent_of_future_play(self):
        observation, mask, _ = self.reach_turn_20()
        snapshot = capture_state(self.env, metadata={"state_id": "test_20"})
        frozen = copy.deepcopy(snapshot)
        self.assertEqual(snapshot["global_game_state"]["cards_drawn"], 20)
        self.assertEqual(snapshot["metadata"]["state_id"], "test_20")

        # Advance the live game: the captured data must remain unchanged.
        legal = np.flatnonzero(mask)
        self.env.step(int(self.rng.choice(legal)))
        self.assertEqual(snapshot, frozen)

    def test_json_disk_round_trip_and_metadata(self):
        observation, mask, source_game = self.reach_turn_20()
        snapshot = capture_state(
            self.env,
            metadata={"state_id": "unit_test_001", "source_game": source_game},
        )
        final_info, decisions = self.play_to_end(
            observation, {"action_mask": mask}
        )
        snapshot["metadata"]["original_final_turn"] = int(final_info["turn_reached"])
        snapshot["metadata"]["original_result"] = final_info["result"]

        filename = Path(self.temp_dir.name) / "turn20.json"
        self.assertEqual(save_snapshot(snapshot, filename), filename)
        self.assertTrue(filename.is_file())
        loaded = load_snapshot(filename)
        self.assertEqual(loaded, snapshot)
        self.assertEqual(loaded["version"], VERSION)
        self.assertEqual(loaded["metadata"]["original_result"], final_info["result"])
        self.assertEqual(
            loaded["metadata"]["original_final_turn"],
            int(final_info["turn_reached"]),
        )

    def test_restore_observation_mask_and_full_snapshot(self):
        original_observation, original_mask, _ = self.reach_turn_20()
        snapshot = capture_state(self.env)

        # Reset deliberately destroys the saved position.
        self.env.reset()
        restore_state(self.env, snapshot)

        restored_observation, restored_mask = self.current_decision()
        np.testing.assert_array_equal(restored_observation, original_observation)
        np.testing.assert_array_equal(restored_mask, original_mask)

        # Check the *entire serialized runtime state*, not only 967 inputs.
        # The RNG is part of the snapshot and should match too.
        restored_snapshot = capture_state(self.env)
        self.assertEqual(restored_snapshot, snapshot)

    def test_restored_game_can_finish(self):
        observation, mask, _ = self.reach_turn_20()
        snapshot = capture_state(self.env)
        path = Path(self.temp_dir.name) / "resume.json"
        save_snapshot(snapshot, path)

        self.env.reset()
        metadata = load_game(self.env, path)
        self.assertEqual(metadata, {})
        np.testing.assert_array_equal(self.current_decision()[0], observation)
        np.testing.assert_array_equal(self.legal_mask(), mask)

        final_info, decisions = self.play_to_end(
            observation, {"action_mask": mask}
        )
        self.assertGreater(decisions, 0)
        self.assertGreaterEqual(int(final_info["turn_reached"]), 20)
        self.assertLessEqual(int(final_info["turn_reached"]), 48)

    def test_same_saved_state_can_be_restored_repeatedly(self):
        observation, mask, _ = self.reach_turn_20()
        snapshot = capture_state(self.env)
        for _ in range(3):
            self.env.reset()
            restore_state(self.env, snapshot)
            np.testing.assert_array_equal(self.current_decision()[0], observation)
            np.testing.assert_array_equal(self.legal_mask(), mask)
            self.assertEqual(capture_state(self.env), snapshot)

    def test_convenience_save_game(self):
        observation, mask, _ = self.reach_turn_20()
        path = Path(self.temp_dir.name) / "direct.json"
        save_game(self.env, path, metadata={"state_id": "direct"})
        self.assertTrue(path.exists())
        self.env.reset()
        self.assertEqual(load_game(self.env, path)["state_id"], "direct")
        np.testing.assert_array_equal(self.current_decision()[0], observation)
        np.testing.assert_array_equal(self.legal_mask(), mask)

    def test_rejects_incompatible_snapshot_version(self):
        observation, mask, _ = self.reach_turn_20()
        snapshot = capture_state(self.env)
        snapshot["version"] = VERSION + 1
        with self.assertRaises(ValueError):
            restore_state(self.env, snapshot)
        path = Path(self.temp_dir.name) / "bad_version.json"
        path.write_text(json.dumps(snapshot), encoding="utf-8")
        with self.assertRaises(ValueError):
            load_snapshot(path)


if __name__ == "__main__":
    unittest.main()

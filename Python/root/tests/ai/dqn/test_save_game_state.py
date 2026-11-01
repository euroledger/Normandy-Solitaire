\
import unittest
import numpy as np

from core.ai.dqn.normandy_env import NormandyEnv
from core.global_game_state import GlobalGameState
from core.ai.dqn.save_game_state import (
    save_game,
    load_saved_observation,
)


class TestSaveGameState(unittest.TestCase):

    def setUp(self):
        self.env = NormandyEnv()
        self.rng = np.random.default_rng(12345)
        self.game_id = "unit_test_001"

    def tearDown(self):
        self.env.close()

    def test_save_turn_20_and_final_result(self):
        """
        Play a complete game.

        Capture the observation at turn 20.
        Continue playing until the game ends.
        Save the turn-20 state together with the final turn.
        Verify the saved data.
        """

        observation, info = self.env.reset()

        saved_state = None
        saved_mask = None
        decisions = 0

        while True:

            current_turn = int(
                GlobalGameState.cards_drawn
            )

            # Capture the first decision at turn 20.
            if current_turn == 20 and saved_state is None:
                saved_state = observation.copy()
                saved_mask = info["action_mask"].copy()

                print("\nCaptured game state at turn 20")

            # Select a legal random action.
            legal_actions = np.flatnonzero(
                info["action_mask"]
            )

            self.assertGreater(
                len(legal_actions), 0,
                "No legal actions available"
            )

            action = int(
                self.rng.choice(legal_actions)
            )

            observation, reward, terminated, truncated, info = (
                self.env.step(action)
            )

            decisions += 1

            self.assertLess(
                decisions, 10000,
                "Game exceeded 10000 decisions"
            )

            if terminated or truncated:
                break

        final_turn = int(info["turn_reached"])
        result = info["result"]

        print(f"\nGame finished at turn {final_turn}")
        print(f"Result: {result}")
        print(f"Decisions: {decisions}")

        # A game that ends before turn 20 cannot
        # provide a turn-20 state.
        if saved_state is None:
            self.skipTest(
                f"Game ended at turn {final_turn}, "
                "before reaching turn 20"
            )

        # Save the captured state and final result.
        filename = save_game(
            observation=saved_state,
            turn=20,
            game_id=self.game_id,
            action_mask=saved_mask,
            original_final_turn=final_turn,
        )

        print(f"Saved game state: {filename}")

        # Reload the file and verify its contents.
        saved = load_saved_observation(filename)

        np.testing.assert_array_equal(
            saved_state,
            saved["state"],
        )

        np.testing.assert_array_equal(
            saved_mask,
            saved["action_mask"],
        )

        self.assertEqual(
            int(saved["turn"]), 20
        )

        self.assertEqual(
            int(saved["original_final_turn"]),
            final_turn,
        )

        self.assertEqual(
            saved["state"].shape,
            (967,),
        )

        self.assertEqual(
            saved["action_mask"].shape,
            (1032,),
        )

        print("All save-game-state checks passed.")


if __name__ == "__main__":
    unittest.main()

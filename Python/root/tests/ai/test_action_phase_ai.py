import unittest
from unittest.mock import patch

import numpy as np

from core.global_game_state import GlobalGameState
from game_phases.action_phase import action_phase_ai


class TestActionPhaseAI(unittest.TestCase):

    def setUp(self):
        GlobalGameState.actions_left_this_turn = 0
        GlobalGameState.reserve_actions = 1
        GlobalGameState.headless = True

    @patch("game_phases.action_phase.common_post_action_phase")
    @patch("game_phases.action_phase.common_pre_action_phase")
    @patch("game_phases.action_phase.get_action_phase_action_mask")
    @patch("game_phases.action_phase.np.random.choice")
    def test_ai_can_pass_and_preserve_reserve_action(
        self,
        mock_choice,
        mock_mask,
        mock_pre,
        mock_post,
    ):
        mock_pre.return_value = True

        # PASS and another action are both available.
        mask = np.zeros(589, dtype=np.float32)
        mask[0] = 1.0
        mask[559] = 1.0
        mock_mask.return_value = mask

        # Force the AI to choose PASS.
        mock_choice.return_value = 0

        action_phase_ai()

        # Reserve AP must survive the end of the Action Phase.
        self.assertEqual(GlobalGameState.actions_left_this_turn, 0)
        self.assertEqual(GlobalGameState.reserve_actions, 1)

        # The AI really made a choice rather than ending because
        # there were no available actions.
        mock_choice.assert_called_once()

        # Normal Action Phase cleanup must still occur.
        mock_post.assert_called_once()


if __name__ == "__main__":
    unittest.main()

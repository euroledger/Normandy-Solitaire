from unittest.mock import patch
import unittest

import numpy as np

from core.ai.normandy_full_game_ai_dqn import advance_to_next_decision
from core.ai.dqn.dqn_game_state_encoder import OBSERVATION_SIZE
from core.ai.headless_action_masks import get_total_actions
from core.global_game_state import GlobalGameState
from reset import reset_game


class TestAdvanceToNextDecision(unittest.TestCase):

    def setUp(self):
        self.draw_deck, self.mid_deck, self.late_deck = reset_game()
        self.total_actions, _ = get_total_actions()

    def test_advances_to_first_action_phase_decision(self):
        decision = advance_to_next_decision(
            self.draw_deck,
            self.mid_deck,
            self.late_deck,
        )

        self.assertEqual(decision["decision_type"], "action_phase")
        self.assertEqual(GlobalGameState.current_step, 6)

        observation = decision["observation"]
        action_mask = decision["action_mask"]

        self.assertEqual(observation.shape, (OBSERVATION_SIZE,))
        self.assertEqual(observation.dtype, np.float32)

        self.assertEqual(len(action_mask), self.total_actions)
        self.assertTrue(np.any(action_mask == 1.0))
        

    def test_action_phase_initialization(self):
        decision = advance_to_next_decision(
            self.draw_deck,
            self.mid_deck,
            self.late_deck,
        )

        self.assertEqual(decision["decision_type"], "action_phase")
        self.assertEqual(GlobalGameState.current_step, 6)

        expected_actions = GlobalGameState.current_card.actions.actions_available

        for effect in GlobalGameState.current_card.actions.conditional_actions:
            if effect.condition is not None and effect.condition.is_met(GlobalGameState):
                expected_actions += effect.value

        self.assertEqual(GlobalGameState.actions_left_this_turn, expected_actions)


    def test_advances_after_action_phase(self):
        # Reach the first Action Phase
        decision = advance_to_next_decision(
            self.draw_deck,
            self.mid_deck,
            self.late_deck,
        )

        self.assertEqual(decision["decision_type"], "action_phase")

        first_card = GlobalGameState.current_card        
        first_cards_drawn = len(GlobalGameState.drawn_cards)

        # Finish the Action Phase
        GlobalGameState.actions_left_this_turn = 0
        GlobalGameState.reserve_actions = 0

        # Resume automatic game processing
        decision = advance_to_next_decision(
            self.draw_deck,
            self.mid_deck,
            self.late_deck,
        )

        # Verify that a new turn has started
        self.assertEqual(decision["decision_type"], "action_phase")
        self.assertEqual(GlobalGameState.current_step, 6)
        self.assertIsNot(GlobalGameState.current_card, first_card)
        self.assertEqual(
            len(GlobalGameState.drawn_cards),
            first_cards_drawn + 1,
        )


    def test_returns_hitler_intervention_decision(self):
        # Simulate Hitler Intervention during pre-action processing
        with patch(
            "core.ai.normandy_full_game_ai_dqn.common_pre_action_phase",
            return_value=False,
        ), patch(
            "core.ai.normandy_full_game_ai_dqn.get_hitler_intervention_decision",
            return_value=(
                np.zeros(OBSERVATION_SIZE, dtype=np.float32),
                np.array([1.0, 1.0], dtype=np.float32),
            ),
        ):
            decision = advance_to_next_decision(
                self.draw_deck,
                self.mid_deck,
                self.late_deck,
            )

        self.assertEqual(
            decision["decision_type"],
            "hitler_intervention",
        )
        self.assertEqual(
            decision["observation"].shape,
            (OBSERVATION_SIZE,),
        )
        np.testing.assert_array_equal(
            decision["action_mask"],
            [1.0, 1.0],
        )

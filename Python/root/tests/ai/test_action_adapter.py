import unittest
import numpy as np

from core.global_game_state import GlobalGameState
from core.map.map_utilities import do_opening_setup
from core.map.map_model import strategic_reserve_box, hitler_approval_track
from core.map.map_spaces_us_1 import carentan
from core.german_units import PZ_LEHR, panzer_divisions_list
from core.ai.headless_action_masks import (
    HeadlessDecisionState,
    get_total_actions,
)
from core.ai.dqn.dqn_game_state_encoder import (
    OBSERVATION_SIZE,
    DECISION_ACTION_PHASE_INDEX,
    DECISION_CASUALTY_INDEX,
    DECISION_HITLER_INTERVENTION_INDEX,
)
from core.ai.dqn.dqn_action_adapter import (
    is_legal_action,
    get_legal_actions,
    get_action_phase_decision,
    execute_action_phase_decision,
    get_casualty_decision,
    execute_casualty_decision,
    get_hitler_intervention_decision,
)
from tests.core_mechanics.testing_utilities import reset_game_state_for_tests


class TestActionAdapter(unittest.TestCase):

    def setUp(self):
        do_opening_setup()
        GlobalGameState.headless = True
        HeadlessDecisionState.reset_hitler_intervention()

    def tearDown(self):
        HeadlessDecisionState.reset_hitler_intervention()
        reset_game_state_for_tests()

    def test_is_legal_action(self):
        mask = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32)

        self.assertFalse(is_legal_action(-1, mask))
        self.assertFalse(is_legal_action(0, mask))
        self.assertTrue(is_legal_action(1, mask))
        self.assertTrue(is_legal_action(3, mask))
        self.assertFalse(is_legal_action(4, mask))

    def test_get_legal_actions(self):
        mask = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32)

        legal_actions = get_legal_actions(mask)

        np.testing.assert_array_equal(legal_actions, np.array([1, 3]))

    def test_get_action_phase_decision(self):
        GlobalGameState.actions_left_this_turn = 1
        GlobalGameState.reserve_actions = 0

        observation, mask = get_action_phase_decision()

        self.assertEqual(observation.shape, (OBSERVATION_SIZE,))
        self.assertEqual(observation[DECISION_ACTION_PHASE_INDEX], 1.0)
        self.assertGreater(len(mask), 0)
        self.assertGreater(np.count_nonzero(mask), 0)

    def test_execute_action_phase_decision(self):
        GlobalGameState.actions_left_this_turn = 1
        GlobalGameState.reserve_actions = 0
        strategic_reserve_box.units.clear()
        strategic_reserve_box.units.append(PZ_LEHR)
        hitler_approval_track.value = 6

        _, german_eligible_spaces = get_total_actions()
        action_id = (
            7
            + panzer_divisions_list.index(PZ_LEHR) * len(german_eligible_spaces)
            + german_eligible_spaces.index(carentan)
        )

        self.assertIn(PZ_LEHR, strategic_reserve_box.units)
        self.assertNotIn(PZ_LEHR, carentan.units)

        result = execute_action_phase_decision(action_id)

        self.assertTrue(result)
        self.assertNotIn(PZ_LEHR, strategic_reserve_box.units)
        self.assertIn(PZ_LEHR, carentan.units)

    def test_execute_action_phase_decision_rejects_illegal_action(self):
        GlobalGameState.actions_left_this_turn = 0
        GlobalGameState.reserve_actions = 0

        result = execute_action_phase_decision(1)

        self.assertFalse(result)

    def test_get_casualty_decision(self):
        german_units = [PZ_LEHR]

        observation, mask = get_casualty_decision(german_units)

        self.assertEqual(observation.shape, (OBSERVATION_SIZE,))
        self.assertEqual(observation[DECISION_CASUALTY_INDEX], 1.0)
        self.assertGreater(np.count_nonzero(mask), 0)

    def test_execute_casualty_decision(self):
        german_units = [PZ_LEHR]

        _, mask = get_casualty_decision(german_units)
        action_id = get_legal_actions(mask)[0]

        casualty = execute_casualty_decision(action_id, german_units)

        self.assertIs(casualty, PZ_LEHR)

    def test_execute_casualty_decision_rejects_illegal_action(self):
        german_units = [PZ_LEHR]

        _, mask = get_casualty_decision(german_units)
        illegal_actions = np.where(mask == 0.0)[0]

        self.assertGreater(len(illegal_actions), 0)

        casualty = execute_casualty_decision(
            illegal_actions[0],
            german_units,
        )

        self.assertIsNone(casualty)

    def test_get_hitler_intervention_decision(self):
        from cards.card_32 import card as card_032

        observation, mask = get_hitler_intervention_decision(card_032)

        self.assertEqual(observation.shape, (OBSERVATION_SIZE,))
        self.assertEqual(
            observation[DECISION_HITLER_INTERVENTION_INDEX],
            1.0,
        )
        self.assertGreater(np.count_nonzero(mask), 0)


if __name__ == "__main__":
    unittest.main()

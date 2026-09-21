import unittest
from unittest.mock import patch

from core.ai.headless_action_masks import (
    execute_flat_action_ai,
    get_mini_action_mask,
    get_total_actions,
)
from core.german_units import (
    PZ_LEHR,
    SS_12,
    SS_1,
    panzer_divisions_list,
)
from core.global_game_state import GlobalGameState
from core.map.map_model import (
    strategic_reserve_box,
    transport_track,
)
from core.map.map_spaces_can_1 import caen
from core.map.map_spaces_us_1 import carentan
from core.map.map_utilities import do_opening_setup
from tests.core_mechanics.testing_utilities import (
    reset_game_state_for_tests,
)


class TestMoveToStrategicReserveActionMask(unittest.TestCase):

    def setUp(self):
        do_opening_setup()

        GlobalGameState.headless = True
        GlobalGameState.actions_left_this_turn = 1
        GlobalGameState.reserve_actions = 0

        carentan.units.append(SS_12)
        caen.units.append(PZ_LEHR)

        _, self.german_eligible_spaces = get_total_actions()

        self.to_reserve_start = (
            7
            + len(panzer_divisions_list)
            * len(self.german_eligible_spaces)
        )

    def tearDown(self):
        reset_game_state_for_tests()

    def get_action_id(self, panzer):
        return (
            self.to_reserve_start
            + panzer_divisions_list.index(panzer)
        )

    def test_mask_contains_panzer_on_map(self):
        mask = get_mini_action_mask()

        self.assertEqual(
            mask[self.get_action_id(SS_12)],
            1.0
        )

        self.assertEqual(
            mask[self.get_action_id(PZ_LEHR)],
            1.0
        )

    def test_mask_excludes_panzer_not_on_map(self):
        strategic_reserve_box.units.append(SS_1)

        mask = get_mini_action_mask()

        self.assertEqual(
            mask[self.get_action_id(SS_1)],
            0.0
        )

    def test_execute_move_to_strategic_reserve(self):
        transport_track.value = 6

        action_id = self.get_action_id(SS_12)

        with patch(
            "core.ai.headless_action_masks.randint",
            return_value=6
        ):
            result = execute_flat_action_ai(action_id)

        self.assertTrue(result)
        self.assertNotIn(SS_12, carentan.units)
        self.assertIn(SS_12, strategic_reserve_box.units)
        self.assertEqual(
            GlobalGameState.actions_left_this_turn,
            0
        )

    def test_failed_transport_check_spends_action_but_does_not_move_panzer(
        self
    ):
        transport_track.value = 1

        action_id = self.get_action_id(SS_12)

        with patch(
            "core.ai.headless_action_masks.randint",
            return_value=6
        ):
            result = execute_flat_action_ai(action_id)

        self.assertTrue(result)
        self.assertIn(SS_12, carentan.units)
        self.assertNotIn(
            SS_12,
            strategic_reserve_box.units
        )
        self.assertEqual(
            GlobalGameState.actions_left_this_turn,
            0
        )


if __name__ == "__main__":
    unittest.main()

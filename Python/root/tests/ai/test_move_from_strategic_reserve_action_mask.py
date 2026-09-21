import unittest

from core.ai.headless_action_masks import (
    execute_flat_action_ai,
    get_mini_action_mask,
    get_total_actions,
    print_mini_action_mask,
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
    hitler_approval_track,
)
from core.map.map_spaces_can_1 import caen
from core.map.map_spaces_us_1 import carentan
from core.map.map_utilities import do_opening_setup
from tests.core_mechanics.testing_utilities import reset_game_state_for_tests


class TestMoveFromStrategicReserveActionMask(unittest.TestCase):

    def setUp(self):
        do_opening_setup()
        GlobalGameState.headless = True
        GlobalGameState.actions_left_this_turn = 1
        GlobalGameState.reserve_actions = 0

        strategic_reserve_box.units.clear()
        strategic_reserve_box.units.extend([PZ_LEHR, SS_12])

        self.total_actions, self.german_eligible_spaces = get_total_actions()
        self.space_count = len(self.german_eligible_spaces)

    def tearDown(self):
        reset_game_state_for_tests()

    def get_action_id(self, panzer, space):
        return (
            7
            + panzer_divisions_list.index(panzer) * self.space_count
            + self.german_eligible_spaces.index(space)
        )

    def test_mask_contains_panzer_lehr_deployment(self):
        mask = get_mini_action_mask()
        action_id = self.get_action_id(PZ_LEHR, carentan)

        print_mini_action_mask(mask, legal_only=True)

        self.assertEqual(mask[action_id], 1.0)

    def test_mask_contains_12th_ss_deployment(self):
        mask = get_mini_action_mask()
        action_id = self.get_action_id(SS_12, caen)

        self.assertEqual(mask[action_id], 1.0)

    def test_mask_excludes_panzer_not_in_strategic_reserve(self):
        mask = get_mini_action_mask()
        action_id = self.get_action_id(SS_1, caen)

        self.assertEqual(mask[action_id], 0.0)

    def test_execute_panzer_lehr_deployment(self):
        hitler_approval_track.value = 6

        action_id = self.get_action_id(PZ_LEHR, carentan)

        self.assertIn(PZ_LEHR, strategic_reserve_box.units)
        self.assertNotIn(PZ_LEHR, carentan.units)

        result = execute_flat_action_ai(action_id)

        self.assertTrue(result)
        self.assertNotIn(PZ_LEHR, strategic_reserve_box.units)
        self.assertIn(PZ_LEHR, carentan.units)
        self.assertEqual(GlobalGameState.actions_left_this_turn, 0)

    def test_execute_12th_ss_deployment(self):
        hitler_approval_track.value = 6

        action_id = self.get_action_id(SS_12, caen)

        result = execute_flat_action_ai(action_id)

        self.assertTrue(result)
        self.assertNotIn(SS_12, strategic_reserve_box.units)
        self.assertIn(SS_12, caen.units)
        self.assertIn(PZ_LEHR, strategic_reserve_box.units)

    def test_failed_hitler_approval_spends_action_but_does_not_move_panzer(self):
        hitler_approval_track.value = 1

        action_id = self.get_action_id(PZ_LEHR, carentan)

        with unittest.mock.patch(
            "core.ai.headless_action_masks.randint",
            return_value=6
        ):
            result = execute_flat_action_ai(action_id)

        self.assertTrue(result)
        self.assertIn(PZ_LEHR, strategic_reserve_box.units)
        self.assertNotIn(PZ_LEHR, carentan.units)
        self.assertEqual(GlobalGameState.actions_left_this_turn, 0)

    def test_only_panzers_in_strategic_reserve_have_deployment_actions(self):
        mask = get_mini_action_mask()

        for panzer in panzer_divisions_list:
            start = (
                7
                + panzer_divisions_list.index(panzer) * self.space_count
            )
            end = start + self.space_count

            if panzer in [PZ_LEHR, SS_12]:
                self.assertGreater(
                    mask[start:end].sum(),
                    0.0,
                    f"{panzer.name} should have deployment actions"
                )
            else:
                self.assertEqual(
                    mask[start:end].sum(),
                    0.0,
                    f"{panzer.name} should have no deployment actions"
                )


if __name__ == "__main__":
    unittest.main()

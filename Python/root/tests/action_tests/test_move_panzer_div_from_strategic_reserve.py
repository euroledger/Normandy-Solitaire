import unittest

from core.actions.actions_helper import get_german_controlled_spaces
from core.actions.strategic_reserve_actions import do_move_panzer_from_strategic_reserve
from core.german_units import PZ_LEHR, SS_12
from core.global_game_state import GlobalGameState
from core.map.map_model import (
    hitler_approval_track,
    strategic_reserve_box,
)
from core.map.map_spaces_can_1 import caen
from core.map.map_spaces_us_1 import carentan
from core.map.map_utilities import do_opening_setup
from tests.core_mechanics.testing_utilities import reset_game_state_for_tests
from core.actions.strategic_reserve_actions import check_stacking
from core.german_units import PZ_LEHR, SS_12, PZ_21, create_kampfgruppe


class TestMovePanzerFromStrategicReserve(unittest.TestCase):
    def setUp(self):
        do_opening_setup()
        GlobalGameState.actions_left_this_turn = 1
        strategic_reserve_box.units.append(SS_12)

    def tearDown(self):
        reset_game_state_for_tests()

    def test_move_succeeds_when_hitler_approval_check_passes(self):
        hitler_approval_track.value = 6

        destination = get_german_controlled_spaces()[0]

        self.assertIn(SS_12, strategic_reserve_box.units)
        self.assertNotIn(SS_12, destination.units)

        do_move_panzer_from_strategic_reserve(
            die_roll=6,
            div_choice=SS_12,
            space_choice=destination,
        )
        self.assertNotIn(SS_12, strategic_reserve_box.units)
        self.assertIn(SS_12, destination.units)
        self.assertEqual(GlobalGameState.actions_left_this_turn, 0)

    def test_move_fails_when_hitler_approval_check_fails(self):
        hitler_approval_track.value = 2

        destination = get_german_controlled_spaces()[0]

        self.assertIn(SS_12, strategic_reserve_box.units)
        self.assertNotIn(SS_12, destination.units)

        do_move_panzer_from_strategic_reserve(
            die_roll=3,
            div_choice=1,
            space_choice=1,
        )

        self.assertIn(SS_12, strategic_reserve_box.units)
        self.assertNotIn(SS_12, destination.units)
        self.assertEqual(GlobalGameState.actions_left_this_turn, 0)

    def test_invalid_panzer_choice_does_nothing(self):
        hitler_approval_track.value = 6

        destination = get_german_controlled_spaces()[0]

        self.assertIn(SS_12, strategic_reserve_box.units)
        self.assertNotIn(SS_12, destination.units)

        do_move_panzer_from_strategic_reserve(
            die_roll=1,
            div_choice=99,
            space_choice=1,
        )

        self.assertIn(SS_12, strategic_reserve_box.units)
        self.assertNotIn(SS_12, destination.units)
        self.assertEqual(GlobalGameState.actions_left_this_turn, 1)

    def test_invalid_destination_choice_does_nothing(self):
        hitler_approval_track.value = 6

        destination = get_german_controlled_spaces()[0]

        self.assertIn(SS_12, strategic_reserve_box.units)
        self.assertNotIn(SS_12, destination.units)

        do_move_panzer_from_strategic_reserve(
            die_roll=1,
            div_choice=1,
            space_choice=99,
        )

        self.assertIn(SS_12, strategic_reserve_box.units)
        self.assertNotIn(SS_12, destination.units)
        self.assertEqual(GlobalGameState.actions_left_this_turn, 1)


    def test_check_stacking_returns_true_when_3_panzer_units(self):
        carentan.units.extend([
            PZ_LEHR,
            SS_12,
            create_kampfgruppe(),
        ])

        self.assertTrue(check_stacking(carentan))


    def test_check_stacking_returns_false_when_4_panzer_units(self):
        carentan.units.extend([
            PZ_LEHR,
            SS_12,
            PZ_21,
            create_kampfgruppe(),
        ])

        self.assertFalse(check_stacking(carentan))
if __name__ == "__main__":
    unittest.main()

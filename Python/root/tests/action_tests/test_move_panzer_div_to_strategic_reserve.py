import unittest

from core.actions.strategic_reserve_actions import do_move_panzer_to_strategic_reserve, get_panzer_divisions_on_map
from core.german_units import PZ_21, PZ_LEHR, SS_12
from core.global_game_state import GlobalGameState
from core.map.map_model import (
    strategic_reserve_box,
    transport_track,
)
from core.map.map_spaces_brit_2 import bayeux
from core.map.map_spaces_can_1 import caen
from core.map.map_spaces_us_1 import carentan
from core.map.map_utilities import do_opening_setup
from tests.core_mechanics.testing_utilities import reset_game_state_for_tests


class TestGetPanzerDivisionsOnMap(unittest.TestCase):
    def setUp(self):
        do_opening_setup()
        carentan.units.append(SS_12)
        caen.units.append(PZ_LEHR)

    def tearDown(self):
        reset_game_state_for_tests()

    def test_get_panzer_divisions_on_map(self):
        panzer_divisions = get_panzer_divisions_on_map()

        self.assertEqual(len(panzer_divisions), 3)

        self.assertIn((carentan, SS_12), panzer_divisions)
        self.assertIn((caen, PZ_LEHR), panzer_divisions)
        self.assertIn((bayeux, PZ_21), panzer_divisions)


class TestMovePanzerToStrategicReserve(unittest.TestCase):
    def setUp(self):
        do_opening_setup()
        GlobalGameState.actions_left_this_turn = 1

        carentan.units.append(SS_12)
        caen.units.append(PZ_LEHR)

    def tearDown(self):
        reset_game_state_for_tests()

    def test_move_succeeds_when_transport_check_passes(self):
        transport_track.value = 6

        self.assertIn(SS_12, carentan.units)
        self.assertIn(PZ_LEHR, caen.units)
        self.assertNotIn(SS_12, strategic_reserve_box.units)

        do_move_panzer_to_strategic_reserve(
            die_roll=6,
            div_choice=SS_12,
        )
        self.assertNotIn(SS_12, carentan.units)
        self.assertIn(SS_12, strategic_reserve_box.units)
        self.assertIn(PZ_LEHR, caen.units)
        self.assertEqual(GlobalGameState.actions_left_this_turn, 0)

    def test_move_fails_when_transport_check_fails(self):
        transport_track.value = 2

        self.assertIn(SS_12, carentan.units)
        self.assertNotIn(SS_12, strategic_reserve_box.units)

        do_move_panzer_to_strategic_reserve(
            die_roll=3,
            div_choice=1,
        )

        self.assertIn(SS_12, carentan.units)
        self.assertNotIn(SS_12, strategic_reserve_box.units)
        self.assertEqual(GlobalGameState.actions_left_this_turn, 0)

    def test_invalid_panzer_choice_does_nothing(self):
        transport_track.value = 6

        self.assertIn(SS_12, carentan.units)
        self.assertIn(PZ_LEHR, caen.units)
        self.assertNotIn(SS_12, strategic_reserve_box.units)
        self.assertNotIn(PZ_LEHR, strategic_reserve_box.units)

        do_move_panzer_to_strategic_reserve(
            die_roll=1,
            div_choice=99,
        )

        self.assertIn(SS_12, carentan.units)
        self.assertIn(PZ_LEHR, caen.units)
        self.assertNotIn(SS_12, strategic_reserve_box.units)
        self.assertNotIn(PZ_LEHR, strategic_reserve_box.units)
        self.assertEqual(GlobalGameState.actions_left_this_turn, 1)


if __name__ == "__main__":
    unittest.main()



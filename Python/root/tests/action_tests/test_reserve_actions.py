import unittest
from unittest.mock import patch
from core.actions.counter_attack_action import do_counter_attack, get_counter_attack_options
from core.actions.move_action_point_to_reserve import do_move_action_point_to_strategic_reserve
from core.actions.move_unit_one_space_action import do_move_unit_one_space
from core.actions.strategic_reserve_actions import do_move_panzer_to_strategic_reserve
from core.german_units import PZ_LEHR, SS_12
from core.global_game_state import GlobalGameState
from core.map.map_spaces_us_1 import cherbourg, valognes
from core.map.map_spaces_can_1 import caen
from core.map.map_utilities import add_units_to_space, do_opening_setup
from tests.core_mechanics.testing_utilities import reset_game_state_for_tests
from core.allied_armies import US_FIRST_ARMY
from core.map.map_utilities import get_eligible_german_units
from core.map.map_spaces_us_1 import carentan, utah_omaha
from cards.card_2 import card as card_002
from core.tables.weather import WEATHER_TABLE
from core.map.map_model import (
    transport_track,
    strategic_reserve_box,
)


class TestReserveActions(unittest.TestCase):
    def setUp(self):
        do_opening_setup()
        cherbourg.units.append(SS_12)
        

    def tearDown(self):
        reset_game_state_for_tests()

    def test_save_action_then_use_normal_action(self):
        GlobalGameState.actions_left_this_turn = 2
        GlobalGameState.reserve_actions = 0

        do_move_action_point_to_strategic_reserve()

        self.assertEqual(GlobalGameState.actions_left_this_turn, 1)
        self.assertEqual(GlobalGameState.reserve_actions, 1)

        do_move_unit_one_space(unit_choice=SS_12, space_choice=valognes)

        self.assertNotIn(SS_12, cherbourg.units)
        self.assertIn(SS_12, valognes.units)
        self.assertEqual(GlobalGameState.actions_left_this_turn, 0)
        self.assertEqual(GlobalGameState.reserve_actions, 1)

    def test_save_action_then_use_reserve_action(self):
        GlobalGameState.actions_left_this_turn = 1
        GlobalGameState.reserve_actions = 0

        do_move_action_point_to_strategic_reserve()

        self.assertEqual(GlobalGameState.actions_left_this_turn, 0)
        self.assertEqual(GlobalGameState.reserve_actions, 1)

        with patch("builtins.input", return_value="Y"):
            do_move_unit_one_space(unit_choice=SS_12, space_choice=valognes)

        self.assertNotIn(SS_12, cherbourg.units)
        self.assertIn(SS_12, valognes.units)
        self.assertEqual(GlobalGameState.actions_left_this_turn, 0)
        self.assertEqual(GlobalGameState.reserve_actions, 0)


    def test_normal_action_only(self):
        GlobalGameState.actions_left_this_turn = 1
        GlobalGameState.reserve_actions = 0

        do_move_unit_one_space(unit_choice=SS_12, space_choice=valognes)

        self.assertEqual(GlobalGameState.actions_left_this_turn, 0)
        self.assertEqual(GlobalGameState.reserve_actions, 0)


    def test_normal_action_with_reserve_action_available(self):
        GlobalGameState.actions_left_this_turn = 1
        GlobalGameState.reserve_actions = 1

        do_move_unit_one_space(unit_choice=SS_12, space_choice=valognes)

        self.assertEqual(GlobalGameState.actions_left_this_turn, 0)
        self.assertEqual(GlobalGameState.reserve_actions, 1)


    def test_reserve_action_only(self):
        GlobalGameState.actions_left_this_turn = 0
        GlobalGameState.reserve_actions = 1

        with patch("builtins.input", return_value="Y"):
            do_move_unit_one_space(unit_choice=SS_12, space_choice=valognes)

        self.assertEqual(GlobalGameState.actions_left_this_turn, 0)
        self.assertEqual(GlobalGameState.reserve_actions, 0)


    def test_two_reserve_actions(self):
        GlobalGameState.actions_left_this_turn = 0
        GlobalGameState.reserve_actions = 2

        with patch("builtins.input", return_value="Y"):
            do_move_unit_one_space(unit_choice=SS_12, space_choice=valognes)

        self.assertEqual(GlobalGameState.actions_left_this_turn, 0)
        self.assertEqual(GlobalGameState.reserve_actions, 1)


    def test_counter_attack_normal_action_with_reserve_available(self):
        # Verifies counter attack spends a normal action first, leaving reserve untouched.
        GlobalGameState.actions_left_this_turn = 1
        GlobalGameState.reserve_actions = 1
        GlobalGameState.cards_drawn = 1
        GlobalGameState.us_1_front_line = utah_omaha.track_number
        add_units_to_space(utah_omaha, US_FIRST_ARMY)
        GlobalGameState.current_card = card_002
        GlobalGameState.current_weather = WEATHER_TABLE[1]

        options = get_counter_attack_options()
        selected_option = next(option for option in options if option["army"] == US_FIRST_ARMY)

        do_counter_attack(selected_option=selected_option, selected_units=get_eligible_german_units(carentan))

        self.assertEqual(GlobalGameState.actions_left_this_turn, 0)
        self.assertEqual(GlobalGameState.reserve_actions, 1)
        

    def test_move_panzer_to_reserve_with_normal_action_and_reserve_available(self):

        GlobalGameState.actions_left_this_turn = 1
        carentan.units.append(SS_12)
        caen.units.append(PZ_LEHR)
        
        GlobalGameState.actions_left_this_turn = 1
        GlobalGameState.reserve_actions = 1
        transport_track.value = 6

        self.assertIn(SS_12, carentan.units)
        self.assertNotIn(SS_12, strategic_reserve_box.units)

        do_move_panzer_to_strategic_reserve(die_roll=6, div_choice=1)

        self.assertNotIn(SS_12, carentan.units)
        self.assertIn(SS_12, strategic_reserve_box.units)
        self.assertEqual(GlobalGameState.actions_left_this_turn, 0)
        self.assertEqual(GlobalGameState.reserve_actions, 1)
if __name__ == "__main__":
    unittest.main()

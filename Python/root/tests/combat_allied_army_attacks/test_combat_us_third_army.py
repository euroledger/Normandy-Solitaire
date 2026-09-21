import unittest

from cards.card_29 import card as card_029

from core.actions.counter_attack_action import do_post_combat, get_counter_attack_options
from core.allied_advances_phase import do_allied_attacks
from core.allied_armies import (
    US_THIRD_ARMY,
    US_VIII_CORPS,
    US_XV_CORPS,
)
from core.enums import SideType
from core.global_game_state import GlobalGameState
from core.map.map_utilities import (
    add_units_to_space,
    do_opening_setup,
)
from core.map.map_spaces_us_3 import (
    us_3_start_box,
    st_malo,
    rennes,
    le_mans,
    lorient
)

from core.tables.weather import WEATHER_TABLE


class TestUsThirdArmyEdgeCases(unittest.TestCase):

    def setUp(self):
        do_opening_setup()

        GlobalGameState.us_third_army_activated = True
        GlobalGameState.us_third_army_merged = False

        US_THIRD_ARMY.merged = False
        US_THIRD_ARMY.location = None
        US_VIII_CORPS.location = None
        US_XV_CORPS.location = None

        for space in [us_3_start_box, st_malo, rennes]:
            for army in [US_THIRD_ARMY, US_VIII_CORPS, US_XV_CORPS]:
                if army in space.units:
                    space.units.remove(army)

        self.weather = WEATHER_TABLE[1]

    def test_viii_and_xv_retreat_to_start_box_without_merging(self):
        add_units_to_space(st_malo, US_VIII_CORPS)
        add_units_to_space(rennes, US_XV_CORPS)

        xv_option = {
            "army": US_XV_CORPS,
            "target_space": rennes,
            "attacking_space": rennes,
        }

        do_post_combat(
            {"result": "WIN"},
            xv_option,
            [],
        )

        self.assertEqual(US_XV_CORPS.location, us_3_start_box)
        self.assertEqual(US_VIII_CORPS.location, st_malo)
        self.assertFalse(US_THIRD_ARMY.merged)

        viii_option = {
            "army": US_VIII_CORPS,
            "target_space": st_malo,
            "attacking_space": st_malo,
        }

        do_post_combat(
            {"result": "WIN"},
            viii_option,
            [],
        )

        self.assertEqual(US_VIII_CORPS.location, us_3_start_box)
        self.assertEqual(US_XV_CORPS.location, us_3_start_box)

        self.assertIn(US_VIII_CORPS, us_3_start_box.units)
        self.assertIn(US_XV_CORPS, us_3_start_box.units)

        self.assertNotIn(US_THIRD_ARMY, us_3_start_box.units)
        self.assertIsNone(US_THIRD_ARMY.location)

        self.assertFalse(US_THIRD_ARMY.merged)
        self.assertFalse(GlobalGameState.us_third_army_merged)

    def test_viii_and_xv_attack_out_of_start_box(self):
        add_units_to_space(us_3_start_box, US_VIII_CORPS)
        add_units_to_space(us_3_start_box, US_XV_CORPS)

        do_allied_attacks(
            [US_VIII_CORPS, US_XV_CORPS],
            card_029,
            self.weather,
            die_roll=6,
        )

        self.assertEqual(US_VIII_CORPS.location, st_malo)
        self.assertEqual(US_XV_CORPS.location, rennes)

        self.assertIn(US_VIII_CORPS, st_malo.units)
        self.assertIn(US_XV_CORPS, rennes.units)

        self.assertNotIn(US_VIII_CORPS, us_3_start_box.units)
        self.assertNotIn(US_XV_CORPS, us_3_start_box.units)

        self.assertFalse(US_THIRD_ARMY.merged)

    def test_merged_third_army_retreats_from_rennes_to_start_box(self):
        US_THIRD_ARMY.merged = True
        GlobalGameState.us_third_army_merged = True

        add_units_to_space(rennes, US_THIRD_ARMY)

        option = {
            "army": US_THIRD_ARMY,
            "target_space": rennes,
            "attacking_space": rennes,
        }

        do_post_combat(
            {"result": "WIN"},
            option,
            [],
        )

        self.assertEqual(
            US_THIRD_ARMY.location,
            us_3_start_box,
        )

        self.assertIn(
            US_THIRD_ARMY,
            us_3_start_box.units,
        )

        self.assertNotIn(
            US_THIRD_ARMY,
            rennes.units,
        )

        self.assertTrue(US_THIRD_ARMY.merged)
        self.assertTrue(
            GlobalGameState.us_third_army_merged
        )

    def test_merged_third_army_attacks_rennes_from_start_box(self):
        US_THIRD_ARMY.merged = True
        GlobalGameState.us_third_army_merged = True

        add_units_to_space(
            us_3_start_box,
            US_THIRD_ARMY,
        )

        do_allied_attacks(
            [US_THIRD_ARMY],
            card_029,
            self.weather,
            die_roll=6,
        )

        self.assertEqual(
            US_THIRD_ARMY.location,
            rennes,
        )

        self.assertIn(
            US_THIRD_ARMY,
            rennes.units,
        )

        self.assertNotIn(
            US_THIRD_ARMY,
            us_3_start_box.units,
        )

        self.assertTrue(US_THIRD_ARMY.merged)
        self.assertTrue(
            GlobalGameState.us_third_army_merged
        )


    def test_viii_advances_without_attack_into_allied_controlled_space(self):
        add_units_to_space(lorient, US_VIII_CORPS)

        lorient.controlling_player = SideType.ALLIED
        rennes.controlling_player = SideType.ALLIED

        GlobalGameState.us_viii_front_line = lorient.track_number

        do_allied_attacks(
            [US_VIII_CORPS],
            card_029,
            self.weather,
            die_roll=1,
        )

        self.assertEqual(US_VIII_CORPS.location, rennes)
        self.assertIn(US_VIII_CORPS, rennes.units)
        self.assertNotIn(US_VIII_CORPS, lorient.units)


    def test_counter_attack_options_exclude_viii_when_xv_is_ahead(self):
        add_units_to_space(rennes, US_VIII_CORPS)
        add_units_to_space(le_mans, US_XV_CORPS)

        rennes.controlling_player = SideType.ALLIED
        le_mans.controlling_player = SideType.ALLIED

        GlobalGameState.us_viii_front_line = rennes.track_number
        GlobalGameState.us_xv_front_line = le_mans.track_number


        GlobalGameState.current_card = card_029
        GlobalGameState.current_weather = self.weather
        options = get_counter_attack_options()

        targeted_armies = [option["army"] for option in options]

        self.assertNotIn(US_VIII_CORPS, targeted_armies)

if __name__ == "__main__":
    unittest.main()

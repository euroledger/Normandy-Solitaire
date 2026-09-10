import unittest

from cards.card_3 import card as card_003
from cards.card_38 import card as card_038



from core.enums import SideType
from core.german_units import PZ_21
from core.tables.weather import WEATHER_TABLE
from core.allied_advances_phase import do_allied_attacks, advance_army_one_space
from core.map.map_utilities import add_units_to_space, do_opening_setup
from core.map.map_spaces_us_1 import utah_omaha, carentan, valognes, cherbourg
from core.map.map_spaces_brit_2 import gold_juno_sword_brit, bayeux, tilly
from core.map.map_spaces_can_1 import gold_juno_sword_can, lebisey_wood
from core.allied_armies import US_FIRST_ARMY, BRITISH_SECOND_ARMY, CANADIAN_FIRST_ARMY
from core.global_game_state import GlobalGameState
from core.models import Strategy
from core.map.map_model import hitler_approval_track
from tests.core_mechanics.testing_utilities import setup_units_for_tests


class TestAlliedAttacks(unittest.TestCase):
    def setUp(self):
        do_opening_setup()

        GlobalGameState.german_casualty_strategy = Strategy.UNIT_TEST

        advance_army_one_space(US_FIRST_ARMY)
        self.weather = WEATHER_TABLE[1]

    def test_us_first_army_attack_carentan_natural_6(self):
        self.assertEqual(US_FIRST_ARMY.location, utah_omaha)
        self.assertEqual(len(carentan.units), 1)

        do_allied_attacks([US_FIRST_ARMY], card_003, self.weather, die_roll=6)

        self.assertEqual(US_FIRST_ARMY.location, carentan)
        self.assertEqual(len(carentan.units), 1)

    def test_us_first_army_attack_carentan_natural_1(self):
        self.assertEqual(carentan.controlling_player, SideType.GERMAN)

        do_allied_attacks([US_FIRST_ARMY], card_003, self.weather, die_roll=1)

        self.assertEqual(US_FIRST_ARMY.location, utah_omaha)
        self.assertEqual(carentan.controlling_player, SideType.GERMAN)

    # Allied Defeat

    def test_us_first_army_attack_carentan_roll_2(self):
        self.assertEqual(US_FIRST_ARMY.location, utah_omaha)
        self.assertEqual(len(carentan.units), 1)

        do_allied_attacks([US_FIRST_ARMY], card_003, self.weather, die_roll=2)

        self.assertEqual(US_FIRST_ARMY.location, utah_omaha)
        self.assertEqual(len(carentan.units), 0)

    # Allied Victory

    def test_us_first_army_attack_carentan_roll_3(self):
        self.assertEqual(carentan.controlling_player, SideType.GERMAN)

        do_allied_attacks([US_FIRST_ARMY], card_003, self.weather, die_roll=3)

        self.assertEqual(US_FIRST_ARMY.location, carentan)
        self.assertEqual(carentan.controlling_player, SideType.ALLIED)

    def test_british_second_army_attack_bayeux_natural_6(self):
        advance_army_one_space(BRITISH_SECOND_ARMY)

        self.assertEqual(bayeux.controlling_player, SideType.GERMAN)

        do_allied_attacks([BRITISH_SECOND_ARMY], card_003, self.weather, die_roll=6)

        self.assertEqual(BRITISH_SECOND_ARMY.location, bayeux)
        self.assertEqual(bayeux.controlling_player, SideType.ALLIED)

    def test_british_second_army_attack_bayeux_natural_1(self):
        advance_army_one_space(BRITISH_SECOND_ARMY)

        self.assertEqual(BRITISH_SECOND_ARMY.location, gold_juno_sword_brit)
        self.assertEqual(len(bayeux.units), 4)

        do_allied_attacks([BRITISH_SECOND_ARMY], card_003, self.weather, die_roll=1)

        self.assertEqual(BRITISH_SECOND_ARMY.location, gold_juno_sword_brit)
        self.assertEqual(len(bayeux.units), 4)

        self.assertEqual(bayeux.units[0].name, "21st Panzer")
        self.assertEqual(bayeux.units[1].name, "Nebelwerfer")
        self.assertEqual(bayeux.units[2].name, "Nebelwerfer")
        self.assertEqual(bayeux.units[3].name, "Flak 88")

        self.assertEqual(bayeux.units[0].combat_value, 1)
        self.assertEqual(bayeux.units[1].combat_value, 1)
        self.assertEqual(bayeux.units[2].combat_value, 1)
        self.assertEqual(bayeux.units[3].combat_value, 2)

    def test_canadian_first_army_attack_lebisey_wood_natural_6(self):
        advance_army_one_space(CANADIAN_FIRST_ARMY)

        self.assertEqual(CANADIAN_FIRST_ARMY.location, gold_juno_sword_can)
        self.assertEqual(len(lebisey_wood.units), 3)

        do_allied_attacks([CANADIAN_FIRST_ARMY], card_003, self.weather, die_roll=6)

        self.assertEqual(CANADIAN_FIRST_ARMY.location, lebisey_wood)
        self.assertEqual(len(lebisey_wood.units), 1)

    def test_canadian_first_army_attack_lebisey_wood_natural_1(self):
        advance_army_one_space(CANADIAN_FIRST_ARMY)

        self.assertEqual(CANADIAN_FIRST_ARMY.location, gold_juno_sword_can)
        self.assertEqual(len(lebisey_wood.units), 3)

        do_allied_attacks([CANADIAN_FIRST_ARMY], card_003, self.weather, die_roll=1)

        self.assertEqual(CANADIAN_FIRST_ARMY.location, gold_juno_sword_can)
        self.assertEqual(len(lebisey_wood.units), 2)

        self.assertEqual(lebisey_wood.units[0].combat_value, 1)
        self.assertEqual(lebisey_wood.units[1].combat_value, 2)

    def test_us_first_army_advance_updates_front_line_space(self):
        add_units_to_space(carentan, US_FIRST_ARMY)

        self.assertEqual(valognes.controlling_player, SideType.GERMAN)

        advance_army_one_space(US_FIRST_ARMY)

        self.assertEqual(US_FIRST_ARMY.location, valognes)
        self.assertEqual(GlobalGameState.us_1_front_line, valognes.track_number)
        self.assertEqual(valognes.controlling_player, SideType.ALLIED)

    def test_us_first_army_attack_carentan_roll_3_no_panzer_no_hitler_approval_loss(self):
        hitler_approval_track.value = 6

        self.assertEqual(carentan.controlling_player, SideType.GERMAN)

        do_allied_attacks([US_FIRST_ARMY], card_003, self.weather, die_roll=3)

        self.assertEqual(US_FIRST_ARMY.location, carentan)
        self.assertEqual(carentan.controlling_player, SideType.ALLIED)
        self.assertEqual(hitler_approval_track.value, 6)

    def test_british_second_army_attack_bayeux_natural_6_panzer_defense_lowers_hitler_approval(self):
        hitler_approval_track.value = 6

        advance_army_one_space(BRITISH_SECOND_ARMY)

        self.assertEqual(bayeux.controlling_player, SideType.GERMAN)

        do_allied_attacks([BRITISH_SECOND_ARMY], card_003, self.weather, die_roll=6)

        self.assertEqual(BRITISH_SECOND_ARMY.location, bayeux)
        self.assertEqual(bayeux.controlling_player, SideType.ALLIED)
        self.assertEqual(hitler_approval_track.value, 5)

    def test_canadian_first_army_attack_lebisey_wood_natural_6_no_panzer_no_hitler_approval_loss(self):
        hitler_approval_track.value = 6

        advance_army_one_space(CANADIAN_FIRST_ARMY)

        do_allied_attacks([CANADIAN_FIRST_ARMY], card_003, self.weather, die_roll=6)

        self.assertEqual(CANADIAN_FIRST_ARMY.location, lebisey_wood)
        self.assertEqual(hitler_approval_track.value, 6)

    def test_british_second_army_attack_bayeux_natural_1_panzer_present_but_no_german_loss_no_hitler_approval_loss(self):
        hitler_approval_track.value = 6

        advance_army_one_space(BRITISH_SECOND_ARMY)

        do_allied_attacks([BRITISH_SECOND_ARMY], card_003, self.weather, die_roll=1)

        self.assertEqual(BRITISH_SECOND_ARMY.location, gold_juno_sword_brit)
        self.assertEqual(hitler_approval_track.value, 6)

    def test_us_first_army_captures_cherbourg_lowers_hitler_approval_once(self):
        GlobalGameState.cherbourg_captured = False

        # Move US First Army to Valognes
        advance_army_one_space(US_FIRST_ARMY)
        advance_army_one_space(US_FIRST_ARMY)


        self.assertEqual(US_FIRST_ARMY.location, valognes)
        self.assertEqual(cherbourg.controlling_player, SideType.GERMAN)

        hitler_approval_track.value = 6

        # First capture of Cherbourg
        do_allied_attacks([US_FIRST_ARMY], card_003, self.weather, die_roll=3)

        self.assertEqual(US_FIRST_ARMY.location, cherbourg)
        self.assertEqual(cherbourg.controlling_player, SideType.ALLIED)
        self.assertEqual(hitler_approval_track.value, 5)
        self.assertTrue(GlobalGameState.cherbourg_captured)

        # Reset the game state
        do_opening_setup()
        GlobalGameState.cherbourg_captured = True
        hitler_approval_track.value = 5
        cherbourg.controlling_player=SideType.GERMAN

        # GlobalGameState.cherbourg_captured = True
        hitler_approval_track.value = 5

        advance_army_one_space(US_FIRST_ARMY)
        advance_army_one_space(US_FIRST_ARMY)
        advance_army_one_space(US_FIRST_ARMY)

        self.assertEqual(US_FIRST_ARMY.location, valognes)
        self.assertEqual(cherbourg.controlling_player, SideType.GERMAN)

        # Capture Cherbourg again
        do_allied_attacks([US_FIRST_ARMY], card_003, self.weather, die_roll=3)

        self.assertEqual(US_FIRST_ARMY.location, cherbourg)
        self.assertEqual(cherbourg.controlling_player, SideType.ALLIED)

        # Hitler Approval should not be reduced a second time
        self.assertEqual(hitler_approval_track.value, 5)
        self.assertTrue(GlobalGameState.cherbourg_captured)
        

    def test_fortified_villages_satisfies_german_step_loss_on_capture(self):
        setup_units_for_tests()
        GlobalGameState.german_casualty_strategy = Strategy.RANDOM
        self.weather = WEATHER_TABLE[1]

        add_units_to_space(bayeux, BRITISH_SECOND_ARMY)
        add_units_to_space(tilly, PZ_21)
        tilly.fortified_village_modifier = 1

        original_combat_value = PZ_21.combat_value

        do_allied_attacks([BRITISH_SECOND_ARMY], card_038, self.weather, die_roll=6)

        self.assertEqual(PZ_21.combat_value, original_combat_value)
        self.assertEqual(tilly.fortified_village_modifier, 0)
        self.assertEqual(BRITISH_SECOND_ARMY.location, tilly)

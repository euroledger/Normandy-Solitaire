import unittest
from unittest.mock import patch

from cards.card_46 import card as card_046
from core.actions.hitler_intervention import check_hitler_intervention_applies, do_hitler_intervention_redeploy, get_german_space_facing_front_line, get_hitler_intervention_targets
from core.allied_armies import US_THIRD_ARMY, US_VIII_CORPS, US_XV_CORPS
from core.global_game_state import GlobalGameState
from core.german_units import PZ_21, PZ_9, PZ_LEHR
from core.map.map_model import hitler_approval_track, strategic_reserve_box
from core.map.map_spaces_us_3 import brest, rennes, st_malo, le_mans
from core.map.map_spaces_brit_2 import bayeux
from core.map.map_spaces_can_1 import caen
from core.map.map_utilities import add_units_to_space, do_opening_setup, get_all_map_spaces, reset_map
from core.models import GermanUnit


class TestHitlerInterventionTargets(unittest.TestCase):
    def setUp(self):
        do_opening_setup()
        GlobalGameState.us_third_army_activated = False
        GlobalGameState.hitler_intervention_no_effect = False

    def tearDown(self):
        reset_map()
        do_opening_setup()
        GlobalGameState.us_third_army_activated = False
        GlobalGameState.hitler_intervention_no_effect = False

    def test_card_46_no_targets_if_us_third_army_not_activated(self):
        targets = get_hitler_intervention_targets(card_046)
        self.assertEqual(targets, [])

    def test_card_46_uses_third_army_when_neither_corps_is_on_map(self):
        GlobalGameState.us_third_army_activated = True
        US_VIII_CORPS.location = None
        US_XV_CORPS.location = None
        add_units_to_space(le_mans, US_THIRD_ARMY)

        targets = get_hitler_intervention_targets(card_046)

        self.assertEqual(targets, [US_THIRD_ARMY])

    def test_card_46_uses_viii_corps_when_it_is_on_map(self):
        GlobalGameState.us_third_army_activated = True
        add_units_to_space(st_malo, US_VIII_CORPS)
        US_XV_CORPS.location = None
        add_units_to_space(rennes, US_THIRD_ARMY)

        targets = get_hitler_intervention_targets(card_046)

        self.assertEqual(targets, [US_VIII_CORPS])

    def test_card_46_uses_xv_corps_when_it_is_on_map(self):
        GlobalGameState.us_third_army_activated = True
        US_VIII_CORPS.location = None
        add_units_to_space(rennes, US_XV_CORPS)
        add_units_to_space(st_malo, US_THIRD_ARMY)

        targets = get_hitler_intervention_targets(card_046)

        self.assertEqual(targets, [US_XV_CORPS])

    def test_card_46_returns_both_corps_when_both_are_on_map(self):
        GlobalGameState.us_third_army_activated = True
        add_units_to_space(st_malo, US_VIII_CORPS)
        add_units_to_space(rennes, US_XV_CORPS)
        add_units_to_space(st_malo, US_THIRD_ARMY)

        targets = get_hitler_intervention_targets(card_046)

        self.assertEqual(targets, [US_VIII_CORPS, US_XV_CORPS])
        self.assertNotIn(US_THIRD_ARMY, targets)

    def test_card_46_prefers_corps_over_third_army_when_all_three_are_on_map(self):
        GlobalGameState.us_third_army_activated = True
        add_units_to_space(st_malo, US_VIII_CORPS)
        add_units_to_space(rennes, US_XV_CORPS)
        add_units_to_space(st_malo, US_THIRD_ARMY)

        targets = get_hitler_intervention_targets(card_046)

        self.assertEqual(targets, [US_VIII_CORPS, US_XV_CORPS])
        self.assertNotIn(US_THIRD_ARMY, targets)

    def test_card_46_prefers_viii_corps_over_third_army_when_both_are_on_map(self):
        GlobalGameState.us_third_army_activated = True
        add_units_to_space(st_malo, US_VIII_CORPS)
        US_XV_CORPS.location = None
        add_units_to_space(rennes, US_THIRD_ARMY)

        targets = get_hitler_intervention_targets(card_046)

        self.assertEqual(targets, [US_VIII_CORPS])
        self.assertNotIn(US_THIRD_ARMY, targets)

    def test_card_46_prefers_xv_corps_over_third_army_when_both_are_on_map(self):
        GlobalGameState.us_third_army_activated = True
        US_VIII_CORPS.location = None
        add_units_to_space(rennes, US_XV_CORPS)
        add_units_to_space(st_malo, US_THIRD_ARMY)

        targets = get_hitler_intervention_targets(card_046)

        self.assertEqual(targets, [US_XV_CORPS])
        self.assertNotIn(US_THIRD_ARMY, targets)

    def test_card_46_excludes_corps_besieging_fortress_but_does_not_fall_back_to_third_army(self):
        GlobalGameState.us_third_army_activated = True
        add_units_to_space(st_malo, US_VIII_CORPS)
        add_units_to_space(rennes, US_XV_CORPS)
        add_units_to_space(st_malo, US_THIRD_ARMY)
        brest.under_siege = True

        targets = get_hitler_intervention_targets(card_046)

        self.assertEqual(targets, [US_XV_CORPS])
        self.assertNotIn(US_THIRD_ARMY, targets)

    def test_card_46_does_not_use_third_army_when_corps_are_on_map(self):
        GlobalGameState.us_third_army_activated = True
        add_units_to_space(st_malo, US_VIII_CORPS)
        add_units_to_space(rennes, US_XV_CORPS)
        add_units_to_space(st_malo, US_THIRD_ARMY)

        targets = get_hitler_intervention_targets(card_046)

        self.assertNotIn(US_THIRD_ARMY, targets)
        self.assertEqual(len(targets), 2)


class TestCheckHitlerInterventionApplies(unittest.TestCase):
    def setUp(self):
        do_opening_setup()
        GlobalGameState.us_third_army_activated = False
        GlobalGameState.hitler_intervention_no_effect = False
        hitler_approval_track.value = 5

    def tearDown(self):
        reset_map()
        do_opening_setup()
        GlobalGameState.us_third_army_activated = False
        GlobalGameState.hitler_intervention_no_effect = False
        hitler_approval_track.value = 6

    def test_no_effect_when_there_are_no_eligible_targets(self):
        result = check_hitler_intervention_applies(card_046, die_roll=1)
        self.assertIsNone(result)
        self.assertTrue(GlobalGameState.hitler_intervention_no_effect)

    @patch("builtins.input", return_value="N")
    def test_intervention_returns_single_us_third_army_when_neither_corps_is_on_map(self, mock_input):
        GlobalGameState.us_third_army_activated = True
        US_VIII_CORPS.location = None
        US_XV_CORPS.location = None
        add_units_to_space(le_mans, US_THIRD_ARMY)

        result = check_hitler_intervention_applies(card_046, die_roll=1)

        self.assertEqual(result, US_THIRD_ARMY)
        self.assertFalse(GlobalGameState.hitler_intervention_no_effect)
        mock_input.assert_called_once_with("ATTEMPT TO CANCEL HITLER INTERVENTION? (Y/N): ")

    @patch("builtins.input", return_value="N")
    def test_intervention_returns_viii_corps_when_corps_are_on_map(self, mock_input):
        GlobalGameState.us_third_army_activated = True
        add_units_to_space(st_malo, US_VIII_CORPS)
        add_units_to_space(rennes, US_XV_CORPS)
        add_units_to_space(st_malo, US_THIRD_ARMY)

        result = check_hitler_intervention_applies(card_046, die_roll=1, target_choice=1)

        self.assertEqual(result, US_VIII_CORPS)
        self.assertFalse(GlobalGameState.hitler_intervention_no_effect)
        mock_input.assert_called_once_with("ATTEMPT TO CANCEL HITLER INTERVENTION? (Y/N): ")

    @patch("builtins.input", return_value="N")
    def test_intervention_returns_xv_corps_when_corps_are_on_map(self, mock_input):
        GlobalGameState.us_third_army_activated = True
        add_units_to_space(st_malo, US_VIII_CORPS)
        add_units_to_space(rennes, US_XV_CORPS)
        add_units_to_space(st_malo, US_THIRD_ARMY)

        result = check_hitler_intervention_applies(card_046, die_roll=1, target_choice=2)

        self.assertEqual(result, US_XV_CORPS)
        self.assertFalse(GlobalGameState.hitler_intervention_no_effect)
        mock_input.assert_called_once_with("ATTEMPT TO CANCEL HITLER INTERVENTION? (Y/N): ")

    @patch("builtins.input", return_value="Y")
    def test_intervention_returns_none_when_hitler_approval_roll_passes(self, mock_input):
        GlobalGameState.us_third_army_activated = True
        US_VIII_CORPS.location = None
        US_XV_CORPS.location = None
        add_units_to_space(st_malo, US_THIRD_ARMY)

        result = check_hitler_intervention_applies(card_046, die_roll=5)

        self.assertIsNone(result)
        self.assertTrue(GlobalGameState.hitler_intervention_no_effect)

    @patch("builtins.input", return_value="Y")
    def test_intervention_returns_third_army_when_roll_fails_and_corps_are_not_on_map(self, mock_input):
        GlobalGameState.us_third_army_activated = True
        US_VIII_CORPS.location = None
        US_XV_CORPS.location = None
        add_units_to_space(le_mans, US_THIRD_ARMY)

        result = check_hitler_intervention_applies(card_046, die_roll=6)

        self.assertEqual(result, US_THIRD_ARMY)
        self.assertFalse(GlobalGameState.hitler_intervention_no_effect)

    @patch("builtins.input", return_value="Y")
    def test_selected_corps_target_can_still_be_cancelled(self, mock_input):
        GlobalGameState.us_third_army_activated = True
        add_units_to_space(st_malo, US_VIII_CORPS)
        add_units_to_space(rennes, US_XV_CORPS)
        add_units_to_space(st_malo, US_THIRD_ARMY)

        result = check_hitler_intervention_applies(card_046, die_roll=5, target_choice=2)

        self.assertIsNone(result)
        self.assertTrue(GlobalGameState.hitler_intervention_no_effect)

    @patch("builtins.input", return_value="N")
    def test_intervention_resets_no_effect_flag_when_corps_target_applies(self, mock_input):
        GlobalGameState.us_third_army_activated = True
        add_units_to_space(st_malo, US_VIII_CORPS)
        GlobalGameState.hitler_intervention_no_effect = True

        result = check_hitler_intervention_applies(card_046, die_roll=1)

        self.assertEqual(result, US_VIII_CORPS)
        self.assertFalse(GlobalGameState.hitler_intervention_no_effect)


class TestHitlerInterventionRedeploy(unittest.TestCase):
    def setUp(self):
        do_opening_setup()
        GlobalGameState.us_third_army_activated = True
        US_THIRD_ARMY.location = None
        US_VIII_CORPS.location = None
        add_units_to_space(rennes, US_XV_CORPS)
        self._remove_unit_from_map_and_reserve(PZ_21)
        self._remove_unit_from_map_and_reserve(PZ_LEHR)
        attacking_space = get_german_space_facing_front_line(US_XV_CORPS)
        source_space = next(space for space in get_all_map_spaces() if space != attacking_space)
        add_units_to_space(source_space, PZ_21)
        strategic_reserve_box.units.append(PZ_LEHR)

    def tearDown(self):
        reset_map()
        do_opening_setup()
        GlobalGameState.us_third_army_activated = False
        US_THIRD_ARMY.location = None
        US_VIII_CORPS.location = None
        US_XV_CORPS.location = None

    @staticmethod
    def _remove_unit_from_map_and_reserve(unit):
        for space in get_all_map_spaces():
            if unit in space.units:
                space.units.remove(unit)
        while unit in strategic_reserve_box.units:
            strategic_reserve_box.units.remove(unit)

    def test_card_46_redeploys_one_panzer_to_attack_xv_corps(self):
        available_panzers = []
        for space in get_all_map_spaces():
            for unit in space.units:
                if isinstance(unit, GermanUnit) and unit.is_panzer():
                    available_panzers.append((unit, space))
        for unit in strategic_reserve_box.units:
            if isinstance(unit, GermanUnit) and unit.is_panzer():
                available_panzers.append((unit, strategic_reserve_box))
        self.assertGreaterEqual(len(available_panzers), 2)

        panzer, original_space = available_panzers[0]
        target_army, attacking_space = do_hitler_intervention_redeploy(card_046, US_XV_CORPS, deployment_choices=[1])

        self.assertEqual(target_army, US_XV_CORPS)
        self.assertIn(panzer, attacking_space.units)
        self.assertNotIn(panzer, original_space.units)


    def test_card_46_redeploys_two_panzers_to_attack_xv_corps(self):
        for space in get_all_map_spaces():
            space.units[:] = [unit for unit in space.units if not (isinstance(unit, GermanUnit) and unit.is_panzer())]

        add_units_to_space(bayeux, PZ_21)
        strategic_reserve_box.units.append(PZ_LEHR)

        target_army, attacking_space = do_hitler_intervention_redeploy(card_046, US_XV_CORPS, deployment_choices=[1, 1])

        self.assertEqual(target_army, US_XV_CORPS)
        self.assertIn(PZ_21, attacking_space.units)
        self.assertIn(PZ_LEHR, attacking_space.units)
        self.assertNotIn(PZ_21, bayeux.units)
        self.assertNotIn(PZ_LEHR, strategic_reserve_box.units)
    
    def test_card_46_does_not_exceed_four_panzer_stacking_limit(self):
        _target_army, attacking_space = do_hitler_intervention_redeploy(card_046, US_XV_CORPS, deployment_choices=[1, 2])
        panzer_count = sum(1 for unit in attacking_space.units if isinstance(unit, GermanUnit) and unit.is_panzer())
        self.assertLessEqual(panzer_count, 4)

    def test_card_46_uses_actual_xv_corps_location(self):
        target_army, attacking_space = do_hitler_intervention_redeploy(card_046, US_XV_CORPS, deployment_choices=[])
        self.assertEqual(target_army, US_XV_CORPS)
        self.assertIsNotNone(attacking_space)


    def test_card_46_does_not_redeploy_panzer_from_space_under_siege(self):
        add_units_to_space(caen, [PZ_9])
        caen.under_siege = True
        panzer_in_caen = next(unit for unit in caen.units if isinstance(unit, GermanUnit) and unit.is_panzer())

        target_army, attacking_space = do_hitler_intervention_redeploy(card_046, US_XV_CORPS, deployment_choices=[1])

        self.assertEqual(target_army, US_XV_CORPS)
        self.assertIn(panzer_in_caen, caen.units)
        self.assertNotIn(panzer_in_caen, attacking_space.units)
if __name__ == "__main__":
    unittest.main()

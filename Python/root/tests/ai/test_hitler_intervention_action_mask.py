import unittest

from cards.card_32 import card as card_032
from cards.card_35 import card as card_035
from cards.card_36 import card as card_036
from cards.card_46 import card as card_046

from core.ai.headless_action_masks import (
    HeadlessDecisionState,
    get_hitler_intervention_mask,
)
from core.allied_armies import (
    armies_list,
    US_FIRST_ARMY,
    US_THIRD_ARMY,
    US_VIII_CORPS,
    US_XV_CORPS,
    BRITISH_SECOND_ARMY,
    CANADIAN_FIRST_ARMY,
)
from core.german_units import (
    PZ_LEHR,
    SS_12,
    panzer_divisions_list,
    create_kampfgruppe,
)
from core.global_game_state import GlobalGameState
from core.map.map_model import strategic_reserve_box
from core.map.map_spaces_us_1 import (
    us_1_start_box,
    utah_omaha,
    carentan,
)
from core.map.map_spaces_brit_2 import (
    brit_2_start_box,
    bayeux,
    gold_juno_sword_brit,
)
from core.map.map_spaces_can_1 import (
    can_1_start_box,
    lebisey_wood,
    gold_juno_sword_can,
)

from core.map.map_spaces_us_3 import (
    us_3_start_box,
    st_malo,
    le_mans,
    rennes,
    brest,
)

from core.map.map_utilities import (
    do_opening_setup,
    add_units_to_space,
    remove_units_from_space,
)
from core.save_load_game import get_all_map_spaces_excluding_boxes




def print_hitler_intervention_mask(mask):
    labels = {
        0: "Continue / finish",
        1: "Attempt cancel",
    }

    for index, army in enumerate(armies_list):
        labels[2 + index] = f"Target {army.display_name}"

    from core.german_units import panzer_divisions_list

    for index, panzer in enumerate(panzer_divisions_list):
        labels[8 + index] = f"Redeploy {panzer.name}"

    labels[19] = "Redeploy Kampfgruppe 1"
    labels[20] = "Redeploy Kampfgruppe 2"

    print()
    print("HITLER INTERVENTION MASK")
    print("------------------------")

    for action_id, value in enumerate(mask):
        print(
            f"{action_id:>2}  "
            f"{int(value)}  "
            f"{labels.get(action_id, 'UNKNOWN')}"
        )

class TestHitlerInterventionActionMask(unittest.TestCase):


    def remove_all_panzer_forces(self):
        from core.models import GermanUnit

        for space in get_all_map_spaces_excluding_boxes():
            space.units[:] = [
                unit for unit in space.units
                if not (
                    isinstance(unit, GermanUnit)
                    and unit.is_panzer()
                )
            ]

        strategic_reserve_box.units[:] = [
            unit for unit in strategic_reserve_box.units
            if not (
                isinstance(unit, GermanUnit)
                and unit.is_panzer()
            )
        ]
    def setUp(self):
        do_opening_setup()
        GlobalGameState.headless = True
        GlobalGameState.us_third_army_activated = False
        GlobalGameState.hitler_intervention_no_effect = False
        HeadlessDecisionState.reset_hitler_intervention()

        remove_units_from_space(us_1_start_box, US_FIRST_ARMY)
        remove_units_from_space(brit_2_start_box, BRITISH_SECOND_ARMY)
        remove_units_from_space(can_1_start_box, CANADIAN_FIRST_ARMY)

        strategic_reserve_box.units.clear()

    def set_target_stage(self):
        HeadlessDecisionState.hitler_intervention_stage = "TARGET"

    def set_redeploy_stage(self, target):
        HeadlessDecisionState.hitler_intervention_stage = "REDEPLOY"
        HeadlessDecisionState.hitler_intervention_target = target
        HeadlessDecisionState.hitler_intervention_redeployments = 0

    def assert_target_mask(self, card, expected_armies):
        self.set_target_stage()
        mask = get_hitler_intervention_mask(card)

        print_hitler_intervention_mask(mask)
        
        for army in armies_list:
            action_id = armies_list.index(army) + 2
            expected = 1.0 if army in expected_armies else 0.0
            self.assertEqual(
                mask[action_id],
                expected,
                f"Unexpected mask for {army.display_name}"
            )

    def test_target_selection_mask_with_five_armies_present(self):
        GlobalGameState.us_third_army_activated = True

        remove_units_from_space(us_3_start_box, US_VIII_CORPS)
        remove_units_from_space(us_3_start_box, US_XV_CORPS)

        add_units_to_space(carentan, US_FIRST_ARMY)
        add_units_to_space(bayeux, BRITISH_SECOND_ARMY)
        add_units_to_space(lebisey_wood, CANADIAN_FIRST_ARMY)
        add_units_to_space(st_malo, US_VIII_CORPS)
        add_units_to_space(rennes, US_XV_CORPS)

        self.assert_target_mask(
            card_035,
            [
                US_FIRST_ARMY,
                BRITISH_SECOND_ARMY,
                CANADIAN_FIRST_ARMY,
                US_VIII_CORPS,
                US_XV_CORPS,
            ]
        )

    def test_target_selection_mask_with_three_armies_when_third_army_inactive(self):
        GlobalGameState.us_third_army_activated = False

        add_units_to_space(carentan, US_FIRST_ARMY)
        add_units_to_space(bayeux, BRITISH_SECOND_ARMY)
        add_units_to_space(lebisey_wood, CANADIAN_FIRST_ARMY)

        self.assert_target_mask(
            card_035,
            [
                US_FIRST_ARMY,
                BRITISH_SECOND_ARMY,
                CANADIAN_FIRST_ARMY,
            ]
        )

    def test_target_selection_mask_with_merged_us_third_army(self):
        GlobalGameState.us_third_army_activated = True

        remove_units_from_space(us_3_start_box, US_VIII_CORPS)
        remove_units_from_space(us_3_start_box, US_XV_CORPS)

        add_units_to_space(carentan, US_FIRST_ARMY)
        add_units_to_space(bayeux, BRITISH_SECOND_ARMY)
        add_units_to_space(lebisey_wood, CANADIAN_FIRST_ARMY)


        add_units_to_space(le_mans, US_THIRD_ARMY)
        US_VIII_CORPS.location = None
        US_XV_CORPS.location = None
        GlobalGameState.us_third_army_merged = True
        
        self.assert_target_mask(
            card_035,
            [
                US_FIRST_ARMY,
                US_THIRD_ARMY,
                BRITISH_SECOND_ARMY,
                CANADIAN_FIRST_ARMY,
            ]
        )


    def test_card_32_automatically_selects_us_first_army(self):
        add_units_to_space(carentan, US_FIRST_ARMY)

        self.set_target_stage()
        mask = get_hitler_intervention_mask(card_032)

        self.assertIs(
            HeadlessDecisionState.hitler_intervention_target,
            US_FIRST_ARMY
        )
        self.assertEqual(
            HeadlessDecisionState.hitler_intervention_stage,
            "REDEPLOY"
        )

        self.assertEqual(mask[2], 0.0)

    def test_card_35_targets_all_available_allied_armies(self):
        GlobalGameState.us_third_army_activated = True

        remove_units_from_space(us_3_start_box, US_VIII_CORPS)
        remove_units_from_space(us_3_start_box, US_XV_CORPS)

        add_units_to_space(carentan, US_FIRST_ARMY)
        add_units_to_space(bayeux, BRITISH_SECOND_ARMY)
        add_units_to_space(lebisey_wood, CANADIAN_FIRST_ARMY)
        add_units_to_space(st_malo, US_VIII_CORPS)
        add_units_to_space(rennes, US_XV_CORPS)

        self.assert_target_mask(
            card_035,
            [
                US_FIRST_ARMY,
                BRITISH_SECOND_ARMY,
                CANADIAN_FIRST_ARMY,
                US_VIII_CORPS,
                US_XV_CORPS,
            ]
        )


    def test_card_36_automatically_selects_canadian_first_army(self):
        add_units_to_space(lebisey_wood, CANADIAN_FIRST_ARMY)

        self.set_target_stage()
        mask = get_hitler_intervention_mask(card_036)
        print_hitler_intervention_mask(mask)

        self.assertIs(
            HeadlessDecisionState.hitler_intervention_target,
            CANADIAN_FIRST_ARMY
        )

        self.assertEqual(
            HeadlessDecisionState.hitler_intervention_stage,
            "REDEPLOY"
        )

        canadian_action = 2 + armies_list.index(CANADIAN_FIRST_ARMY)
        self.assertEqual(mask[canadian_action], 0.0)

        self.assertGreater(mask[8:21].sum(), 0.0)

    def test_card_46_targets_us_third_army_corps(self):
        GlobalGameState.us_third_army_activated = True

        remove_units_from_space(us_3_start_box, US_VIII_CORPS)
        remove_units_from_space(us_3_start_box, US_XV_CORPS)

        add_units_to_space(st_malo, US_VIII_CORPS)
        add_units_to_space(rennes, US_XV_CORPS)

        self.assert_target_mask(
            card_046,
            [
                US_VIII_CORPS,
                US_XV_CORPS,
            ]
        )


    def test_card_46_automatically_selects_merged_us_third_army(self):
        GlobalGameState.us_third_army_activated = True

        US_VIII_CORPS.location = None
        US_XV_CORPS.location = None
        GlobalGameState.us_third_army_merged = True

        add_units_to_space(le_mans, US_THIRD_ARMY)

        self.set_target_stage()
        mask = get_hitler_intervention_mask(card_046)
        print_hitler_intervention_mask(mask)

        self.assertIs(
            HeadlessDecisionState.hitler_intervention_target,
            US_THIRD_ARMY
        )

        self.assertEqual(
            HeadlessDecisionState.hitler_intervention_stage,
            "REDEPLOY"
        )

        us_3_action = 2 + armies_list.index(US_THIRD_ARMY)
        self.assertEqual(mask[us_3_action], 0.0)

        self.assertGreater(mask[8:21].sum(), 0.0)

    def test_target_mask_excludes_fortress_beach_and_start_box(self):
        GlobalGameState.us_third_army_activated = True

        remove_units_from_space(us_3_start_box, US_VIII_CORPS)
        remove_units_from_space(us_3_start_box, US_XV_CORPS)

        add_units_to_space(utah_omaha, US_FIRST_ARMY)
        add_units_to_space(gold_juno_sword_brit, BRITISH_SECOND_ARMY)
        add_units_to_space(gold_juno_sword_can, CANADIAN_FIRST_ARMY)
        add_units_to_space(brest, US_VIII_CORPS)
        add_units_to_space(us_3_start_box, US_XV_CORPS)

        self.set_target_stage()
        mask = get_hitler_intervention_mask(card_035)

        for army in armies_list:
            self.assertEqual(
                mask[armies_list.index(army) + 2],
                0.0
            )

    def test_redeploy_mask_contains_available_panzer_divisions(self):
        add_units_to_space(carentan, US_FIRST_ARMY)

        strategic_reserve_box.units.extend([
            PZ_LEHR,
            SS_12,
        ])

        self.set_redeploy_stage(US_FIRST_ARMY)
        mask = get_hitler_intervention_mask(card_032)

        self.assertEqual(
            mask[8 + panzer_divisions_list.index(PZ_LEHR)],
            1.0
        )
        self.assertEqual(
            mask[8 + panzer_divisions_list.index(SS_12)],
            1.0
        )

    def test_redeploy_mask_excludes_unavailable_panzer_divisions(self):
        add_units_to_space(carentan, US_FIRST_ARMY)

        strategic_reserve_box.units.append(PZ_LEHR)

        self.set_redeploy_stage(US_FIRST_ARMY)
        mask = get_hitler_intervention_mask(card_032)

        self.assertEqual(
            mask[8 + panzer_divisions_list.index(PZ_LEHR)],
            1.0
        )
        self.assertEqual(
            mask[8 + panzer_divisions_list.index(SS_12)],
            0.0
        )


    def test_redeploy_mask_finishes_when_no_panzer_forces_available(self):
        add_units_to_space(carentan, US_FIRST_ARMY)

        self.remove_all_panzer_forces()

        self.set_redeploy_stage(US_FIRST_ARMY)

        mask = get_hitler_intervention_mask(card_032)
        print_hitler_intervention_mask(mask)

        self.assertEqual(mask[0], 1.0)
        self.assertEqual(mask[8:21].sum(), 0.0)


    def test_redeploy_mask_contains_one_kampfgruppe(self):
        GlobalGameState.us_third_army_activated = True

        remove_units_from_space(us_3_start_box, US_VIII_CORPS)
        US_XV_CORPS.location = None

        add_units_to_space(st_malo, US_VIII_CORPS)

        strategic_reserve_box.units.append(
            create_kampfgruppe()
        )

        self.set_redeploy_stage(US_VIII_CORPS)

        mask = get_hitler_intervention_mask(card_046)
        print_hitler_intervention_mask(mask)

        self.assertEqual(mask[19], 1.0)
        self.assertEqual(mask[20], 0.0)


    def test_redeploy_mask_contains_two_kampfgruppen(self):
        GlobalGameState.us_third_army_activated = True

        remove_units_from_space(us_3_start_box, US_VIII_CORPS)
        US_XV_CORPS.location = None

        add_units_to_space(st_malo, US_VIII_CORPS)

        strategic_reserve_box.units.extend([
            create_kampfgruppe(),
            create_kampfgruppe(),
        ])

        self.set_redeploy_stage(US_VIII_CORPS)

        mask = get_hitler_intervention_mask(card_046)
        print_hitler_intervention_mask(mask)

        self.assertEqual(mask[19], 1.0)
        self.assertEqual(mask[20], 1.0)

    def test_cancel_mask_for_card_35(self):
        add_units_to_space(carentan, US_FIRST_ARMY)

        mask = get_hitler_intervention_mask(card_035)

        self.assertEqual(mask[0], 1.0)
        self.assertEqual(mask[1], 1.0)
        self.assertEqual(mask[2:].sum(), 0.0)

    def test_cancel_mask_for_card_36(self):
        add_units_to_space(lebisey_wood, CANADIAN_FIRST_ARMY)

        mask = get_hitler_intervention_mask(card_036)

        self.assertEqual(mask[0], 1.0)
        self.assertEqual(mask[1], 1.0)
        self.assertEqual(mask[2:].sum(), 0.0)

    def test_cancel_mask_for_card_46(self):
        GlobalGameState.us_third_army_activated = True

        remove_units_from_space(us_3_start_box, US_VIII_CORPS)
        add_units_to_space(st_malo, US_VIII_CORPS)

        mask = get_hitler_intervention_mask(card_046)

        self.assertEqual(mask[0], 1.0)
        self.assertEqual(mask[1], 1.0)
        self.assertEqual(mask[2:].sum(), 0.0)

    def test_card_32_has_no_cancel_action(self):
        GlobalGameState.us_third_army_activated = True
        add_units_to_space(carentan, US_FIRST_ARMY)
        add_units_to_space(st_malo, US_VIII_CORPS)

        mask = get_hitler_intervention_mask(card_032)

        self.assertEqual(mask[0], 1.0)
        self.assertEqual(mask[1], 0.0)

    def test_no_effect_mask_when_no_valid_target_exists(self):
        add_units_to_space(utah_omaha, US_FIRST_ARMY)

        mask = get_hitler_intervention_mask(card_032)

        self.assertEqual(mask[0], 1.0)
        self.assertEqual(mask[1:].sum(), 0.0)


if __name__ == "__main__":
    unittest.main()

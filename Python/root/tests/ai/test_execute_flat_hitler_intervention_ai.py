import unittest
from unittest.mock import patch

from cards.card_32 import card as card_032
from cards.card_35 import card as card_035
from cards.card_36 import card as card_036
from cards.card_46 import card as card_046

from core.ai.headless_action_masks import (
    HeadlessDecisionState,
    execute_flat_hitler_intervention_ai,
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

from core.global_game_state import GlobalGameState

from core.german_units import (
    PZ_LEHR,
    panzer_divisions_list,
    create_kampfgruppe,
)

from core.map.map_model import (
    strategic_reserve_box,
    hitler_approval_track,
)

from core.map.map_spaces_us_1 import carentan
from core.map.map_spaces_brit_2 import bayeux
from core.map.map_spaces_can_1 import lebisey_wood
from core.map.map_spaces_us_3 import (
    st_malo,
    rennes,
    le_mans,
)

from core.map.map_utilities import (
    do_opening_setup,
    add_units_to_space,
    remove_units_from_space,
    get_all_map_spaces,
)

from core.models import GermanUnit
from core.tables.weather import get_weather_result


class TestExecuteFlatHitlerInterventionAI(unittest.TestCase):

    def setUp(self):
        do_opening_setup()
        GlobalGameState.headless = True
        GlobalGameState.us_third_army_activated = False
        GlobalGameState.us_third_army_merged = False
        GlobalGameState.hitler_intervention_no_effect = False
        HeadlessDecisionState.reset_hitler_intervention()
        self.weather = get_weather_result(4)

        for army in armies_list:
            if army.location is not None:
                remove_units_from_space(army.location, army)

        strategic_reserve_box.units.clear()

    def tearDown(self):
        HeadlessDecisionState.reset_hitler_intervention()
        GlobalGameState.headless = False

    def remove_all_panzer_forces(self):
        for space in get_all_map_spaces():
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

    def setup_five_targets(self):
        GlobalGameState.us_third_army_activated = True
        add_units_to_space(carentan, US_FIRST_ARMY)
        add_units_to_space(bayeux, BRITISH_SECOND_ARMY)
        add_units_to_space(lebisey_wood, CANADIAN_FIRST_ARMY)
        add_units_to_space(st_malo, US_VIII_CORPS)
        add_units_to_space(rennes, US_XV_CORPS)

    def setup_card_32(self):
        GlobalGameState.us_third_army_activated = True
        add_units_to_space(carentan, US_FIRST_ARMY)
        add_units_to_space(st_malo, US_VIII_CORPS)

    def setup_card_46_corps(self):
        GlobalGameState.us_third_army_activated = True
        add_units_to_space(st_malo, US_VIII_CORPS)
        add_units_to_space(rennes, US_XV_CORPS)

    def setup_card_46_merged(self):
        GlobalGameState.us_third_army_activated = True
        GlobalGameState.us_third_army_merged = True
        US_VIII_CORPS.location = None
        US_XV_CORPS.location = None
        add_units_to_space(le_mans, US_THIRD_ARMY)

    def test_card_32_starts_intervention_and_selects_us_first_army(self):
        self.setup_card_32()

        result = execute_flat_hitler_intervention_ai(
            0,
            card_032,
            self.weather
        )

        self.assertTrue(result)
        self.assertIs(
            HeadlessDecisionState.hitler_intervention_target,
            US_FIRST_ARMY
        )
        self.assertEqual(
            HeadlessDecisionState.hitler_intervention_stage,
            "REDEPLOY"
        )

    def test_card_32_no_effect_when_us_third_army_not_activated(self):
        add_units_to_space(carentan, US_FIRST_ARMY)

        result = execute_flat_hitler_intervention_ai(
            0,
            card_032,
            self.weather
        )

        self.assertFalse(result)
        self.assertTrue(GlobalGameState.hitler_intervention_no_effect)
        self.assertIsNone(
            HeadlessDecisionState.hitler_intervention_stage
        )

    def test_card_32_no_effect_when_st_malo_not_taken(self):
        GlobalGameState.us_third_army_activated = True
        add_units_to_space(carentan, US_FIRST_ARMY)

        result = execute_flat_hitler_intervention_ai(
            0,
            card_032,
            self.weather
        )

        self.assertFalse(result)
        self.assertTrue(GlobalGameState.hitler_intervention_no_effect)
        self.assertIsNone(
            HeadlessDecisionState.hitler_intervention_stage
        )

    @patch(
        "core.ai.headless_action_masks.do_hitler_intervention_redeploy"
    )
    def test_card_32_executes_selected_panzer_redeployment(
        self,
        mock_redeploy
    ):
        self.setup_card_32()
        self.remove_all_panzer_forces()
        strategic_reserve_box.units.append(PZ_LEHR)

        HeadlessDecisionState.hitler_intervention_stage = "REDEPLOY"
        HeadlessDecisionState.hitler_intervention_target = US_FIRST_ARMY

        action_id = 8 + panzer_divisions_list.index(PZ_LEHR)

        result = execute_flat_hitler_intervention_ai(
            action_id,
            card_032,
            self.weather
        )

        self.assertTrue(result)
        mock_redeploy.assert_called_once_with(
            card_032,
            US_FIRST_ARMY,
            deployment_choices=[1]
        )
        self.assertEqual(
            HeadlessDecisionState.hitler_intervention_redeployments,
            1
        )

    @patch(
        "core.ai.headless_action_masks.do_hitler_intervention_attack"
    )
    def test_card_32_finishes_intervention_with_attack(
        self,
        mock_attack
    ):
        self.setup_card_32()

        HeadlessDecisionState.hitler_intervention_stage = "REDEPLOY"
        HeadlessDecisionState.hitler_intervention_target = US_FIRST_ARMY
        HeadlessDecisionState.hitler_intervention_redeployments = 2

        result = execute_flat_hitler_intervention_ai(
            0,
            card_032,
            self.weather
        )

        self.assertFalse(result)
        mock_attack.assert_called_once()
        self.assertIsNone(
            HeadlessDecisionState.hitler_intervention_stage
        )
        self.assertIsNone(
            HeadlessDecisionState.hitler_intervention_target
        )
        self.assertEqual(
            HeadlessDecisionState.hitler_intervention_redeployments,
            0
        )

    def test_card_35_action_0_declines_cancellation(self):
        self.setup_five_targets()
        HeadlessDecisionState.hitler_intervention_stage = "CANCEL"

        result = execute_flat_hitler_intervention_ai(
            0,
            card_035,
            self.weather
        )

        self.assertTrue(result)
        self.assertEqual(
            HeadlessDecisionState.hitler_intervention_stage,
            "TARGET"
        )
        self.assertIsNone(
            HeadlessDecisionState.hitler_intervention_target
        )

    @patch(
        "core.actions.hitler_intervention.randint",
        return_value=1
    )
    def test_card_35_successful_cancellation_ends_intervention(
        self,
        mock_randint
    ):
        self.setup_five_targets()
        hitler_approval_track.value = 3
        HeadlessDecisionState.hitler_intervention_stage = "CANCEL"

        result = execute_flat_hitler_intervention_ai(
            1,
            card_035,
            self.weather
        )

        self.assertFalse(result)
        self.assertTrue(GlobalGameState.hitler_intervention_no_effect)
        self.assertIsNone(
            HeadlessDecisionState.hitler_intervention_stage
        )

    @patch(
        "core.actions.hitler_intervention.randint",
        return_value=6
    )
    def test_card_35_failed_cancellation_moves_to_target_stage(
        self,
        mock_randint
    ):
        self.setup_five_targets()
        hitler_approval_track.value = 3
        HeadlessDecisionState.hitler_intervention_stage = "CANCEL"

        result = execute_flat_hitler_intervention_ai(
            1,
            card_035,
            self.weather
        )

        self.assertTrue(result)
        self.assertFalse(GlobalGameState.hitler_intervention_no_effect)
        self.assertEqual(
            HeadlessDecisionState.hitler_intervention_stage,
            "TARGET"
        )
        self.assertIsNone(
            HeadlessDecisionState.hitler_intervention_target
        )

    def test_card_35_selects_us_xv_corps_target(self):
        self.setup_five_targets()
        HeadlessDecisionState.hitler_intervention_stage = "TARGET"

        action_id = 2 + armies_list.index(US_XV_CORPS)

        result = execute_flat_hitler_intervention_ai(
            action_id,
            card_035,
            self.weather
        )

        self.assertTrue(result)
        self.assertIs(
            HeadlessDecisionState.hitler_intervention_target,
            US_XV_CORPS
        )
        self.assertEqual(
            HeadlessDecisionState.hitler_intervention_stage,
            "REDEPLOY"
        )

    def test_card_35_rejects_target_not_in_current_targets(self):
        add_units_to_space(carentan, US_FIRST_ARMY)
        add_units_to_space(bayeux, BRITISH_SECOND_ARMY)
        HeadlessDecisionState.hitler_intervention_stage = "TARGET"

        action_id = 2 + armies_list.index(US_XV_CORPS)

        result = execute_flat_hitler_intervention_ai(
            action_id,
            card_035,
            self.weather
        )

        self.assertTrue(result)
        self.assertIsNone(
            HeadlessDecisionState.hitler_intervention_target
        )
        self.assertEqual(
            HeadlessDecisionState.hitler_intervention_stage,
            "TARGET"
        )

    def test_card_36_action_0_declines_cancellation(self):
        add_units_to_space(
            lebisey_wood,
            CANADIAN_FIRST_ARMY
        )

        HeadlessDecisionState.hitler_intervention_stage = "CANCEL"

        result = execute_flat_hitler_intervention_ai(
            0,
            card_036,
            self.weather
        )

        self.assertTrue(result)
        self.assertEqual(
            HeadlessDecisionState.hitler_intervention_stage,
            "TARGET"
        )

    @patch(
        "core.actions.hitler_intervention.randint",
        return_value=6
    )
    def test_card_36_failed_cancellation_automatically_selects_canadian(
        self,
        mock_randint
    ):
        add_units_to_space(
            lebisey_wood,
            CANADIAN_FIRST_ARMY
        )

        hitler_approval_track.value = 3
        HeadlessDecisionState.hitler_intervention_stage = "CANCEL"

        result = execute_flat_hitler_intervention_ai(
            1,
            card_036,
            self.weather
        )

        self.assertTrue(result)
        self.assertIs(
            HeadlessDecisionState.hitler_intervention_target,
            CANADIAN_FIRST_ARMY
        )
        self.assertEqual(
            HeadlessDecisionState.hitler_intervention_stage,
            "REDEPLOY"
        )

    def test_card_46_selects_viii_corps_target(self):
        self.setup_card_46_corps()
        HeadlessDecisionState.hitler_intervention_stage = "TARGET"

        action_id = 2 + armies_list.index(US_VIII_CORPS)

        result = execute_flat_hitler_intervention_ai(
            action_id,
            card_046,
            self.weather
        )

        self.assertTrue(result)
        self.assertIs(
            HeadlessDecisionState.hitler_intervention_target,
            US_VIII_CORPS
        )
        self.assertEqual(
            HeadlessDecisionState.hitler_intervention_stage,
            "REDEPLOY"
        )

    def test_card_46_selects_xv_corps_target(self):
        self.setup_card_46_corps()
        HeadlessDecisionState.hitler_intervention_stage = "TARGET"

        action_id = 2 + armies_list.index(US_XV_CORPS)

        result = execute_flat_hitler_intervention_ai(
            action_id,
            card_046,
            self.weather
        )

        self.assertTrue(result)
        self.assertIs(
            HeadlessDecisionState.hitler_intervention_target,
            US_XV_CORPS
        )
        self.assertEqual(
            HeadlessDecisionState.hitler_intervention_stage,
            "REDEPLOY"
        )

    @patch(
        "core.actions.hitler_intervention.randint",
        return_value=6
    )
    def test_card_46_failed_cancellation_with_two_corps_moves_to_target_stage(
        self,
        mock_randint
    ):
        self.setup_card_46_corps()
        hitler_approval_track.value = 3
        HeadlessDecisionState.hitler_intervention_stage = "CANCEL"

        result = execute_flat_hitler_intervention_ai(
            1,
            card_046,
            self.weather
        )

        self.assertTrue(result)
        self.assertEqual(
            HeadlessDecisionState.hitler_intervention_stage,
            "TARGET"
        )
        self.assertIsNone(
            HeadlessDecisionState.hitler_intervention_target
        )

    @patch(
        "core.actions.hitler_intervention.randint",
        return_value=6
    )
    def test_card_46_failed_cancellation_automatically_selects_merged_third_army(
        self,
        mock_randint
    ):
        self.setup_card_46_merged()
        hitler_approval_track.value = 3
        HeadlessDecisionState.hitler_intervention_stage = "CANCEL"

        result = execute_flat_hitler_intervention_ai(
            1,
            card_046,
            self.weather
        )

        self.assertTrue(result)
        self.assertIs(
            HeadlessDecisionState.hitler_intervention_target,
            US_THIRD_ARMY
        )
        self.assertEqual(
            HeadlessDecisionState.hitler_intervention_stage,
            "REDEPLOY"
        )

    @patch(
        "core.ai.headless_action_masks.do_hitler_intervention_redeploy"
    )
    def test_card_46_redeploys_first_kampfgruppe(
        self,
        mock_redeploy
    ):
        self.setup_card_46_corps()
        self.remove_all_panzer_forces()

        strategic_reserve_box.units.append(
            create_kampfgruppe()
        )

        HeadlessDecisionState.hitler_intervention_stage = "REDEPLOY"
        HeadlessDecisionState.hitler_intervention_target = US_VIII_CORPS

        result = execute_flat_hitler_intervention_ai(
            19,
            card_046,
            self.weather
        )

        self.assertTrue(result)
        mock_redeploy.assert_called_once_with(
            card_046,
            US_VIII_CORPS,
            deployment_choices=[1]
        )
        self.assertEqual(
            HeadlessDecisionState.hitler_intervention_redeployments,
            1
        )

    @patch(
        "core.ai.headless_action_masks.do_hitler_intervention_redeploy"
    )
    def test_card_46_redeploys_second_kampfgruppe(
        self,
        mock_redeploy
    ):
        self.setup_card_46_corps()
        self.remove_all_panzer_forces()

        strategic_reserve_box.units.extend([
            create_kampfgruppe(),
            create_kampfgruppe(),
        ])

        HeadlessDecisionState.hitler_intervention_stage = "REDEPLOY"
        HeadlessDecisionState.hitler_intervention_target = US_VIII_CORPS

        result = execute_flat_hitler_intervention_ai(
            20,
            card_046,
            self.weather
        )

        self.assertTrue(result)
        mock_redeploy.assert_called_once_with(
            card_046,
            US_VIII_CORPS,
            deployment_choices=[2]
        )
        self.assertEqual(
            HeadlessDecisionState.hitler_intervention_redeployments,
            1
        )

    @patch(
        "core.ai.headless_action_masks.do_hitler_intervention_attack"
    )
    def test_card_46_finishes_redeployment_and_attacks(
        self,
        mock_attack
    ):
        self.setup_card_46_corps()

        HeadlessDecisionState.hitler_intervention_stage = "REDEPLOY"
        HeadlessDecisionState.hitler_intervention_target = US_VIII_CORPS
        HeadlessDecisionState.hitler_intervention_redeployments = 2

        result = execute_flat_hitler_intervention_ai(
            0,
            card_046,
            self.weather
        )

        self.assertFalse(result)
        mock_attack.assert_called_once()
        self.assertIsNone(
            HeadlessDecisionState.hitler_intervention_stage
        )
        self.assertIsNone(
            HeadlessDecisionState.hitler_intervention_target
        )

    def test_no_valid_target_ends_intervention_as_no_effect(self):
        result = execute_flat_hitler_intervention_ai(
            0,
            card_035,
            self.weather
        )

        self.assertFalse(result)
        self.assertTrue(GlobalGameState.hitler_intervention_no_effect)
        self.assertIsNone(
            HeadlessDecisionState.hitler_intervention_stage
        )


    @patch(
        "core.ai.headless_action_masks.do_hitler_intervention_redeploy"
    )
    def test_panzer_action_redeploys_selected_division(self, mock_redeploy):
        self.setup_card_32()
        self.remove_all_panzer_forces()

        selected_panzer = PZ_LEHR
        strategic_reserve_box.units.append(selected_panzer)

        HeadlessDecisionState.hitler_intervention_stage = "REDEPLOY"
        HeadlessDecisionState.hitler_intervention_target = US_FIRST_ARMY

        action_id = 8 + panzer_divisions_list.index(selected_panzer)

        result = execute_flat_hitler_intervention_ai(
            action_id,
            card_032,
            self.weather
        )

        self.assertTrue(result)
        mock_redeploy.assert_called_once_with(
            card_032,
            US_FIRST_ARMY,
            deployment_choices=[1]
        )
        self.assertEqual(
            HeadlessDecisionState.hitler_intervention_redeployments,
            1
        )
if __name__ == "__main__":
    unittest.main()

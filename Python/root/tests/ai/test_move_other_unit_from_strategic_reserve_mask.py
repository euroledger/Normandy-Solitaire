import unittest
from unittest.mock import patch

import numpy as np

from core.ai.headless_action_masks import (
    OTHER_RESERVE_UNIT_TYPES,
    execute_flat_move_other_units_from_strategic_reserve_ai,
    german_eligible_spaces,
)
from core.enums import ReinforcementType
from core.german_units import create_flak88, create_kampfgruppe
from core.global_game_state import GlobalGameState
from core.map.map_model import strategic_reserve_box
from core.map.map_utilities import do_opening_setup
from tests.core_mechanics.testing_utilities import reset_game_state_for_tests


class TestMoveOtherUnitFromStrategicReserveAI(unittest.TestCase):

    def setUp(self):
        do_opening_setup()
        GlobalGameState.headless = True

    def tearDown(self):
        reset_game_state_for_tests()

    def test_ai_can_stop_without_deploying(self):
        flak = create_flak88()
        strategic_reserve_box.units.append(flak)

        with patch(
            "core.ai.headless_action_masks.randint",
            return_value=0,
        ):
            execute_flat_move_other_units_from_strategic_reserve_ai()

        self.assertIn(flak, strategic_reserve_box.units)

    def test_ai_deploys_legal_unit(self):
        flak = create_flak88()
        strategic_reserve_box.units.append(flak)

        flak_index = next(
            index
            for index, (_, unit_type, _) in enumerate(OTHER_RESERVE_UNIT_TYPES)
            if unit_type == ReinforcementType.FLAK_88
        )

        destination_index = 0
        destination = german_eligible_spaces[destination_index]

        local_action_id = (
            flak_index * len(german_eligible_spaces)
            + destination_index
        )

        mask = np.zeros(
            len(OTHER_RESERVE_UNIT_TYPES) * len(german_eligible_spaces),
            dtype=np.float32,
        )
        mask[local_action_id] = 1.0

        with patch(
            "core.ai.headless_action_masks.get_move_other_unit_from_strategic_reserve_mask",
            side_effect=[mask, np.zeros_like(mask)],
        ), patch(
            "core.ai.headless_action_masks.randint",
            return_value=1,
        ):
            execute_flat_move_other_units_from_strategic_reserve_ai()

        self.assertNotIn(flak, strategic_reserve_box.units)
        self.assertIn(flak, destination.units)

    def test_ai_can_deploy_multiple_units(self):
        flak = create_flak88()
        kampfgruppe = create_kampfgruppe()

        strategic_reserve_box.units.append(flak)
        strategic_reserve_box.units.append(kampfgruppe)

        flak_index = next(
            index
            for index, (_, unit_type, _) in enumerate(OTHER_RESERVE_UNIT_TYPES)
            if unit_type == ReinforcementType.FLAK_88
        )

        kampfgruppe_index = next(
            index
            for index, (_, unit_type, _) in enumerate(OTHER_RESERVE_UNIT_TYPES)
            if unit_type == ReinforcementType.KAMPFGRUPPE
        )

        destination_1 = german_eligible_spaces[0]
        destination_2 = german_eligible_spaces[1]

        flak_action_id = (
            flak_index * len(german_eligible_spaces)
            + 0
        )

        kampfgruppe_action_id = (
            kampfgruppe_index * len(german_eligible_spaces)
            + 1
        )

        mask_1 = np.zeros(
            len(OTHER_RESERVE_UNIT_TYPES) * len(german_eligible_spaces),
            dtype=np.float32,
        )
        mask_1[flak_action_id] = 1.0

        mask_2 = np.zeros_like(mask_1)
        mask_2[kampfgruppe_action_id] = 1.0

        empty_mask = np.zeros_like(mask_1)

        with patch(
            "core.ai.headless_action_masks.get_move_other_unit_from_strategic_reserve_mask",
            side_effect=[mask_1, mask_2, empty_mask],
        ), patch(
            "core.ai.headless_action_masks.randint",
            return_value=1,
        ):
            execute_flat_move_other_units_from_strategic_reserve_ai()

        self.assertNotIn(flak, strategic_reserve_box.units)
        self.assertNotIn(kampfgruppe, strategic_reserve_box.units)

        self.assertIn(flak, destination_1.units)
        self.assertIn(kampfgruppe, destination_2.units)

    def test_deployment_costs_no_actions(self):
        flak = create_flak88()
        strategic_reserve_box.units.append(flak)

        GlobalGameState.actions_left_this_turn = 3
        GlobalGameState.reserve_actions = 2

        flak_index = next(
            index
            for index, (_, unit_type, _) in enumerate(OTHER_RESERVE_UNIT_TYPES)
            if unit_type == ReinforcementType.FLAK_88
        )

        local_action_id = (
            flak_index * len(german_eligible_spaces)
        )

        mask = np.zeros(
            len(OTHER_RESERVE_UNIT_TYPES) * len(german_eligible_spaces),
            dtype=np.float32,
        )
        mask[local_action_id] = 1.0

        with patch(
            "core.ai.headless_action_masks.get_move_other_unit_from_strategic_reserve_mask",
            side_effect=[mask, np.zeros_like(mask)],
        ), patch(
            "core.ai.headless_action_masks.randint",
            return_value=1,
        ):
            execute_flat_move_other_units_from_strategic_reserve_ai()

        self.assertEqual(GlobalGameState.actions_left_this_turn, 3)
        self.assertEqual(GlobalGameState.reserve_actions, 2)


if __name__ == "__main__":
    unittest.main()

# tests/test_action_mask.py

import unittest
import numpy as np

from core.map.map_model import transport_track, supply_track, hitler_approval_track, strategic_reserve_box

from core.global_game_state import GlobalGameState
from core.ai.headless_action_masks import (
    augmentation_roll_mask,
    get_mini_action_mask,
    get_total_actions,
    print_mini_action_mask,
    OTHER_RESERVE_UNIT_TYPES,
)
from core.allied_armies import US_FIRST_ARMY

import core.map.map_model

from core.german_units import (
    PZ_LEHR,
    SS_12,
    FS_3,
    FS_5,
    ROMMEL,
    MEYER,
    MODEL,
    TIGER_101,
    create_flak88,
    create_kampfgruppe,
    create_nebelwerfer,
    panzer_divisions_list,
)

from core.map.map_model import strategic_reserve_box


class TestActionMask(unittest.TestCase):

    def setUp(self):
        GlobalGameState.actions_left_this_turn = 1
        GlobalGameState.reserve_actions = 0
        GlobalGameState.counter_attacked_armies.clear()
        GlobalGameState.headless = True

        # Make Strategic Reserve deterministic for every test.
        strategic_reserve_box.units.clear()

        np.set_printoptions(
            formatter={'float': '{:0.1f}'.format}
        )

    # ---------------------------------------------------------
    # HELPERS
    # ---------------------------------------------------------

    def get_other_reserve_start(self):
        _, german_eligible_spaces = get_total_actions()

        return (
            7
            + len(panzer_divisions_list) * len(german_eligible_spaces)
            + len(panzer_divisions_list)
        )

    def get_other_unit_action_slice(self, unit_type_index):
        _, german_eligible_spaces = get_total_actions()

        space_count = len(german_eligible_spaces)
        start = (
            self.get_other_reserve_start()
            + unit_type_index * space_count
        )
        end = start + space_count

        return start, end

    # ---------------------------------------------------------
    # BASIC MASK TESTS
    # ---------------------------------------------------------

    def test_action_mask_baseline_pass_only(self):
        """
        With zero AP and nothing in Strategic Reserve,
        only PASS should be legal.
        """

        GlobalGameState.actions_left_this_turn = 0
        GlobalGameState.reserve_actions = 0
        strategic_reserve_box.units.clear()

        mask = get_mini_action_mask()

        print(f"\n[DEBUG] Test Baseline Mask Output: {mask}")

        self.assertEqual(mask[0], 1.0)
        self.assertEqual(np.sum(mask), 1.0)


    def test_action_mask_blocks_already_attacked_army(self):
        # Verify that if an army was hit, its corresponding button locks out
        GlobalGameState.counter_attacked_armies.add(
            US_FIRST_ARMY.name
        )

        mask = get_mini_action_mask()

        print(
            f"\n[DEBUG] Test Attacked Army Mask Output: {mask}"
        )

        # US 1st Army has already been counter-attacked,
        # so its action must be unavailable.
        self.assertEqual(mask[1], 0.0)

        # AP remains, so PASS must NOT yet be available.
        self.assertEqual(mask[0], 0.0)

    def test_action_mask_output_structure(self):
        mask = get_mini_action_mask()

        self.assertIsInstance(mask, np.ndarray)

        total_actions, _ = get_total_actions()

        self.assertEqual(mask.shape, (total_actions,))
        self.assertEqual(mask.dtype, np.float32)

    # ---------------------------------------------------------
    # TOTAL ACTION SPACE
    # ---------------------------------------------------------

    def test_total_actions_and_german_eligible_spaces(self):
        total_actions, german_eligible_spaces = get_total_actions()

        self.assertEqual(
            len(german_eligible_spaces),
            27
        )

        self.assertEqual(
            len({id(space) for space in german_eligible_spaces}),
            len(german_eligible_spaces)
        )

        # 1 PASS
        # 6 counter-attacks
        # 11 * 27 Panzer FROM reserve
        # 11 Panzer TO reserve
        # 9 * 27 other units FROM reserve
        self.assertEqual(total_actions, 561)

        for space in german_eligible_spaces:
            self.assertNotIn(
                space.terrain,
                [
                    core.map.map_model.TerrainType.START_BOX,
                    core.map.map_model.TerrainType.BEACH
                ]
            )

    # ---------------------------------------------------------
    # PANZER FROM STRATEGIC RESERVE
    # ---------------------------------------------------------

    def test_action_mask_two_panzers_in_strategic_reserve(self):
        strategic_reserve_box.units.extend(
            [PZ_LEHR, SS_12]
        )

        _, german_eligible_spaces = get_total_actions()
        mask = get_mini_action_mask()

        print_mini_action_mask(
            mask=mask,
            legal_only=True
        )

        space_count = len(german_eligible_spaces)

        for panzer_index, panzer in enumerate(
            panzer_divisions_list
        ): 
            start = 7 + panzer_index * space_count
            end = start + space_count

            panzer_mask = mask[start:end]

            if panzer in [PZ_LEHR, SS_12]:
                self.assertGreater(
                    np.sum(panzer_mask),
                    0.0,
                    f"{panzer.name} should have legal deployment actions"
                )
            else:
                self.assertEqual(
                    np.sum(panzer_mask),
                    0.0,
                    f"{panzer.name} should have no deployment actions"
                )

    # ---------------------------------------------------------
    # GENERIC OTHER UNITS FROM STRATEGIC RESERVE
    # ---------------------------------------------------------

    def test_flak_in_reserve_enables_flak_actions(self):
        flak = create_flak88()
        strategic_reserve_box.units.append(flak)

        mask = get_mini_action_mask()

        flak_index = 0

        start, end = self.get_other_unit_action_slice(
            flak_index
        )

        self.assertGreater(
            np.sum(mask[start:end]),
            0.0,
            "Flak 88 should have legal deployment actions"
        )

    def test_no_flak_in_reserve_disables_flak_actions(self):
        strategic_reserve_box.units.append(
            create_nebelwerfer()
        )

        mask = get_mini_action_mask()

        flak_index = 0

        start, end = self.get_other_unit_action_slice(
            flak_index
        )

        self.assertEqual(
            np.sum(mask[start:end]),
            0.0,
            "Flak 88 actions should be disabled when no Flak is in reserve"
        )

    def test_multiple_flaks_use_same_action_block(self):
        strategic_reserve_box.units.extend(
            [
                create_flak88(),
                create_flak88(),
                create_flak88(),
            ]
        )

        mask = get_mini_action_mask()

        flak_index = 0

        start, end = self.get_other_unit_action_slice(
            flak_index
        )

        flak_mask = mask[start:end]

        self.assertGreater(
            np.sum(flak_mask),
            0.0
        )

        # The action block is still exactly one destination
        # block, irrespective of how many physical Flak
        # counters are present.
        self.assertEqual(
            len(flak_mask),
            27
        )

    def test_kampfgruppe_in_reserve_enables_only_kampfgruppe_block(self):
        strategic_reserve_box.units.append(
            create_kampfgruppe()
        )

        mask = get_mini_action_mask()

        flak_start, flak_end = (
            self.get_other_unit_action_slice(0)
        )

        kg_start, kg_end = (
            self.get_other_unit_action_slice(1)
        )

        neb_start, neb_end = (
            self.get_other_unit_action_slice(2)
        )

        self.assertEqual(
            np.sum(mask[flak_start:flak_end]),
            0.0
        )

        self.assertGreater(
            np.sum(mask[kg_start:kg_end]),
            0.0
        )

        self.assertEqual(
            np.sum(mask[neb_start:neb_end]),
            0.0
        )

    def test_nebelwerfer_in_reserve_enables_nebelwerfer_actions(self):
        strategic_reserve_box.units.append(
            create_nebelwerfer()
        )

        mask = get_mini_action_mask()

        nebelwerfer_index = 2

        start, end = self.get_other_unit_action_slice(
            nebelwerfer_index
        )

        self.assertGreater(
            np.sum(mask[start:end]),
            0.0
        )

    # ---------------------------------------------------------
    # NAMED OTHER UNITS FROM STRATEGIC RESERVE
    # ---------------------------------------------------------

    def test_fs3_in_reserve_enables_fs3_but_not_fs5(self):
        strategic_reserve_box.units.append(FS_3)

        mask = get_mini_action_mask()

        fs3_start, fs3_end = (
            self.get_other_unit_action_slice(3)
        )

        fs5_start, fs5_end = (
            self.get_other_unit_action_slice(4)
        )

        self.assertGreater(
            np.sum(mask[fs3_start:fs3_end]),
            0.0,
            "3rd Fallschirmjager should have legal deployment actions"
        )

        self.assertEqual(
            np.sum(mask[fs5_start:fs5_end]),
            0.0,
            "5th Fallschirmjager should remain unavailable"
        )

    def test_fs5_in_reserve_enables_fs5_but_not_fs3(self):
        strategic_reserve_box.units.append(FS_5)

        mask = get_mini_action_mask()

        fs3_start, fs3_end = (
            self.get_other_unit_action_slice(3)
        )

        fs5_start, fs5_end = (
            self.get_other_unit_action_slice(4)
        )

        self.assertEqual(
            np.sum(mask[fs3_start:fs3_end]),
            0.0
        )

        self.assertGreater(
            np.sum(mask[fs5_start:fs5_end]),
            0.0
        )

    def test_named_commanders_have_separate_action_blocks(self):
        strategic_reserve_box.units.append(ROMMEL)

        mask = get_mini_action_mask()

        rommel_start, rommel_end = (
            self.get_other_unit_action_slice(5)
        )

        meyer_start, meyer_end = (
            self.get_other_unit_action_slice(6)
        )

        model_start, model_end = (
            self.get_other_unit_action_slice(7)
        )

        self.assertGreater(
            np.sum(mask[rommel_start:rommel_end]),
            0.0
        )

        self.assertEqual(
            np.sum(mask[meyer_start:meyer_end]),
            0.0
        )

        self.assertEqual(
            np.sum(mask[model_start:model_end]),
            0.0
        )

    def test_tiger_101_has_own_action_block(self):
        strategic_reserve_box.units.append(TIGER_101)

        mask = get_mini_action_mask()

        tiger_start, tiger_end = (
            self.get_other_unit_action_slice(8)
        )

        self.assertGreater(
            np.sum(mask[tiger_start:tiger_end]),
            0.0
        )

    # ---------------------------------------------------------
    # ZERO AP BEHAVIOUR
    # ---------------------------------------------------------

    def test_other_reserve_units_remain_legal_with_zero_ap(self):
        """
        Other-unit deployment from Strategic Reserve costs
        no Action Point, so these actions should remain legal
        when actions_left_this_turn == 0.
        """

        GlobalGameState.actions_left_this_turn = 0

        strategic_reserve_box.units.append(
            create_flak88()
        )

        mask = get_mini_action_mask()

        start, end = self.get_other_unit_action_slice(0)

        self.assertGreater(
            np.sum(mask[start:end]),
            0.0,
            "Zero-cost Flak deployment should remain legal with 0 AP"
        )

        # Counter-attacks should still be disabled.
        self.assertEqual(
            np.sum(mask[1:7]),
            0.0
        )

    # ---------------------------------------------------------
    # ACTION BLOCK STRUCTURE
    # ---------------------------------------------------------

    def test_other_reserve_action_block_size(self):
        _, german_eligible_spaces = get_total_actions()

        expected_size = (
            len(OTHER_RESERVE_UNIT_TYPES)
            * len(german_eligible_spaces)
        )

        self.assertEqual(
            expected_size,
            243
        )

    def test_other_reserve_block_ends_at_total_actions(self):
        total_actions, german_eligible_spaces = (
            get_total_actions()
        )

        other_start = self.get_other_reserve_start()

        other_end = (
            other_start
            + len(OTHER_RESERVE_UNIT_TYPES)
            * len(german_eligible_spaces)
        )

        self.assertEqual(
            other_start,
            315
        )

        self.assertEqual(
            other_end,
            total_actions - 3
        )

        self.assertEqual(
            other_end,
            558
        )

    # ---------------------------------------------------------
    # AUGMENTATION ROLL
    # ---------------------------------------------------------

    def test_augmentation_roll_mask_all_available(self):

        transport_track.value = transport_track.maximum - 1
        supply_track.value = supply_track.maximum - 1
        hitler_approval_track.value = hitler_approval_track.maximum - 1

        mask = augmentation_roll_mask()

        self.assertEqual(mask.shape, (3,))
        self.assertEqual(mask.dtype, np.float32)

        np.testing.assert_array_equal(
            mask,
            np.array(
                [1.0, 1.0, 1.0],
                dtype=np.float32
            )
        )

    def test_augmentation_roll_mask_blocks_tracks_at_maximum(self):

        transport_track.value = transport_track.maximum
        supply_track.value = supply_track.maximum - 1
        hitler_approval_track.value = hitler_approval_track.maximum

        mask = augmentation_roll_mask()

        np.testing.assert_array_equal(
            mask,
            np.array(
                [0.0, 1.0, 0.0],
                dtype=np.float32
            )
        )

    def test_augmentation_roll_mask_all_at_maximum(self):

        transport_track.value = transport_track.maximum
        supply_track.value = supply_track.maximum
        hitler_approval_track.value = hitler_approval_track.maximum

        mask = augmentation_roll_mask()

        np.testing.assert_array_equal(
            mask,
            np.array(
                [0.0, 0.0, 0.0],
                dtype=np.float32
            )
        )

if __name__ == "__main__":
    unittest.main()

# tests/resources/test_augmentation_rolls.py

import unittest

from core.actions.resource_actions import (
    do_resource_augmentation_roll,
)

from core.global_game_state import GlobalGameState

from core.map.map_model import (
    transport_track,
    supply_track,
    hitler_approval_track,
)


class TestAugmentationRolls(unittest.TestCase):

    def setUp(self):

        # Save original state.

        self.original_transport_value = transport_track.value
        self.original_supply_value = supply_track.value
        self.original_hitler_approval_value = hitler_approval_track.value

        self.original_transport_base = (
            GlobalGameState.transport_base_level
        )
        self.original_supply_base = (
            GlobalGameState.supply_base_level
        )
        self.original_hitler_base = (
            GlobalGameState.hitler_approval_base_level
        )

        self.original_transport_drm = (
            GlobalGameState.transport_roll_drm
        )
        self.original_supply_drm = (
            GlobalGameState.supply_roll_drm
        )

        self.original_actions = (
            GlobalGameState.actions_left_this_turn
        )
        self.original_reserve_actions = (
            GlobalGameState.reserve_actions
        )

        # Standard deterministic test state.

        transport_track.value = 3
        supply_track.value = 3
        hitler_approval_track.value = 3

        GlobalGameState.transport_base_level = 3
        GlobalGameState.supply_base_level = 3
        GlobalGameState.hitler_approval_base_level = 3

        GlobalGameState.transport_roll_drm = 0
        GlobalGameState.supply_roll_drm = 0

        GlobalGameState.actions_left_this_turn = 10
        GlobalGameState.reserve_actions = 0

    def tearDown(self):

        transport_track.value = self.original_transport_value
        supply_track.value = self.original_supply_value
        hitler_approval_track.value = (
            self.original_hitler_approval_value
        )

        GlobalGameState.transport_base_level = (
            self.original_transport_base
        )
        GlobalGameState.supply_base_level = (
            self.original_supply_base
        )
        GlobalGameState.hitler_approval_base_level = (
            self.original_hitler_base
        )

        GlobalGameState.transport_roll_drm = (
            self.original_transport_drm
        )
        GlobalGameState.supply_roll_drm = (
            self.original_supply_drm
        )

        GlobalGameState.actions_left_this_turn = (
            self.original_actions
        )
        GlobalGameState.reserve_actions = (
            self.original_reserve_actions
        )

    # ---------------------------------------------------------
    # TRANSPORT
    # ---------------------------------------------------------

    def test_transport_roll_without_drm(self):

        # Base level = 3.
        # Roll 3 fails because the roll must EXCEED base level.

        do_resource_augmentation_roll(
            choice=1,
            die_roll=3
        )

        self.assertEqual(
            transport_track.value,
            3
        )

        # Reset AP because the failed augmentation roll
        # still spends an action.

        GlobalGameState.actions_left_this_turn = 10

        # Roll 4 succeeds.

        do_resource_augmentation_roll(
            choice=1,
            die_roll=4
        )

        self.assertEqual(
            transport_track.value,
            4
        )

    def test_transport_roll_with_drm(self):

        GlobalGameState.transport_roll_drm = 1

        # Roll 3 + 1 DRM = 4.
        # 4 exceeds base level 3, so Transport increases.

        do_resource_augmentation_roll(
            choice=1,
            die_roll=3
        )

        self.assertEqual(
            transport_track.value,
            4
        )

    # ---------------------------------------------------------
    # SUPPLY
    # ---------------------------------------------------------

    def test_supply_roll_without_drm(self):

        # Base level = 3.
        # Roll 3 fails.

        do_resource_augmentation_roll(
            choice=2,
            die_roll=3
        )

        self.assertEqual(
            supply_track.value,
            3
        )

        GlobalGameState.actions_left_this_turn = 10

        # Roll 4 succeeds.

        do_resource_augmentation_roll(
            choice=2,
            die_roll=4
        )

        self.assertEqual(
            supply_track.value,
            4
        )

    def test_supply_roll_with_drm(self):

        GlobalGameState.supply_roll_drm = 1

        # Roll 3 + 1 DRM = 4.
        # 4 exceeds base level 3.

        do_resource_augmentation_roll(
            choice=2,
            die_roll=3
        )

        self.assertEqual(
            supply_track.value,
            4
        )

    # ---------------------------------------------------------
    # HITLER APPROVAL
    # ---------------------------------------------------------

    def test_hitler_approval_before_assassination(self):

        # Before Hitler Assassination:
        # base level = 3.

        GlobalGameState.hitler_approval_base_level = 3

        # Roll 3 fails.

        do_resource_augmentation_roll(
            choice=3,
            die_roll=3
        )

        self.assertEqual(
            hitler_approval_track.value,
            3
        )

        GlobalGameState.actions_left_this_turn = 10

        # Roll 4 succeeds.

        do_resource_augmentation_roll(
            choice=3,
            die_roll=4
        )

        self.assertEqual(
            hitler_approval_track.value,
            4
        )

    def test_hitler_approval_after_assassination(self):

        # Hitler Assassination changes the base level
        # from 3 to 4.

        GlobalGameState.hitler_approval_base_level = 4

        # Roll 4 now fails.

        do_resource_augmentation_roll(
            choice=3,
            die_roll=4
        )

        self.assertEqual(
            hitler_approval_track.value,
            3
        )

        GlobalGameState.actions_left_this_turn = 10

        # Roll 5 succeeds.

        do_resource_augmentation_roll(
            choice=3,
            die_roll=5
        )

        self.assertEqual(
            hitler_approval_track.value,
            4
        )

    # ---------------------------------------------------------
    # MAXIMUM RESOURCE LEVEL
    # ---------------------------------------------------------

    def test_transport_cannot_augment_at_maximum(self):

        transport_track.value = transport_track.maximum

        actions_before = (
            GlobalGameState.actions_left_this_turn
        )

        result = do_resource_augmentation_roll(
            choice=1,
            die_roll=6
        )

        self.assertFalse(result)

        self.assertEqual(
            transport_track.value,
            transport_track.maximum
        )

        # Illegal augmentation must not spend an action.

        self.assertEqual(
            GlobalGameState.actions_left_this_turn,
            actions_before
        )

    def test_supply_cannot_augment_at_maximum(self):

        supply_track.value = supply_track.maximum

        actions_before = (
            GlobalGameState.actions_left_this_turn
        )

        result = do_resource_augmentation_roll(
            choice=2,
            die_roll=6
        )

        self.assertFalse(result)

        self.assertEqual(
            supply_track.value,
            supply_track.maximum
        )

        self.assertEqual(
            GlobalGameState.actions_left_this_turn,
            actions_before
        )

    def test_hitler_approval_cannot_augment_at_maximum(self):

        hitler_approval_track.value = (
            hitler_approval_track.maximum
        )

        actions_before = (
            GlobalGameState.actions_left_this_turn
        )

        result = do_resource_augmentation_roll(
            choice=3,
            die_roll=6
        )

        self.assertFalse(result)

        self.assertEqual(
            hitler_approval_track.value,
            hitler_approval_track.maximum
        )

        self.assertEqual(
            GlobalGameState.actions_left_this_turn,
            actions_before
        )


if __name__ == "__main__":
    unittest.main()

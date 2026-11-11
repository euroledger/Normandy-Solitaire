
import copy
import unittest
from unittest.mock import patch

from core.ai import headless_action_masks as masks
from core.german_units import SS_12


class TestPanzerIdentityMask(unittest.TestCase):

    def test_duplicate_panzer_object_is_not_selectable(self):
        # Create a distinct object with identical unit attributes.
        duplicate = copy.copy(SS_12)

        # Confirm that the objects compare equal.
        self.assertEqual(duplicate, SS_12)

        # Confirm that they are NOT the same object.
        self.assertIsNot(duplicate, SS_12)

        # Find the action-mask index for 12th SS Panzer.
        panzer_index = next(
            i
            for i, unit in enumerate(masks.panzer_divisions_list)
            if unit is SS_12
        )

        # Simulate the map reporting the duplicate object.
        # No actual game state is modified.
        with patch.object(
            masks,
            "get_panzer_divisions_on_map",
            return_value=[(None, duplicate)]
        ):
            action_mask = (
                masks.get_move_panzer_to_strategic_reserve_mask()
            )

        # The executor requires the canonical SS_12 object.
        # Therefore the duplicate must not be selectable.
        self.assertEqual(
            action_mask[panzer_index],
            0.0,
            "BUG: Action mask accepts a different Panzer "
            "object because it compares by value rather "
            "than object identity."
        )

    def test_canonical_panzer_object_is_selectable(self):
        # Find the canonical 12th SS Panzer object.
        panzer_index = next(
            i
            for i, unit in enumerate(masks.panzer_divisions_list)
            if unit is SS_12
        )

        # Simulate the canonical object being on the map.
        with patch.object(
            masks,
            "get_panzer_divisions_on_map",
            return_value=[(None, SS_12)]
        ):
            action_mask = (
                masks.get_move_panzer_to_strategic_reserve_mask()
            )

        self.assertEqual(
            action_mask[panzer_index],
            1.0,
            "Canonical Panzer object should be selectable."
        )


if __name__ == "__main__":
    unittest.main()

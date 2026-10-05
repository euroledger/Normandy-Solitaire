import unittest

import numpy as np

from core.ai.headless_action_masks import (
    GERMAN_LOSS_TYPES,
    get_casualty_mask,
    get_casualty_from_action,
    select_random_casualty_ai,
)
from core.enums import ReinforcementType
from core.german_units import (
    PZ_LEHR,
    SS_12,
    PZ_21,
    FS_3,
    FS_5,
    create_flak88,
    create_kampfgruppe,
    create_nebelwerfer,
)


class TestCasualtyMask(unittest.TestCase):

    def get_action_id(
        self,
        unit_type=None,
        named_unit=None,
        combat_value=None,
    ):
        for action_id, (
            _,
            loss_unit_type,
            loss_named_unit,
            loss_combat_value,
        ) in enumerate(GERMAN_LOSS_TYPES):

            if (
                loss_unit_type == unit_type
                and loss_named_unit is named_unit
                and loss_combat_value == combat_value
            ):
                return action_id

        self.fail(
            "Could not find casualty action in GERMAN_LOSS_TYPES"
        )

    def test_mask_many_different_units(self):
        flak = create_flak88()
        kampfgruppe = create_kampfgruppe()
        nebelwerfer = create_nebelwerfer()

        german_units = [
            PZ_LEHR,
            SS_12,
            PZ_21,
            FS_3,
            FS_5,
            flak,
            kampfgruppe,
            nebelwerfer,
        ]

        mask = get_casualty_mask(german_units)

        expected_actions = {
            self.get_action_id(
                named_unit=PZ_LEHR,
                combat_value=2,
            ),
            self.get_action_id(
                named_unit=SS_12,
                combat_value=2,
            ),
            self.get_action_id(
                named_unit=PZ_21,
                combat_value=2,
            ),
            self.get_action_id(
                named_unit=FS_3,
                combat_value=2,
            ),
            self.get_action_id(
                named_unit=FS_5,
                combat_value=1,
            ),
            self.get_action_id(
                unit_type=ReinforcementType.FLAK_88,
                combat_value=2,
            ),
            self.get_action_id(
                unit_type=ReinforcementType.KAMPFGRUPPE,
                combat_value=1,
            ),
            self.get_action_id(
                unit_type=ReinforcementType.NEBELWERFER,
                combat_value=1,
            ),
        }

        legal_actions = set(
            np.where(mask == 1.0)[0]
        )

        self.assertEqual(
            legal_actions,
            expected_actions,
        )

        self.assertEqual(
            int(mask.sum()),
            8,
        )

    def test_mask_distinguishes_full_and_reduced(self):
        flak_full = create_flak88()
        flak_reduced = create_flak88()

        flak_reduced.combat_value = 1

        original_lehr_value = PZ_LEHR.combat_value
        original_fs3_value = FS_3.combat_value

        try:
            PZ_LEHR.combat_value = 1
            FS_3.combat_value = 1

            german_units = [
                PZ_LEHR,
                FS_3,
                flak_full,
                flak_reduced,
            ]

            mask = get_casualty_mask(german_units)

            self.assertEqual(
                mask[
                    self.get_action_id(
                        named_unit=PZ_LEHR,
                        combat_value=1,
                    )
                ],
                1.0,
            )

            self.assertEqual(
                mask[
                    self.get_action_id(
                        named_unit=PZ_LEHR,
                        combat_value=2,
                    )
                ],
                0.0,
            )

            self.assertEqual(
                mask[
                    self.get_action_id(
                        named_unit=FS_3,
                        combat_value=1,
                    )
                ],
                1.0,
            )

            self.assertEqual(
                mask[
                    self.get_action_id(
                        named_unit=FS_3,
                        combat_value=2,
                    )
                ],
                0.0,
            )

            self.assertEqual(
                mask[
                    self.get_action_id(
                        unit_type=ReinforcementType.FLAK_88,
                        combat_value=2,
                    )
                ],
                1.0,
            )

            self.assertEqual(
                mask[
                    self.get_action_id(
                        unit_type=ReinforcementType.FLAK_88,
                        combat_value=1,
                    )
                ],
                1.0,
            )

            self.assertEqual(
                int(mask.sum()),
                4,
            )

        finally:
            PZ_LEHR.combat_value = original_lehr_value
            FS_3.combat_value = original_fs3_value

    def test_get_casualty_from_action_named_unit(self):
        german_units = [
            PZ_LEHR,
            SS_12,
            PZ_21,
        ]

        action_id = self.get_action_id(
            named_unit=SS_12,
            combat_value=2,
        )

        casualty = get_casualty_from_action(
            action_id,
            german_units,
        )

        self.assertIs(
            casualty,
            SS_12,
        )

    def test_get_casualty_from_action_generic_unit(self):
        flak_full = create_flak88()
        flak_reduced = create_flak88()

        flak_reduced.combat_value = 1

        german_units = [
            PZ_LEHR,
            flak_full,
            flak_reduced,
            create_kampfgruppe(),
        ]

        action_id = self.get_action_id(
            unit_type=ReinforcementType.FLAK_88,
            combat_value=1,
        )

        casualty = get_casualty_from_action(
            action_id,
            german_units,
        )

        self.assertIs(
            casualty,
            flak_reduced,
        )

        self.assertEqual(
            casualty.combat_value,
            1,
        )

    def test_random_casualty_is_always_eligible(self):
        flak = create_flak88()
        kampfgruppe = create_kampfgruppe()
        nebelwerfer = create_nebelwerfer()

        german_units = [
            PZ_LEHR,
            SS_12,
            FS_3,
            FS_5,
            flak,
            kampfgruppe,
            nebelwerfer,
        ]

        for _ in range(500):
            casualty = select_random_casualty_ai(
                german_units
            )

            self.assertIn(
                casualty,
                german_units,
            )

    def test_random_casualty_uses_multiple_choices(self):
        flak = create_flak88()
        kampfgruppe = create_kampfgruppe()
        nebelwerfer = create_nebelwerfer()

        german_units = [
            PZ_LEHR,
            SS_12,
            FS_3,
            FS_5,
            flak,
            kampfgruppe,
            nebelwerfer,
        ]

        casualties_seen = set()

        for _ in range(1000):
            casualty = select_random_casualty_ai(
                german_units
            )

            casualties_seen.add(
                id(casualty)
            )

        self.assertGreater(
            len(casualties_seen),
            1,
        )


if __name__ == "__main__":
    unittest.main()

import unittest

import numpy as np

from core.models import GermanUnit
from core.enums import ReinforcementType

from core.map.map_model import (
    strategic_reserve_box,
    eliminated_units_box,
)

from core.ai.dqn.dqn_game_state_encoder import (
    OBSERVATION_SIZE,
    GENERIC_BOX_UNITS_START,
    GENERIC_BOX_UNITS_END,
    GENERIC_FEATURES_PER_BOX,
    MAX_TOTAL_FLAK,
    MAX_TOTAL_KAMPFGRUPPE,
    MAX_TOTAL_NEBELWERFER,
    encode_generic_box_units,
)


class TestEncodeGenericBoxUnits(unittest.TestCase):

    def setUp(self):
        # Remove generic units from the two boxes where they can legally occur.

        for box in [
            strategic_reserve_box,
            eliminated_units_box,
        ]:
            box.units[:] = [
                unit
                for unit in box.units
                if not (
                    isinstance(unit, GermanUnit)
                    and unit.type in {
                        ReinforcementType.FLAK_88,
                        ReinforcementType.KAMPFGRUPPE,
                        ReinforcementType.NEBELWERFER,
                    }
                )
            ]

        self.observation = np.zeros(
            OBSERVATION_SIZE,
            dtype=np.float32,
        )

    def test_encode_generic_box_units_empty_boxes(self):
        encode_generic_box_units(self.observation)

        box_state = self.observation[
            GENERIC_BOX_UNITS_START:
            GENERIC_BOX_UNITS_END
        ]

        self.assertEqual(
            np.count_nonzero(box_state),
            0,
        )

    def test_encode_generic_box_units_in_transit_slots_remain_zero(self):
        encode_generic_box_units(self.observation)

        in_transit_state = self.observation[
            GENERIC_BOX_UNITS_START:
            GENERIC_BOX_UNITS_START + GENERIC_FEATURES_PER_BOX
        ]

        self.assertEqual(
            np.count_nonzero(in_transit_state),
            0,
        )

    def test_encode_generic_box_units_strategic_reserve_full_flak(self):
        strategic_reserve_box.units.append(
            GermanUnit(
                type=ReinforcementType.FLAK_88,
                combat_value=2,
            )
        )

        encode_generic_box_units(self.observation)

        start = (
            GENERIC_BOX_UNITS_START
            + GENERIC_FEATURES_PER_BOX
        )

        self.assertAlmostEqual(
            self.observation[start],
            1 / MAX_TOTAL_FLAK,
        )

        self.assertEqual(
            self.observation[start + 1],
            0.0,
        )

    def test_encode_generic_box_units_strategic_reserve_reduced_flak(self):
        strategic_reserve_box.units.append(
            GermanUnit(
                type=ReinforcementType.FLAK_88,
                combat_value=1,
            )
        )

        encode_generic_box_units(self.observation)

        start = (
            GENERIC_BOX_UNITS_START
            + GENERIC_FEATURES_PER_BOX
        )

        self.assertEqual(
            self.observation[start],
            0.0,
        )

        self.assertAlmostEqual(
            self.observation[start + 1],
            1 / MAX_TOTAL_FLAK,
        )

    def test_encode_generic_box_units_strategic_reserve_kampfgruppe(self):
        strategic_reserve_box.units.append(
            GermanUnit(
                type=ReinforcementType.KAMPFGRUPPE,
                combat_value=1,
            )
        )

        encode_generic_box_units(self.observation)

        start = (
            GENERIC_BOX_UNITS_START
            + GENERIC_FEATURES_PER_BOX
        )

        self.assertAlmostEqual(
            self.observation[start + 2],
            1 / MAX_TOTAL_KAMPFGRUPPE,
        )

    def test_encode_generic_box_units_strategic_reserve_nebelwerfer(self):
        strategic_reserve_box.units.append(
            GermanUnit(
                type=ReinforcementType.NEBELWERFER,
                combat_value=1,
            )
        )

        encode_generic_box_units(self.observation)

        start = (
            GENERIC_BOX_UNITS_START
            + GENERIC_FEATURES_PER_BOX
        )

        self.assertAlmostEqual(
            self.observation[start + 3],
            1 / MAX_TOTAL_NEBELWERFER,
        )

    def test_encode_generic_box_units_strategic_reserve_mixed_counts(self):
        for _ in range(2):
            strategic_reserve_box.units.append(
                GermanUnit(
                    type=ReinforcementType.FLAK_88,
                    combat_value=2,
                )
            )

        strategic_reserve_box.units.append(
            GermanUnit(
                type=ReinforcementType.FLAK_88,
                combat_value=1,
            )
        )

        for _ in range(3):
            strategic_reserve_box.units.append(
                GermanUnit(
                    type=ReinforcementType.KAMPFGRUPPE,
                    combat_value=1,
                )
            )

        for _ in range(2):
            strategic_reserve_box.units.append(
                GermanUnit(
                    type=ReinforcementType.NEBELWERFER,
                    combat_value=1,
                )
            )

        encode_generic_box_units(self.observation)

        start = (
            GENERIC_BOX_UNITS_START
            + GENERIC_FEATURES_PER_BOX
        )

        self.assertAlmostEqual(
            self.observation[start],
            2 / MAX_TOTAL_FLAK,
        )

        self.assertAlmostEqual(
            self.observation[start + 1],
            1 / MAX_TOTAL_FLAK,
        )

        self.assertAlmostEqual(
            self.observation[start + 2],
            3 / MAX_TOTAL_KAMPFGRUPPE,
        )

        self.assertAlmostEqual(
            self.observation[start + 3],
            2 / MAX_TOTAL_NEBELWERFER,
        )

    def test_encode_generic_box_units_eliminated_mixed_counts(self):
        eliminated_units_box.units.append(
            GermanUnit(
                type=ReinforcementType.FLAK_88,
                combat_value=2,
            )
        )

        eliminated_units_box.units.append(
            GermanUnit(
                type=ReinforcementType.FLAK_88,
                combat_value=1,
            )
        )

        for _ in range(2):
            eliminated_units_box.units.append(
                GermanUnit(
                    type=ReinforcementType.KAMPFGRUPPE,
                    combat_value=1,
                )
            )

        for _ in range(3):
            eliminated_units_box.units.append(
                GermanUnit(
                    type=ReinforcementType.NEBELWERFER,
                    combat_value=1,
                )
            )

        encode_generic_box_units(self.observation)

        start = (
            GENERIC_BOX_UNITS_START
            + 2 * GENERIC_FEATURES_PER_BOX
        )

        self.assertAlmostEqual(
            self.observation[start],
            1 / MAX_TOTAL_FLAK,
        )

        self.assertAlmostEqual(
            self.observation[start + 1],
            1 / MAX_TOTAL_FLAK,
        )

        self.assertAlmostEqual(
            self.observation[start + 2],
            2 / MAX_TOTAL_KAMPFGRUPPE,
        )

        self.assertAlmostEqual(
            self.observation[start + 3],
            3 / MAX_TOTAL_NEBELWERFER,
        )

    def test_encode_generic_box_units_reserve_and_eliminated_are_independent(self):
        strategic_reserve_box.units.append(
            GermanUnit(
                type=ReinforcementType.FLAK_88,
                combat_value=2,
            )
        )

        eliminated_units_box.units.append(
            GermanUnit(
                type=ReinforcementType.NEBELWERFER,
                combat_value=1,
            )
        )

        encode_generic_box_units(self.observation)

        reserve_start = (
            GENERIC_BOX_UNITS_START
            + GENERIC_FEATURES_PER_BOX
        )

        eliminated_start = (
            GENERIC_BOX_UNITS_START
            + 2 * GENERIC_FEATURES_PER_BOX
        )

        self.assertAlmostEqual(
            self.observation[reserve_start],
            1 / MAX_TOTAL_FLAK,
        )

        self.assertEqual(
            self.observation[reserve_start + 3],
            0.0,
        )

        self.assertEqual(
            self.observation[eliminated_start],
            0.0,
        )

        self.assertAlmostEqual(
            self.observation[eliminated_start + 3],
            1 / MAX_TOTAL_NEBELWERFER,
        )

    def test_encode_generic_box_units_only_changes_generic_box_block(self):
        strategic_reserve_box.units.append(
            GermanUnit(
                type=ReinforcementType.KAMPFGRUPPE,
                combat_value=1,
            )
        )

        eliminated_units_box.units.append(
            GermanUnit(
                type=ReinforcementType.NEBELWERFER,
                combat_value=1,
            )
        )

        encode_generic_box_units(self.observation)

        before_block = self.observation[
            :GENERIC_BOX_UNITS_START
        ]

        after_block = self.observation[
            GENERIC_BOX_UNITS_END:
        ]

        self.assertEqual(
            np.count_nonzero(before_block),
            0,
        )

        self.assertEqual(
            np.count_nonzero(after_block),
            0,
        )


if __name__ == "__main__":
    unittest.main()

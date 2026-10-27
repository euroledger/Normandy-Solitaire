import unittest

import numpy as np

from core.models import GermanUnit
from core.enums import ReinforcementType

from core.map.map_spaces_brit_2 import bayeux
from core.map.map_spaces_us_1 import carentan

from core.ai.dqn.dqn_game_state_encoder import (
    OBSERVATION_SIZE,
    GENERIC_MAP_UNITS_START,
    GENERIC_MAP_UNITS_END,
    GENERIC_FEATURES_PER_SPACE,
    GERMAN_MAP_SPACES,
    MAX_FLAK_STACK,
    MAX_KAMPFGRUPPE_STACK,
    MAX_NEBELWERFER_STACK,
    encode_generic_map_units,
)


class TestEncodeGenericMapUnits(unittest.TestCase):

    def setUp(self):
        # Remove generic units from all encoded map spaces.

        for space in GERMAN_MAP_SPACES:
            space.units[:] = [
                unit
                for unit in space.units
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

    def test_encode_generic_map_units_empty_space(self):
        encode_generic_map_units(self.observation)

        bayeux_index = GERMAN_MAP_SPACES.index(bayeux)

        start = (
            GENERIC_MAP_UNITS_START
            + bayeux_index * GENERIC_FEATURES_PER_SPACE
        )

        self.assertEqual(
            self.observation[start],
            0.0,
        )

        self.assertEqual(
            self.observation[start + 1],
            0.0,
        )

        self.assertEqual(
            self.observation[start + 2],
            0.0,
        )

        self.assertEqual(
            self.observation[start + 3],
            0.0,
        )

    def test_encode_generic_map_units_full_flak(self):
        bayeux.units.append(
            GermanUnit(
                type=ReinforcementType.FLAK_88,
                combat_value=2,
            )
        )

        encode_generic_map_units(self.observation)

        bayeux_index = GERMAN_MAP_SPACES.index(bayeux)

        start = (
            GENERIC_MAP_UNITS_START
            + bayeux_index * GENERIC_FEATURES_PER_SPACE
        )

        self.assertAlmostEqual(
            self.observation[start],
            1 / MAX_FLAK_STACK,
        )

        self.assertEqual(
            self.observation[start + 1],
            0.0,
        )

    def test_encode_generic_map_units_reduced_flak(self):
        bayeux.units.append(
            GermanUnit(
                type=ReinforcementType.FLAK_88,
                combat_value=1,
            )
        )

        encode_generic_map_units(self.observation)

        bayeux_index = GERMAN_MAP_SPACES.index(bayeux)

        start = (
            GENERIC_MAP_UNITS_START
            + bayeux_index * GENERIC_FEATURES_PER_SPACE
        )

        self.assertEqual(
            self.observation[start],
            0.0,
        )

        self.assertAlmostEqual(
            self.observation[start + 1],
            1 / MAX_FLAK_STACK,
        )

    def test_encode_generic_map_units_full_and_reduced_flak_separate(self):
        bayeux.units.append(
            GermanUnit(
                type=ReinforcementType.FLAK_88,
                combat_value=2,
            )
        )

        bayeux.units.append(
            GermanUnit(
                type=ReinforcementType.FLAK_88,
                combat_value=1,
            )
        )

        encode_generic_map_units(self.observation)

        bayeux_index = GERMAN_MAP_SPACES.index(bayeux)

        start = (
            GENERIC_MAP_UNITS_START
            + bayeux_index * GENERIC_FEATURES_PER_SPACE
        )

        self.assertAlmostEqual(
            self.observation[start],
            1 / MAX_FLAK_STACK,
        )

        self.assertAlmostEqual(
            self.observation[start + 1],
            1 / MAX_FLAK_STACK,
        )

    def test_encode_generic_map_units_kampfgruppe(self):
        bayeux.units.append(
            GermanUnit(
                type=ReinforcementType.KAMPFGRUPPE,
                combat_value=1,
            )
        )

        encode_generic_map_units(self.observation)

        bayeux_index = GERMAN_MAP_SPACES.index(bayeux)

        start = (
            GENERIC_MAP_UNITS_START
            + bayeux_index * GENERIC_FEATURES_PER_SPACE
        )

        self.assertAlmostEqual(
            self.observation[start + 2],
            1 / MAX_KAMPFGRUPPE_STACK,
        )

    def test_encode_generic_map_units_nebelwerfer(self):
        bayeux.units.append(
            GermanUnit(
                type=ReinforcementType.NEBELWERFER,
                combat_value=1,
            )
        )

        encode_generic_map_units(self.observation)

        bayeux_index = GERMAN_MAP_SPACES.index(bayeux)

        start = (
            GENERIC_MAP_UNITS_START
            + bayeux_index * GENERIC_FEATURES_PER_SPACE
        )

        self.assertAlmostEqual(
            self.observation[start + 3],
            1 / MAX_NEBELWERFER_STACK,
        )

    def test_encode_generic_map_units_normalizes_maximum_stacks(self):
        for _ in range(MAX_FLAK_STACK):
            bayeux.units.append(
                GermanUnit(
                    type=ReinforcementType.FLAK_88,
                    combat_value=2,
                )
            )

        for _ in range(MAX_FLAK_STACK):
            bayeux.units.append(
                GermanUnit(
                    type=ReinforcementType.FLAK_88,
                    combat_value=1,
                )
            )

        for _ in range(MAX_KAMPFGRUPPE_STACK):
            bayeux.units.append(
                GermanUnit(
                    type=ReinforcementType.KAMPFGRUPPE,
                    combat_value=1,
                )
            )

        for _ in range(MAX_NEBELWERFER_STACK):
            bayeux.units.append(
                GermanUnit(
                    type=ReinforcementType.NEBELWERFER,
                    combat_value=1,
                )
            )

        encode_generic_map_units(self.observation)

        bayeux_index = GERMAN_MAP_SPACES.index(bayeux)

        start = (
            GENERIC_MAP_UNITS_START
            + bayeux_index * GENERIC_FEATURES_PER_SPACE
        )

        self.assertEqual(
            self.observation[start],
            1.0,
        )

        self.assertEqual(
            self.observation[start + 1],
            1.0,
        )

        self.assertEqual(
            self.observation[start + 2],
            1.0,
        )

        self.assertEqual(
            self.observation[start + 3],
            1.0,
        )

    def test_encode_generic_map_units_mixed_counts(self):
        for _ in range(2):
            bayeux.units.append(
                GermanUnit(
                    type=ReinforcementType.FLAK_88,
                    combat_value=2,
                )
            )

        bayeux.units.append(
            GermanUnit(
                type=ReinforcementType.FLAK_88,
                combat_value=1,
            )
        )

        for _ in range(3):
            bayeux.units.append(
                GermanUnit(
                    type=ReinforcementType.KAMPFGRUPPE,
                    combat_value=1,
                )
            )

        for _ in range(2):
            bayeux.units.append(
                GermanUnit(
                    type=ReinforcementType.NEBELWERFER,
                    combat_value=1,
                )
            )

        encode_generic_map_units(self.observation)

        bayeux_index = GERMAN_MAP_SPACES.index(bayeux)

        start = (
            GENERIC_MAP_UNITS_START
            + bayeux_index * GENERIC_FEATURES_PER_SPACE
        )

        self.assertAlmostEqual(
            self.observation[start],
            2 / 3,
        )

        self.assertAlmostEqual(
            self.observation[start + 1],
            1 / 3,
        )

        self.assertAlmostEqual(
            self.observation[start + 2],
            3 / 4,
        )

        self.assertAlmostEqual(
            self.observation[start + 3],
            2 / 3,
        )

    def test_encode_generic_map_units_spaces_are_independent(self):
        bayeux.units.append(
            GermanUnit(
                type=ReinforcementType.FLAK_88,
                combat_value=2,
            )
        )

        carentan.units.append(
            GermanUnit(
                type=ReinforcementType.KAMPFGRUPPE,
                combat_value=1,
            )
        )

        encode_generic_map_units(self.observation)

        bayeux_index = GERMAN_MAP_SPACES.index(bayeux)
        carentan_index = GERMAN_MAP_SPACES.index(carentan)

        bayeux_start = (
            GENERIC_MAP_UNITS_START
            + bayeux_index * GENERIC_FEATURES_PER_SPACE
        )

        carentan_start = (
            GENERIC_MAP_UNITS_START
            + carentan_index * GENERIC_FEATURES_PER_SPACE
        )

        self.assertAlmostEqual(
            self.observation[bayeux_start],
            1 / 3,
        )

        self.assertEqual(
            self.observation[bayeux_start + 2],
            0.0,
        )

        self.assertEqual(
            self.observation[carentan_start],
            0.0,
        )

        self.assertAlmostEqual(
            self.observation[carentan_start + 2],
            1 / 4,
        )

    def test_encode_generic_map_units_named_units_are_ignored(self):
        named_unit = GermanUnit(
            type=ReinforcementType.PZ_DIV,
            name="TEST PANZER",
            combat_value=2,
        )

        bayeux.units.append(named_unit)

        encode_generic_map_units(self.observation)

        bayeux_index = GERMAN_MAP_SPACES.index(bayeux)

        start = (
            GENERIC_MAP_UNITS_START
            + bayeux_index * GENERIC_FEATURES_PER_SPACE
        )

        space_state = self.observation[
            start:
            start + GENERIC_FEATURES_PER_SPACE
        ]

        self.assertEqual(
            np.count_nonzero(space_state),
            0,
        )

    def test_encode_generic_map_units_only_changes_generic_map_block(self):
        bayeux.units.append(
            GermanUnit(
                type=ReinforcementType.FLAK_88,
                combat_value=2,
            )
        )

        carentan.units.append(
            GermanUnit(
                type=ReinforcementType.NEBELWERFER,
                combat_value=1,
            )
        )

        encode_generic_map_units(self.observation)

        before_block = self.observation[
            :GENERIC_MAP_UNITS_START
        ]

        after_block = self.observation[
            GENERIC_MAP_UNITS_END:
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

import unittest

import numpy as np

from core.german_units import (
    FS_3,
    FS_5,
    TIGER_101,
)

from core.map.map_model import (
    in_transit_box,
    strategic_reserve_box,
    eliminated_units_box,
)

from core.map.map_spaces_brit_2 import bayeux
from core.map.map_spaces_us_1 import carentan

from core.ai.dqn.dqn_game_state_encoder import (
    OBSERVATION_SIZE,
    SPECIAL_GERMAN_UNITS_START,
    SPECIAL_GERMAN_UNITS_END,
    NAMED_UNIT_LOCATION_COUNT,
    FS3_START,
    FS3_REDUCED_INDEX,
    FS5_START,
    TIGER_101_START,
    LOCATION_NOT_AVAILABLE,
    LOCATION_IN_TRANSIT,
    LOCATION_STRATEGIC_RESERVE,
    LOCATION_ELIMINATED,
    GERMAN_LOCATION_INDEX,
    GERMAN_MAP_SPACES,
    encode_special_german_units,
)


class TestEncodeSpecialGermanUnits(unittest.TestCase):

    def setUp(self):
        self.special_units = [
            FS_3,
            FS_5,
            TIGER_101,
        ]

        # Remove the three units from all encoded map spaces.

        for space in GERMAN_MAP_SPACES:
            for unit in self.special_units:
                if unit in space.units:
                    space.units.remove(unit)

        # Remove them from all off-map boxes.

        for box in [
            in_transit_box,
            strategic_reserve_box,
            eliminated_units_box,
        ]:
            for unit in self.special_units:
                if unit in box.units:
                    box.units.remove(unit)

        # Restore normal combat values.

        FS_3.combat_value = 2
        FS_5.combat_value = 1
        TIGER_101.combat_value = 3

        self.observation = np.zeros(
            OBSERVATION_SIZE,
            dtype=np.float32,
        )

    def test_encode_special_units_fs3_not_available(self):
        encode_special_german_units(self.observation)

        location_state = self.observation[
            FS3_START:
            FS3_START + NAMED_UNIT_LOCATION_COUNT
        ]

        self.assertEqual(
            np.count_nonzero(location_state),
            1,
        )

        self.assertEqual(
            self.observation[
                FS3_START + LOCATION_NOT_AVAILABLE
            ],
            1.0,
        )

    def test_encode_special_units_fs3_in_transit(self):
        in_transit_box.units.append(FS_3)

        encode_special_german_units(self.observation)

        self.assertEqual(
            self.observation[
                FS3_START + LOCATION_IN_TRANSIT
            ],
            1.0,
        )

        location_state = self.observation[
            FS3_START:
            FS3_START + NAMED_UNIT_LOCATION_COUNT
        ]

        self.assertEqual(
            np.count_nonzero(location_state),
            1,
        )

    def test_encode_special_units_fs3_strategic_reserve(self):
        strategic_reserve_box.units.append(FS_3)

        encode_special_german_units(self.observation)

        self.assertEqual(
            self.observation[
                FS3_START + LOCATION_STRATEGIC_RESERVE
            ],
            1.0,
        )

        location_state = self.observation[
            FS3_START:
            FS3_START + NAMED_UNIT_LOCATION_COUNT
        ]

        self.assertEqual(
            np.count_nonzero(location_state),
            1,
        )

    def test_encode_special_units_fs3_eliminated(self):
        eliminated_units_box.units.append(FS_3)

        encode_special_german_units(self.observation)

        self.assertEqual(
            self.observation[
                FS3_START + LOCATION_ELIMINATED
            ],
            1.0,
        )

        location_state = self.observation[
            FS3_START:
            FS3_START + NAMED_UNIT_LOCATION_COUNT
        ]

        self.assertEqual(
            np.count_nonzero(location_state),
            1,
        )

    def test_encode_special_units_fs3_map_location(self):
        bayeux.units.append(FS_3)

        encode_special_german_units(self.observation)

        bayeux_location_index = GERMAN_LOCATION_INDEX[id(bayeux)]

        self.assertEqual(
            self.observation[
                FS3_START + bayeux_location_index
            ],
            1.0,
        )

        location_state = self.observation[
            FS3_START:
            FS3_START + NAMED_UNIT_LOCATION_COUNT
        ]

        self.assertEqual(
            np.count_nonzero(location_state),
            1,
        )

    def test_encode_special_units_fs3_full_strength(self):
        strategic_reserve_box.units.append(FS_3)
        FS_3.combat_value = 2

        encode_special_german_units(self.observation)

        self.assertEqual(
            self.observation[FS3_REDUCED_INDEX],
            0.0,
        )

    def test_encode_special_units_fs3_reduced_strength(self):
        strategic_reserve_box.units.append(FS_3)
        FS_3.combat_value = 1

        encode_special_german_units(self.observation)

        self.assertEqual(
            self.observation[FS3_REDUCED_INDEX],
            1.0,
        )

    def test_encode_special_units_fs5_not_available(self):
        encode_special_german_units(self.observation)

        location_state = self.observation[
            FS5_START:
            FS5_START + NAMED_UNIT_LOCATION_COUNT
        ]

        self.assertEqual(
            np.count_nonzero(location_state),
            1,
        )

        self.assertEqual(
            self.observation[
                FS5_START + LOCATION_NOT_AVAILABLE
            ],
            1.0,
        )

    def test_encode_special_units_fs5_map_location(self):
        carentan.units.append(FS_5)

        encode_special_german_units(self.observation)

        carentan_location_index = GERMAN_LOCATION_INDEX[
            id(carentan)
        ]

        self.assertEqual(
            self.observation[
                FS5_START + carentan_location_index
            ],
            1.0,
        )

        location_state = self.observation[
            FS5_START:
            FS5_START + NAMED_UNIT_LOCATION_COUNT
        ]

        self.assertEqual(
            np.count_nonzero(location_state),
            1,
        )

    def test_encode_special_units_tiger_not_available(self):
        encode_special_german_units(self.observation)

        location_state = self.observation[
            TIGER_101_START:
            TIGER_101_START + NAMED_UNIT_LOCATION_COUNT
        ]

        self.assertEqual(
            np.count_nonzero(location_state),
            1,
        )

        self.assertEqual(
            self.observation[
                TIGER_101_START + LOCATION_NOT_AVAILABLE
            ],
            1.0,
        )

    def test_encode_special_units_tiger_strategic_reserve(self):
        strategic_reserve_box.units.append(TIGER_101)

        encode_special_german_units(self.observation)

        self.assertEqual(
            self.observation[
                TIGER_101_START + LOCATION_STRATEGIC_RESERVE
            ],
            1.0,
        )

        location_state = self.observation[
            TIGER_101_START:
            TIGER_101_START + NAMED_UNIT_LOCATION_COUNT
        ]

        self.assertEqual(
            np.count_nonzero(location_state),
            1,
        )

    def test_encode_special_units_tiger_map_location(self):
        bayeux.units.append(TIGER_101)

        encode_special_german_units(self.observation)

        bayeux_location_index = GERMAN_LOCATION_INDEX[id(bayeux)]

        self.assertEqual(
            self.observation[
                TIGER_101_START + bayeux_location_index
            ],
            1.0,
        )

        location_state = self.observation[
            TIGER_101_START:
            TIGER_101_START + NAMED_UNIT_LOCATION_COUNT
        ]

        self.assertEqual(
            np.count_nonzero(location_state),
            1,
        )

    def test_encode_special_units_all_three_locations_independent(self):
        in_transit_box.units.append(FS_3)
        carentan.units.append(FS_5)
        bayeux.units.append(TIGER_101)

        FS_3.combat_value = 1

        encode_special_german_units(self.observation)

        carentan_location_index = GERMAN_LOCATION_INDEX[
            id(carentan)
        ]

        bayeux_location_index = GERMAN_LOCATION_INDEX[
            id(bayeux)
        ]

        self.assertEqual(
            self.observation[
                FS3_START + LOCATION_IN_TRANSIT
            ],
            1.0,
        )

        self.assertEqual(
            self.observation[FS3_REDUCED_INDEX],
            1.0,
        )

        self.assertEqual(
            self.observation[
                FS5_START + carentan_location_index
            ],
            1.0,
        )

        self.assertEqual(
            self.observation[
                TIGER_101_START + bayeux_location_index
            ],
            1.0,
        )

    def test_encode_special_units_each_unit_has_one_location(self):
        encode_special_german_units(self.observation)

        for start_index in [
            FS3_START,
            FS5_START,
            TIGER_101_START,
        ]:
            location_state = self.observation[
                start_index:
                start_index + NAMED_UNIT_LOCATION_COUNT
            ]

            self.assertEqual(
                np.count_nonzero(location_state),
                1,
            )

    def test_encode_special_units_only_changes_special_unit_block(self):
        strategic_reserve_box.units.append(FS_3)
        carentan.units.append(FS_5)
        bayeux.units.append(TIGER_101)

        FS_3.combat_value = 1

        encode_special_german_units(self.observation)

        before_special_units = self.observation[
            :SPECIAL_GERMAN_UNITS_START
        ]

        after_special_units = self.observation[
            SPECIAL_GERMAN_UNITS_END:
        ]

        self.assertEqual(
            np.count_nonzero(before_special_units),
            0,
        )

        self.assertEqual(
            np.count_nonzero(after_special_units),
            0,
        )


if __name__ == "__main__":
    unittest.main()

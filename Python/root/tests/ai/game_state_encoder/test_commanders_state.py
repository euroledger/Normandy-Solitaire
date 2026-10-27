import unittest

import numpy as np

from core.german_units import (
    ROMMEL,
    MEYER,
    MODEL,
)

from core.global_game_state import GlobalGameState

from core.map.map_model import (
    in_transit_box,
    strategic_reserve_box,
    eliminated_units_box,
)

from core.map.map_spaces_brit_2 import bayeux
from core.map.map_spaces_us_1 import carentan

from core.ai.dqn.dqn_game_state_encoder import (
    OBSERVATION_SIZE,
    COMMANDERS_START,
    COMMANDERS_END,
    NAMED_UNIT_LOCATION_COUNT,
    ROMMEL_START,
    MEYER_START,
    MODEL_START,
    MEYER_AVAILABLE_INDEX,
    MODEL_IN_COMMAND_INDEX,
    LOCATION_NOT_AVAILABLE,
    LOCATION_IN_TRANSIT,
    LOCATION_STRATEGIC_RESERVE,
    LOCATION_ELIMINATED,
    GERMAN_LOCATION_INDEX,
    GERMAN_MAP_SPACES,
    encode_commanders,
)


class TestEncodeCommanders(unittest.TestCase):

    def setUp(self):
        self.commanders = [
            ROMMEL,
            MEYER,
            MODEL,
        ]

        # Remove commanders from all encoded map spaces.

        for space in GERMAN_MAP_SPACES:
            for commander in self.commanders:
                if commander in space.units:
                    space.units.remove(commander)

        # Remove commanders from all off-map boxes.

        for box in [
            in_transit_box,
            strategic_reserve_box,
            eliminated_units_box,
        ]:
            for commander in self.commanders:
                if commander in box.units:
                    box.units.remove(commander)

        GlobalGameState.meyer_available = False
        GlobalGameState.model_in_command = False

        self.observation = np.zeros(
            OBSERVATION_SIZE,
            dtype=np.float32,
        )

    def test_encode_commanders_rommel_not_available(self):
        encode_commanders(self.observation)

        location_state = self.observation[
            ROMMEL_START:
            ROMMEL_START + NAMED_UNIT_LOCATION_COUNT
        ]

        self.assertEqual(
            np.count_nonzero(location_state),
            1,
        )

        self.assertEqual(
            self.observation[
                ROMMEL_START + LOCATION_NOT_AVAILABLE
            ],
            1.0,
        )

    def test_encode_commanders_rommel_in_transit(self):
        in_transit_box.units.append(ROMMEL)

        encode_commanders(self.observation)

        self.assertEqual(
            self.observation[
                ROMMEL_START + LOCATION_IN_TRANSIT
            ],
            1.0,
        )

        location_state = self.observation[
            ROMMEL_START:
            ROMMEL_START + NAMED_UNIT_LOCATION_COUNT
        ]

        self.assertEqual(
            np.count_nonzero(location_state),
            1,
        )

    def test_encode_commanders_rommel_strategic_reserve(self):
        strategic_reserve_box.units.append(ROMMEL)

        encode_commanders(self.observation)

        self.assertEqual(
            self.observation[
                ROMMEL_START + LOCATION_STRATEGIC_RESERVE
            ],
            1.0,
        )

        location_state = self.observation[
            ROMMEL_START:
            ROMMEL_START + NAMED_UNIT_LOCATION_COUNT
        ]

        self.assertEqual(
            np.count_nonzero(location_state),
            1,
        )

    def test_encode_commanders_rommel_eliminated(self):
        eliminated_units_box.units.append(ROMMEL)

        encode_commanders(self.observation)

        self.assertEqual(
            self.observation[
                ROMMEL_START + LOCATION_ELIMINATED
            ],
            1.0,
        )

        location_state = self.observation[
            ROMMEL_START:
            ROMMEL_START + NAMED_UNIT_LOCATION_COUNT
        ]

        self.assertEqual(
            np.count_nonzero(location_state),
            1,
        )

    def test_encode_commanders_rommel_map_location(self):
        bayeux.units.append(ROMMEL)

        encode_commanders(self.observation)

        bayeux_location_index = GERMAN_LOCATION_INDEX[id(bayeux)]

        self.assertEqual(
            self.observation[
                ROMMEL_START + bayeux_location_index
            ],
            1.0,
        )

        location_state = self.observation[
            ROMMEL_START:
            ROMMEL_START + NAMED_UNIT_LOCATION_COUNT
        ]

        self.assertEqual(
            np.count_nonzero(location_state),
            1,
        )

    def test_encode_commanders_meyer_not_available(self):
        encode_commanders(self.observation)

        location_state = self.observation[
            MEYER_START:
            MEYER_START + NAMED_UNIT_LOCATION_COUNT
        ]

        self.assertEqual(
            np.count_nonzero(location_state),
            1,
        )

        self.assertEqual(
            self.observation[
                MEYER_START + LOCATION_NOT_AVAILABLE
            ],
            1.0,
        )

    def test_encode_commanders_meyer_map_location(self):
        carentan.units.append(MEYER)

        encode_commanders(self.observation)

        carentan_location_index = GERMAN_LOCATION_INDEX[
            id(carentan)
        ]

        self.assertEqual(
            self.observation[
                MEYER_START + carentan_location_index
            ],
            1.0,
        )

        location_state = self.observation[
            MEYER_START:
            MEYER_START + NAMED_UNIT_LOCATION_COUNT
        ]

        self.assertEqual(
            np.count_nonzero(location_state),
            1,
        )

    def test_encode_commanders_model_not_available(self):
        encode_commanders(self.observation)

        location_state = self.observation[
            MODEL_START:
            MODEL_START + NAMED_UNIT_LOCATION_COUNT
        ]

        self.assertEqual(
            np.count_nonzero(location_state),
            1,
        )

        self.assertEqual(
            self.observation[
                MODEL_START + LOCATION_NOT_AVAILABLE
            ],
            1.0,
        )

    def test_encode_commanders_model_map_location(self):
        bayeux.units.append(MODEL)

        encode_commanders(self.observation)

        bayeux_location_index = GERMAN_LOCATION_INDEX[id(bayeux)]

        self.assertEqual(
            self.observation[
                MODEL_START + bayeux_location_index
            ],
            1.0,
        )

        location_state = self.observation[
            MODEL_START:
            MODEL_START + NAMED_UNIT_LOCATION_COUNT
        ]

        self.assertEqual(
            np.count_nonzero(location_state),
            1,
        )

    def test_encode_commanders_meyer_available_false(self):
        GlobalGameState.meyer_available = False

        encode_commanders(self.observation)

        self.assertEqual(
            self.observation[MEYER_AVAILABLE_INDEX],
            0.0,
        )

    def test_encode_commanders_meyer_available_true(self):
        GlobalGameState.meyer_available = True

        encode_commanders(self.observation)

        self.assertEqual(
            self.observation[MEYER_AVAILABLE_INDEX],
            1.0,
        )

    def test_encode_commanders_model_in_command_false(self):
        GlobalGameState.model_in_command = False

        encode_commanders(self.observation)

        self.assertEqual(
            self.observation[MODEL_IN_COMMAND_INDEX],
            0.0,
        )

    def test_encode_commanders_model_in_command_true(self):
        GlobalGameState.model_in_command = True

        encode_commanders(self.observation)

        self.assertEqual(
            self.observation[MODEL_IN_COMMAND_INDEX],
            1.0,
        )

    def test_encode_commanders_locations_are_independent(self):
        in_transit_box.units.append(ROMMEL)
        carentan.units.append(MEYER)
        bayeux.units.append(MODEL)

        encode_commanders(self.observation)

        carentan_location_index = GERMAN_LOCATION_INDEX[
            id(carentan)
        ]

        bayeux_location_index = GERMAN_LOCATION_INDEX[
            id(bayeux)
        ]

        self.assertEqual(
            self.observation[
                ROMMEL_START + LOCATION_IN_TRANSIT
            ],
            1.0,
        )

        self.assertEqual(
            self.observation[
                MEYER_START + carentan_location_index
            ],
            1.0,
        )

        self.assertEqual(
            self.observation[
                MODEL_START + bayeux_location_index
            ],
            1.0,
        )

    def test_encode_commanders_each_commander_has_one_location(self):
        encode_commanders(self.observation)

        for start_index in [
            ROMMEL_START,
            MEYER_START,
            MODEL_START,
        ]:
            location_state = self.observation[
                start_index:
                start_index + NAMED_UNIT_LOCATION_COUNT
            ]

            self.assertEqual(
                np.count_nonzero(location_state),
                1,
            )

    def test_encode_commanders_flags_independent_of_locations(self):
        strategic_reserve_box.units.append(MEYER)
        in_transit_box.units.append(MODEL)

        GlobalGameState.meyer_available = True
        GlobalGameState.model_in_command = True

        encode_commanders(self.observation)

        self.assertEqual(
            self.observation[
                MEYER_START + LOCATION_STRATEGIC_RESERVE
            ],
            1.0,
        )

        self.assertEqual(
            self.observation[
                MODEL_START + LOCATION_IN_TRANSIT
            ],
            1.0,
        )

        self.assertEqual(
            self.observation[MEYER_AVAILABLE_INDEX],
            1.0,
        )

        self.assertEqual(
            self.observation[MODEL_IN_COMMAND_INDEX],
            1.0,
        )

    def test_encode_commanders_only_changes_commanders_block(self):
        in_transit_box.units.append(ROMMEL)
        carentan.units.append(MEYER)
        bayeux.units.append(MODEL)

        GlobalGameState.meyer_available = True
        GlobalGameState.model_in_command = True

        encode_commanders(self.observation)

        before_commanders = self.observation[:COMMANDERS_START]
        after_commanders = self.observation[COMMANDERS_END:]

        self.assertEqual(
            np.count_nonzero(before_commanders),
            0,
        )

        self.assertEqual(
            np.count_nonzero(after_commanders),
            0,
        )


if __name__ == "__main__":
    unittest.main()

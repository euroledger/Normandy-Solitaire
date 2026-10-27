import unittest

import numpy as np

from core.german_units import (
    PZ_LEHR,
    SS_12,
    panzer_divisions_list,
)

from core.map.map_model import (
    in_transit_box,
    strategic_reserve_box,
    eliminated_units_box,
)

from core.map.map_spaces_brit_2 import bayeux

from core.ai.dqn.dqn_game_state_encoder import (
    OBSERVATION_SIZE,
    PANZERS_START,
    PANZERS_END,
    PANZER_FEATURES_PER_UNIT,
    NAMED_UNIT_LOCATION_COUNT,
    LOCATION_NOT_AVAILABLE,
    LOCATION_IN_TRANSIT,
    LOCATION_STRATEGIC_RESERVE,
    LOCATION_ELIMINATED,
    GERMAN_LOCATION_INDEX,
    GERMAN_MAP_SPACES,
    encode_panzer_divisions,
)


class TestEncodePanzerDivisions(unittest.TestCase):

    def setUp(self):
        # Remove all Panzer divisions from every encoded location.

        for space in GERMAN_MAP_SPACES:
            for panzer in panzer_divisions_list:
                if panzer in space.units:
                    space.units.remove(panzer)

        for box in [
            in_transit_box,
            strategic_reserve_box,
            eliminated_units_box,
        ]:
            for panzer in panzer_divisions_list:
                if panzer in box.units:
                    box.units.remove(panzer)

        # Restore all Panzer divisions to full strength.

        for panzer in panzer_divisions_list:
            panzer.combat_value = 2

        self.observation = np.zeros(
            OBSERVATION_SIZE,
            dtype=np.float32,
        )

    def test_encode_panzer_divisions_not_available(self):
        encode_panzer_divisions(self.observation)

        panzer_lehr_start = PANZERS_START

        location_state = self.observation[
            panzer_lehr_start:
            panzer_lehr_start + NAMED_UNIT_LOCATION_COUNT
        ]

        self.assertEqual(
            np.count_nonzero(location_state),
            1,
        )

        self.assertEqual(
            self.observation[
                panzer_lehr_start + LOCATION_NOT_AVAILABLE
            ],
            1.0,
        )

    def test_encode_panzer_divisions_in_transit(self):
        in_transit_box.units.append(PZ_LEHR)

        encode_panzer_divisions(self.observation)

        panzer_lehr_start = PANZERS_START

        self.assertEqual(
            self.observation[
                panzer_lehr_start + LOCATION_IN_TRANSIT
            ],
            1.0,
        )

        location_state = self.observation[
            panzer_lehr_start:
            panzer_lehr_start + NAMED_UNIT_LOCATION_COUNT
        ]

        self.assertEqual(
            np.count_nonzero(location_state),
            1,
        )

    def test_encode_panzer_divisions_strategic_reserve(self):
        strategic_reserve_box.units.append(PZ_LEHR)

        encode_panzer_divisions(self.observation)

        panzer_lehr_start = PANZERS_START

        self.assertEqual(
            self.observation[
                panzer_lehr_start + LOCATION_STRATEGIC_RESERVE
            ],
            1.0,
        )

        location_state = self.observation[
            panzer_lehr_start:
            panzer_lehr_start + NAMED_UNIT_LOCATION_COUNT
        ]

        self.assertEqual(
            np.count_nonzero(location_state),
            1,
        )

    def test_encode_panzer_divisions_eliminated(self):
        eliminated_units_box.units.append(PZ_LEHR)

        encode_panzer_divisions(self.observation)

        panzer_lehr_start = PANZERS_START

        self.assertEqual(
            self.observation[
                panzer_lehr_start + LOCATION_ELIMINATED
            ],
            1.0,
        )

        location_state = self.observation[
            panzer_lehr_start:
            panzer_lehr_start + NAMED_UNIT_LOCATION_COUNT
        ]

        self.assertEqual(
            np.count_nonzero(location_state),
            1,
        )

    def test_encode_panzer_divisions_map_location(self):
        bayeux.units.append(PZ_LEHR)

        encode_panzer_divisions(self.observation)

        panzer_lehr_start = PANZERS_START
        bayeux_location_index = GERMAN_LOCATION_INDEX[id(bayeux)]

        self.assertEqual(
            self.observation[
                panzer_lehr_start + bayeux_location_index
            ],
            1.0,
        )

        location_state = self.observation[
            panzer_lehr_start:
            panzer_lehr_start + NAMED_UNIT_LOCATION_COUNT
        ]

        self.assertEqual(
            np.count_nonzero(location_state),
            1,
        )

    def test_encode_panzer_divisions_full_strength(self):
        strategic_reserve_box.units.append(PZ_LEHR)
        PZ_LEHR.combat_value = 2

        encode_panzer_divisions(self.observation)

        panzer_lehr_reduced_index = (
            PANZERS_START + NAMED_UNIT_LOCATION_COUNT
        )

        self.assertEqual(
            self.observation[panzer_lehr_reduced_index],
            0.0,
        )

    def test_encode_panzer_divisions_reduced_strength(self):
        strategic_reserve_box.units.append(PZ_LEHR)
        PZ_LEHR.combat_value = 1

        encode_panzer_divisions(self.observation)

        panzer_lehr_reduced_index = (
            PANZERS_START + NAMED_UNIT_LOCATION_COUNT
        )

        self.assertEqual(
            self.observation[panzer_lehr_reduced_index],
            1.0,
        )

    def test_encode_panzer_divisions_second_panzer_uses_next_block(self):
        strategic_reserve_box.units.append(PZ_LEHR)
        bayeux.units.append(SS_12)

        PZ_LEHR.combat_value = 2
        SS_12.combat_value = 1

        encode_panzer_divisions(self.observation)

        panzer_lehr_start = PANZERS_START

        ss_12_start = (
            PANZERS_START + PANZER_FEATURES_PER_UNIT
        )

        self.assertEqual(
            self.observation[
                panzer_lehr_start + LOCATION_STRATEGIC_RESERVE
            ],
            1.0,
        )

        bayeux_location_index = GERMAN_LOCATION_INDEX[id(bayeux)]

        self.assertEqual(
            self.observation[
                ss_12_start + bayeux_location_index
            ],
            1.0,
        )

        self.assertEqual(
            self.observation[
                panzer_lehr_start + NAMED_UNIT_LOCATION_COUNT
            ],
            0.0,
        )

        self.assertEqual(
            self.observation[
                ss_12_start + NAMED_UNIT_LOCATION_COUNT
            ],
            1.0,
        )

    def test_encode_panzer_divisions_every_panzer_has_one_location(self):
        encode_panzer_divisions(self.observation)

        for panzer_index in range(len(panzer_divisions_list)):
            start = (
                PANZERS_START
                + panzer_index * PANZER_FEATURES_PER_UNIT
            )

            location_state = self.observation[
                start:start + NAMED_UNIT_LOCATION_COUNT
            ]

            self.assertEqual(
                np.count_nonzero(location_state),
                1,
                msg=(
                    f"{panzer_divisions_list[panzer_index].name} "
                    f"does not have exactly one encoded location"
                ),
            )

    def test_encode_panzer_divisions_only_changes_panzer_block(self):
        strategic_reserve_box.units.append(PZ_LEHR)
        bayeux.units.append(SS_12)
        SS_12.combat_value = 1

        encode_panzer_divisions(self.observation)

        before_panzers = self.observation[:PANZERS_START]
        after_panzers = self.observation[PANZERS_END:]

        self.assertEqual(
            np.count_nonzero(before_panzers),
            0,
        )

        self.assertEqual(
            np.count_nonzero(after_panzers),
            0,
        )


if __name__ == "__main__":
    unittest.main()

import unittest

import numpy as np

from core.enums import SideType

# Use the actual import path for do_opening_setup in your project

from core.ai.dqn.dqn_game_state_encoder import (
    OBSERVATION_SIZE,
    MAP_STATE_START,
    MAP_STATE_END,
    MAP_FEATURES_PER_SPACE,
    GERMAN_MAP_SPACES,
    MAX_FORTIFIED_VILLAGE_MODIFIER,
    encode_map_state,
)
from core.map.map_utilities import do_opening_setup


class TestEncodeMapState(unittest.TestCase):

    def setUp(self):
        do_opening_setup()

        self.observation = np.zeros(
            OBSERVATION_SIZE,
            dtype=np.float32,
        )

    def get_space_start(self, space):
        space_index = GERMAN_MAP_SPACES.index(space)
        return MAP_STATE_START + space_index * MAP_FEATURES_PER_SPACE

    def test_encode_map_state_opening_setup(self):
        encode_map_state(self.observation)

        for space in GERMAN_MAP_SPACES:
            start = self.get_space_start(space)

            self.assertEqual(
                self.observation[start],
                float(space.controlling_player == SideType.GERMAN),
                msg=f"Incorrect control state for {space.name}",
            )

            self.assertEqual(
                self.observation[start + 1],
                float(space.under_siege),
                msg=f"Incorrect siege state for {space.name}",
            )

            self.assertAlmostEqual(
                self.observation[start + 2],
                space.fortified_village_modifier
                / MAX_FORTIFIED_VILLAGE_MODIFIER,
                msg=f"Incorrect fortified village modifier for {space.name}",
            )

            self.assertEqual(
                self.observation[start + 3],
                0.0,
                msg=f"Reserved slot not zero for {space.name}",
            )

    def test_encode_map_state_german_control(self):
        space = GERMAN_MAP_SPACES[0]
        space.controlling_player = SideType.GERMAN

        encode_map_state(self.observation)

        start = self.get_space_start(space)

        self.assertEqual(
            self.observation[start],
            1.0,
        )

    def test_encode_map_state_allied_control(self):
        space = GERMAN_MAP_SPACES[0]
        space.controlling_player = SideType.ALLIED

        encode_map_state(self.observation)

        start = self.get_space_start(space)

        self.assertEqual(
            self.observation[start],
            0.0,
        )

    def test_encode_map_state_under_siege(self):
        space = GERMAN_MAP_SPACES[0]
        space.under_siege = True

        encode_map_state(self.observation)

        start = self.get_space_start(space)

        self.assertEqual(
            self.observation[start + 1],
            1.0,
        )

    def test_encode_map_state_not_under_siege(self):
        space = GERMAN_MAP_SPACES[0]
        space.under_siege = False

        encode_map_state(self.observation)

        start = self.get_space_start(space)

        self.assertEqual(
            self.observation[start + 1],
            0.0,
        )

    def test_encode_map_state_fortified_village_modifier(self):
        space = GERMAN_MAP_SPACES[0]
        space.fortified_village_modifier = 1

        encode_map_state(self.observation)

        start = self.get_space_start(space)

        self.assertAlmostEqual(
            self.observation[start + 2],
            1 / MAX_FORTIFIED_VILLAGE_MODIFIER,
        )

    def test_encode_map_state_fortified_village_modifier_maximum(self):
        space = GERMAN_MAP_SPACES[0]
        space.fortified_village_modifier = (
            MAX_FORTIFIED_VILLAGE_MODIFIER
        )

        encode_map_state(self.observation)

        start = self.get_space_start(space)

        self.assertEqual(
            self.observation[start + 2],
            1.0,
        )

    def test_encode_map_state_reserved_slots_remain_zero(self):
        encode_map_state(self.observation)

        for space in GERMAN_MAP_SPACES:
            start = self.get_space_start(space)

            self.assertEqual(
                self.observation[start + 3],
                0.0,
                msg=f"Reserved slot not zero for {space.name}",
            )

    def test_encode_map_state_spaces_are_independent(self):
        first_space = GERMAN_MAP_SPACES[0]
        second_space = GERMAN_MAP_SPACES[1]

        first_space.controlling_player = SideType.GERMAN
        first_space.under_siege = True
        first_space.fortified_village_modifier = 1

        second_space.controlling_player = SideType.ALLIED
        second_space.under_siege = False
        second_space.fortified_village_modifier = (
            MAX_FORTIFIED_VILLAGE_MODIFIER
        )

        encode_map_state(self.observation)

        first_start = self.get_space_start(first_space)
        second_start = self.get_space_start(second_space)

        self.assertEqual(
            self.observation[first_start],
            1.0,
        )

        self.assertEqual(
            self.observation[first_start + 1],
            1.0,
        )

        self.assertAlmostEqual(
            self.observation[first_start + 2],
            1 / MAX_FORTIFIED_VILLAGE_MODIFIER,
        )

        self.assertEqual(
            self.observation[first_start + 3],
            0.0,
        )

        self.assertEqual(
            self.observation[second_start],
            0.0,
        )

        self.assertEqual(
            self.observation[second_start + 1],
            0.0,
        )

        self.assertEqual(
            self.observation[second_start + 2],
            1.0,
        )

        self.assertEqual(
            self.observation[second_start + 3],
            0.0,
        )

    def test_encode_map_state_only_changes_map_state_block(self):
        space = GERMAN_MAP_SPACES[0]

        space.controlling_player = SideType.GERMAN
        space.under_siege = True
        space.fortified_village_modifier = 1

        encode_map_state(self.observation)

        before_block = self.observation[
            :MAP_STATE_START
        ]

        after_block = self.observation[
            MAP_STATE_END:
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

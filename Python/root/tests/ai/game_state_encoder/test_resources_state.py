import unittest

import numpy as np

from core.global_game_state import GlobalGameState

from core.map.map_model import (
    transport_track,
    supply_track,
    hitler_approval_track,
)

from core.ai.dqn.dqn_game_state_encoder import (
    OBSERVATION_SIZE,
    RESOURCES_START,
    RESOURCES_END,
    TRANSPORT_INDEX,
    SUPPLY_INDEX,
    HITLER_APPROVAL_INDEX,
    TRANSPORT_BASE_INDEX,
    SUPPLY_BASE_INDEX,
    HITLER_APPROVAL_BASE_INDEX,
    ACTIONS_LEFT_INDEX,
    RESERVE_ACTIONS_INDEX,
    encode_resources,
)


class TestEncodeResources(unittest.TestCase):

    def setUp(self):
        transport_track.value = 5
        supply_track.value = 4
        hitler_approval_track.value = 6

        GlobalGameState.transport_base_level = 3
        GlobalGameState.supply_base_level = 3
        GlobalGameState.hitler_approval_base_level = 3

        GlobalGameState.actions_left_this_turn = 0
        GlobalGameState.reserve_actions = 0

        self.observation = np.zeros(
            OBSERVATION_SIZE,
            dtype=np.float32,
        )

    def test_encode_resources_current_values(self):
        transport_track.value = 3
        supply_track.value = 3
        hitler_approval_track.value = 2

        encode_resources(self.observation)

        self.assertEqual(
            self.observation[TRANSPORT_INDEX],
            0.5,
        )

        self.assertEqual(
            self.observation[SUPPLY_INDEX],
            0.5,
        )

        self.assertEqual(
            self.observation[HITLER_APPROVAL_INDEX],
            0.5,
        )

    def test_encode_resources_minimum_values(self):
        transport_track.value = 0
        supply_track.value = 0
        hitler_approval_track.value = -2

        encode_resources(self.observation)

        self.assertEqual(
            self.observation[TRANSPORT_INDEX],
            0.0,
        )

        self.assertEqual(
            self.observation[SUPPLY_INDEX],
            0.0,
        )

        self.assertEqual(
            self.observation[HITLER_APPROVAL_INDEX],
            0.0,
        )

    def test_encode_resources_maximum_values(self):
        transport_track.value = 6
        supply_track.value = 6
        hitler_approval_track.value = 6

        encode_resources(self.observation)

        self.assertEqual(
            self.observation[TRANSPORT_INDEX],
            1.0,
        )

        self.assertEqual(
            self.observation[SUPPLY_INDEX],
            1.0,
        )

        self.assertEqual(
            self.observation[HITLER_APPROVAL_INDEX],
            1.0,
        )

    def test_encode_resources_base_levels(self):
        GlobalGameState.transport_base_level = 3
        GlobalGameState.supply_base_level = 3
        GlobalGameState.hitler_approval_base_level = 2

        encode_resources(self.observation)

        self.assertEqual(
            self.observation[TRANSPORT_BASE_INDEX],
            0.5,
        )

        self.assertEqual(
            self.observation[SUPPLY_BASE_INDEX],
            0.5,
        )

        self.assertEqual(
            self.observation[HITLER_APPROVAL_BASE_INDEX],
            0.5,
        )

    def test_encode_resources_actions_left(self):
        GlobalGameState.actions_left_this_turn = 2

        encode_resources(self.observation)

        self.assertEqual(
            self.observation[ACTIONS_LEFT_INDEX],
            0.5,
        )

    def test_encode_resources_actions_left_maximum(self):
        GlobalGameState.actions_left_this_turn = 4

        encode_resources(self.observation)

        self.assertEqual(
            self.observation[ACTIONS_LEFT_INDEX],
            1.0,
        )

    def test_encode_resources_reserve_actions(self):
        GlobalGameState.reserve_actions = 1

        encode_resources(self.observation)

        self.assertEqual(
            self.observation[RESERVE_ACTIONS_INDEX],
            0.5,
        )

    def test_encode_resources_reserve_actions_maximum(self):
        GlobalGameState.reserve_actions = 2

        encode_resources(self.observation)

        self.assertEqual(
            self.observation[RESERVE_ACTIONS_INDEX],
            1.0,
        )

    def test_encode_resources_only_changes_resources_block(self):
        encode_resources(self.observation)

        before_resources = self.observation[:RESOURCES_START]
        after_resources = self.observation[RESOURCES_END:]

        self.assertEqual(
            np.count_nonzero(before_resources),
            0,
        )

        self.assertEqual(
            np.count_nonzero(after_resources),
            0,
        )


if __name__ == "__main__":
    unittest.main()

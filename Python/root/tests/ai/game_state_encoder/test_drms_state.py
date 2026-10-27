import unittest

import numpy as np

from core.global_game_state import GlobalGameState
from core.map.map_utilities import do_opening_setup

from core.ai.dqn.dqn_game_state_encoder import (
    OBSERVATION_SIZE,
    DRM_START,
    DRM_END,
    TRANSPORT_ROLL_DRM_INDEX,
    SUPPLY_ROLL_DRM_INDEX,
    TRANSPORT_CHECK_DRM_INDEX,
    SUPPLY_CHECK_DRM_INDEX,
    HITLER_APPROVAL_CHECK_DRM_INDEX,
    MIN_TRANSPORT_ROLL_DRM,
    MAX_TRANSPORT_ROLL_DRM,
    MIN_SUPPLY_ROLL_DRM,
    MAX_SUPPLY_ROLL_DRM,
    MIN_TRANSPORT_CHECK_DRM,
    MAX_TRANSPORT_CHECK_DRM,
    MIN_SUPPLY_CHECK_DRM,
    MAX_SUPPLY_CHECK_DRM,
    MIN_HITLER_APPROVAL_CHECK_DRM,
    MAX_HITLER_APPROVAL_CHECK_DRM,
    encode_drms,
)


class TestEncodeDrms(unittest.TestCase):

    def setUp(self):
        do_opening_setup()

        self.observation = np.zeros(
            OBSERVATION_SIZE,
            dtype=np.float32,
        )

    def test_encode_drms_opening_setup(self):
        encode_drms(self.observation)

        self.assertEqual(
            self.observation[TRANSPORT_ROLL_DRM_INDEX],
            0.5,
        )

        self.assertEqual(
            self.observation[SUPPLY_ROLL_DRM_INDEX],
            0.5,
        )

        self.assertEqual(
            self.observation[TRANSPORT_CHECK_DRM_INDEX],
            0.5,
        )

        self.assertEqual(
            self.observation[SUPPLY_CHECK_DRM_INDEX],
            0.5,
        )

        self.assertEqual(
            self.observation[HITLER_APPROVAL_CHECK_DRM_INDEX],
            0.5,
        )

    def test_encode_drms_minimum_values(self):
        GlobalGameState.transport_roll_drm = MIN_TRANSPORT_ROLL_DRM
        GlobalGameState.supply_roll_drm = MIN_SUPPLY_ROLL_DRM
        GlobalGameState.transport_check_drm = MIN_TRANSPORT_CHECK_DRM
        GlobalGameState.supply_check_drm = MIN_SUPPLY_CHECK_DRM
        GlobalGameState.hitler_approval_check_drm = (
            MIN_HITLER_APPROVAL_CHECK_DRM
        )

        encode_drms(self.observation)

        np.testing.assert_allclose(
            self.observation[DRM_START:DRM_END],
            np.zeros(5),
        )

    def test_encode_drms_maximum_values(self):
        GlobalGameState.transport_roll_drm = MAX_TRANSPORT_ROLL_DRM
        GlobalGameState.supply_roll_drm = MAX_SUPPLY_ROLL_DRM
        GlobalGameState.transport_check_drm = MAX_TRANSPORT_CHECK_DRM
        GlobalGameState.supply_check_drm = MAX_SUPPLY_CHECK_DRM
        GlobalGameState.hitler_approval_check_drm = (
            MAX_HITLER_APPROVAL_CHECK_DRM
        )

        encode_drms(self.observation)

        np.testing.assert_allclose(
            self.observation[DRM_START:DRM_END],
            np.ones(5),
        )

    def test_encode_drms_individual_positions(self):
        GlobalGameState.transport_roll_drm = MAX_TRANSPORT_ROLL_DRM
        GlobalGameState.supply_roll_drm = MIN_SUPPLY_ROLL_DRM
        GlobalGameState.transport_check_drm = MAX_TRANSPORT_CHECK_DRM
        GlobalGameState.supply_check_drm = MIN_SUPPLY_CHECK_DRM
        GlobalGameState.hitler_approval_check_drm = (
            MAX_HITLER_APPROVAL_CHECK_DRM
        )

        encode_drms(self.observation)

        np.testing.assert_allclose(
            self.observation[DRM_START:DRM_END],
            [
                1.0,
                0.0,
                1.0,
                0.0,
                1.0,
            ],
        )

    def test_encode_drms_zero_is_midpoint(self):
        GlobalGameState.transport_roll_drm = 0
        GlobalGameState.supply_roll_drm = 0
        GlobalGameState.transport_check_drm = 0
        GlobalGameState.supply_check_drm = 0
        GlobalGameState.hitler_approval_check_drm = 0

        encode_drms(self.observation)

        np.testing.assert_allclose(
            self.observation[DRM_START:DRM_END],
            np.full(5, 0.5),
        )

    def test_encode_drms_values_are_independent(self):
        GlobalGameState.transport_roll_drm = -3
        GlobalGameState.supply_roll_drm = -2
        GlobalGameState.transport_check_drm = 0
        GlobalGameState.supply_check_drm = 2
        GlobalGameState.hitler_approval_check_drm = 3

        encode_drms(self.observation)

        np.testing.assert_allclose(
            self.observation[DRM_START:DRM_END],
            [
                0.0,
                1 / 6,
                0.5,
                5 / 6,
                1.0,
            ],
        )

    def test_encode_drms_only_changes_drm_block(self):
        GlobalGameState.transport_roll_drm = 3
        GlobalGameState.supply_roll_drm = -3
        GlobalGameState.transport_check_drm = 1
        GlobalGameState.supply_check_drm = -1
        GlobalGameState.hitler_approval_check_drm = 2

        encode_drms(self.observation)

        self.assertEqual(
            np.count_nonzero(
                self.observation[:DRM_START]
            ),
            0,
        )

        self.assertEqual(
            np.count_nonzero(
                self.observation[DRM_END:]
            ),
            0,
        )


if __name__ == "__main__":
    unittest.main()

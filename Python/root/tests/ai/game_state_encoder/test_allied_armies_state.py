import unittest

import numpy as np

from core.global_game_state import GlobalGameState

from core.allied_armies import (
    US_FIRST_ARMY,
    BRITISH_SECOND_ARMY,
    CANADIAN_FIRST_ARMY,
    US_THIRD_ARMY,
    US_VIII_CORPS,
    US_XV_CORPS,
)

from core.map.map_spaces_us_1 import (
    us_1_track,
)

from core.map.map_spaces_brit_2 import (
    brit_2_track,
)

from core.map.map_spaces_can_1 import (
    can_1_track,
)

from core.map.map_spaces_us_3 import (
    us_viii_track,
    us_xv_track,
)

from core.ai.dqn.dqn_game_state_encoder import (
    OBSERVATION_SIZE,
    ALLIED_ARMIES_START,
    ALLIED_ARMIES_END,
    ALLIED_FEATURES_PER_ARMY,
    US_THIRD_ACTIVATED_INDEX,
    US_THIRD_MERGED_INDEX,
    ARMIES_UPGRADED_INDEX,
    encode_allied_armies,
)


class TestEncodeAlliedArmies(unittest.TestCase):

    def setUp(self):
        self.armies = [
            US_FIRST_ARMY,
            BRITISH_SECOND_ARMY,
            CANADIAN_FIRST_ARMY,
            US_THIRD_ARMY,
            US_VIII_CORPS,
            US_XV_CORPS,
        ]

        for army in self.armies:
            army.location = None
            army.flipped = False
            army.merged = False

        GlobalGameState.counter_attacked_armies = set()
        GlobalGameState.us_third_army_activated = False
        GlobalGameState.us_third_army_merged = False
        GlobalGameState.armies_upgraded = False

        self.observation = np.zeros(
            OBSERVATION_SIZE,
            dtype=np.float32,
        )

    def test_encode_allied_armies_inactive_army(self):
        encode_allied_armies(self.observation)

        us_first_start = ALLIED_ARMIES_START

        self.assertEqual(
            self.observation[us_first_start],
            0.0,
        )

        self.assertEqual(
            self.observation[us_first_start + 1],
            0.0,
        )

        self.assertEqual(
            self.observation[us_first_start + 2],
            0.0,
        )

        self.assertEqual(
            self.observation[us_first_start + 3],
            0.0,
        )

    def test_encode_allied_armies_active_army(self):
        US_FIRST_ARMY.location = us_1_track[0]

        encode_allied_armies(self.observation)

        us_first_start = ALLIED_ARMIES_START

        self.assertEqual(
            self.observation[us_first_start + 1],
            1.0,
        )

    def test_encode_allied_armies_progress_start(self):
        US_FIRST_ARMY.location = us_1_track[0]

        encode_allied_armies(self.observation)

        us_first_start = ALLIED_ARMIES_START

        self.assertEqual(
            self.observation[us_first_start],
            0.0,
        )

    def test_encode_allied_armies_progress_end(self):
        US_FIRST_ARMY.location = us_1_track[-1]

        encode_allied_armies(self.observation)

        us_first_start = ALLIED_ARMIES_START

        self.assertEqual(
            self.observation[us_first_start],
            1.0,
        )

    def test_encode_allied_armies_progress_middle(self):
        middle_index = len(us_1_track) // 2

        US_FIRST_ARMY.location = us_1_track[middle_index]

        encode_allied_armies(self.observation)

        expected_progress = (
            middle_index / (len(us_1_track) - 1)
        )

        us_first_start = ALLIED_ARMIES_START

        self.assertAlmostEqual(
            self.observation[us_first_start],
            expected_progress,
        )

    def test_encode_allied_armies_flipped(self):
        US_FIRST_ARMY.location = us_1_track[0]
        US_FIRST_ARMY.flipped = True

        encode_allied_armies(self.observation)

        us_first_start = ALLIED_ARMIES_START

        self.assertEqual(
            self.observation[us_first_start + 2],
            1.0,
        )

    def test_encode_allied_armies_counterattacked(self):
        US_FIRST_ARMY.location = us_1_track[0]

        GlobalGameState.counter_attacked_armies = {
            US_FIRST_ARMY.name
        }

        encode_allied_armies(self.observation)

        us_first_start = ALLIED_ARMIES_START

        self.assertEqual(
            self.observation[us_first_start + 3],
            1.0,
        )

    def test_encode_allied_armies_each_army_uses_correct_block(self):
        army_tracks = [
            (US_FIRST_ARMY, us_1_track),
            (BRITISH_SECOND_ARMY, brit_2_track),
            (CANADIAN_FIRST_ARMY, can_1_track),
            (US_THIRD_ARMY, us_xv_track),
            (US_VIII_CORPS, us_viii_track),
            (US_XV_CORPS, us_xv_track),
        ]

        for army, track in army_tracks:
            army.location = track[-1]

        encode_allied_armies(self.observation)

        for army_index in range(6):
            start = (
                ALLIED_ARMIES_START
                + army_index * ALLIED_FEATURES_PER_ARMY
            )

            self.assertEqual(
                self.observation[start],
                1.0,
            )

            self.assertEqual(
                self.observation[start + 1],
                1.0,
            )

    def test_encode_allied_armies_us_third_activated(self):
        GlobalGameState.us_third_army_activated = True

        encode_allied_armies(self.observation)

        self.assertEqual(
            self.observation[US_THIRD_ACTIVATED_INDEX],
            1.0,
        )

    def test_encode_allied_armies_us_third_merged(self):
        GlobalGameState.us_third_army_merged = True

        encode_allied_armies(self.observation)

        self.assertEqual(
            self.observation[US_THIRD_MERGED_INDEX],
            1.0,
        )

    def test_encode_allied_armies_armies_upgraded(self):
        GlobalGameState.armies_upgraded = True

        encode_allied_armies(self.observation)

        self.assertEqual(
            self.observation[ARMIES_UPGRADED_INDEX],
            1.0,
        )

    def test_encode_allied_armies_global_flags_false(self):
        encode_allied_armies(self.observation)

        self.assertEqual(
            self.observation[US_THIRD_ACTIVATED_INDEX],
            0.0,
        )

        self.assertEqual(
            self.observation[US_THIRD_MERGED_INDEX],
            0.0,
        )

        self.assertEqual(
            self.observation[ARMIES_UPGRADED_INDEX],
            0.0,
        )

    def test_encode_allied_armies_only_changes_allied_block(self):
        US_FIRST_ARMY.location = us_1_track[-1]
        BRITISH_SECOND_ARMY.location = brit_2_track[-1]
        CANADIAN_FIRST_ARMY.location = can_1_track[-1]

        GlobalGameState.us_third_army_activated = True
        GlobalGameState.armies_upgraded = True

        encode_allied_armies(self.observation)

        before_allied = self.observation[:ALLIED_ARMIES_START]
        after_allied = self.observation[ALLIED_ARMIES_END:]

        self.assertEqual(
            np.count_nonzero(before_allied),
            0,
        )

        self.assertEqual(
            np.count_nonzero(after_allied),
            0,
        )


if __name__ == "__main__":
    unittest.main()

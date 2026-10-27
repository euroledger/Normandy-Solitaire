import unittest

import numpy as np

from core.global_game_state import GlobalGameState
from core.map.map_utilities import do_opening_setup

from core.ai.dqn.dqn_game_state_encoder import (
    OBSERVATION_SIZE,
    STRATEGIC_STATE_START,
    STRATEGIC_STATE_END,
    CHERBOURG_CAPTURED_INDEX,
    HITLER_ASSASSINATION_INDEX,
    HITLER_INTERVENTION_NO_EFFECT_INDEX,
    BOCAGE_DEFENSE_INDEX,
    US_FIRST_FURTHEST_ADVANCE_INDEX,
    CURRENT_CARPET_BOMBING_INDEX,
    MAX_BOCAGE_DEFENSE_MODIFIER,
    MAX_CARPET_BOMBING,
    encode_global_game_state,
)


class TestEncodeGlobalGameState(unittest.TestCase):

    def setUp(self):
        do_opening_setup()

        self.observation = np.zeros(
            OBSERVATION_SIZE,
            dtype=np.float32,
        )

    def test_encode_global_game_state_opening_setup(self):
        encode_global_game_state(self.observation)

        self.assertEqual(
            self.observation[CHERBOURG_CAPTURED_INDEX],
            float(GlobalGameState.cherbourg_captured),
        )

        self.assertEqual(
            self.observation[HITLER_ASSASSINATION_INDEX],
            float(GlobalGameState.hitler_assassination),
        )

        self.assertEqual(
            self.observation[HITLER_INTERVENTION_NO_EFFECT_INDEX],
            float(GlobalGameState.hitler_intervention_no_effect),
        )

        self.assertAlmostEqual(
            self.observation[BOCAGE_DEFENSE_INDEX],
            GlobalGameState.bocage_defense_modifier
            / MAX_BOCAGE_DEFENSE_MODIFIER,
        )

        self.assertAlmostEqual(
            self.observation[US_FIRST_FURTHEST_ADVANCE_INDEX],
            (
                11
                - GlobalGameState.us_first_army_furthest_advance
            ) / 11,
        )

        self.assertAlmostEqual(
            self.observation[CURRENT_CARPET_BOMBING_INDEX],
            GlobalGameState.current_carpet_bombing
            / MAX_CARPET_BOMBING,
        )

    def test_encode_global_game_state_cherbourg_captured(self):
        GlobalGameState.cherbourg_captured = True

        encode_global_game_state(self.observation)

        self.assertEqual(
            self.observation[CHERBOURG_CAPTURED_INDEX],
            1.0,
        )

    def test_encode_global_game_state_hitler_assassination(self):
        GlobalGameState.hitler_assassination = True

        encode_global_game_state(self.observation)

        self.assertEqual(
            self.observation[HITLER_ASSASSINATION_INDEX],
            1.0,
        )

    def test_encode_global_game_state_hitler_intervention_no_effect(self):
        GlobalGameState.hitler_intervention_no_effect = True

        encode_global_game_state(self.observation)

        self.assertEqual(
            self.observation[HITLER_INTERVENTION_NO_EFFECT_INDEX],
            1.0,
        )

    def test_encode_global_game_state_bocage_defense(self):
        GlobalGameState.bocage_defense_modifier = 1

        encode_global_game_state(self.observation)

        self.assertAlmostEqual(
            self.observation[BOCAGE_DEFENSE_INDEX],
            1 / MAX_BOCAGE_DEFENSE_MODIFIER,
        )

    def test_encode_global_game_state_bocage_defense_maximum(self):
        GlobalGameState.bocage_defense_modifier = (
            MAX_BOCAGE_DEFENSE_MODIFIER
        )

        encode_global_game_state(self.observation)

        self.assertEqual(
            self.observation[BOCAGE_DEFENSE_INDEX],
            1.0,
        )

    def test_encode_global_game_state_us_first_furthest_advance_start(self):
        GlobalGameState.us_first_army_furthest_advance = 11

        encode_global_game_state(self.observation)

        self.assertEqual(
            self.observation[US_FIRST_FURTHEST_ADVANCE_INDEX],
            0.0,
        )

    def test_encode_global_game_state_us_first_furthest_advance_midway(self):
        GlobalGameState.us_first_army_furthest_advance = 6

        encode_global_game_state(self.observation)

        self.assertAlmostEqual(
            self.observation[US_FIRST_FURTHEST_ADVANCE_INDEX],
            5 / 11,
        )

    def test_encode_global_game_state_us_first_furthest_advance_maximum(self):
        GlobalGameState.us_first_army_furthest_advance = 0

        encode_global_game_state(self.observation)

        self.assertEqual(
            self.observation[US_FIRST_FURTHEST_ADVANCE_INDEX],
            1.0,
        )

    def test_encode_global_game_state_carpet_bombing(self):
        GlobalGameState.current_carpet_bombing = 1

        encode_global_game_state(self.observation)

        self.assertAlmostEqual(
            self.observation[CURRENT_CARPET_BOMBING_INDEX],
            1 / MAX_CARPET_BOMBING,
        )

    def test_encode_global_game_state_carpet_bombing_maximum(self):
        GlobalGameState.current_carpet_bombing = MAX_CARPET_BOMBING

        encode_global_game_state(self.observation)

        self.assertEqual(
            self.observation[CURRENT_CARPET_BOMBING_INDEX],
            1.0,
        )

    def test_encode_global_game_state_only_changes_its_block(self):
        GlobalGameState.cherbourg_captured = True
        GlobalGameState.hitler_assassination = True
        GlobalGameState.bocage_defense_modifier = 1
        GlobalGameState.current_carpet_bombing = 1

        encode_global_game_state(self.observation)

        before_block = self.observation[
            :STRATEGIC_STATE_START
        ]

        after_block = self.observation[
            STRATEGIC_STATE_END:
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
import unittest

import numpy as np

from core.map.map_utilities import do_opening_setup
from core.ai.headless_action_masks import HeadlessDecisionState

from core.ai.dqn.dqn_game_state_encoder import (
    OBSERVATION_SIZE,
    DECISION_ACTION_PHASE_INDEX,
    DECISION_ARMY_START,
    DECISION_ALLIED_ATTACK_INDEX,
    DecisionType,
    DecisionContext,
    US_FIRST_ARMY,
    encode_game_state,
)


class TestEncodeGameState(unittest.TestCase):

    def setUp(self):
        do_opening_setup()
        HeadlessDecisionState.reset_hitler_intervention()

    def tearDown(self):
        HeadlessDecisionState.reset_hitler_intervention()

    def test_encode_complete_opening_game_state(self):
        observation = encode_game_state()

        self.assertEqual(
            observation.shape,
            (OBSERVATION_SIZE,),
        )

        self.assertEqual(
            observation.dtype,
            np.float32,
        )

        self.assertTrue(
            np.all(np.isfinite(observation))
        )

        self.assertTrue(
            np.all(observation >= 0.0)
        )

        self.assertTrue(
            np.all(observation <= 1.0)
        )

        self.assertGreater(
            np.count_nonzero(observation),
            0,
        )

    def test_encode_complete_game_state_with_decision_context(self):
        context = DecisionContext(
            decision_type=DecisionType.ACTION_PHASE,
            relevant_army=US_FIRST_ARMY,
            allied_attack=True,
        )

        observation = encode_game_state(context)

        self.assertEqual(
            observation[DECISION_ACTION_PHASE_INDEX],
            1.0,
        )

        self.assertEqual(
            observation[DECISION_ARMY_START],
            1.0,
        )

        self.assertEqual(
            observation[DECISION_ALLIED_ATTACK_INDEX],
            1.0,
        )

    def test_encode_game_state_returns_fresh_observation(self):
        first_observation = encode_game_state()
        second_observation = encode_game_state()

        self.assertIsNot(
            first_observation,
            second_observation,
        )

        np.testing.assert_array_equal(
            first_observation,
            second_observation,
        )


if __name__ == "__main__":
    unittest.main()

import unittest

import numpy as np

from core.allied_armies import (
    US_FIRST_ARMY,
    BRITISH_SECOND_ARMY,
    CANADIAN_FIRST_ARMY,
    US_THIRD_ARMY,
    US_VIII_CORPS,
    US_XV_CORPS,
)

from core.ai.headless_action_masks import HeadlessDecisionState

from core.ai.dqn.dqn_game_state_encoder import (
    OBSERVATION_SIZE,
    HITLER_STATE_START,
    HITLER_STATE_END,
    HITLER_STAGE_NONE_INDEX,
    HITLER_STAGE_CANCEL_INDEX,
    HITLER_STAGE_TARGET_INDEX,
    HITLER_STAGE_REDEPLOY_INDEX,
    HITLER_TARGET_START,
    HITLER_TARGET_END,
    HITLER_REDEPLOYMENTS_INDEX,
    encode_hitler_intervention_state,
)


class TestEncodeHitlerInterventionState(unittest.TestCase):

    def setUp(self):
        HeadlessDecisionState.reset_hitler_intervention()

        self.observation = np.zeros(
            OBSERVATION_SIZE,
            dtype=np.float32,
        )

    def tearDown(self):
        HeadlessDecisionState.reset_hitler_intervention()

    def test_encode_hitler_intervention_initial_state(self):
        encode_hitler_intervention_state(
            self.observation
        )

        self.assertEqual(
            self.observation[HITLER_STAGE_NONE_INDEX],
            1.0,
        )

        self.assertEqual(
            np.count_nonzero(
                self.observation[
                    HITLER_STAGE_NONE_INDEX:
                    HITLER_STAGE_REDEPLOY_INDEX + 1
                ]
            ),
            1,
        )

        self.assertEqual(
            np.count_nonzero(
                self.observation[
                    HITLER_TARGET_START:
                    HITLER_TARGET_END
                ]
            ),
            0,
        )

        self.assertEqual(
            self.observation[HITLER_REDEPLOYMENTS_INDEX],
            0.0,
        )

    def test_encode_hitler_intervention_cancel_stage(self):
        HeadlessDecisionState.hitler_intervention_stage = "CANCEL"

        encode_hitler_intervention_state(
            self.observation
        )

        self.assertEqual(
            self.observation[HITLER_STAGE_CANCEL_INDEX],
            1.0,
        )

        self.assertEqual(
            np.count_nonzero(
                self.observation[
                    HITLER_STAGE_NONE_INDEX:
                    HITLER_STAGE_REDEPLOY_INDEX + 1
                ]
            ),
            1,
        )

    def test_encode_hitler_intervention_target_stage(self):
        HeadlessDecisionState.hitler_intervention_stage = "TARGET"

        encode_hitler_intervention_state(
            self.observation
        )

        self.assertEqual(
            self.observation[HITLER_STAGE_TARGET_INDEX],
            1.0,
        )

        self.assertEqual(
            np.count_nonzero(
                self.observation[
                    HITLER_STAGE_NONE_INDEX:
                    HITLER_STAGE_REDEPLOY_INDEX + 1
                ]
            ),
            1,
        )

    def test_encode_hitler_intervention_redeploy_stage(self):
        HeadlessDecisionState.hitler_intervention_stage = "REDEPLOY"

        encode_hitler_intervention_state(
            self.observation
        )

        self.assertEqual(
            self.observation[HITLER_STAGE_REDEPLOY_INDEX],
            1.0,
        )

        self.assertEqual(
            np.count_nonzero(
                self.observation[
                    HITLER_STAGE_NONE_INDEX:
                    HITLER_STAGE_REDEPLOY_INDEX + 1
                ]
            ),
            1,
        )

    def test_encode_hitler_intervention_us_first_target(self):
        HeadlessDecisionState.hitler_intervention_stage = "TARGET"
        HeadlessDecisionState.hitler_intervention_target = US_FIRST_ARMY

        encode_hitler_intervention_state(
            self.observation
        )

        self.assertEqual(
            self.observation[HITLER_TARGET_START],
            1.0,
        )

        self.assertEqual(
            np.count_nonzero(
                self.observation[
                    HITLER_TARGET_START:
                    HITLER_TARGET_END
                ]
            ),
            1,
        )

    def test_encode_hitler_intervention_all_army_targets(self):
        armies = [
            US_FIRST_ARMY,
            BRITISH_SECOND_ARMY,
            CANADIAN_FIRST_ARMY,
            US_THIRD_ARMY,
            US_VIII_CORPS,
            US_XV_CORPS,
        ]

        for expected_offset, army in enumerate(armies):
            observation = np.zeros(
                OBSERVATION_SIZE,
                dtype=np.float32,
            )

            HeadlessDecisionState.hitler_intervention_target = army

            encode_hitler_intervention_state(
                observation
            )

            self.assertEqual(
                observation[
                    HITLER_TARGET_START + expected_offset
                ],
                1.0,
                msg=f"Incorrect target encoding for {army.name}",
            )

            self.assertEqual(
                np.count_nonzero(
                    observation[
                        HITLER_TARGET_START:
                        HITLER_TARGET_END
                    ]
                ),
                1,
            )

    def test_encode_hitler_intervention_zero_redeployments(self):
        HeadlessDecisionState.hitler_intervention_redeployments = 0

        encode_hitler_intervention_state(
            self.observation
        )

        self.assertEqual(
            self.observation[HITLER_REDEPLOYMENTS_INDEX],
            0.0,
        )

    def test_encode_hitler_intervention_one_redeployment(self):
        HeadlessDecisionState.hitler_intervention_redeployments = 1

        encode_hitler_intervention_state(
            self.observation
        )

        self.assertEqual(
            self.observation[HITLER_REDEPLOYMENTS_INDEX],
            0.5,
        )

    def test_encode_hitler_intervention_two_redeployments(self):
        HeadlessDecisionState.hitler_intervention_redeployments = 2

        encode_hitler_intervention_state(
            self.observation
        )

        self.assertEqual(
            self.observation[HITLER_REDEPLOYMENTS_INDEX],
            1.0,
        )

    def test_encode_hitler_intervention_complete_redeploy_state(self):
        HeadlessDecisionState.hitler_intervention_stage = "REDEPLOY"
        HeadlessDecisionState.hitler_intervention_target = BRITISH_SECOND_ARMY
        HeadlessDecisionState.hitler_intervention_redeployments = 1

        encode_hitler_intervention_state(
            self.observation
        )

        self.assertEqual(
            self.observation[HITLER_STAGE_REDEPLOY_INDEX],
            1.0,
        )

        self.assertEqual(
            self.observation[HITLER_TARGET_START + 1],
            1.0,
        )

        self.assertEqual(
            self.observation[HITLER_REDEPLOYMENTS_INDEX],
            0.5,
        )

    def test_encode_hitler_intervention_invalid_stage_raises(self):
        HeadlessDecisionState.hitler_intervention_stage = "INVALID"

        with self.assertRaises(ValueError):
            encode_hitler_intervention_state(
                self.observation
            )

    def test_encode_hitler_intervention_only_changes_its_block(self):
        HeadlessDecisionState.hitler_intervention_stage = "REDEPLOY"
        HeadlessDecisionState.hitler_intervention_target = US_FIRST_ARMY
        HeadlessDecisionState.hitler_intervention_redeployments = 2

        encode_hitler_intervention_state(
            self.observation
        )

        self.assertEqual(
            np.count_nonzero(
                self.observation[:HITLER_STATE_START]
            ),
            0,
        )

        self.assertEqual(
            np.count_nonzero(
                self.observation[HITLER_STATE_END:]
            ),
            0,
        )


if __name__ == "__main__":
    unittest.main()

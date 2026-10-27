import unittest

import numpy as np

from core.allied_armies import (
    US_FIRST_ARMY,
    BRITISH_SECOND_ARMY,
    CANADIAN_FIRST_ARMY,
)

from core.map.map_spaces_us_1 import (
    carentan,
    cherbourg,
)

from core.map.map_spaces_brit_2 import (
    bayeux,
)

from core.ai.dqn.dqn_game_state_encoder import (
    OBSERVATION_SIZE,
    DECISION_TYPE_START,
    DECISION_TYPE_END,
    DECISION_CONTEXT_START,
    DECISION_CONTEXT_END,
    DECISION_ACTION_PHASE_INDEX,
    DECISION_RESERVE_DEPLOYMENT_INDEX,
    DECISION_CASUALTY_INDEX,
    DECISION_HITLER_INTERVENTION_INDEX,
    DECISION_ARMY_START,
    DECISION_ARMY_END,
    DECISION_SPACE_START,
    DECISION_SPACE_END,
    DECISION_ALLIED_ATTACK_INDEX,
    DECISION_GERMAN_COUNTERATTACK_INDEX,
    DecisionType,
    DecisionContext,
    encode_decision_type,
    encode_decision_context,
)


class TestEncodeDecisionState(unittest.TestCase):

    def setUp(self):
        self.observation = np.zeros(
            OBSERVATION_SIZE,
            dtype=np.float32,
        )

    # ------------------------------------------------------------
    # DECISION TYPE
    # ------------------------------------------------------------

    def test_encode_decision_type_none_context(self):
        encode_decision_type(
            self.observation,
            None,
        )

        self.assertEqual(
            np.count_nonzero(
                self.observation[
                    DECISION_TYPE_START:
                    DECISION_TYPE_END
                ]
            ),
            0,
        )

    def test_encode_decision_type_none_type(self):
        context = DecisionContext()

        encode_decision_type(
            self.observation,
            context,
        )

        self.assertEqual(
            np.count_nonzero(
                self.observation[
                    DECISION_TYPE_START:
                    DECISION_TYPE_END
                ]
            ),
            0,
        )

    def test_encode_decision_type_action_phase(self):
        context = DecisionContext(
            decision_type=DecisionType.ACTION_PHASE,
        )

        encode_decision_type(
            self.observation,
            context,
        )

        self.assertEqual(
            self.observation[
                DECISION_ACTION_PHASE_INDEX
            ],
            1.0,
        )

        self.assertEqual(
            np.count_nonzero(
                self.observation[
                    DECISION_TYPE_START:
                    DECISION_TYPE_END
                ]
            ),
            1,
        )

    def test_encode_decision_type_reserve_deployment(self):
        context = DecisionContext(
            decision_type=DecisionType.RESERVE_DEPLOYMENT,
        )

        encode_decision_type(
            self.observation,
            context,
        )

        self.assertEqual(
            self.observation[
                DECISION_RESERVE_DEPLOYMENT_INDEX
            ],
            1.0,
        )

        self.assertEqual(
            np.count_nonzero(
                self.observation[
                    DECISION_TYPE_START:
                    DECISION_TYPE_END
                ]
            ),
            1,
        )

    def test_encode_decision_type_casualty(self):
        context = DecisionContext(
            decision_type=DecisionType.CASUALTY,
        )

        encode_decision_type(
            self.observation,
            context,
        )

        self.assertEqual(
            self.observation[
                DECISION_CASUALTY_INDEX
            ],
            1.0,
        )

        self.assertEqual(
            np.count_nonzero(
                self.observation[
                    DECISION_TYPE_START:
                    DECISION_TYPE_END
                ]
            ),
            1,
        )

    def test_encode_decision_type_hitler_intervention(self):
        context = DecisionContext(
            decision_type=DecisionType.HITLER_INTERVENTION,
        )

        encode_decision_type(
            self.observation,
            context,
        )

        self.assertEqual(
            self.observation[
                DECISION_HITLER_INTERVENTION_INDEX
            ],
            1.0,
        )

        self.assertEqual(
            np.count_nonzero(
                self.observation[
                    DECISION_TYPE_START:
                    DECISION_TYPE_END
                ]
            ),
            1,
        )

    # ------------------------------------------------------------
    # DECISION CONTEXT
    # ------------------------------------------------------------

    def test_encode_decision_context_none(self):
        encode_decision_context(
            self.observation,
            None,
        )

        self.assertEqual(
            np.count_nonzero(
                self.observation[
                    DECISION_CONTEXT_START:
                    DECISION_CONTEXT_END
                ]
            ),
            0,
        )

    def test_encode_decision_context_relevant_army(self):
        context = DecisionContext(
            relevant_army=US_FIRST_ARMY,
        )

        encode_decision_context(
            self.observation,
            context,
        )

        self.assertEqual(
            self.observation[
                DECISION_ARMY_START
            ],
            1.0,
        )

        self.assertEqual(
            np.count_nonzero(
                self.observation[
                    DECISION_ARMY_START:
                    DECISION_ARMY_END
                ]
            ),
            1,
        )

    def test_encode_decision_context_different_armies(self):
        armies = [
            US_FIRST_ARMY,
            BRITISH_SECOND_ARMY,
            CANADIAN_FIRST_ARMY,
        ]

        for expected_index, army in enumerate(armies):
            observation = np.zeros(
                OBSERVATION_SIZE,
                dtype=np.float32,
            )

            context = DecisionContext(
                relevant_army=army,
            )

            encode_decision_context(
                observation,
                context,
            )

            self.assertEqual(
                observation[
                    DECISION_ARMY_START
                    + expected_index
                ],
                1.0,
            )

            self.assertEqual(
                np.count_nonzero(
                    observation[
                        DECISION_ARMY_START:
                        DECISION_ARMY_END
                    ]
                ),
                1,
            )

    def test_encode_decision_context_relevant_space(self):
        context = DecisionContext(
            relevant_space=carentan,
        )

        encode_decision_context(
            self.observation,
            context,
        )

        self.assertEqual(
            self.observation[
                DECISION_SPACE_START
            ],
            1.0,
        )

        self.assertEqual(
            np.count_nonzero(
                self.observation[
                    DECISION_SPACE_START:
                    DECISION_SPACE_END
                ]
            ),
            1,
        )

    def test_encode_decision_context_different_spaces(self):
        spaces = [
            carentan,
            cherbourg,
            bayeux,
        ]

        expected_offsets = [
            0,
            2,
            10,
        ]

        for space, expected_offset in zip(
            spaces,
            expected_offsets,
        ):
            observation = np.zeros(
                OBSERVATION_SIZE,
                dtype=np.float32,
            )

            context = DecisionContext(
                relevant_space=space,
            )

            encode_decision_context(
                observation,
                context,
            )

            self.assertEqual(
                observation[
                    DECISION_SPACE_START
                    + expected_offset
                ],
                1.0,
            )

            self.assertEqual(
                np.count_nonzero(
                    observation[
                        DECISION_SPACE_START:
                        DECISION_SPACE_END
                    ]
                ),
                1,
            )

    def test_encode_decision_context_allied_attack(self):
        context = DecisionContext(
            allied_attack=True,
        )

        encode_decision_context(
            self.observation,
            context,
        )

        self.assertEqual(
            self.observation[
                DECISION_ALLIED_ATTACK_INDEX
            ],
            1.0,
        )

        self.assertEqual(
            self.observation[
                DECISION_GERMAN_COUNTERATTACK_INDEX
            ],
            0.0,
        )

    def test_encode_decision_context_german_counterattack(self):
        context = DecisionContext(
            german_counterattack=True,
        )

        encode_decision_context(
            self.observation,
            context,
        )

        self.assertEqual(
            self.observation[
                DECISION_ALLIED_ATTACK_INDEX
            ],
            0.0,
        )

        self.assertEqual(
            self.observation[
                DECISION_GERMAN_COUNTERATTACK_INDEX
            ],
            1.0,
        )

    def test_encode_decision_context_complete_context(self):
        context = DecisionContext(
            decision_type=DecisionType.CASUALTY,
            relevant_army=US_FIRST_ARMY,
            relevant_space=carentan,
            allied_attack=True,
            german_counterattack=False,
        )

        encode_decision_type(
            self.observation,
            context,
        )

        encode_decision_context(
            self.observation,
            context,
        )

        self.assertEqual(
            self.observation[
                DECISION_CASUALTY_INDEX
            ],
            1.0,
        )

        self.assertEqual(
            self.observation[
                DECISION_ARMY_START
            ],
            1.0,
        )

        self.assertEqual(
            self.observation[
                DECISION_SPACE_START
            ],
            1.0,
        )

        self.assertEqual(
            self.observation[
                DECISION_ALLIED_ATTACK_INDEX
            ],
            1.0,
        )

        self.assertEqual(
            self.observation[
                DECISION_GERMAN_COUNTERATTACK_INDEX
            ],
            0.0,
        )

    def test_encode_decision_state_only_changes_decision_blocks(self):
        context = DecisionContext(
            decision_type=DecisionType.CASUALTY,
            relevant_army=US_FIRST_ARMY,
            relevant_space=carentan,
            allied_attack=True,
        )

        encode_decision_type(
            self.observation,
            context,
        )

        encode_decision_context(
            self.observation,
            context,
        )

        self.assertEqual(
            np.count_nonzero(
                self.observation[
                    :DECISION_TYPE_START
                ]
            ),
            0,
        )

        self.assertEqual(
            np.count_nonzero(
                self.observation[
                    DECISION_CONTEXT_END:
                ]
            ),
            0,
        )


if __name__ == "__main__":
    unittest.main()

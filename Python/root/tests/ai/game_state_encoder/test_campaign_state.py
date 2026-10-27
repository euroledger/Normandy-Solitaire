import unittest

import numpy as np

from cards.card_10 import card as card_010
from cards.card_20 import card as card_020
from cards.card_30 import card as card_030
from core.global_game_state import GlobalGameState

from core.ai.dqn.dqn_game_state_encoder import (
    OBSERVATION_SIZE,
    CARDS_DRAWN_INDEX,
    CURRENT_CARD_START,
    CURRENT_CARD_END,
    DRAWN_CARDS_START,
    DRAWN_CARDS_END,
    MID_DECK_ADDED_INDEX,
    LATE_DECK_ADDED_INDEX,
    encode_campaign_state,
    print_observation_summary,
)


class TestEncodeCampaignState(unittest.TestCase):

    def setUp(self):
        GlobalGameState.cards_drawn = 0
        GlobalGameState.current_card = None
        GlobalGameState.drawn_cards = []
        GlobalGameState.mid_deck_added = False
        GlobalGameState.late_deck_added = False

        self.observation = np.zeros(
            OBSERVATION_SIZE,
            dtype=np.float32,
        )

    def test_encode_campaign_state_initial_state(self):
        encode_campaign_state(self.observation)

        self.assertEqual(
            self.observation[CARDS_DRAWN_INDEX],
            0.0,
        )

        self.assertTrue(
            np.all(
                self.observation[
                    CURRENT_CARD_START:CURRENT_CARD_END
                ] == 0.0
            )
        )

        self.assertTrue(
            np.all(
                self.observation[
                    DRAWN_CARDS_START:DRAWN_CARDS_END
                ] == 0.0
            )
        )

        self.assertEqual(
            self.observation[MID_DECK_ADDED_INDEX],
            0.0,
        )

        self.assertEqual(
            self.observation[LATE_DECK_ADDED_INDEX],
            0.0,
        )

    def test_encode_campaign_state_cards_drawn(self):
        GlobalGameState.cards_drawn = 24

        encode_campaign_state(self.observation)

        self.assertEqual(
            self.observation[CARDS_DRAWN_INDEX],
            0.5,
        )
        print_observation_summary(self.observation)

    def test_encode_campaign_state_current_card(self):
        GlobalGameState.current_card = card_020

        encode_campaign_state(self.observation)

        current_card_state = self.observation[
            CURRENT_CARD_START:CURRENT_CARD_END
        ]

        self.assertEqual(
            np.count_nonzero(current_card_state),
            1,
        )

        self.assertEqual(
            self.observation[
                CURRENT_CARD_START + 19
            ],
            1.0,
        )

    def test_encode_campaign_state_drawn_cards(self):
        GlobalGameState.drawn_cards = [
            card_010,
            card_020,
            card_030,
        ]

        encode_campaign_state(self.observation)

        drawn_card_state = self.observation[
            DRAWN_CARDS_START:DRAWN_CARDS_END
        ]

        self.assertEqual(
            np.count_nonzero(drawn_card_state),
            3,
        )

        self.assertEqual(
            self.observation[
                DRAWN_CARDS_START + 9
            ],
            1.0,
        )

        self.assertEqual(
            self.observation[
                DRAWN_CARDS_START + 19
            ],
            1.0,
        )

        self.assertEqual(
            self.observation[
                DRAWN_CARDS_START + 29
            ],
            1.0,
        )

    def test_encode_campaign_state_deck_flags(self):
        GlobalGameState.mid_deck_added = True
        GlobalGameState.late_deck_added = True

        encode_campaign_state(self.observation)

        self.assertEqual(
            self.observation[MID_DECK_ADDED_INDEX],
            1.0,
        )

        self.assertEqual(
            self.observation[LATE_DECK_ADDED_INDEX],
            1.0,
        )


if __name__ == "__main__":
    unittest.main()

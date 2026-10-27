from random import shuffle

from core.global_game_state import GlobalGameState
from core.map.map_utilities import do_opening_setup


def reset_game():
    GlobalGameState.headless = True
    GlobalGameState.cards_drawn = 0
    GlobalGameState.drawn_cards = []
    GlobalGameState.current_card = None
    GlobalGameState.current_weather = None
    GlobalGameState.current_step = 1
    GlobalGameState.mid_deck_added = False
    GlobalGameState.late_deck_added = False

    import cards.decks

    draw_deck = list(cards.decks.draw_deck)
    mid_deck = list(cards.decks.mid_deck)
    late_deck = list(cards.decks.late_deck)

    do_opening_setup()

    opening_cards = draw_deck[:2]
    random_cards = draw_deck[2:]
    shuffle(random_cards)
    draw_deck[:] = opening_cards + random_cards

    return draw_deck, mid_deck, late_deck
  
import numpy as np
from core.actions.counter_attack_action import get_counter_attack_options
from core.ai.headless_action_masks import execute_flat_hitler_intervention_ai, get_hitler_intervention_mask, get_mini_action_mask, execute_flat_action_ai
from datetime import datetime

from core.actions.actions_menu import do_action_phase, common_pre_action_phase, set_available_actions
from core.ai.headless_env_manager import inject_ai_turn_arguments
from core.card_utilities import remove_meyer, remove_model, remove_rommel, remove_wittmann
from core.game_constants import CYAN, RESET
from core.global_game_state import GlobalGameState
from core.save_load_game import save_game


def action_phase_fork():
    if GlobalGameState.headless == False:
        action_phase_manual()
    else:
        action_phase_ai()

def action_phase_ai():
    proceed = common_pre_action_phase(
        GlobalGameState.current_card,
        GlobalGameState.current_weather
    )

    if proceed == False:
        while True:
            action_mask = get_hitler_intervention_mask(
                GlobalGameState.current_card
            )
            legal_actions = np.where(action_mask == 1.0)[0]
            ai_chosen_flat_id = np.random.choice(legal_actions)
            inject_ai_turn_arguments([""])

            action_success = execute_flat_hitler_intervention_ai(
                ai_chosen_flat_id,
                GlobalGameState.current_card,
                GlobalGameState.current_weather
            )

            if not action_success:
                break

        if GlobalGameState.current_card.card_id != 32:
            common_post_action_phase()
            return

        set_available_actions(GlobalGameState.current_card)

    while True:
        action_mask = get_mini_action_mask()
        legal_actions = np.where(action_mask == 1.0)[0]
        tactical_actions = [aid for aid in legal_actions if aid != 0]

        if not tactical_actions:
            break

        ai_chosen_flat_id = np.random.choice(tactical_actions)
        target_option = get_counter_attack_options()[ai_chosen_flat_id - 1]
        execute_flat_action_ai(
            ai_chosen_flat_id,
            target_option=target_option
        )

    common_post_action_phase()
        
def action_phase_manual():
    print(CYAN)
    do_action_phase(
        GlobalGameState.current_card,
        GlobalGameState.current_weather,
    )
    print(RESET)
    input("Press ENTER to continue...")
    common_post_action_phase()


def common_post_action_phase():
    # AutoSave game at end of each turn
    # save_name = (f"{datetime.now():%y-%m-%d}-end-turn{GlobalGameState.cards_drawn}-card{GlobalGameState.current_card.card_id}").upper()
    GlobalGameState.cards_drawn += 1

    remove_wittmann()
    remove_meyer()
    remove_rommel()
    remove_model()

    GlobalGameState.current_step = 1
    GlobalGameState.counter_attacked_armies.clear()

    if GlobalGameState.headless == False:
        save_name = f"{datetime.now():%d-%b}-END-TURN-{GlobalGameState.cards_drawn}-CARD-{GlobalGameState.current_card.card_id}".upper()
        save_game(save_name)

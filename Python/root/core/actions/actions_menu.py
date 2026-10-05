from random import randint

from core.actions.actions_helper import do_panzer_transport_check
from core.actions.counter_attack_action import do_counter_attack
from core.actions.fortified_villages_action import do_build_fortified_villages
from core.actions.hitler_intervention import check_hitler_intervention_applies, do_hitler_intervention_redeploy, do_hitler_intervention_attack
from core.actions.move_action_point_to_reserve import do_move_action_point_to_strategic_reserve
from core.actions.move_unit_one_space_action import do_move_unit_one_space
from core.actions.refit_panzer_division_action import do_refit_panzer_division
from core.actions.resource_actions import do_resource_augmentation_roll
from core.actions.strategic_reserve_actions import do_move_other_unit_from_strategic_reserve, do_move_panzer_from_strategic_reserve, do_move_panzer_to_strategic_reserve, get_panzer_divisions_in_strategic_reserve
from core.ai.headless_env_manager import get_automated_choice
from core.enums import ReinforcementType
from core.game_summary import print_game_summary
from core.german_units import SS_12, TIGER_101
from core.global_game_state import GlobalGameState
from core.map.map_model import (
    in_transit_box,
    strategic_reserve_box,
)
# from game_phases.action_phase import common_post_action_phase

YELLOW = "\033[33m"
GREEN = "\033[32m"
RESET = "\033[0m"


def list_user_actions():
    print(YELLOW)
    print("AVAILABLE ACTIONS")
    if GlobalGameState.reserve_actions > 0:
        print(f"ACTIONS REMAINING: {GlobalGameState.actions_left_this_turn} (+ {GlobalGameState.reserve_actions} IN RESERVE)")
    else:
        print(f"ACTIONS REMAINING: {GlobalGameState.actions_left_this_turn}")
    print()

    if TIGER_101 in strategic_reserve_box.units:
        print(f"{GREEN}*** 101st TIGER BN AVAILABLE IN STRATEGIC RESERVE ***{YELLOW}")

    if GlobalGameState.meyer_available and SS_12 in strategic_reserve_box.units:
        print(f"{GREEN}*** MEYER AVAILABLE WITH 12th SS PANZER IN STRATEGIC RESERVE ***{YELLOW}")

    print()

    print("1. Counter-Attack (1)")
    print("2. Move Panzer Division From Strategic Reserve (1)")
    print("3. Move Panzer Division To Strategic Reserve (1)")
    print("4. Move Other Unit from Strategic Reserve (0)")
    print("5. Do Resource Augmentation Roll (1)")
    print("6. Place or Upgrade Fortified Villages Marker (3)")
    print("7. Move Unit One Space (1)")
    print("8. Refit Panzer Division (1)")
    print("9. Move Action Point to Strategic Reserve (1)")
    if GlobalGameState.actions_left_this_turn == 0:
        print("0. End Turn")
    print("G. Game Summary")

    print(RESET)


def choose_user_action():
    choice = get_automated_choice("Choose Action: ", ai_slot=0, default_fallback="0")

    if choice == "1":
        do_counter_attack()
    elif choice == "2":
        do_move_panzer_from_strategic_reserve(randint(1, 6))
    elif choice == "3":
        do_move_panzer_to_strategic_reserve(randint(1, 6))
    elif choice == "4":
        do_move_other_unit_from_strategic_reserve()
    elif choice == "5":
        do_resource_augmentation_roll()
    elif choice == "6":
        do_build_fortified_villages()
    elif choice == "7":
        do_move_unit_one_space()
    elif choice == "8":
        do_refit_panzer_division()
    elif choice == "9":
        do_move_action_point_to_strategic_reserve()
    elif choice == "0":
        return False
    elif choice == "G" or choice == "g":
        print_game_summary()
        print()
        input("Press ENTER to continue...")
        return True
    else:
        print("INVALID ACTION")
        return True
    return True


def set_available_actions(card):
    GlobalGameState.actions_left_this_turn = card.actions.actions_available
    for effect in card.actions.conditional_actions:
        if effect.condition is not None and effect.condition.is_met(GlobalGameState):
            GlobalGameState.actions_left_this_turn += effect.value


def common_pre_action_phase(card, weather):
    # Executes mandatory card overrides and structural troop arrivals.
       
    # Execute In Transit Resource Rolls for arrived Panzers
    panzer_divisions = [unit for unit in in_transit_box.units[:] if unit.type == ReinforcementType.PZ_DIV]
    for unit in panzer_divisions:
        die_roll = randint(1, 6)
        print("IN TRANSIT CHECK")
        print("================")
        do_panzer_transport_check(unit, die_roll)

    if card.hitler_intervention:
        # If running headlessly, DO NOT let the game engine execute the choice.
        # Return False to halt normal actions, but let action_phase_ai handle it.
        if GlobalGameState.headless:
            return False
        target_army = check_hitler_intervention_applies(card)
        if target_army is not None:
            target_army, attacking_space = do_hitler_intervention_redeploy(card, target_army)
            do_hitler_intervention_attack(card, weather, target_army, attacking_space)
            return False  # Returns False indicating the phase was overridden/ended by Hitler
        
    # Load Action Points from Card modifiers
    set_available_actions(card)
    print(f"ACTIONS AVAILABLE: {GlobalGameState.actions_left_this_turn}")
    return True  # Returns True indicating the turn proceeds to tactical choices

def do_action_phase(card, weather):
    proceed = common_pre_action_phase(GlobalGameState.current_card, GlobalGameState.current_weather)

    if proceed == False:
        # ◄── FIX: Intercepts the forced historical events
        # execute_flat_hitler_intervention_ai(GlobalGameState.current_card, GlobalGameState.current_weather)
        
        # TODO add calls to execute_flat_hitler_intervention_ai for each phase
        # ◄── FIX: Triggers card/turn counters and cleanup scripts
        # common_post_action_phase()
        return  # ◄── Safely exits the function, advancing to the next card draw.
    cont = True
    while cont and GlobalGameState.actions_left_this_turn + GlobalGameState.reserve_actions > 0:
        # 4. List Menu of Actions
        list_user_actions()

        # 5. Choose an action -> Implement Action
        cont = choose_user_action()

    print()
    print("NO ACTIONS LEFT")

from core.actions.actions_helper import build_display_spaces, get_german_controlled_spaces
from core.actions.actions_helper import print_display_spaces
from core.enums import SideType
from core.global_game_state import GlobalGameState
from core.map.map_model import TerrainType


def get_fortified_village_options():
    return [
        space
        for space in get_german_controlled_spaces()
        if space.controlling_player == SideType.GERMAN
        and space.terrain != TerrainType.FORTRESS
        and space.fortified_village_modifier < 2
    ]

def print_fortified_village_options(spaces):
    print()
    print("SELECT GERMAN-CONTROLLED SPACE")
    print()
    for index, space in enumerate(spaces, start=1):
        print(f"{index}. {space.name} (FORTIFIED VILLAGES +{space.fortified_village_modifier})")
    print()
    print("0. Return to main menu")


def do_build_fortified_villages(space=None):
    print("PLACE / UPGRADE FORTIFIED VILLAGES")
    print()

    total_actions = GlobalGameState.actions_left_this_turn + GlobalGameState.reserve_actions
    if total_actions < 3:
        print("NOT ENOUGH ACTIONS")
        print(f"ACTIONS AVAILABLE: {GlobalGameState.actions_left_this_turn}")
        print(f"RESERVE ACTIONS: {GlobalGameState.reserve_actions}")
        return

    spaces = get_fortified_village_options()
    if not spaces:
        print("NO VALID SPACES")
        return

    if space is None:
        display_spaces = build_display_spaces(spaces)
        print_display_spaces(display_spaces)
        space_choice = input("Choice: ").strip()

        if space_choice == "0":
            return

        if not space_choice.isdigit():
            print("INVALID CHOICE")
            return

        selected_index = int(space_choice) - 1
        if selected_index < 0 or selected_index >= len(display_spaces):
            print("INVALID CHOICE")
            return

        selected_space = display_spaces[selected_index]
    else:
        selected_space = space

    if selected_space.fortified_village_modifier == 2:
        print("FORTIFIED VILLAGE IS ALREADY +2")
        return

    if selected_space.fortified_village_modifier == 0:
        selected_space.fortified_village_modifier = 1
        print(f"{selected_space.name} FORTIFIED VILLAGES +1")
    elif selected_space.fortified_village_modifier == 1:
        selected_space.fortified_village_modifier = 2
        print(f"{selected_space.name} FORTIFIED VILLAGES UPGRADED TO +2")

    actions_to_use = 3
    actions_used = min(GlobalGameState.actions_left_this_turn, actions_to_use)
    GlobalGameState.actions_left_this_turn -= actions_used
    GlobalGameState.reserve_actions -= actions_to_use - actions_used

    print()
    print(f"ACTIONS USED: {actions_used}")
    print(f"ACTIONS REMAINING: {GlobalGameState.actions_left_this_turn}")
    print(f"RESERVE ACTIONS USED: {actions_to_use - actions_used}")
    print(f"RESERVE ACTIONS REMAINING: {GlobalGameState.reserve_actions}")

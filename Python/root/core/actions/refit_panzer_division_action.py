from core.actions.strategic_reserve_actions import get_panzer_divisions_in_strategic_reserve
from core.global_game_state import GlobalGameState


def do_refit_panzer_division(unit_choice=None):
    panzer_divisions = get_panzer_divisions_in_strategic_reserve()
    panzer_divisions = [unit for unit in panzer_divisions if unit.combat_value == 1]

    if not panzer_divisions:
        print("NO PANZER DIVISIONS AVAILABLE TO REFIT")
        return

    print("REFIT PANZER DIVISION")
    print()

    for index, unit in enumerate(panzer_divisions, start=1):
        print(f"{index}. {unit.name} (+1)")

    print()
    print("0. Return to main menu")

    if unit_choice is None:
        unit_choice = input("Choice: ").strip()
    else:
        unit_choice = str(unit_choice).strip()

    if unit_choice == "0":
        return

    if not unit_choice.isdigit():
        print("INVALID CHOICE")
        return

    selected_index = int(unit_choice) - 1

    if selected_index < 0 or selected_index >= len(panzer_divisions):
        print("INVALID CHOICE")
        return

    selected_unit = panzer_divisions[selected_index]
    selected_unit.combat_value = 2
    GlobalGameState.actions_left_this_turn -= 1

    print(f"{selected_unit.name} REFITTED TO FULL STRENGTH (+2)")
    print(f"ACTIONS REMAINING: {GlobalGameState.actions_left_this_turn}")

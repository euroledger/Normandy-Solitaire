from core.actions.actions_helper import RED, RESET, can_add_unit_to_space, get_display_spaces, get_german_controlled_spaces, use_action
from core.actions.stacking_limits import PANZER_STACKING_LIMIT
from core.german_units import MEYER, SS_12
from core.models import GermanUnit, ReinforcementType
from core.global_game_state import GlobalGameState
from core.map.map_model import (
    MapSpace,
    eliminated_units_box,
    hitler_approval_track,
    transport_track,
    in_transit_box,
    strategic_reserve_box,
)
from core.map.map_utilities import (
    get_all_map_spaces,
)


def get_panzer_divisions_on_map():
    panzer_divisions = []
    seen_spaces = set()
    for space in get_all_map_spaces():
        if id(space) in seen_spaces:
            continue
        seen_spaces.add(id(space))
        if space in [in_transit_box, strategic_reserve_box, eliminated_units_box]:
            continue
        if space.under_siege:
            continue
        for unit in space.units:
            if isinstance(unit, GermanUnit) and unit.type == ReinforcementType.PZ_DIV:
                panzer_divisions.append((space, unit))
    return panzer_divisions


def get_panzer_divisions_in_strategic_reserve():
    return [
        unit
        for unit in strategic_reserve_box.units
        if isinstance(unit, GermanUnit) and unit.type in [ReinforcementType.PZ_DIV]
    ]


def get_other_units_in_strategic_reserve():
    return [
        unit
        for unit in strategic_reserve_box.units
        if isinstance(unit, GermanUnit) and unit.type != ReinforcementType.PZ_DIV
    ]


def do_move_other_unit_from_strategic_reserve(
    unit_choice=None,
    space_choice=None
):
    print("MOVE OTHER UNIT FROM STRATEGIC RESERVE")
    print()

    units = get_other_units_in_strategic_reserve()

    if not units:
        print("No Units in Strategic Reserve")
        return False

    print("SELECT UNIT")
    print()

    for index, unit in enumerate(units, start=1):
        print(f"{index}. {unit.name} ({unit.combat_value})")

    print()
    print("0. Return to main menu")
    print()

    # ---------------------------------------------------------
    # SELECT UNIT
    # ---------------------------------------------------------

    if isinstance(unit_choice, GermanUnit):
        if unit_choice not in units:
            print("INVALID CHOICE")
            return False

        selected_unit = unit_choice

    else:
        if unit_choice is None:
            unit_choice = input("Choice: ").strip()
        else:
            unit_choice = str(unit_choice).strip()

        if unit_choice == "0":
            return False

        if not unit_choice.isdigit():
            print("INVALID CHOICE")
            return False

        selected_index = int(unit_choice) - 1

        if selected_index < 0 or selected_index >= len(units):
            print("INVALID CHOICE")
            return False

        selected_unit = units[selected_index]

    print()
    print(f"SELECTED: {selected_unit.name}")

    # ---------------------------------------------------------
    # SELECT DESTINATION
    # ---------------------------------------------------------

    german_spaces = get_german_controlled_spaces()

    if not german_spaces:
        print("No German-controlled spaces available")
        return False

    display_spaces = get_display_spaces(space_choice is None)

    if display_spaces is None:
        return False

    if isinstance(space_choice, MapSpace):
        if space_choice not in display_spaces:
            print("INVALID CHOICE")
            return False

        selected_space = space_choice

    else:
        if space_choice is None:
            space_choice = input("Choice: ").strip()
        else:
            space_choice = str(space_choice).strip()

        if space_choice == "0":
            return False

        if not space_choice.isdigit():
            print("INVALID CHOICE")
            return False

        selected_index = int(space_choice) - 1

        if selected_index < 0 or selected_index >= len(display_spaces):
            print("INVALID CHOICE")
            return False

        selected_space = display_spaces[selected_index]

    # ---------------------------------------------------------
    # STACKING
    # ---------------------------------------------------------

    if not can_add_unit_to_space(selected_space, selected_unit):
        print("STACKING LIMIT REACHED, INVALID MOVE")
        return True

    # ---------------------------------------------------------
    # MOVE UNIT
    # ---------------------------------------------------------

    print()
    print(f"{selected_unit.name} moved to {selected_space.name}")

    strategic_reserve_box.units.remove(selected_unit)
    selected_space.units.append(selected_unit)

    return True


def do_move_panzer_to_strategic_reserve(die_roll, div_choice=None):
    print("MOVE PANZER DIVISION TO STRATEGIC RESERVE")
    print()

    panzer_divisions = get_panzer_divisions_on_map()

    if not panzer_divisions:
        print("No Panzer Divisions on the map")
        return

    print("SELECT PANZER DIVISION")
    print()

    for index, (space, unit) in enumerate(panzer_divisions, start=1):
        print(f"{index}. {unit.name} ({unit.combat_value}) - {space.name}")

    print()
    print("0. Return to main menu")
    print()

    if isinstance(div_choice, GermanUnit):
        selected_pair = next(
            (
                (space, unit)
                for space, unit in panzer_divisions
                if unit is div_choice
            ),
            None
        )

        if selected_pair is None:
            print("INVALID CHOICE")
            return

        selected_space, selected_panzer = selected_pair

    else:
        if div_choice is None:
            div_choice = input("Choice: ").strip()
        else:
            div_choice = str(div_choice).strip()

        if div_choice == "0":
            return

        if not div_choice.isdigit():
            print("INVALID CHOICE")
            return

        selected_index = int(div_choice) - 1

        if selected_index < 0 or selected_index >= len(panzer_divisions):
            print("INVALID CHOICE")
            return

        selected_space, selected_panzer = panzer_divisions[selected_index]

    if not use_action():
        print("NOT ENOUGH ACTIONS")
        return

    print()
    print(f"SELECTED: {selected_panzer.name}")
    print()
    print("TRANSPORT CHECK")
    print(f"ROLL: {die_roll}")
    print(f"TRANSPORT LEVEL: {transport_track.value}")

    if die_roll > transport_track.value:
        print(f"{RED}RESULT: FAILED{RESET}")
        return

    print("RESULT: PASSED")
    print()
    print(f"{selected_panzer.name} moved to Strategic Reserve")

    selected_space.units.remove(selected_panzer)
    strategic_reserve_box.units.append(selected_panzer)

    if selected_panzer == SS_12 and MEYER in selected_space.units:
        selected_space.units.remove(MEYER)
        GlobalGameState.meyer_available = True
        print(
            "MEYER REMOVED - "
            "12th SS PANZER MOVED TO STRATEGIC RESERVE"
        )

# TODO move this into separate helper file to be used in move one unit action
# Need to add types of units to check (eg Panzer, FlaK, Fallschirmjager etc)
def check_stacking(space):
    panzer_units = [
        unit
        for unit in space.units
        if isinstance(unit, GermanUnit)
        and unit.type in [ReinforcementType.PZ_DIV, ReinforcementType.KAMPFGRUPPE]
    ]

    if len(panzer_units) >= PANZER_STACKING_LIMIT:
        print("STACKING LIMIT REACHED, INVALID MOVE")
        return False

    return True


def do_move_panzer_from_strategic_reserve(die_roll, div_choice=None, space_choice=None):
    print("MOVE PANZER DIVISION FROM STRATEGIC RESERVE")
    print()
    panzer_divisions = get_panzer_divisions_in_strategic_reserve()
    if not panzer_divisions:
        print("No Panzer Divisions in Strategic Reserve")
        return
    print("SELECT PANZER DIVISION")
    print()
    for index, unit in enumerate(panzer_divisions, start=1):
        print(f"{index}. {unit.name} ({unit.combat_value})")
    print()
    print("0. Return to main menu")
    print()
    if isinstance(div_choice, GermanUnit):
        if div_choice not in panzer_divisions:
            print("INVALID CHOICE")
            return
        selected_panzer = div_choice
    else:
        if div_choice is None:
            div_choice = input("Choice: ").strip()
        else:
            div_choice = str(div_choice).strip()
        if div_choice == "0":
            return
        if not div_choice.isdigit():
            print("INVALID CHOICE")
            return
        selected_index = int(div_choice) - 1
        if selected_index < 0 or selected_index >= len(panzer_divisions):
            print("INVALID CHOICE")
            return
        selected_panzer = panzer_divisions[selected_index]
    
    print()
    print(f"SELECTED: {selected_panzer.name}")
    german_spaces = get_german_controlled_spaces()
    if not german_spaces:
        print("No German-controlled spaces available")
        return

    display_spaces = get_display_spaces(space_choice is None)
    if display_spaces is None:
        return

    if isinstance(space_choice, MapSpace):
        if space_choice not in display_spaces:
            print("INVALID CHOICE")
            return
        selected_space = space_choice
    else:
        if space_choice is None:
            space_choice = input("Choice: ").strip()
        else:
            space_choice = str(space_choice).strip()

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

    if not can_add_unit_to_space(selected_space, selected_panzer):
        print("STACKING LIMIT REACHED, INVALID MOVE")
        return

    if not use_action():
        print("NOT ENOUGH ACTIONS")
        return

    print()
    print("HITLER APPROVAL CHECK")
    print(f"ROLL: {die_roll}")
    print(f"HITLER APPROVAL: {hitler_approval_track.value}")

    if die_roll > hitler_approval_track.value:
        print(f"{RED}RESULT: FAILED{RESET}")
        return

    print("RESULT: PASSED")
    print()
    print(f"{selected_panzer.name} moved to {selected_space.name}")

    strategic_reserve_box.units.remove(selected_panzer)
    selected_space.units.append(selected_panzer)

    if selected_panzer == SS_12 and GlobalGameState.meyer_available:
        selected_space.units.append(MEYER)
        print(f"MEYER DEPLOYED TO {selected_space.name} WITH 12th SS PANZER")
        GlobalGameState.meyer_available = False

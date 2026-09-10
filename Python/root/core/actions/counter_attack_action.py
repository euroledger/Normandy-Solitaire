
from random import randint

from core.actions.actions_helper import BLUE, BOLD, RED, RESET, use_action
from core.allied_armies import US_VIII_CORPS, US_XV_CORPS
from core.card_utilities import calculate_defense_modifiers, get_all_defending_armies
from core.enums import SideType
from core.global_game_state import GlobalGameState
from core.map.map_model import (
    hitler_approval_track,
    supply_track,
    TerrainType,
)
from core.map.map_utilities import (
    can_counter_attack,
    calculate_german_attack_strength,
    get_eligible_german_units,
    update_front_line_for_army,
)
from core.allied_advances_phase import get_front_line_space, get_track_for, do_german_losses, check_and_merge_us_third_army
from core.models import GermanUnit

def choose_attacking_units(units):
    selected = list(units)  # default: all selected

    while True:
        print("\nSELECT ATTACKING UNITS (number of unit to toggle, X=Participating)")
        print("Press ENTER to confirm\n")

        for i, unit in enumerate(units, start=1):
            marker = "[X]" if unit in selected else "[ ]"
            print(f"{i}. {marker} {unit} ({unit.combat_value})")

        choice = input("Choice: ").strip()

        if choice == "":
            break

        if not choice.isdigit():
            continue

        idx = int(choice) - 1
        if 0 <= idx < len(units):
            unit = units[idx]
            if unit in selected:
                selected.remove(unit)
            else:
                selected.append(unit)

    return list(selected)


def get_german_space_facing_front_line(army):
    track = get_track_for(army)
    front_line_space = get_front_line_space(army)
    
    # space = next(space for space in track if space.track_number == front_line_space.track_number - 1)
 # --- DIAGNOSTIC DEBUG BLOCK ---
    print(f"\n[DEBUG] Entering get_german_space_facing_front_line for: {army.name}")
    print(f"[DEBUG] front_line_space: {front_line_space.name} (Track # {front_line_space.track_number})")
    print(f"[DEBUG] Full Track Spaces Available: {[f'{s.name} (Track # {s.track_number})' for s in track]}")

    try:
        # Your original evaluation line statement
        space = next(space for space in track if space.track_number == front_line_space.track_number - 1)
        print(f"[DEBUG] Successfully located backward facing space: {space.name}")
    except StopIteration:
        print(f"\n" + "!" * 60)
        print(f"[CRITICAL DETECTED] StopIteration triggered!")
        print(f"Failed to find a space where track_number == {front_line_space.track_number - 1}")
        print("!" * 60 + "\n")

        # Re-raise the exception so your main script's logging system catches and records it safely
        raise

    return space
   
    return space


def get_counter_attack_options():
    options = []

    for army in get_all_defending_armies():
        if army.name in GlobalGameState.counter_attacked_armies:
            continue

        target_space = get_front_line_space(army)

        if not can_counter_attack(target_space):
            continue

        track = get_track_for(army)
        attacking_space = next((space for space in track if space.track_number == target_space.track_number - 1), None)

        if attacking_space is None:
            continue

        # attack = german_attack_strength(attacking_space)
        eligible = get_eligible_german_units(attacking_space)

        attack = calculate_german_attack_strength(attacking_space, eligible)
        defense = calculate_defense_modifiers(
            card=GlobalGameState.current_card,
            army=army,
            weather=GlobalGameState.current_weather,
        )["defense_strength"]

        options.append(
            {
                "army": army,
                "target_space": target_space,
                "attacking_space": attacking_space,
                "german_attack": attack,
                "allied_defense": defense,
            }
        )

    return options


def print_counter_attack_options(options):
    print()
    if supply_track.value == 0 and any(unit.is_panzer() for option in options for unit in option["attacking_space"].units):
        print(f"{RED}NO SUPPLY: PANZER UNITS CANNOT PARTICIPATE")
    print(f"{BOLD}{BLUE}COUNTER-ATTACK OPTIONS{RESET}")
    print(f"{BLUE}0. Return to main menu{RESET}")

    for index, option in enumerate(options, start=1):
        print(
            f"{BLUE}{index}. {option['army'].name} at "
            f"{option['target_space'].name} "
            f"({option['german_attack']} attack from "
            f"{option['attacking_space'].name} vs "
            f"{option['allied_defense']} defense){RESET}"
        )


def choose_counter_attack_option(options):
    while True:
        choice = input("Choose counter-attack target: ")

        if choice == "0":
            return None

        if not choice.isdigit():
            print("POO INVALID CHOICE")
            continue

        choice_index = int(choice) - 1

        if 0 <= choice_index < len(options):
            return options[choice_index]

        print("POO INVALID CHOICE")


def resolve_counter_attack(attack, defense, selected_units, die_roll):
    print()
    print("RESOLVING COUNTER-ATTACK")
    print(f"Attack Strength: {attack}")
    print(f"Defense Strength: {defense}")
    print()
    print(f"Die Roll: {die_roll}")

    # --- Supply consumption (ANY Panzer unit) ---
    panzer_used = any(isinstance(unit, GermanUnit) and unit.is_panzer() for unit in selected_units)

    if panzer_used:
        supply_track.value -= 1
        print("SUPPLY -1 (Attacking Panzer Unit)")

    # --- Resolution ---
    if die_roll == 6:
        result = "WIN"
        print("Natural 6 → AUTOMATIC GERMAN WIN")

    elif die_roll == 1:
        result = "LOSS"
        print("Natural 1 → AUTOMATIC GERMAN LOSS")

    else:
        total = attack + die_roll
        print(f"Modified Total: {attack} + {die_roll} = {total}")

        if total > defense:
            result = "WIN"
            print("Result: GERMAN WIN")
        else:
            result = "LOSS"
            print("Result: GERMAN LOSS")

    print()

    return {
        "result": result,
        "attack_total": attack + die_roll,
        "defense": defense,
        "natural_roll": die_roll,
    }


def do_post_combat(result, selected_option, selected_units):

    army = selected_option["army"]
    target_space = selected_option["target_space"]
    attacking_space = selected_option["attacking_space"]

    print()
    print("POST-COMBAT")
    print("===========")

    if result["result"] == "WIN":
        print("GERMAN VICTORY")
        
        if attacking_space.under_siege:
            attacking_space.under_siege = False
            print(f"{RED}{attacking_space.name} NO LONGER BESIEGED{RESET}")


        track = get_track_for(army)
        current_space = army.location

        try:
            idx = track.index(current_space)
            retreat_space = track[idx - 1] if idx > 0 else None
        except ValueError:
            retreat_space = None

        if retreat_space:
            current_space.units.remove(army)
            retreat_space.units.append(army)
            army.location = retreat_space

            if army in [US_VIII_CORPS, US_XV_CORPS]:
                check_and_merge_us_third_army(retreat_space)
            update_front_line_for_army(army, retreat_space.track_number)

            print(f"{army.display_name} RETREATS TO {retreat_space.name}")
        else:
            print("NO RETREAT POSSIBLE")

        # --- German Advance (ONLY selected units) ---
        if (
            target_space.terrain != TerrainType.BEACH
            and target_space.terrain != TerrainType.FORTRESS
            and target_space.terrain != TerrainType.START_BOX
        ):
            for unit in selected_units:
                if unit in attacking_space.units:
                    attacking_space.units.remove(unit)
                    target_space.units.append(unit)
            target_space.controlling_player = SideType.GERMAN
            print(f"GERMANS ADVANCE INTO {target_space.name}")
        else:
            print("GERMANS DO NOT ADVANCE INTO BEACH SPACE OR FORTRESS")

        # --- Hitler Approval +1 (max 6) ---
        if hitler_approval_track.value < 6:
            hitler_approval_track.value += 1
            print(f"HITLER APPROVAL +1 → {hitler_approval_track.value}")
        else:
            print("HITLER APPROVAL ALREADY AT MAX (6)")

    else:
        print("GERMAN DEFEAT")

        # --- Apply 1 step loss if units attacking ---
        if selected_units:
            do_german_losses(attacking_space, selected_units)
        else:
            print("NO GERMAN UNITS ENGAGED - NO LOSSES APPLIED")


def do_counter_attack(selected_option=None, selected_units=None):
    options = get_counter_attack_options()

    if not options:
        print("NO VALID COUNTER-ATTACK TARGETS")
        return

    if selected_option is None:
        print_counter_attack_options(options)
        selected_option = choose_counter_attack_option(options)

    if selected_option is None:
        return

    if not use_action():
        print("NOT ENOUGH ACTIONS")
        return

    GlobalGameState.counter_attacked_armies.add(selected_option["army"].name)

    eligible = get_eligible_german_units(selected_option["attacking_space"])

    if selected_units is None:
        selected_units = choose_attacking_units(eligible)

    attack = calculate_german_attack_strength(selected_option["attacking_space"], selected_units)
    defense = selected_option["allied_defense"]

    print(f"COUNTER-ATTACK: {selected_option['army'].name} at {selected_option['target_space'].name}")
    print(f"Final attack strength: {attack} vs defense: {defense}")

    result = resolve_counter_attack(
        attack=attack,
        defense=defense,
        selected_units=selected_units,
        die_roll=randint(1, 6),
    )
    do_post_combat(result, selected_option, selected_units)


from random import randint
from core.actions.counter_attack_action import do_post_combat, get_german_space_facing_front_line, resolve_counter_attack
from core.actions.stacking_limits import PANZER_STACKING_LIMIT
from core.allied_advances_phase import get_track_for
from core.card_utilities import calculate_defense_modifiers
from core.global_game_state import GlobalGameState
from core.map.map_spaces_us_1 import us_1_track
from core.map.map_spaces_us_3 import us_viii_track, us_xv_track
from core.map.map_spaces_can_1 import can_1_track
from core.map.map_spaces_brit_2 import brit_2_track
from core.map.map_model import hitler_approval_track, strategic_reserve_box, TerrainType
from core.allied_armies import US_FIRST_ARMY, US_THIRD_ARMY, US_VIII_CORPS, US_XV_CORPS
from core.map.map_utilities import calculate_german_attack_strength, get_eligible_german_units
from core.models import GermanUnit

def is_valid_target(army):
    if army.location is None:
        return False

    if army.location.terrain in [TerrainType.FORTRESS, TerrainType.BEACH, TerrainType.START_BOX]:
        return False

    target_space = next((space for space in get_track_for(army) if space.track_number == army.location.track_number - 1), None)
    return not (target_space is None or target_space.under_siege)

def get_hitler_intervention_targets(card):
    targets = []

    for army in card.hitler_intervention_target_armies:
        if army == US_THIRD_ARMY:
            if GlobalGameState.us_third_army_activated == False:
                continue
            corps_on_map = [corps for corps in [US_VIII_CORPS, US_XV_CORPS] if corps.location is not None]
            if corps_on_map:
                for corps in corps_on_map:
                    if is_valid_target(corps):
                        targets.append(corps)
            else:
                if US_THIRD_ARMY.location is not None:  # noqa: SIM102
                    if is_valid_target(army):
                        targets.append(army)
            continue

        if is_valid_target(army):
            targets.append(army)
    return targets


def has_us_third_army_taken_st_malo():
    if US_THIRD_ARMY.location is not None:
        return US_THIRD_ARMY.location.track_number <= 7
    if US_VIII_CORPS.location is not None:
        return US_VIII_CORPS.location.track_number <= 7
    return False

def check_hitler_intervention_applies(card, die_roll=None, target_choice=None, cancel_choice=None):
    GlobalGameState.hitler_intervention_no_effect = False
    targets = get_hitler_intervention_targets(card)
    if not targets:
        print("HITLER INTERVENTION - NO EFFECT")
        GlobalGameState.hitler_intervention_no_effect = True
        return None
    
    if card.card_id == 32:
        # Mortain Counter-Attack
        if not GlobalGameState.us_third_army_activated:
            print("HITLER INTERVENTION - NO EFFECT")
            GlobalGameState.hitler_intervention_no_effect = True
            return None

        if not has_us_third_army_taken_st_malo():
            print("HITLER INTERVENTION - NO EFFECT")
            GlobalGameState.hitler_intervention_no_effect = True
            return None
        
        print(f"SELECTED TARGET: {US_FIRST_ARMY.display_name}")
        return US_FIRST_ARMY
        
    if card.card_id != 32:
        if cancel_choice is None:
            cancel_choice = input("ATTEMPT TO CANCEL HITLER INTERVENTION? (Y/N): ").strip().upper()
        else:
            cancel_choice = str(cancel_choice).strip().upper()

        if cancel_choice == "Y":
            actual_die_roll = die_roll if die_roll is not None else randint(1, 6)

            print()
            print("HITLER APPROVAL CHECK")
            print("====================")
            print(f"ROLL: {actual_die_roll}")
            print(f"HITLER APPROVAL: {hitler_approval_track.value}")

            if actual_die_roll <= hitler_approval_track.value:
                print("RESULT: PASSED - INTERVENTION CANCELLED")
                GlobalGameState.hitler_intervention_no_effect = True
                return None

            print("RESULT: FAILED - INTERVENTION PROCEEDS")
            
    if len(targets) == 1:
        selected_target = targets[0]
    else:
        print()
        print("SELECT HITLER INTERVENTION TARGET")
        print("=================================")

        for index, army in enumerate(targets, start=1):
            print(f"{index}. {army.display_name} at {army.location.name}")

        if target_choice is None:
            target_choice = input("Choice: ").strip()

        if isinstance(target_choice, str):
            if not target_choice.isdigit():
                print("INVALID CHOICE")
                return None
            target_choice = int(target_choice)
        if target_choice < 1 or target_choice > len(targets):
            print("INVALID CHOICE")
            return None

        selected_target = targets[target_choice - 1]

    print(f"SELECTED TARGET: {selected_target.display_name}")
    return selected_target


def choose_hitler_intervention_units(available_units, count_needed=2):
    # Sort units so that higher strength divisions (e.g. 21st Panzer over Kampfgruppen) are prioritized
    sorted_units = sorted(available_units, key=lambda x: getattr(x, 'strength', 1), reverse=True)
    return sorted_units[:count_needed]


def create_available_panzers_list(attacking_space):
    available_panzers = []

    # Change your track loop logic to track string names instead of object instances
    seen_spaces = set()
    for track in [us_1_track, brit_2_track, can_1_track, us_viii_track, us_xv_track]:
        for space in track:
            # BUG FIX: Check against the string space name identifier instead of the raw object
            if space.name in seen_spaces or space == attacking_space or space.under_siege:
                continue

            # BUG FIX: Add the string representation to the set
            seen_spaces.add(space.name)

            for unit in space.units:
                if isinstance(unit, GermanUnit) and unit.is_panzer():
                    available_panzers.append((unit, space))


    for unit in strategic_reserve_box.units:
        if isinstance(unit, GermanUnit) and unit.is_panzer():
            available_panzers.append((unit, strategic_reserve_box))
    
    return available_panzers
    

def do_hitler_intervention_redeploy(card, target_army, deployment_choices=None):
    print()
    print("HITLER INTERVENTION")
    print("===================")
    print()
    print(f"TARGET ARMY: {target_army.display_name}")

    attacking_space = get_german_space_facing_front_line(target_army)

    print(f"TARGET LOCATION: {target_army.location.name}")
    print(f"ATTACKING SPACE: {attacking_space.name}\n")

    available_panzers = create_available_panzers_list(attacking_space)
    panzer_count = sum(
        1 for unit in attacking_space.units
        if isinstance(unit, GermanUnit) and unit.is_panzer()
    )
    max_redeployments = min(
        card.hitler_intervention_panzer_count,
        PANZER_STACKING_LIMIT - panzer_count
    )

    provided_choices = deployment_choices is not None
    selected_space_units = []
    redeployed = 0

    for deployment_num in range(max_redeployments):
        if not available_panzers:
            break

        print("AVAILABLE PANZER FORCES")
        print("=======================")
        for index, (unit, space) in enumerate(available_panzers, start=1):
            print(f"{index}. {unit.name} ({unit.combat_value}) - {space.name}")
        print()

        if provided_choices:
            if deployment_num >= len(deployment_choices):
                break
            choice = str(deployment_choices[deployment_num])
        elif GlobalGameState.headless:
            best_selections = choose_hitler_intervention_units(
                available_panzers,
                count_needed=1
            )
            if not best_selections:
                break
            choice = str(available_panzers.index(best_selections[0]) + 1)
        else:
            print(f"Choose Panzer force to redeploy (Choice {deployment_num + 1}): ")
            choice = input("Choice: ").strip()

        if not choice.isdigit():
            continue

        selected_index = int(choice) - 1
        if selected_index < 0 or selected_index >= len(available_panzers):
            print("INVALID CHOICE")
            continue

        selected_unit, selected_space = available_panzers.pop(selected_index)
        selected_space.units.remove(selected_unit)
        attacking_space.units.append(selected_unit)
        selected_space_units.append(selected_unit)
        redeployed += 1

        print(
            f"Moved {selected_unit.name} from {selected_space.name} "
            f"to {attacking_space.name}"
        )

    print(f"REDEPLOYED: {redeployed} PANZER FORCES\n")
    return target_army, attacking_space

def do_hitler_intervention_attack(card, weather, target_army, attacking_space):

    eligible_units = get_eligible_german_units(attacking_space)
    attack = calculate_german_attack_strength(attacking_space, eligible_units)
    defense = calculate_defense_modifiers(card=card, army=target_army, weather=weather)["defense_strength"]

    die_roll = randint(1, 6)

    selected_option = {
        "army": target_army,
        "target_space": target_army.location,
        "attacking_space": attacking_space,
        "german_attack": attack,
        "allied_defense": defense,
    }

    result = resolve_counter_attack(attack, defense, eligible_units, die_roll)
    do_post_combat(result, selected_option, eligible_units)

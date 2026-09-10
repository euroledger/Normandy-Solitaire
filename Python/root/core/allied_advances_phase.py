from core.actions.actions_helper import CYAN, RED, RESET
from core.actions.stacking_limits import FALLSCHIRMJAGER_STACKING_LIMIT, FLAK_88_STACKING_LIMIT, NEBELWERFER_STACKING_LIMIT, PANZER_STACKING_LIMIT
from core.card_utilities import get_armies_as_objects
from random import randint
from core.allied_armies import (
    US_FIRST_ARMY,
    BRITISH_SECOND_ARMY,
    CANADIAN_FIRST_ARMY,
    US_THIRD_ARMY,
    US_VIII_CORPS,
    US_XV_CORPS,
)
from core.game_constants import BLUE, GREEN, LIGHT_BROWN
from core.tables.carpet_bombing import ATTACK_CANCELLED, get_carpet_bombing_result
from core.map.map_model import TerrainType, eliminated_units_box, hitler_approval_track
from core.card_utilities import calculate_attack_modifiers
from core.map.map_utilities import add_units_to_space, german_defense_strength
from random import choice
from core.global_game_state import GlobalGameState
from core.map.map_utilities import update_front_line_for_army
from core.models import GermanUnit, Strategy
from core.enums import Nation, ReinforcementType, SideType
from core.map.map_spaces_us_1 import us_1_track
from core.map.map_spaces_brit_2 import brit_2_track
from core.map.map_spaces_can_1 import can_1_track
from core.map.map_spaces_us_3 import us_viii_track, us_xv_track, st_malo, rennes, us_3_start_box
from core.tables.siege import calculate_siege_drm, get_siege_result
from core.tables.us_third_army_activation import (
    calculate_us_third_army_activation_drm,
    get_us_third_army_activation_result,
)


def get_track_for(army):
    if army.nation == Nation.US_1:
        track = us_1_track
    elif army.nation == Nation.BRIT_2:
        track = brit_2_track
    elif army.nation == Nation.CAN_1:
        track = can_1_track
    elif army.nation == Nation.US_3:
        track = us_xv_track
    elif army.nation == Nation.US_VIII:
        track = us_viii_track
    elif army.nation == Nation.US_XV:
        track = us_xv_track
    else:
        raise ValueError(f"Unknown army nation: {army.nation}")
    return track


def get_front_line_space(army):
    track = get_track_for(army)

    if army == US_FIRST_ARMY:
        front_line = GlobalGameState.us_1_front_line
    elif army == BRITISH_SECOND_ARMY:
        front_line = GlobalGameState.brit_2_front_line
    elif army == CANADIAN_FIRST_ARMY:
        front_line = GlobalGameState.can_1_front_line
    elif army == US_THIRD_ARMY:
        front_line = GlobalGameState.us_3_front_line
    elif army == US_VIII_CORPS:
        front_line = GlobalGameState.us_viii_front_line
    elif army == US_XV_CORPS:
        front_line = GlobalGameState.us_xv_front_line
    else:
        raise ValueError(f"Unknown army: {army}")
    return next(space for space in track if space.track_number == front_line)


def advance_army_one_space(army):
    track = get_track_for(army)
    current_space = army.location
    if army in [US_VIII_CORPS, US_XV_CORPS]:
        current_index = track.index(current_space)
        next_space = track[current_index + 1] if current_index + 1 < len(track) else None
    else:
        next_space = next((space for space in track if space.track_number == current_space.track_number - 1), None)
    if next_space is None:
        return
    current_space.units.remove(army)
    next_space.units.append(army)
    army.location = next_space
    if hasattr(next_space, "controlling_player"):
        next_space.controlling_player = SideType.ALLIED
    new_furthest_advance = update_front_line_for_army(army, next_space.track_number)
    print(f">>>>>>> AFTER ALLIED ADVANCE -> ALLIED ARMY LOCATION:{army.display_name} IS AT {army.location.name}")
    if army in [US_VIII_CORPS, US_XV_CORPS]:
        check_and_merge_us_third_army(next_space)
        
    return new_furthest_advance

def dday_landings_first_wave():
    print()
    print("D-DAY LANDINGS FIRST WAVE")
    print("=========================")
    print()

    for army, target_roll in [(US_FIRST_ARMY, 3), (BRITISH_SECOND_ARMY, 2), (CANADIAN_FIRST_ARMY, 2)]:
        roll = randint(1, 6)
        print(f"{army.display_name}")
        print(f"ROLL: {roll}")
        if roll >= target_roll:
            advance_army_one_space(army)
            print(f"SUCCESS -> {army.location.name}")
        else:
            print(f"FAILURE -> {army.location.name}")
        print()


def dday_landings_second_wave(card, weather):
    print()
    print("D-DAY LANDINGS SECOND WAVE")
    print("==========================")
    print()

    all_armies = [US_FIRST_ARMY, BRITISH_SECOND_ARMY, CANADIAN_FIRST_ARMY]
    attacking_armies = []

    for army in all_armies:
        if army.location.terrain == TerrainType.START_BOX:
            advance_army_one_space(army)
            print(f"{army.display_name} -> {army.location.name}")
        else:
            attacking_armies.append(army)

    if attacking_armies:
        do_allied_attacks(attacking_armies, card, weather, pause_after_attack=True)


def adjust_hitler_approval(space):
    panzer_present_before_losses = any(isinstance(unit, GermanUnit) and unit.is_panzer() for unit in space.units)

    if panzer_present_before_losses:
        hitler_approval_track.value = max(0, hitler_approval_track.value - 1)
        print(f"HITLER APPROVAL -1 → {hitler_approval_track.value}")


def check_hitler_approval(space):
    adjustment = 0

    if space.name == "CHERBOURG":
        if not GlobalGameState.cherbourg_captured:
            adjustment = -1
            GlobalGameState.cherbourg_captured = True

    elif space.name == "CAEN":
        adjustment = -1

    elif space.name in ("BREST", "LORIENT"):
        adjustment = -2

    if adjustment == 0:
        return

    old_value = hitler_approval_track.value
    hitler_approval_track.value = max(-2, min(6, old_value + adjustment))

    print(f"Hitler Approval: LOSE {abs(adjustment)} ({old_value} -> {hitler_approval_track.value})")


def do_german_losses(space, selected_units=None):
    if selected_units is None:
        german_units = [unit for unit in space.units if unit.type != ReinforcementType.COMMANDER]
    else:
        german_units = [unit for unit in selected_units if unit.type != ReinforcementType.COMMANDER]
    if not german_units:
        print("NO UNITS IN SPACE - NO LOSSES")
        return

    print()
    print("GERMAN LOSSES")
    print("=============")

    print(">>>> GlobalGameState.german_casualty_strategy=", GlobalGameState.german_casualty_strategy)
    if GlobalGameState.german_casualty_strategy == Strategy.HUMAN:
        for i, unit in enumerate(german_units, start=1):
            print(f"{i}. {unit} ({unit.combat_value})")

        selection = int(input("Select unit to take loss: "))
        casualty = german_units[selection - 1]
    elif GlobalGameState.german_casualty_strategy == Strategy.UNIT_TEST:
        # Always take first unit in list as loss for unit tests
        casualty = german_units[0]
    elif GlobalGameState.german_casualty_strategy == Strategy.RANDOM:
        casualty = choice(german_units)

    else:
        raise ValueError(
            f"Strategy.HUMAN={Strategy.HUMAN} Unknown strategy: {GlobalGameState.german_casualty_strategy}"
        )

    if casualty.combat_value > 1:
        casualty.combat_value -= 1
        print(f"REDUCED: {casualty} now {casualty.combat_value}")
    else:
        space.units.remove(casualty)
        eliminated_units_box.units.append(casualty)
        print(f"ELIMINATED: {casualty}")


def choose_german_excess_unit(units, choice=None):
    print()
    print("SELECT UNIT TO ELIMINATE")
    for i, unit in enumerate(units, start=1):
        print(f"{i}. {unit} ({unit.combat_value})")

    if choice is None:
        selection = int(input("Select unit to eliminate: "))
    else:
        selection = choice

    return units[selection - 1]


def enforce_german_stacking_limit(space, choice=None):
    stacking_groups = [
        ("PANZER / KAMPFGRUPPE", lambda unit: unit.is_panzer(), PANZER_STACKING_LIMIT),
        ("FLAK 88", lambda unit: unit.type == ReinforcementType.FLAK_88, FLAK_88_STACKING_LIMIT),
        ("NEBELWERFER", lambda unit: unit.type == ReinforcementType.NEBELWERFER, NEBELWERFER_STACKING_LIMIT),
        ("FALLSCHIRMJAGER", lambda unit: unit.type == ReinforcementType.FALLSCHIRMJAGER, FALLSCHIRMJAGER_STACKING_LIMIT)
    ]


    # for group_name, selector, limit in stacking_groups:
    #     units = [unit for unit in space.units if selector(unit)]
    #     excess = len(units) - limit
    for group_name, selector, limit in stacking_groups:
        units = [unit for unit in space.units if isinstance(unit, GermanUnit) and selector(unit)]
        excess = len(units) - limit
        if excess <= 0:
            continue

        print()
        print(f"STACKING LIMIT EXCEEDED — {space.name}")
        print(f"{group_name} LIMIT: {limit}")
        print(f"UNITS PRESENT: {len(units)}")
        print(f"EXCESS UNITS: {excess}")

        if group_name == "NEBELWERFER":
            for _ in range(excess):
                casualty = units.pop()
                space.units.remove(casualty)
                eliminated_units_box.units.append(casualty)
                print(f"ELIMINATED: {casualty}")
            continue

        for _ in range(excess):
            available_units = [unit for unit in units if unit in space.units]
            casualty = choose_german_excess_unit(available_units, choice)
            space.units.remove(casualty)
            eliminated_units_box.units.append(casualty)
            print(f"ELIMINATED: {casualty}")

def retreat_german_units(space, track):
    retreat_space = next((s for s in track if s.track_number == space.track_number - 1), None)
    if retreat_space is None:
        print("NO RETREAT POSSIBLE")
        return
    retreating_units = list(space.units)
    for unit in retreating_units:
        space.units.remove(unit)
        retreat_space.units.append(unit)
    print(f"GERMANS RETREAT FROM {space.name} TO {retreat_space.name}")
    enforce_german_stacking_limit(retreat_space)
    
# if two US Corps occupy same space -> merge
def check_and_merge_us_third_army(space):
    if US_VIII_CORPS not in space.units or US_XV_CORPS not in space.units:
        return False

    space.units.remove(US_VIII_CORPS)
    space.units.remove(US_XV_CORPS)

    US_VIII_CORPS.location = None
    US_XV_CORPS.location = None

    space.units.append(US_THIRD_ARMY)
    US_THIRD_ARMY.location = space
    US_THIRD_ARMY.merged = True
    GlobalGameState.us_third_army_merged = True

    update_front_line_for_army(US_THIRD_ARMY, space.track_number)

    print()
    print(f">>>>>>> US VIII CORPS AND US XV CORPS MERGE AT {space.name}")
    print(f">>>>>>> US 3RD ARMY IS NOW AT {space.name}")

    return True


def activate_us_third_army():
    GlobalGameState.us_third_army_activated = True
    add_units_to_space(us_3_start_box, US_VIII_CORPS)
    add_units_to_space(us_3_start_box, US_XV_CORPS)
    GlobalGameState.us_viii_front_line = st_malo.track_number
    GlobalGameState.us_xv_front_line = rennes.track_number
    print("US 3rd Army ACTIVATED")
    print("US VIII Corps in Start Box")
    print("US XV Corps in Start Box")




def us_third_army_activation_die_roll(track_number, die_roll=None):
    drm_result = calculate_us_third_army_activation_drm(track_number)

    # Automatic activation at Mortain
    if drm_result.auto_activate:
        print()
        print("US 3RD ARMY ACTIVATION")
        print("Automatic Activation")
        activate_us_third_army()

        return
    if die_roll is None:
        die_roll = randint(1, 6) + randint(1, 6)

    modified_roll = die_roll + drm_result.drm

    print()
    print("US 3RD ARMY ACTIVATION")
    print(f"Roll: {die_roll} {' '.join(drm_result.reasons)} = {modified_roll}")

    # QUACK TEST HARD WIRE MODIFIED ROLL to 12
    # modified_roll= 12
    # print("TEST!!!!!!!!!!!!!!!!!!!!!! roll= 12")
    result = get_us_third_army_activation_result(modified_roll)

    if result.activated:
        activate_us_third_army()
    else:
        print("US 3rd Army NOT ACTIVATED")

    return result


def do_allied_victory(army, target_space, activation_die_roll=None):
    print("ALLIED VICTORY")
    track = get_track_for(army)

    new_furthest_advance = (
        army == US_FIRST_ARMY
        and target_space.track_number < GlobalGameState.us_first_army_furthest_advance
    )

    if target_space.units:
        retreat_german_units(target_space, track)
    else:
        print("<<SPACE CLEARED>>")


    if target_space.fortified_village_modifier > 0:
        target_space.fortified_village_modifier = 0
        print(f"FORTIFIED VILLAGES REMOVED FROM {target_space.name}")
    advance_army_one_space(army)
    print(f"{army.display_name} -> {army.location.name}")
    if army.location.terrain == TerrainType.FALAISE_GAP:
        return True

    if (
        GlobalGameState.us_third_army_activated == False
        and new_furthest_advance
        and target_space.track_number in [6, 5, 4, 3]
    ):
        us_third_army_activation_die_roll(target_space.track_number, die_roll=activation_die_roll)

    check_hitler_approval(target_space)

def get_carpet_bombing_modifier(card, weather, die_roll=None):
    if not card.air_power.has_carpet_bombing():
        return 0
    if weather.available_jabos == 0:
        return 0
    actual_die_roll = die_roll if die_roll is not None else randint(1, 6)
    result = get_carpet_bombing_result(die_roll=actual_die_roll, drm=weather.carpet_bombing_drm)

    if result.attack_modifier == ATTACK_CANCELLED:
        return 0

    return result.attack_modifier


def do_siege_roll(space, army, card, weather, carpet_bombing, defense_strength, die_roll=None):
    print()
    print("========================================")
    print(f"{army.display_name} - SIEGE ROLL")
    print("========================================")
    print()
    siege_roll = die_roll if die_roll is not None else randint(1, 6)
    modified_roll = siege_roll

    result = calculate_attack_modifiers(
        card=card, army=army, num_jabos=weather.available_jabos, carpet_bombing=carpet_bombing
    )
    attack_strength = result["attack_strength"]
    has_air_support = result["has_air_support"]

    drm_result = calculate_siege_drm(
        attack_strength=attack_strength, defense_strength=defense_strength, has_air_support=has_air_support
    )
    modified_roll += drm_result.drm

    # =====================================================
    # DISPLAY DRMS
    # =====================================================
    print(f"DEFENSE-ATTACK DIFFERENTIAL: {defense_strength - attack_strength}")
    for reason in drm_result.reasons:
        print(f"DRM: {reason}")

    modified_roll = max(1, min(6, modified_roll))
    siege_result = get_siege_result(modified_roll)

    print()
    print(f"BASE ROLL: {siege_roll}")
    print(f"MODIFIED ROLL: {modified_roll}")
    print()
    print(f"RESULT: {siege_result.result_type.value}")

    # =====================================================
    # APPLY COMBAT STEP LOSSES
    # =====================================================
    if siege_result.combat_steps_eliminated > 0:
        steps_elim = min(defense_strength - 4, siege_result.combat_steps_eliminated)
        print(f"COMBAT STEPS ELIMINATED: {steps_elim}")

        for _ in range(steps_elim):
            do_german_losses(space)

    # =====================================================
    # SPACE CAPTURED
    # =====================================================
    if siege_result.space_captured:
        space.under_siege = False
        do_allied_victory(army, space)


def do_allied_attacks(armies, card, weather, carpet_bombing=0, die_roll=None, pause_after_attack=False):
    
    print()
    print("ALLIED ATTACKS")
    print("==============")
    print()

    for army in armies:        
        
        if army.location is None:
            print("************* CONTINUE POO *****************")
            continue
        
        # --- CLEAN STRUCTURAL MERGE BYPASS ---
        if army in [US_VIII_CORPS, US_XV_CORPS]:
            # Identify the sister corps
            sister_corps = US_XV_CORPS if army == US_VIII_CORPS else US_VIII_CORPS
            target_space = get_track_for(army)[1] if army.location.terrain == TerrainType.START_BOX else next(
                (space for space in get_track_for(army) if space.track_number == army.location.track_number - 1), None)

            # If the sister corps is already at the target destination, skip combat and merge immediately
            if sister_corps.location == target_space and target_space is not None:
                print(f"{army.display_name} moves to join {sister_corps.display_name} at {target_space.name} for Third Army Merge.")

                # Move the unit physically without triggering combat
                army.location.units.remove(army)
                target_space.units.append(army)
                army.location = target_space

                # Execute the merge code immediately
                check_and_merge_us_third_army(target_space)
                continue
        # --------------------------------------
        if army == US_FIRST_ARMY:
            army_color = LIGHT_BROWN
        elif army == BRITISH_SECOND_ARMY:
            army_color = BLUE
        elif army == CANADIAN_FIRST_ARMY:
            army_color = RED
        elif army in [US_THIRD_ARMY, US_VIII_CORPS, US_XV_CORPS]:
            army_color = GREEN
        else:
            army_color = CYAN
        print(army_color) 
        # NEW RULE TO ENSURE two US 3rd ARMY CORPS NEED TO ATTACK OUT OF THEIR BOX
        if army in [US_VIII_CORPS, US_XV_CORPS] and army.location and army.location.terrain == TerrainType.START_BOX:
            target_space = get_track_for(army)[1]
        elif army.location and army.location.terrain == TerrainType.START_BOX:
        # elif army.location.terrain == TerrainType.START_BOX:
            advance_army_one_space(army)
            print(f"{army.display_name} -> {army.location.name}")
            continue
        else:
            target_space = next((space for space in get_track_for(army) if space.track_number == army.location.track_number - 1), None)

        attack_result = calculate_attack_modifiers(card=card, army=army, num_jabos=weather.available_jabos, carpet_bombing=carpet_bombing)
        attack_strength = attack_result["attack_strength"]
        attacking_from = army.location.name
        print()
        print(f"{army.display_name} ADVANCING FROM {attacking_from}")
        print("=" * len(f"{army.display_name} ADVANCING FROM {attacking_from}"))
        defense_strength = german_defense_strength(target_space)
        
        
        actual_die_roll = die_roll if die_roll is not None else randint(1, 6)
        attack_total = attack_strength + actual_die_roll
        had_fortified_villages = target_space.fortified_village_modifier > 0

        print(f"{army.display_name}")
        print(f"TARGET: {target_space.name}")
        print(f"ATTACK STRENGTH: {attack_strength}")
        print(f"DEFENSE STRENGTH: {defense_strength}")
        print()
      
        if target_space.under_siege or (defense_strength - attack_strength >= 6 and target_space.terrain == TerrainType.FORTRESS):
            target_space.under_siege = True
            do_siege_roll(space=target_space, army=army, card=card, weather=weather, carpet_bombing=carpet_bombing, defense_strength=defense_strength, die_roll=actual_die_roll)
        else:
            # QUACK TEST DIE ROLL!!!!!!!!!!
            # actual_die_roll=6
            print(f"DIE ROLL: {actual_die_roll}")
            print(f"ATTACK TOTAL: {attack_total}")
            print()
            
            if actual_die_roll == 6:
                adjust_hitler_approval(target_space)
                if not had_fortified_villages:
                    do_german_losses(target_space)
                do_allied_victory(army, target_space)
                if army.location.terrain == TerrainType.FALAISE_GAP:
                    return True
            elif actual_die_roll == 1:
                do_german_losses(target_space)
                print("ALLIED DEFEAT (UNMODIFIED 1)")
            elif attack_total > defense_strength:
                adjust_hitler_approval(target_space)
                if not had_fortified_villages:
                    do_german_losses(target_space)
                do_allied_victory(army, target_space)
                if army.location.terrain == TerrainType.FALAISE_GAP:
                    return True
            else:
                do_german_losses(target_space)
                print("ALLIED DEFEAT")
                
        if pause_after_attack:
            input("Press ENTER to continue...")
        print(RESET)


def do_allied_advances_phase(card, weather):
    print()
    print("========================================")
    print("ALLIED ADVANCES PHASE")
    print("========================================")
    print()

    if card.card_id == 1:
        dday_landings_first_wave()
        return

    if card.card_id == 2:
        dday_landings_second_wave(card, weather)
        return

    armies = get_armies_as_objects(card)

    if not armies:
        print("NO ATTACKING ARMIES")
        return

    print("ADVANCING ARMIES")

    for army in armies:
        if army in [US_THIRD_ARMY, US_VIII_CORPS, US_XV_CORPS] and GlobalGameState.us_third_army_activated == False:
            print("US 3rd Army not activated - NO ADVANCE")
        else:
            print(f" - {army}")

    advancing_armies = [army for army in armies if army not in [US_THIRD_ARMY, US_VIII_CORPS, US_XV_CORPS] or GlobalGameState.us_third_army_activated]
    carpet_bombing = get_carpet_bombing_modifier(card, weather)
    # do_allied_attacks(advancing_armies, card, weather, carpet_bombing, pause_after_attack=True)
    
    game_over = do_allied_attacks(advancing_armies, card, weather, carpet_bombing,pause_after_attack=True)
    if game_over:
        return True

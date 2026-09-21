from random import randint
from core.actions.actions_helper import use_action
from core.global_game_state import GlobalGameState
from core.map.map_model import transport_track, supply_track, hitler_approval_track


def do_transport_augmentation_roll(die_roll):
    print()
    print("TRANSPORT AUGMENTATION ROLL")
    print(f"ROLL: {die_roll}")
    if GlobalGameState.transport_roll_drm > 0:
        modified_roll = die_roll + GlobalGameState.transport_roll_drm
        print(f"DRM: +{GlobalGameState.transport_roll_drm}")
        print(f"MODIFIED ROLL: {modified_roll}")
    else:
        modified_roll = die_roll
    print(f"BASE LEVEL: {GlobalGameState.transport_base_level}")
    if modified_roll > GlobalGameState.transport_base_level:
        transport_track.value += 1
        print(f"TRANSPORT INCREASED TO {transport_track.value}")
    else:
        print("NO INCREASE")


def do_supply_augmentation_roll(die_roll):
    print()
    print("SUPPLY AUGMENTATION ROLL")
    print(f"ROLL: {die_roll}")
    if GlobalGameState.supply_roll_drm > 0:
        modified_roll = die_roll + GlobalGameState.supply_roll_drm
        print(f"DRM: +{GlobalGameState.supply_roll_drm}")
        print(f"MODIFIED ROLL: {modified_roll}")
    else:
        modified_roll = die_roll
    print(f"BASE LEVEL: {GlobalGameState.supply_base_level}")
    if modified_roll > GlobalGameState.supply_base_level:
        supply_track.value += 1
        print(f"SUPPLY INCREASED TO {supply_track.value}")
    else:
        print("NO INCREASE")


def do_hitler_approval_augmentation_roll(die_roll):
    print()
    print("HITLER APPROVAL AUGMENTATION ROLL")
    print(f"ROLL: {die_roll}")
    print(f"BASE LEVEL: {GlobalGameState.hitler_approval_base_level}")
    if die_roll > GlobalGameState.hitler_approval_base_level:
        hitler_approval_track.value += 1
        print(f"HITLER APPROVAL INCREASED TO {hitler_approval_track.value}")
    else:
        print("NO INCREASE")


def do_resource_augmentation_roll(choice=None, die_roll=None):

    # Human player chooses the resource.
    if choice is None:
        print()
        print("RESOURCE AUGMENTATION ROLL")
        print("==========================")
        print()
        print(f"1. {transport_track.name}: {transport_track.value}")
        print(f"2. {supply_track.name}: {supply_track.value}")
        print(
            f"3. {hitler_approval_track.name}: "
            f"{hitler_approval_track.value}"
        )
        print()
        print("0. Return to main menu")
        print()

        choice = input("Choose Resource: ").strip()

    choice = str(choice)

    if choice == "0":
        return False

    # Generate the die roll only if one was not supplied.
    if die_roll is None:
        die_roll = randint(1, 6)

    # TRANSPORT
    if choice == "1":
        if transport_track.value >= transport_track.maximum:
            print("TRANSPORT IS ALREADY AT MAXIMUM")
            return False

        if not use_action():
            return False

        do_transport_augmentation_roll(die_roll)
        return True

    # SUPPLY
    if choice == "2":
        if supply_track.value >= supply_track.maximum:
            print("SUPPLY IS ALREADY AT MAXIMUM")
            return False

        if not use_action():
            return False

        do_supply_augmentation_roll(die_roll)
        return True

    # HITLER APPROVAL
    if choice == "3":
        if hitler_approval_track.value >= hitler_approval_track.maximum:
            print("HITLER APPROVAL IS ALREADY AT MAXIMUM")
            return False

        if not use_action():
            return False

        do_hitler_approval_augmentation_roll(die_roll)
        return True

    return False


from core.ai.headless_action_masks import (
    german_eligible_spaces,
    OTHER_RESERVE_UNIT_TYPES,
    MOVE_UNIT_TYPES,
)
from core.allied_armies import armies_list
from core.german_units import panzer_divisions_list


# Action group starting positions
PASS_START = 0
COUNTER_ATTACK_START = 1
PANZER_FROM_RESERVE_START = COUNTER_ATTACK_START + len(armies_list)
PANZER_TO_RESERVE_START = PANZER_FROM_RESERVE_START + len(panzer_divisions_list) * len(german_eligible_spaces)
OTHER_FROM_RESERVE_START = PANZER_TO_RESERVE_START + len(panzer_divisions_list)
MOVE_UNIT_START = OTHER_FROM_RESERVE_START + len(OTHER_RESERVE_UNIT_TYPES) * len(german_eligible_spaces)
REFIT_PANZER_START = MOVE_UNIT_START + len(MOVE_UNIT_TYPES) * len(german_eligible_spaces)
STORE_ACTION_POINT_START = REFIT_PANZER_START + len(panzer_divisions_list)
AUGMENTATION_START = STORE_ACTION_POINT_START + 1
FORTIFIED_VILLAGE_START = AUGMENTATION_START + 3


def get_pass_action_id():
    return PASS_START


def get_counter_attack_action_id(army):
    return COUNTER_ATTACK_START + armies_list.index(army)


def get_panzer_from_reserve_action_id(panzer, destination):
    return (
        PANZER_FROM_RESERVE_START
        + panzer_divisions_list.index(panzer) * len(german_eligible_spaces)
        + german_eligible_spaces.index(destination)
    )


def get_panzer_to_reserve_action_id(panzer):
    return PANZER_TO_RESERVE_START + panzer_divisions_list.index(panzer)


def get_unit_type_index(unit, unit_types):
    for i, (_, unit_type, named_unit) in enumerate(unit_types):
        if named_unit is unit or (
            named_unit is None and unit_type == getattr(unit, "type", None)
        ):
            return i

    raise ValueError(f"Unit not found in action layout: {unit}")


def get_other_from_reserve_action_id(unit, destination):
    return (
        OTHER_FROM_RESERVE_START
        + get_unit_type_index(unit, OTHER_RESERVE_UNIT_TYPES) * len(german_eligible_spaces)
        + german_eligible_spaces.index(destination)
    )


def get_move_unit_action_id(unit, destination):
    return (
        MOVE_UNIT_START
        + get_unit_type_index(unit, MOVE_UNIT_TYPES) * len(german_eligible_spaces)
        + german_eligible_spaces.index(destination)
    )


def get_refit_panzer_action_id(panzer):
    return REFIT_PANZER_START + panzer_divisions_list.index(panzer)


def get_store_action_point_action_id():
    return STORE_ACTION_POINT_START


def get_augmentation_action_id(choice):
    # 1 = Transport, 2 = Supply, 3 = Hitler Approval
    if choice not in (1, 2, 3):
        raise ValueError("Augmentation choice must be 1, 2 or 3")

    return AUGMENTATION_START + choice - 1


def get_fortified_village_action_id(space):
    return FORTIFIED_VILLAGE_START + german_eligible_spaces.index(space)


from core.actions.resource_actions import do_resource_augmentation_roll
from core.german_units import MEYER, MODEL, ROMMEL, TIGER_101, FS_3, FS_5
from random import randint

import numpy as np
from core.actions.actions_helper import can_add_unit_to_space, get_german_controlled_spaces
from core.actions.strategic_reserve_actions import do_move_other_unit_from_strategic_reserve, do_move_panzer_from_strategic_reserve, do_move_panzer_to_strategic_reserve, get_other_units_in_strategic_reserve, get_panzer_divisions_in_strategic_reserve, get_panzer_divisions_on_map
from core.allied_armies import armies_list
from core.actions.hitler_intervention import (
    check_hitler_intervention_applies,
    do_hitler_intervention_attack,
    do_hitler_intervention_redeploy,
    get_hitler_intervention_targets,
    create_available_panzers_list,
    get_german_space_facing_front_line,
)
from core.actions.counter_attack_action import (
    do_counter_attack,
    get_counter_attack_options,
)
from core.map.map_model import TerrainType, transport_track, supply_track, hitler_approval_track, strategic_reserve_box
from core.actions.stacking_limits import PANZER_STACKING_LIMIT
from core.enums import ReinforcementType
from core.global_game_state import GlobalGameState
from core.german_units import GermanUnit
from core.save_load_game import get_all_map_spaces_excluding_boxes
from core.german_units import panzer_divisions_list


OTHER_RESERVE_UNIT_TYPES = [
    ("FLAK 88", ReinforcementType.FLAK_88, None),
    ("KAMPFGRUPPE", ReinforcementType.KAMPFGRUPPE, None),
    ("NEBELWERFER", ReinforcementType.NEBELWERFER, None),
    ("3rd FALLSCHIRMJAGER", None, FS_3),
    ("5th FALLSCHIRMJAGER", None, FS_5),
    ("ROMMEL", None, ROMMEL),
    ("MEYER", None, MEYER),
    ("MODEL", None, MODEL),
    ("101st TIGER BATTALION", None, TIGER_101),
]

# ============================================================
# HEADLESS AGENT DECISION STATE
# ============================================================


class HeadlessDecisionState:

    # Temporary state used only to coordinate multi-step decisions made by
    # the headless agent.

    # This is deliberately separate from GlobalGameState because these
    # attributes describe the agent's current decision process rather than
    # the underlying game state.
    hitler_intervention_stage = None
    hitler_intervention_target = None
    hitler_intervention_redeployments = 0

    @classmethod
    def reset_hitler_intervention(cls):
        cls.hitler_intervention_stage = None
        cls.hitler_intervention_target = None
        cls.hitler_intervention_redeployments = 0


# TOTAL_ACTIONS = 160
HITLER_INTERVENTION_ACTIONS = 21


def get_total_actions():
    all_map_spaces = []
    seen_spaces = set()

    for space in get_all_map_spaces_excluding_boxes():
        if id(space) in seen_spaces:
            continue
        seen_spaces.add(id(space))
        all_map_spaces.append(space)

    german_eligible_spaces = [
        space
        for space in all_map_spaces
        if space.terrain not in [
            TerrainType.START_BOX,
            TerrainType.BEACH,
        ]
    ]

    actions = (
        7
        + len(panzer_divisions_list) * len(german_eligible_spaces)
        + len(panzer_divisions_list)
        + len(OTHER_RESERVE_UNIT_TYPES) * len(german_eligible_spaces)
        + 3 # augmentation rolls
    )

    return actions, german_eligible_spaces


TOTAL_ACTIONS, german_eligible_spaces = get_total_actions()


def print_mini_action_mask(mask, legal_only=False):
    action_names = []

    # ---------------------------------------------------------
    # PASS
    # ---------------------------------------------------------

    action_names.append("PASS")

    # ---------------------------------------------------------
    # COUNTER-ATTACK
    # ---------------------------------------------------------

    for army in armies_list:
        action_names.append(
            f"COUNTER-ATTACK: {army.name}"
        )

    # ---------------------------------------------------------
    # MOVE PANZER FROM STRATEGIC RESERVE
    # ---------------------------------------------------------

    for panzer in panzer_divisions_list:
        for space in german_eligible_spaces:
            action_names.append(
                f"FROM STRATEGIC RESERVE: "
                f"{panzer.name} -> {space.name}"
            )

    # ---------------------------------------------------------
    # MOVE PANZER TO STRATEGIC RESERVE
    # ---------------------------------------------------------

    for panzer in panzer_divisions_list:
        action_names.append(
            f"TO STRATEGIC RESERVE: {panzer.name}"
        )

    # ---------------------------------------------------------
    # MOVE OTHER UNIT FROM STRATEGIC RESERVE
    # ---------------------------------------------------------

    for unit_name, _, _ in OTHER_RESERVE_UNIT_TYPES:
        for space in german_eligible_spaces:
            action_names.append(
                f"FROM STRATEGIC RESERVE: "
                f"{unit_name} -> {space.name}"
            )

    # AUGMENTATION ROLLS

    action_names.append("AUGMENTATION ROLL: TRANSPORT")
    action_names.append("AUGMENTATION ROLL: SUPPLY")
    action_names.append("AUGMENTATION ROLL: HITLER APPROVAL")

    assert len(action_names) == len(mask), (
        f"Action-name count ({len(action_names)}) "
        f"does not match mask size ({len(mask)})"
    )
    # ---------------------------------------------------------
    # SAFETY CHECK
    # ---------------------------------------------------------

    assert len(action_names) == len(mask), (
        f"Action-name count ({len(action_names)}) "
        f"does not match mask size ({len(mask)})"
    )

    # ---------------------------------------------------------
    # PRINT
    # ---------------------------------------------------------

    print()
    print("AI ACTION MASK")
    print("--------------")

    for action_id, is_legal in enumerate(mask):

        if legal_only and is_legal != 1.0:
            continue

        print(
            f"{action_id}: "
            f"{action_names[action_id]} "
            f"[{is_legal:.1f}]"
        )

    print()


def get_pass_action_mask(is_legal):
    # PASS has one local action slot.
    mask = np.zeros(1, dtype=np.float32)

    if is_legal:
        mask[0] = 1.0

    return mask


def get_counter_attack_mask():
    # Local mask layout:
    # 0..5 = armies_list[0]..armies_list[5]
    mask = np.zeros(len(armies_list), dtype=np.float32)

    legal_options = get_counter_attack_options()

    options_by_army_name = {
        option["army"].name: option
        for option in legal_options
    }

    for army_index, army in enumerate(armies_list):
        if army.name in options_by_army_name:
            mask[army_index] = 1.0

    return mask


def get_move_panzer_from_strategic_reserve_mask():
    # Local mask layout:
    # panzer_index * number_of_spaces + space_index
    mask_size = (
        len(panzer_divisions_list)
        * len(german_eligible_spaces)
    )

    mask = np.zeros(mask_size, dtype=np.float32)

    german_controlled_spaces = get_german_controlled_spaces()

    panzer_divisions_in_reserve = (
        get_panzer_divisions_in_strategic_reserve()
    )

    for panzer_index, panzer in enumerate(
        panzer_divisions_list
    ):
        if panzer not in panzer_divisions_in_reserve:
            continue

        for space_index, space in enumerate(
            german_eligible_spaces
        ):
            if space not in german_controlled_spaces:
                continue

            local_action_id = (
                panzer_index
                * len(german_eligible_spaces)
                + space_index
            )

            mask[local_action_id] = 1.0

    return mask


def get_move_panzer_to_strategic_reserve_mask():
    # Local mask layout:
    # 0..10 = panzer_divisions_list[0]..panzer_divisions_list[10]
    mask = np.zeros(
        len(panzer_divisions_list),
        dtype=np.float32
    )

    panzer_divisions_on_map = [
        panzer
        for _, panzer in get_panzer_divisions_on_map()
    ]

    for panzer_index, panzer in enumerate(
        panzer_divisions_list
    ):
        if panzer in panzer_divisions_on_map:
            mask[panzer_index] = 1.0

    return mask


def get_move_other_unit_from_strategic_reserve_mask():
    # Local mask layout:
    # unit_type_index * number_of_spaces + space_index
    #
    # This mask contains no Action Phase offsets or AP logic.
    # It can therefore be reused unchanged by the Resources Phase.
    mask_size = (
        len(OTHER_RESERVE_UNIT_TYPES)
        * len(german_eligible_spaces)
    )

    mask = np.zeros(mask_size, dtype=np.float32)

    german_controlled_spaces = get_german_controlled_spaces()

    other_units_in_reserve = (
        get_other_units_in_strategic_reserve()
    )

    for unit_type_index, (_, unit_type, named_unit) in enumerate(
        OTHER_RESERVE_UNIT_TYPES
    ):

        # Named unit: find the exact object.
        if named_unit is not None:
            available_unit = next(
                (
                    unit
                    for unit in other_units_in_reserve
                    if unit is named_unit
                ),
                None
            )

        # Generic unit type: find any physical counter
        # of the requested type currently in reserve.
        else:
            available_unit = next(
                (
                    unit
                    for unit in other_units_in_reserve
                    if unit.type == unit_type
                ),
                None
            )

        if available_unit is None:
            continue

        for space_index, space in enumerate(
            german_eligible_spaces
        ):
            if space not in german_controlled_spaces:
                continue

            if not can_add_unit_to_space(
                space,
                available_unit
            ):
                continue

            local_action_id = (
                unit_type_index
                * len(german_eligible_spaces)
                + space_index
            )

            mask[local_action_id] = 1.0

    return mask


def get_mini_action_mask():

    total_ap = (
        GlobalGameState.actions_left_this_turn
        + GlobalGameState.reserve_actions
    )

    has_ap = total_ap > 0

    # ---------------------------------------------------------
    # PASS
    # ---------------------------------------------------------

    pass_mask = get_pass_action_mask(
        is_legal=not has_ap
    )

    # ---------------------------------------------------------
    # ACTIONS THAT REQUIRE AP
    # ---------------------------------------------------------

    if has_ap:

        counter_attack_mask = (
            get_counter_attack_mask()
        )

        panzer_from_reserve_mask = (
            get_move_panzer_from_strategic_reserve_mask()
        )

        panzer_to_reserve_mask = (
            get_move_panzer_to_strategic_reserve_mask()
        )

        augmentation_mask = (
            augmentation_roll_mask()
        )

    else:

        counter_attack_mask = np.zeros(
            len(armies_list),
            dtype=np.float32
        )

        panzer_from_reserve_mask = np.zeros(
            len(panzer_divisions_list)
            * len(german_eligible_spaces),
            dtype=np.float32
        )

        panzer_to_reserve_mask = np.zeros(
            len(panzer_divisions_list),
            dtype=np.float32
        )

        augmentation_mask = np.zeros(
            3,
            dtype=np.float32
        )

    # ---------------------------------------------------------
    # ACTIONS THAT DO NOT REQUIRE AP
    # ---------------------------------------------------------

    other_unit_from_reserve_mask = (
        get_move_other_unit_from_strategic_reserve_mask()
    )

    # ---------------------------------------------------------
    # BUILD COMPLETE ACTION PHASE MASK
    # ---------------------------------------------------------

    mask = np.concatenate(
        (
            pass_mask,
            counter_attack_mask,
            panzer_from_reserve_mask,
            panzer_to_reserve_mask,
            other_unit_from_reserve_mask,
            augmentation_mask,
        )
    )

    return mask


def augmentation_roll_mask():
    mask = np.zeros(3, dtype=np.float32)

    # 0 = Transport augmentation roll
    # 1 = Supply augmentation roll
    # 2 = Hitler Approval augmentation roll

    if transport_track.value < transport_track.maximum:
        mask[0] = 1.0

    if supply_track.value < supply_track.maximum:
        mask[1] = 1.0

    if hitler_approval_track.value < hitler_approval_track.maximum:
        mask[2] = 1.0

    return mask

def get_action_mask():
    mask = np.zeros(TOTAL_ACTIONS, dtype=np.float32)
    # Action 0 = end/pass action phase.
    mask[0] = 1.0
    return mask


def execute_flat_action_ai(action_id, target_option=None):

    # ---------------------------------------------------------
    # PASS
    # ---------------------------------------------------------

    if action_id == 0:
        GlobalGameState.actions_left_this_turn = 0
        return True

    # ---------------------------------------------------------
    # COUNTER-ATTACK
    # ---------------------------------------------------------

    if 1 <= action_id <= 6:
        if target_option is None:
            selected_army = armies_list[action_id - 1]

            target_option = next(
                (
                    option
                    for option in get_counter_attack_options()
                    if option["army"] is selected_army
                ),
                None
            )

        if target_option is None:
            return False

        do_counter_attack(selected_option=target_option)
        return True

    # ---------------------------------------------------------
    # MOVE PANZER FROM STRATEGIC RESERVE
    # ---------------------------------------------------------

    panzer_action_start = 7

    panzer_action_end = (
        panzer_action_start
        + len(panzer_divisions_list) * len(german_eligible_spaces)
    )

    if panzer_action_start <= action_id < panzer_action_end:
        relative_id = action_id - panzer_action_start

        panzer_index = (
            relative_id // len(german_eligible_spaces)
        )

        space_index = (
            relative_id % len(german_eligible_spaces)
        )

        selected_panzer = panzer_divisions_list[panzer_index]
        selected_space = german_eligible_spaces[space_index]

        do_move_panzer_from_strategic_reserve(
            randint(1, 6),
            div_choice=selected_panzer,
            space_choice=selected_space
        )

        return True

    # ---------------------------------------------------------
    # MOVE PANZER TO STRATEGIC RESERVE
    # ---------------------------------------------------------

    panzer_to_reserve_start = panzer_action_end

    panzer_to_reserve_end = (
        panzer_to_reserve_start
        + len(panzer_divisions_list)
    )

    if panzer_to_reserve_start <= action_id < panzer_to_reserve_end:
        panzer_index = action_id - panzer_to_reserve_start

        selected_panzer = panzer_divisions_list[panzer_index]

        do_move_panzer_to_strategic_reserve(
            randint(1, 6),
            div_choice=selected_panzer
        )

        return True

    # ---------------------------------------------------------
    # MOVE OTHER UNIT FROM STRATEGIC RESERVE
    # ---------------------------------------------------------

    other_reserve_start = panzer_to_reserve_end

    other_reserve_end = (
        other_reserve_start
        + len(OTHER_RESERVE_UNIT_TYPES)
        * len(german_eligible_spaces)
    )

    if other_reserve_start <= action_id < other_reserve_end:
        relative_id = action_id - other_reserve_start

        unit_type_index = (
            relative_id // len(german_eligible_spaces)
        )

        space_index = (
            relative_id % len(german_eligible_spaces)
        )

        _, unit_type, named_unit = (
            OTHER_RESERVE_UNIT_TYPES[unit_type_index]
        )

        selected_space = german_eligible_spaces[space_index]

        # Named units use their exact object.
        if named_unit is not None:
            selected_unit = named_unit

            if selected_unit not in strategic_reserve_box.units:
                return False

        # Generic units use any matching physical counter
        # currently in Strategic Reserve.
        else:
            selected_unit = next(
                (
                    unit
                    for unit in strategic_reserve_box.units
                    if isinstance(unit, GermanUnit)
                    and unit.type == unit_type
                ),
                None
            )

            if selected_unit is None:
                return False

        return do_move_other_unit_from_strategic_reserve(
            unit_choice=selected_unit,
            space_choice=selected_space,
        )
    # ---------------------------------------------------------
    # AUGMENTATION ROLLS
    # ---------------------------------------------------------

    augmentation_start = other_reserve_end
    augmentation_end = augmentation_start + 3

    if augmentation_start <= action_id < augmentation_end:

        augmentation_index = action_id - augmentation_start

        # Local augmentation mask:
        # 0 = Transport
        # 1 = Supply
        # 2 = Hitler Approval
        #
        # do_resource_augmentation_roll choices:
        # 1 = Transport
        # 2 = Supply
        # 3 = Hitler Approval

        augmentation_choice = augmentation_index + 1

        return do_resource_augmentation_roll(
            choice=augmentation_choice
        )
        
    # ---------------------------------------------------------
    # UNKNOWN / INVALID ACTION ID
    # ---------------------------------------------------------

    return False

# ============================================================
# HITLER INTERVENTION ACTION MASK
#
# ACTION SPACE
#
# 0      Continue / finish current Hitler Intervention stage
# 1      Attempt to cancel Hitler Intervention
# 2..7   Select Allied target using armies_list
# 8..18  Select permanent Panzer division
# 19..20 Select a Kampfgruppe
#
# ============================================================


def get_hitler_intervention_mask(card):
    mask = np.zeros(HITLER_INTERVENTION_ACTIONS, dtype=np.float32)

    targets = get_hitler_intervention_targets(card)

    # --------------------------------------------------------
    # No legal target.
    #
    # Expose action 0 so the executor can close the sequence
    # cleanly and set the normal no-effect flag.
    # --------------------------------------------------------
    if not targets:
        mask[0] = 1.0
        return mask

    stage = HeadlessDecisionState.hitler_intervention_stage

    # --------------------------------------------------------
    # Initialise a new Hitler Intervention sequence.
    # --------------------------------------------------------
    if stage is None:
        GlobalGameState.hitler_intervention_no_effect = False

        # Card 32 is special:
        # - no cancellation roll
        # - fixed target
        # - extra applicability conditions are checked by
        #   check_hitler_intervention_applies()
        #
        # Action 0 therefore means "process/start Card 32".
        if card.card_id == 32:
            mask[0] = 1.0
            return mask

        HeadlessDecisionState.hitler_intervention_stage = "CANCEL"
        stage = "CANCEL"

    # --------------------------------------------------------
    # STAGE 1: cancellation decision
    #
    # 0 = do not attempt cancellation
    # 1 = attempt cancellation
    # --------------------------------------------------------
    if stage == "CANCEL":
        mask[0] = 1.0
        mask[1] = 1.0
        return mask

    # --------------------------------------------------------
    # STAGE 2: target selection
    # --------------------------------------------------------
    if stage == "TARGET":

        # If only one target is legal, no agent decision is
        # necessary. Advance directly to redeployment.
        if len(targets) == 1:
            HeadlessDecisionState.hitler_intervention_target = targets[0]
            HeadlessDecisionState.hitler_intervention_stage = "REDEPLOY"

            return get_hitler_intervention_mask(card)

        # Fixed action identity:
        #
        # action = 2 + position in armies_list
        #
        # This does NOT depend on the current length/order of
        # the dynamic targets list.
        for index, army in enumerate(armies_list):
            if army in targets:
                mask[2 + index] = 1.0

        return mask

    # --------------------------------------------------------
    # STAGE 3: Panzer redeployment
    # --------------------------------------------------------
    if stage == "REDEPLOY":
        target_army = HeadlessDecisionState.hitler_intervention_target

        if target_army is None:
            mask[0] = 1.0
            return mask

        attacking_space = get_german_space_facing_front_line(
            target_army
        )

        current_panzer_count = sum(
            1
            for unit in attacking_space.units
            if (
                isinstance(unit, GermanUnit)
                and unit.is_panzer()
            )
        )

        # The card specifies how many forces may be redeployed
        # during THIS Hitler Intervention. Do not allow repeated
        # headless decisions to exceed that number.
        if (
            HeadlessDecisionState.hitler_intervention_redeployments
            >= card.hitler_intervention_panzer_count
        ):
            mask[0] = 1.0
            return mask

        # Nor may the German stacking limit be exceeded.
        if current_panzer_count >= PANZER_STACKING_LIMIT:
            mask[0] = 1.0
            return mask

        available_panzers = create_available_panzers_list(
            attacking_space
        )

        # Nothing else can be brought in. Resolve the attack.
        if not available_panzers:
            mask[0] = 1.0
            return mask

        legal_units = [
            unit
            for unit, source_space in available_panzers
        ]

        # Permanent Panzer divisions occupy actions 8..18.
        for index, panzer in enumerate(panzer_divisions_list):
            if panzer in legal_units:
                mask[8 + index] = 1.0

        # Kampfgruppen count as Panzer forces but are dynamically
        # created, so action 19 represents an available Kampfgruppe.
        kampfgruppen = [
            unit for unit in legal_units
            if unit.type == ReinforcementType.KAMPFGRUPPE
        ]

        for index in range(min(len(kampfgruppen), 2)):
            mask[19 + index] = 1.0

        # Notice: action 0 is NOT legal while another mandatory
        # redeployment can still be made. Once the card limit,
        # stacking limit, or available-unit limit is reached,
        # the next mask exposes only action 0 to resolve the attack.
        return mask

    # Defensive fallback.
    mask[0] = 1.0
    return mask


def execute_flat_hitler_intervention_ai(action_id, card, weather):
    targets = get_hitler_intervention_targets(card)
    stage = HeadlessDecisionState.hitler_intervention_stage

    if action_id == 0:
        if not targets:
            check_hitler_intervention_applies(
                card,
                cancel_choice="N"
            )
            HeadlessDecisionState.reset_hitler_intervention()
            return False

        if stage is None and card.card_id == 32:
            target_army = check_hitler_intervention_applies(card)

            if target_army is None:
                HeadlessDecisionState.reset_hitler_intervention()
                return False

            HeadlessDecisionState.hitler_intervention_target = target_army
            HeadlessDecisionState.hitler_intervention_stage = "REDEPLOY"
            return True

        if stage == "CANCEL":
            HeadlessDecisionState.hitler_intervention_stage = "TARGET"
            return True

        if stage == "TARGET":
            if len(targets) == 1:
                HeadlessDecisionState.hitler_intervention_target = targets[0]
                HeadlessDecisionState.hitler_intervention_stage = "REDEPLOY"
            return True

        if stage == "REDEPLOY":
            target_army = HeadlessDecisionState.hitler_intervention_target

            if target_army is None:
                HeadlessDecisionState.reset_hitler_intervention()
                return False

            attacking_space = get_german_space_facing_front_line(target_army)

            do_hitler_intervention_attack(
                card,
                weather,
                target_army,
                attacking_space
            )

            HeadlessDecisionState.reset_hitler_intervention()
            return False

        return True

    if action_id == 1:
        if card.card_id == 32:
            return True

        result = check_hitler_intervention_applies(
            card,
            cancel_choice="Y",
            target_choice=1
        )

        if result is None:
            HeadlessDecisionState.reset_hitler_intervention()
            return False

        targets = get_hitler_intervention_targets(card)

        if len(targets) == 1:
            HeadlessDecisionState.hitler_intervention_target = targets[0]
            HeadlessDecisionState.hitler_intervention_stage = "REDEPLOY"
        else:
            HeadlessDecisionState.hitler_intervention_stage = "TARGET"

        return True

    if 2 <= action_id <= 7:
        selected_army = armies_list[action_id - 2]

        if selected_army not in targets:
            return True

        HeadlessDecisionState.hitler_intervention_target = selected_army
        HeadlessDecisionState.hitler_intervention_stage = "REDEPLOY"

        print(f"SELECTED TARGET: {selected_army.display_name}")
        return True

    if 8 <= action_id <= 18:
        target_army = HeadlessDecisionState.hitler_intervention_target

        if target_army is None:
            return True

        attacking_space = get_german_space_facing_front_line(target_army)
        available_panzers = create_available_panzers_list(attacking_space)

        from core.german_units import panzer_divisions_list

        selected_panzer = panzer_divisions_list[action_id - 8]

        selected_index = next(
            (
                index
                for index, (unit, source_space)
                in enumerate(available_panzers)
                if unit is selected_panzer
            ),
            None
        )

        if selected_index is None:
            return True

        do_hitler_intervention_redeploy(
            card,
            target_army,
            deployment_choices=[selected_index + 1]
        )

        HeadlessDecisionState.hitler_intervention_redeployments += 1
        return True

    if 19 <= action_id <= 20:
        target_army = HeadlessDecisionState.hitler_intervention_target

        if target_army is None:
            return True

        attacking_space = get_german_space_facing_front_line(target_army)
        available_panzers = create_available_panzers_list(attacking_space)

        kampfgruppen = [
            (unit, source_space)
            for unit, source_space in available_panzers
            if unit.type == ReinforcementType.KAMPFGRUPPE
        ]

        kampfgruppe_index = action_id - 19

        if kampfgruppe_index >= len(kampfgruppen):
            return True

        selected_unit, selected_space = kampfgruppen[kampfgruppe_index]

        selected_index = next(
            (
                index
                for index, (unit, source_space)
                in enumerate(available_panzers)
                if unit is selected_unit
                and source_space is selected_space
            ),
            None
        )

        if selected_index is None:
            return True

        do_hitler_intervention_redeploy(
            card,
            target_army,
            deployment_choices=[selected_index + 1]
        )

        HeadlessDecisionState.hitler_intervention_redeployments += 1
        return True

    return False

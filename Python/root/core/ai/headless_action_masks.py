
import numpy as np
from core.allied_armies import armies_list
from core.actions.hitler_intervention import (
    check_hitler_intervention_applies,
    choose_hitler_intervention_units,
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
from core.actions.stacking_limits import PANZER_STACKING_LIMIT
from core.enums import ReinforcementType
from core.global_game_state import GlobalGameState
from core.german_units import GermanUnit


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


def get_mini_action_mask():
    mask = np.zeros(7, dtype=np.float32)

    # Action 0 = end/pass action phase.
    mask[0] = 1.0

    total_ap = (
        GlobalGameState.actions_left_this_turn
        + GlobalGameState.reserve_actions
    )

    if total_ap <= 0:
        return mask

    legal_options = get_counter_attack_options()

    options_by_army_name = {
        option["army"].name: option
        for option in legal_options
    }

    for index, army in enumerate(armies_list):
        if army.name in options_by_army_name:
            mask[index + 1] = 1.0

    return mask


TOTAL_ACTIONS = 160
HITLER_INTERVENTION_ACTIONS = 21


def get_action_mask():
    mask = np.zeros(TOTAL_ACTIONS, dtype=np.float32)
    # Action 0 = end/pass action phase.
    mask[0] = 1.0
    return mask


def execute_flat_action_ai(action_id, target_option=None):
    if action_id == 0:
        GlobalGameState.actions_left_this_turn = 0
        return True

    if 1 <= action_id <= 6:
        do_counter_attack(selected_option=target_option)
        return True

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

        from core.german_units import panzer_divisions_list

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

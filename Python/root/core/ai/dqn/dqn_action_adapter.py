import numpy as np

from core.ai.dqn.dqn_game_state_encoder import (
    DecisionType,
    DecisionContext,
    encode_game_state,
)

from core.ai.headless_action_masks import (
    get_action_phase_action_mask,
    execute_flat_action_ai,
    get_casualty_mask,
    get_casualty_from_action,
    get_hitler_intervention_mask,
    execute_flat_hitler_intervention_ai,
)


def is_legal_action(action_id, mask):
    if action_id < 0 or action_id >= len(mask):
        return False

    return mask[action_id] == 1.0


def get_legal_actions(mask):
    return np.flatnonzero(mask == 1.0)


# ============================================================
# ACTION PHASE
# ============================================================

def get_action_phase_decision():
    context = DecisionContext(
        decision_type=DecisionType.ACTION_PHASE,
    )

    observation = encode_game_state(context)
    mask = get_action_phase_action_mask()

    return observation, mask


def execute_action_phase_decision(action_id):
    mask = get_action_phase_action_mask()

    if not is_legal_action(action_id, mask):
        return False

    return execute_flat_action_ai(action_id)


# ============================================================
# CASUALTY
# ============================================================

def get_casualty_decision(
    german_units,
    relevant_army=None,
    relevant_space=None,
    allied_attack=False,
    german_counterattack=False,
):
    context = DecisionContext(
        decision_type=DecisionType.CASUALTY,
        relevant_army=relevant_army,
        relevant_space=relevant_space,
        allied_attack=allied_attack,
        german_counterattack=german_counterattack,
    )

    observation = encode_game_state(context)
    mask = get_casualty_mask(german_units)

    return observation, mask


def execute_casualty_decision(action_id, german_units):
    mask = get_casualty_mask(german_units)

    if not is_legal_action(action_id, mask):
        return None

    return get_casualty_from_action(action_id, german_units)


# ============================================================
# HITLER INTERVENTION
# ============================================================

def get_hitler_intervention_decision(card):
    context = DecisionContext(
        decision_type=DecisionType.HITLER_INTERVENTION,
    )

    # The mask function may initialise or advance the
    # Hitler Intervention decision stage, so build it first.
    print(
        "HITLER INTERVENTION DEBUG:",
        "card =", card,
        "targets =", card.hitler_intervention_target_armies,
    )
    mask = get_hitler_intervention_mask(card)
    observation = encode_game_state(context)

    return observation, mask


def execute_hitler_intervention_decision(action_id, card, weather):
    mask = get_hitler_intervention_mask(card)

    if not is_legal_action(action_id, mask):
        return False

    return execute_flat_hitler_intervention_ai(
        action_id,
        card,
        weather,
    )

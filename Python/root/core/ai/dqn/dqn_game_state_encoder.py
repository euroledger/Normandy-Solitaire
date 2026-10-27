from dataclasses import dataclass
from enum import Enum, auto

import numpy as np

from core.global_game_state import GlobalGameState
from core.models import GermanUnit
from core.enums import ReinforcementType, SideType

from core.allied_armies import (
    US_FIRST_ARMY,
    BRITISH_SECOND_ARMY,
    CANADIAN_FIRST_ARMY,
    US_THIRD_ARMY,
    US_VIII_CORPS,
    US_XV_CORPS,
)

from core.german_units import (
    PZ_LEHR,
    SS_12,
    SS_1,
    SS_9,
    SS_10,
    SS_2,
    TIGER_101,
    PZ_21,
    PZ_116,
    SS_17_PZGRD,
    PZ_2,
    PZ_9,
    FS_3,
    FS_5,
    ROMMEL,
    MEYER,
    MODEL,
    panzer_divisions_list,
)

from core.map.map_model import (
    falaise_gap,
    in_transit_box,
    strategic_reserve_box,
    eliminated_units_box,
    transport_track,
    supply_track,
    hitler_approval_track,
)

from core.map.map_spaces_us_1 import (
    us_1_track,
    carentan,
    valognes,
    cherbourg,
    st_lo,
    coutances,
    avranches,
    mortain,
    vire,
    flers,
)

from core.map.map_spaces_brit_2 import (
    brit_2_track,
    bayeux,
    tilly,
    villers_bocage,
    mont_pincon,
    thury_harcourt,
)

from core.map.map_spaces_can_1 import (
    can_1_track,
    lebisey_wood,
    caen,
    cagny,
    bourguebus_ridge,
    falaise,
)

from core.map.map_spaces_us_3 import (
    us_viii_track,
    us_xv_track,
    st_malo,
    brest,
    lorient,
    rennes,
    le_mans,
    alencon,
    argentan,
)

from core.tables.weather import WeatherType

from core.ai.headless_action_masks import HeadlessDecisionState


# ============================================================
# OBSERVATION SIZE
# ============================================================

OBSERVATION_SIZE = 967


# ============================================================
# OBSERVATION BLOCKS
# ============================================================

# Campaign / deck state
CAMPAIGN_START = 0
CAMPAIGN_END = 99

CARDS_DRAWN_INDEX = 0

CURRENT_CARD_START = 1
CURRENT_CARD_END = 49

DRAWN_CARDS_START = 49
DRAWN_CARDS_END = 97

MID_DECK_ADDED_INDEX = 97
LATE_DECK_ADDED_INDEX = 98


# Weather
WEATHER_START = 99
WEATHER_END = 102

WEATHER_CLEAR_INDEX = 99
WEATHER_PARTLY_CLEAR_INDEX = 100
WEATHER_OVERCAST_INDEX = 101


# Resources / action economy
RESOURCES_START = 102
RESOURCES_END = 110

TRANSPORT_INDEX = 102
SUPPLY_INDEX = 103
HITLER_APPROVAL_INDEX = 104

TRANSPORT_BASE_INDEX = 105
SUPPLY_BASE_INDEX = 106
HITLER_APPROVAL_BASE_INDEX = 107

ACTIONS_LEFT_INDEX = 108
RESERVE_ACTIONS_INDEX = 109


# Allied armies
ALLIED_ARMIES_START = 110
ALLIED_ARMIES_END = 137

ALLIED_FEATURES_PER_ARMY = 4

US_THIRD_ACTIVATED_INDEX = 134
US_THIRD_MERGED_INDEX = 135
ARMIES_UPGRADED_INDEX = 136


# Panzer divisions
PANZERS_START = 137
PANZERS_END = 489

NAMED_UNIT_LOCATION_COUNT = 31
PANZER_FEATURES_PER_UNIT = 32


# FS3, FS5, Tiger 101
SPECIAL_GERMAN_UNITS_START = 489
SPECIAL_GERMAN_UNITS_END = 583

FS3_START = 489
FS3_REDUCED_INDEX = 520

FS5_START = 521

TIGER_101_START = 552


# Commanders
COMMANDERS_START = 583
COMMANDERS_END = 678

ROMMEL_START = 583
MEYER_START = 614
MODEL_START = 645

MEYER_AVAILABLE_INDEX = 676
MODEL_IN_COMMAND_INDEX = 677


# Generic German units on map
GENERIC_MAP_UNITS_START = 678
GENERIC_MAP_UNITS_END = 786

GENERIC_FEATURES_PER_SPACE = 4


# Generic German units in off-map boxes
GENERIC_BOX_UNITS_START = 786
GENERIC_BOX_UNITS_END = 798

GENERIC_FEATURES_PER_BOX = 4


# Dynamic map state
MAP_STATE_START = 798
MAP_STATE_END = 906

MAP_FEATURES_PER_SPACE = 4


# Persistent strategic state
STRATEGIC_STATE_START = 906
STRATEGIC_STATE_END = 912

CHERBOURG_CAPTURED_INDEX = 906
HITLER_ASSASSINATION_INDEX = 907
HITLER_INTERVENTION_NO_EFFECT_INDEX = 908
BOCAGE_DEFENSE_INDEX = 909
US_FIRST_FURTHEST_ADVANCE_INDEX = 910
CURRENT_CARPET_BOMBING_INDEX = 911


# Resource / check DRMs
DRM_START = 912
DRM_END = 917

TRANSPORT_ROLL_DRM_INDEX = 912
SUPPLY_ROLL_DRM_INDEX = 913
TRANSPORT_CHECK_DRM_INDEX = 914
SUPPLY_CHECK_DRM_INDEX = 915
HITLER_APPROVAL_CHECK_DRM_INDEX = 916


# Decision type
DECISION_TYPE_START = 917
DECISION_TYPE_END = 921

DECISION_ACTION_PHASE_INDEX = 917
DECISION_RESERVE_DEPLOYMENT_INDEX = 918
DECISION_CASUALTY_INDEX = 919
DECISION_HITLER_INTERVENTION_INDEX = 920


# Decision context
DECISION_CONTEXT_START = 921
DECISION_CONTEXT_END = 956

DECISION_ARMY_START = 921
DECISION_ARMY_END = 927

DECISION_SPACE_START = 927
DECISION_SPACE_END = 954

DECISION_ALLIED_ATTACK_INDEX = 954
DECISION_GERMAN_COUNTERATTACK_INDEX = 955


# Hitler Intervention substate
HITLER_STATE_START = 956
HITLER_STATE_END = 967

HITLER_STAGE_NONE_INDEX = 956
HITLER_STAGE_CANCEL_INDEX = 957
HITLER_STAGE_TARGET_INDEX = 958
HITLER_STAGE_REDEPLOY_INDEX = 959

HITLER_TARGET_START = 960
HITLER_TARGET_END = 966

HITLER_REDEPLOYMENTS_INDEX = 966


# ============================================================
# NORMALIZATION CONSTANTS
# ============================================================

MAX_CARDS = 48

MAX_TRANSPORT = 6
MAX_SUPPLY = 6

MIN_HITLER_APPROVAL = -2
MAX_HITLER_APPROVAL = 6

MAX_ACTIONS = 4
MAX_RESERVE_ACTIONS = 2

MAX_FLAK_STACK = 3
MAX_KAMPFGRUPPE_STACK = 4
MAX_NEBELWERFER_STACK = 3

# These are total counter pools rather than stacking limits.
# Verify these against all reinforcement/event creation paths
# before freezing them for training.

MAX_TOTAL_FLAK = 9
MAX_TOTAL_KAMPFGRUPPE = 5
MAX_TOTAL_NEBELWERFER = 8

MAX_FORTIFIED_VILLAGE_MODIFIER = 2

# Model modifier exists in MapSpace but currently appears unused.
# Leave the feature allocated while the mechanic is checked.

MAX_MODEL_MODIFIER = 1

# These ranges should be verified against every card/event before
# training. Keeping them here makes that correction local.

MIN_TRANSPORT_ROLL_DRM = -3
MAX_TRANSPORT_ROLL_DRM = 3

MIN_SUPPLY_ROLL_DRM = -3
MAX_SUPPLY_ROLL_DRM = 3

MIN_TRANSPORT_CHECK_DRM = -3
MAX_TRANSPORT_CHECK_DRM = 3

MIN_SUPPLY_CHECK_DRM = -3
MAX_SUPPLY_CHECK_DRM = 3

MIN_HITLER_APPROVAL_CHECK_DRM = -3
MAX_HITLER_APPROVAL_CHECK_DRM = 3

MIN_BOCAGE_DEFENSE_MODIFIER = 0
MAX_BOCAGE_DEFENSE_MODIFIER = 2

MIN_CARPET_BOMBING = 0
MAX_CARPET_BOMBING = 3


# ============================================================
# ALLIED ARMIES
# ============================================================

ALLIED_ARMIES = [
    US_FIRST_ARMY,
    BRITISH_SECOND_ARMY,
    CANADIAN_FIRST_ARMY,
    US_THIRD_ARMY,
    US_VIII_CORPS,
    US_XV_CORPS,
]


# ============================================================
# GERMAN MAP SPACES
# ============================================================

# These 27 spaces are exactly the German-eligible location
# vocabulary used by the observation.

GERMAN_MAP_SPACES = [
    carentan,
    valognes,
    cherbourg,
    st_lo,
    coutances,
    avranches,
    mortain,
    vire,
    flers,
    falaise_gap,
    bayeux,
    tilly,
    villers_bocage,
    mont_pincon,
    thury_harcourt,
    lebisey_wood,
    caen,
    cagny,
    bourguebus_ridge,
    falaise,
    st_malo,
    brest,
    lorient,
    rennes,
    le_mans,
    alencon,
    argentan,
]


# ============================================================
# NAMED-UNIT LOCATION ENCODING
# ============================================================

# Each named German unit has 31 possible location states:
#
# 0  NOT AVAILABLE
# 1  IN TRANSIT
# 2  STRATEGIC RESERVE
# 3  ELIMINATED
# 4-30  one of the 27 German map spaces

LOCATION_NOT_AVAILABLE = 0
LOCATION_IN_TRANSIT = 1
LOCATION_STRATEGIC_RESERVE = 2
LOCATION_ELIMINATED = 3

GERMAN_LOCATION_INDEX = {
    id(space): index + 4
    for index, space in enumerate(GERMAN_MAP_SPACES)
}


# ============================================================
# OFF-MAP BOXES
# ============================================================

GENERIC_UNIT_BOXES = [
    in_transit_box,
    strategic_reserve_box,
    eliminated_units_box,
]


# ============================================================
# DECISION CONTEXT
# ============================================================

class DecisionType(Enum):
    ACTION_PHASE = auto()
    RESERVE_DEPLOYMENT = auto()
    CASUALTY = auto()
    HITLER_INTERVENTION = auto()


@dataclass
class DecisionContext:
    decision_type: DecisionType = None
    relevant_army: object = None
    relevant_space: object = None
    allied_attack: bool = False
    german_counterattack: bool = False


# ============================================================
# GENERIC HELPERS
# ============================================================

def normalize(value, minimum, maximum):
    if maximum == minimum:
        return 0.0

    normalized = (value - minimum) / (maximum - minimum)

    return float(np.clip(normalized, 0.0, 1.0))


def get_unit_location(unit):
    # Search the 27 German map spaces first.

    for space in GERMAN_MAP_SPACES:
        if unit in space.units:
            return space

    # Then search the three off-map boxes.

    if unit in in_transit_box.units:
        return in_transit_box

    if unit in strategic_reserve_box.units:
        return strategic_reserve_box

    if unit in eliminated_units_box.units:
        return eliminated_units_box

    # None means that the unit is not yet available.

    return None


def encode_named_unit_location(observation, start_index, unit):
    location = get_unit_location(unit)

    if location is None:
        location_index = LOCATION_NOT_AVAILABLE

    elif location is in_transit_box:
        location_index = LOCATION_IN_TRANSIT

    elif location is strategic_reserve_box:
        location_index = LOCATION_STRATEGIC_RESERVE

    elif location is eliminated_units_box:
        location_index = LOCATION_ELIMINATED

    else:
        location_index = GERMAN_LOCATION_INDEX.get(id(location))

        if location_index is None:
            raise ValueError(
                f"German unit {unit.name} is in unsupported "
                f"location {location.name}"
            )

    observation[start_index + location_index] = 1.0


def count_generic_units(space, unit_type, combat_value=None):
    count = 0

    for unit in space.units:
        if not isinstance(unit, GermanUnit):
            continue

        if unit.type != unit_type:
            continue

        if combat_value is not None and unit.combat_value != combat_value:
            continue

        count += 1

    return count


def get_allied_track(army):
    if army is US_FIRST_ARMY:
        return us_1_track

    if army is BRITISH_SECOND_ARMY:
        return brit_2_track

    if army is CANADIAN_FIRST_ARMY:
        return can_1_track

    if army is US_THIRD_ARMY:
        return us_xv_track

    if army is US_VIII_CORPS:
        return us_viii_track

    if army is US_XV_CORPS:
        return us_xv_track

    raise ValueError(
        f"Unknown Allied army: {army}"
    )


def get_allied_progress(army):
    if army.location is None:
        return 0.0

    track = get_allied_track(army)

    try:
        current_index = track.index(army.location)
    except ValueError:
        return 0.0

    if len(track) <= 1:
        return 0.0

    return current_index / (len(track) - 1)


def army_was_counterattacked(army_name):
    counterattacked = GlobalGameState.counter_attacked_armies
    return army_name in counterattacked


# ============================================================
# CAMPAIGN / DECK STATE
# ============================================================

def encode_campaign_state(observation):
    observation[CARDS_DRAWN_INDEX] = (
        GlobalGameState.cards_drawn / MAX_CARDS
    )

    current_card = GlobalGameState.current_card

    if current_card is not None:
        card_id = current_card.card_id

        if 1 <= card_id <= MAX_CARDS:
            observation[
                CURRENT_CARD_START + card_id - 1
            ] = 1.0

    for card in GlobalGameState.drawn_cards:
        if hasattr(card, "card_id"):
            card_id = card.card_id
        else:
            card_id = int(card)

        if 1 <= card_id <= MAX_CARDS:
            observation[
                DRAWN_CARDS_START + card_id - 1
            ] = 1.0

    observation[MID_DECK_ADDED_INDEX] = float(
        GlobalGameState.mid_deck_added
    )

    observation[LATE_DECK_ADDED_INDEX] = float(
        GlobalGameState.late_deck_added
    )


# ============================================================
# WEATHER
# ============================================================

def encode_weather(observation):
    weather = GlobalGameState.current_weather

    if weather is None:
        return

    if weather.weather_type == WeatherType.CLEAR:
        observation[WEATHER_CLEAR_INDEX] = 1.0

    elif weather.weather_type == WeatherType.PARTLY_CLEAR:
        observation[WEATHER_PARTLY_CLEAR_INDEX] = 1.0

    elif weather.weather_type == WeatherType.OVERCAST:
        observation[WEATHER_OVERCAST_INDEX] = 1.0


# ============================================================
# RESOURCES / ACTION ECONOMY
# ============================================================

def encode_resources(observation):
    observation[TRANSPORT_INDEX] = (
        transport_track.value / MAX_TRANSPORT
    )

    observation[SUPPLY_INDEX] = (
        supply_track.value / MAX_SUPPLY
    )

    observation[HITLER_APPROVAL_INDEX] = normalize(
        hitler_approval_track.value,
        MIN_HITLER_APPROVAL,
        MAX_HITLER_APPROVAL,
    )

    observation[TRANSPORT_BASE_INDEX] = (
        GlobalGameState.transport_base_level
        / MAX_TRANSPORT
    )

    observation[SUPPLY_BASE_INDEX] = (
        GlobalGameState.supply_base_level
        / MAX_SUPPLY
    )

    observation[HITLER_APPROVAL_BASE_INDEX] = normalize(
        GlobalGameState.hitler_approval_base_level,
        MIN_HITLER_APPROVAL,
        MAX_HITLER_APPROVAL,
    )

    observation[ACTIONS_LEFT_INDEX] = normalize(
        GlobalGameState.actions_left_this_turn,
        0,
        MAX_ACTIONS,
    )

    observation[RESERVE_ACTIONS_INDEX] = normalize(
        GlobalGameState.reserve_actions,
        0,
        MAX_RESERVE_ACTIONS,
    )


# ============================================================
# ALLIED ARMIES
# ============================================================

def encode_allied_armies(observation):
    index = ALLIED_ARMIES_START

    for army in ALLIED_ARMIES:
        observation[index] = get_allied_progress(army)

        observation[index + 1] = float(
            army.location is not None
        )

        observation[index + 2] = float(
            army.flipped
        )

        observation[index + 3] = float(
            army_was_counterattacked(army.name)
        )

        index += ALLIED_FEATURES_PER_ARMY

    observation[US_THIRD_ACTIVATED_INDEX] = float(
        GlobalGameState.us_third_army_activated
    )

    observation[US_THIRD_MERGED_INDEX] = float(
        GlobalGameState.us_third_army_merged
    )

    observation[ARMIES_UPGRADED_INDEX] = float(
        GlobalGameState.armies_upgraded
    )


# ============================================================
# PANZER DIVISIONS
# ============================================================

def encode_panzer_divisions(observation):
    index = PANZERS_START

    for unit in panzer_divisions_list:
        encode_named_unit_location(
            observation,
            index,
            unit,
        )

        observation[index + 31] = float(
            unit.combat_value == 1
        )

        index += PANZER_FEATURES_PER_UNIT


# ============================================================
# FS3 / FS5 / TIGER 101
# ============================================================

def encode_special_german_units(observation):
    # 3rd Fallschirmjager:
    # 31 location bits + reduced flag.

    encode_named_unit_location(
        observation,
        FS3_START,
        FS_3,
    )

    observation[FS3_REDUCED_INDEX] = float(
        FS_3.combat_value == 1
    )

    # 5th Fallschirmjager:
    # 31 location bits only.

    encode_named_unit_location(
        observation,
        FS5_START,
        FS_5,
    )

    # 101st Tiger Battalion:
    # 31 location bits only.

    encode_named_unit_location(
        observation,
        TIGER_101_START,
        TIGER_101,
    )


# ============================================================
# COMMANDERS
# ============================================================

def encode_commanders(observation):
    encode_named_unit_location(
        observation,
        ROMMEL_START,
        ROMMEL,
    )

    encode_named_unit_location(
        observation,
        MEYER_START,
        MEYER,
    )

    encode_named_unit_location(
        observation,
        MODEL_START,
        MODEL,
    )

    observation[MEYER_AVAILABLE_INDEX] = float(
        GlobalGameState.meyer_available
    )

    observation[MODEL_IN_COMMAND_INDEX] = float(
        GlobalGameState.model_in_command
    )


# ============================================================
# GENERIC GERMAN UNITS ON MAP
# ============================================================

def encode_generic_map_units(observation):
    index = GENERIC_MAP_UNITS_START

    for space in GERMAN_MAP_SPACES:
        full_flak = count_generic_units(
            space,
            ReinforcementType.FLAK_88,
            combat_value=2,
        )

        reduced_flak = count_generic_units(
            space,
            ReinforcementType.FLAK_88,
            combat_value=1,
        )

        kampfgruppen = count_generic_units(
            space,
            ReinforcementType.KAMPFGRUPPE,
        )

        nebelwerfers = count_generic_units(
            space,
            ReinforcementType.NEBELWERFER,
        )

        observation[index] = normalize(
            full_flak,
            0,
            MAX_FLAK_STACK,
        )

        observation[index + 1] = normalize(
            reduced_flak,
            0,
            MAX_FLAK_STACK,
        )

        observation[index + 2] = normalize(
            kampfgruppen,
            0,
            MAX_KAMPFGRUPPE_STACK,
        )

        observation[index + 3] = normalize(
            nebelwerfers,
            0,
            MAX_NEBELWERFER_STACK,
        )

        index += GENERIC_FEATURES_PER_SPACE


# ============================================================
# GENERIC GERMAN UNITS IN OFF-MAP BOXES
# ============================================================

def encode_generic_box_units(observation):
    index = GENERIC_BOX_UNITS_START

    for box in GENERIC_UNIT_BOXES:
        full_flak = count_generic_units(
            box,
            ReinforcementType.FLAK_88,
            combat_value=2,
        )

        reduced_flak = count_generic_units(
            box,
            ReinforcementType.FLAK_88,
            combat_value=1,
        )

        kampfgruppen = count_generic_units(
            box,
            ReinforcementType.KAMPFGRUPPE,
        )

        nebelwerfers = count_generic_units(
            box,
            ReinforcementType.NEBELWERFER,
        )

        observation[index] = normalize(
            full_flak,
            0,
            MAX_TOTAL_FLAK,
        )

        observation[index + 1] = normalize(
            reduced_flak,
            0,
            MAX_TOTAL_FLAK,
        )

        observation[index + 2] = normalize(
            kampfgruppen,
            0,
            MAX_TOTAL_KAMPFGRUPPE,
        )

        observation[index + 3] = normalize(
            nebelwerfers,
            0,
            MAX_TOTAL_NEBELWERFER,
        )

        index += GENERIC_FEATURES_PER_BOX


# ============================================================
# DYNAMIC MAP STATE
# ============================================================

def encode_map_state(observation):
    index = MAP_STATE_START

    for space in GERMAN_MAP_SPACES:
        observation[index] = float(
            space.controlling_player == SideType.GERMAN
        )

        observation[index + 1] = float(
            space.under_siege
        )

        observation[index + 2] = normalize(
            space.fortified_village_modifier,
            0,
            MAX_FORTIFIED_VILLAGE_MODIFIER,
        )
        # Reserved to preserve existing observation offsets
        observation[index + 3] = 0.0

        index += MAP_FEATURES_PER_SPACE


# ============================================================
# PERSISTENT STRATEGIC STATE
# ============================================================

def encode_global_game_state(observation):
    observation[CHERBOURG_CAPTURED_INDEX] = float(
        GlobalGameState.cherbourg_captured
    )

    observation[HITLER_ASSASSINATION_INDEX] = float(
        GlobalGameState.hitler_assassination
    )

    observation[HITLER_INTERVENTION_NO_EFFECT_INDEX] = float(
        GlobalGameState.hitler_intervention_no_effect
    )

    observation[BOCAGE_DEFENSE_INDEX] = normalize(
        GlobalGameState.bocage_defense_modifier,
        MIN_BOCAGE_DEFENSE_MODIFIER,
        MAX_BOCAGE_DEFENSE_MODIFIER,
    )

    observation[US_FIRST_FURTHEST_ADVANCE_INDEX] = normalize(
        11 - GlobalGameState.us_first_army_furthest_advance,
        0,
        11,
    )

    observation[CURRENT_CARPET_BOMBING_INDEX] = normalize(
        GlobalGameState.current_carpet_bombing,
        MIN_CARPET_BOMBING,
        MAX_CARPET_BOMBING,
    )


# ============================================================
# RESOURCE / CHECK DRMS
# ============================================================

def encode_drms(observation):
    observation[TRANSPORT_ROLL_DRM_INDEX] = normalize(
        GlobalGameState.transport_roll_drm,
        MIN_TRANSPORT_ROLL_DRM,
        MAX_TRANSPORT_ROLL_DRM,
    )

    observation[SUPPLY_ROLL_DRM_INDEX] = normalize(
        GlobalGameState.supply_roll_drm,
        MIN_SUPPLY_ROLL_DRM,
        MAX_SUPPLY_ROLL_DRM,
    )

    observation[TRANSPORT_CHECK_DRM_INDEX] = normalize(
        GlobalGameState.transport_check_drm,
        MIN_TRANSPORT_CHECK_DRM,
        MAX_TRANSPORT_CHECK_DRM,
    )

    observation[SUPPLY_CHECK_DRM_INDEX] = normalize(
        GlobalGameState.supply_check_drm,
        MIN_SUPPLY_CHECK_DRM,
        MAX_SUPPLY_CHECK_DRM,
    )

    observation[HITLER_APPROVAL_CHECK_DRM_INDEX] = normalize(
        GlobalGameState.hitler_approval_check_drm,
        MIN_HITLER_APPROVAL_CHECK_DRM,
        MAX_HITLER_APPROVAL_CHECK_DRM,
    )


# ============================================================
# DECISION TYPE
# ============================================================

def encode_decision_type(observation, context):
    if context is None:
        return

    if context.decision_type is None:
        return

    if context.decision_type == DecisionType.ACTION_PHASE:
        observation[DECISION_ACTION_PHASE_INDEX] = 1.0

    elif context.decision_type == DecisionType.RESERVE_DEPLOYMENT:
        observation[DECISION_RESERVE_DEPLOYMENT_INDEX] = 1.0

    elif context.decision_type == DecisionType.CASUALTY:
        observation[DECISION_CASUALTY_INDEX] = 1.0

    elif context.decision_type == DecisionType.HITLER_INTERVENTION:
        observation[DECISION_HITLER_INTERVENTION_INDEX] = 1.0


# ============================================================
# DECISION CONTEXT
# ============================================================

def encode_decision_context(observation, context):
    if context is None:
        return

    if context.relevant_army is not None:
        try:
            army_index = ALLIED_ARMIES.index(
                context.relevant_army
            )

            observation[
                DECISION_ARMY_START + army_index
            ] = 1.0

        except ValueError:
            pass

    if context.relevant_space is not None:
        try:
            space_index = GERMAN_MAP_SPACES.index(
                context.relevant_space
            )

            observation[
                DECISION_SPACE_START + space_index
            ] = 1.0

        except ValueError:
            pass

    observation[DECISION_ALLIED_ATTACK_INDEX] = float(
        context.allied_attack
    )

    observation[DECISION_GERMAN_COUNTERATTACK_INDEX] = float(
        context.german_counterattack
    )


# ============================================================
# HITLER INTERVENTION SUBSTATE
# ============================================================

def encode_hitler_intervention_state(observation):
    stage = HeadlessDecisionState.hitler_intervention_stage

    if stage is None:
        observation[HITLER_STAGE_NONE_INDEX] = 1.0

    elif stage == "CANCEL":
        observation[HITLER_STAGE_CANCEL_INDEX] = 1.0

    elif stage == "TARGET":
        observation[HITLER_STAGE_TARGET_INDEX] = 1.0

    elif stage == "REDEPLOY":
        observation[HITLER_STAGE_REDEPLOY_INDEX] = 1.0

    else:
        raise ValueError(
            f"Unknown Hitler Intervention stage: {stage}"
        )

    target = HeadlessDecisionState.hitler_intervention_target

    if target is not None:
        try:
            target_index = ALLIED_ARMIES.index(target)

            observation[
                HITLER_TARGET_START + target_index
            ] = 1.0

        except ValueError:
            pass

    observation[HITLER_REDEPLOYMENTS_INDEX] = normalize(
        HeadlessDecisionState.hitler_intervention_redeployments,
        0,
        2,
    )


# ============================================================
# MAIN ENCODER
# ============================================================

def encode_game_state(context=None):
    observation = np.zeros(
        OBSERVATION_SIZE,
        dtype=np.float32,
    )

    encode_campaign_state(observation)

    encode_weather(observation)

    encode_resources(observation)

    encode_allied_armies(observation)

    encode_panzer_divisions(observation)

    encode_special_german_units(observation)

    encode_commanders(observation)

    encode_generic_map_units(observation)

    encode_generic_box_units(observation)

    encode_map_state(observation)

    encode_global_game_state(observation)

    encode_drms(observation)

    encode_decision_type(
        observation,
        context,
    )

    encode_decision_context(
        observation,
        context,
    )

    encode_hitler_intervention_state(
        observation
    )

    validate_observation(observation)

    return observation


# ============================================================
# VALIDATION
# ============================================================

def validate_observation(observation):
    if observation.shape != (OBSERVATION_SIZE,):
        raise ValueError(
            f"Observation has shape {observation.shape}; "
            f"expected ({OBSERVATION_SIZE},)"
        )

    if observation.dtype != np.float32:
        raise ValueError(
            f"Observation dtype is {observation.dtype}; "
            f"expected float32"
        )

    if not np.all(np.isfinite(observation)):
        raise ValueError(
            "Observation contains NaN or infinite values"
        )

    if np.any(observation < 0.0):
        bad_indices = np.where(
            observation < 0.0
        )[0]

        raise ValueError(
            f"Observation contains values below 0 "
            f"at indices {bad_indices.tolist()}"
        )

    if np.any(observation > 1.0):
        bad_indices = np.where(
            observation > 1.0
        )[0]

        raise ValueError(
            f"Observation contains values above 1 "
            f"at indices {bad_indices.tolist()}"
        )


# ============================================================
# OPTIONAL DEBUGGING HELPERS
# ============================================================

def get_active_observation_indices(observation):
    return np.where(
        observation != 0.0
    )[0].tolist()


def print_observation_summary(observation):
    print()
    print("========================================")
    print("DQN OBSERVATION")
    print("========================================")
    print(f"Shape: {observation.shape}")
    print(f"Dtype: {observation.dtype}")
    print(f"Minimum: {observation.min():.3f}")
    print(f"Maximum: {observation.max():.3f}")
    print(
        f"Non-zero features: "
        f"{np.count_nonzero(observation)}"
    )
    print("========================================")

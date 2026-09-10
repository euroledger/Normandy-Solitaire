import json
import os
from pathlib import Path
from cards.decks import draw_deck, mid_deck, late_deck

from core.enums import SideType
from core.global_game_state import GlobalGameState

from core.allied_armies import (
    US_FIRST_ARMY,
    BRITISH_SECOND_ARMY,
    CANADIAN_FIRST_ARMY,
    US_THIRD_ARMY,
    US_VIII_CORPS,
    US_XV_CORPS,
)

from core.german_units import (
    FS_3,
    FS_5,
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
    create_nebelwerfer,
    create_flak88,
    create_kampfgruppe,
)

from core.map.map_model import (
    transport_track,
    supply_track,
    hitler_approval_track,
    in_transit_box,
    strategic_reserve_box,
    eliminated_units_box,
)

from core.map.map_spaces_us_1 import us_1_track
from core.map.map_spaces_brit_2 import brit_2_track
from core.map.map_spaces_can_1 import can_1_track
from core.map.map_spaces_us_3 import (
    us_viii_track,
    us_xv_track,
)
from core.models import Strategy
from core.tables.weather import WeatherResult, WeatherType


# ---------------------------------------------------------
# CARD LOOKUP
# ---------------------------------------------------------

CARDS_BY_ID = {card.card_id: card for card in draw_deck + mid_deck + late_deck}


# ---------------------------------------------------------
# UNIT LOOKUPS
# ---------------------------------------------------------

GERMAN_UNITS_BY_SAVE_NAME = {
    unit.name: unit
    for unit in [
        FS_3,
        FS_5,
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
    ]
}


ALLIED_UNITS_BY_SAVE_NAME = {
    unit.name: unit
    for unit in [
        US_FIRST_ARMY,
        BRITISH_SECOND_ARMY,
        CANADIAN_FIRST_ARMY,
        US_THIRD_ARMY,
        US_VIII_CORPS,
        US_XV_CORPS,
    ]
}


def get_unit_by_save_name(unit_name):
    if unit_name in ALLIED_UNITS_BY_SAVE_NAME:
        return ALLIED_UNITS_BY_SAVE_NAME[unit_name]

    if unit_name in GERMAN_UNITS_BY_SAVE_NAME:
        return GERMAN_UNITS_BY_SAVE_NAME[unit_name]

    if unit_name == "Nebelwerfer":
        return create_nebelwerfer()

    if unit_name == "Flak 88":
        return create_flak88()

    if unit_name == "Kampfgruppe":
        return create_kampfgruppe()

    raise ValueError(f"Unknown unit in save file: {unit_name}")


# ---------------------------------------------------------
# MAP HELPERS
# ---------------------------------------------------------


def get_all_map_spaces():
    spaces = []

    for track in [
        us_1_track,
        brit_2_track,
        can_1_track,
        us_viii_track,
        us_xv_track,
    ]:
        spaces.extend(track)

    return spaces


def clear_all_units_from_map():
    for space in get_all_map_spaces():
        space.units.clear()


# ---------------------------------------------------------
# GLOBAL GAME STATE SERIALIZATION
# ---------------------------------------------------------


def serialize_value(value):

    if value is None:
        return None

    if hasattr(value, "card_id"):
        return value.card_id

    if isinstance(value, list):
        return [serialize_value(item) for item in value]

    if hasattr(value, "weather_type"):
        return {
            "__type__": "WeatherResult",
            "weather_type": value.weather_type.name,
            "available_jabos": value.available_jabos,
            "resource_drm": value.resource_drm,
            "carpet_bombing_drm": value.carpet_bombing_drm,
        }

    if hasattr(value, "name"):
        return value.name

    return value


def load_army_flip_state():
    for army in [US_FIRST_ARMY, BRITISH_SECOND_ARMY, CANADIAN_FIRST_ARMY]:
        army.flipped = GlobalGameState.armies_upgraded


def load_army_merged_state():
    US_THIRD_ARMY.merged = GlobalGameState.us_third_army_merged
        
def save_global_game_state():
    return {
        name: serialize_value(value)
        for name, value in GlobalGameState.__dict__.items()
        if not name.startswith("_") and not callable(value) and name != "counter_attacked_armies"

    }


def load_global_game_state(saved_state):

    for name, saved_value in saved_state.items():
        if name == "current_card":
            value = None if saved_value is None else CARDS_BY_ID[saved_value]
        elif name == "drawn_cards":
            value = [CARDS_BY_ID[card_id] for card_id in saved_value]
        elif name == "german_casualty_strategy":
            value = Strategy[saved_value]
            print(f"SETTING GLOBAL GAME STATE saved_value={saved_value} -> name={name} value={value}")
        elif isinstance(saved_value, dict) and saved_value.get("__type__") == "WeatherResult":
            value = WeatherResult(weather_type=WeatherType[saved_value["weather_type"]], available_jabos=saved_value["available_jabos"],
                          resource_drm=saved_value["resource_drm"], carpet_bombing_drm=saved_value["carpet_bombing_drm"])
        else:
            value = saved_value

        setattr(
            GlobalGameState,
            name,
            value,
        )


# ---------------------------------------------------------
# SAVE
# ---------------------------------------------------------


BASE_DIR = Path(__file__).resolve().parent.parent   # goes up from /core → /root
DATA_DIR = BASE_DIR / "data"


def save_game(save_name=None):

    print("SAVE GAME")
    print()

    if save_name is not None:
        save_file_name = save_name
    else:
        save_file_name = input("Save file name: ").strip()

        if not save_file_name:
            print("SAVE CANCELLED")
            return

    if not save_file_name.endswith(".json"):
        save_file_name += ".json"

    # ensure directory exists
    DATA_DIR.mkdir(exist_ok=True)

    save_path = DATA_DIR / save_file_name

    save_data = {
        "draw_deck": [card.card_id for card in draw_deck],
        "global_game_state": save_global_game_state(),
        "resource_tracks": {
            "transport": transport_track.value,
            "supply": supply_track.value,
            "hitler_approval": hitler_approval_track.value,
        },
        "unit_boxes": {
            "in_transit": [unit.name for unit in in_transit_box.units],
            "strategic_reserve": [unit.name for unit in strategic_reserve_box.units],
            "eliminated_units": [unit.name for unit in eliminated_units_box.units],
        },
        "map_spaces": {
            space.name: {
                "controlling_player": space.controlling_player.name,
                "fortified_village_modifier": space.fortified_village_modifier,
                "fortified": space.fortified,
                "under_siege": space.under_siege,
                "units": [unit.name for unit in space.units],
            }
            for space in get_all_map_spaces()
        },
    }

    with open(
        save_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            save_data,
            file,
            indent=4,
        )

    print()
    print(f"GAME SAVED: {save_path}")


# ---------------------------------------------------------
# LOAD
# ---------------------------------------------------------


def load_game():

    print("LOAD GAME")
    print()

    save_file_name = input("Save file name: ").strip()

    if not save_file_name:
        print("LOAD CANCELLED")
        return

    if not save_file_name.endswith(".json"):
        save_file_name += ".json"

    save_path = DATA_DIR / save_file_name
    if not os.path.exists(save_path):
        print(f"SAVE FILE NOT FOUND: {save_path}")
        return

    with open(
        save_path,
        "r",
        encoding="utf-8",
    ) as file:
        save_data = json.load(file)

    draw_deck[:] = [CARDS_BY_ID[card_id] for card_id in save_data["draw_deck"]]

    load_global_game_state(save_data["global_game_state"])
    load_army_flip_state()
    load_army_merged_state()
    resources = save_data["resource_tracks"]

    transport_track.value = resources["transport"]
    supply_track.value = resources["supply"]
    hitler_approval_track.value = resources["hitler_approval"]

    clear_all_units_from_map()

    boxes = save_data["unit_boxes"]
    in_transit_box.units[:] = [get_unit_by_save_name(unit) for unit in boxes["in_transit"]]
    strategic_reserve_box.units[:] = [get_unit_by_save_name(unit) for unit in boxes["strategic_reserve"]]
    eliminated_units_box.units[:] = [get_unit_by_save_name(unit) for unit in boxes["eliminated_units"]]
    spaces = {space.name: space for space in get_all_map_spaces()}

    for name, saved_space in save_data["map_spaces"].items():
        space = spaces[name]
        space.controlling_player = SideType[saved_space["controlling_player"]]
        space.fortified_village_modifier = saved_space["fortified_village_modifier"]
        space.fortified = saved_space["fortified"]
        space.under_siege = saved_space["under_siege"]
        space.units[:] = [get_unit_by_save_name(unit) for unit in saved_space["units"]]

        for unit in space.units:
            if hasattr(unit, "location"):
                unit.location = space

    print()
    print(f"GAME LOADED: {save_path}")

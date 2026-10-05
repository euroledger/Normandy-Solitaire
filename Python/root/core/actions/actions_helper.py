from core.actions.stacking_limits import FALLSCHIRMJAGER_STACKING_LIMIT, FLAK_88_STACKING_LIMIT, NEBELWERFER_STACKING_LIMIT, PANZER_STACKING_LIMIT
from core.enums import ReinforcementType, SideType
from core.german_units import SS_12
from core.global_game_state import GlobalGameState
from core.map.map_model import in_transit_box, strategic_reserve_box, transport_track, eliminated_units_box
from core.map.map_utilities import get_all_map_spaces
from core.models import GermanUnit


LIGHT_BROWN = "\033[38;5;180m"
BLUE = "\033[94m"
RED = "\033[91m"
GREEN = "\033[92m"
GREY = "\033[90m"
RESET = "\033[0m"
BOLD = "\033[1m"
CYAN = "\033[96m"


def get_german_units_on_map():
    german_units = []
    for space in get_all_map_spaces():
        if space in [in_transit_box, strategic_reserve_box, eliminated_units_box]:
            continue
        if space.under_siege:
            continue
        for unit in space.units:
            if isinstance(unit, GermanUnit):
                german_units.append((space, unit))
    return german_units


def can_add_unit_to_space(space, unit):
    counts = {
        ReinforcementType.PZ_DIV: sum(1 for u in space.units if u.type in [ReinforcementType.PZ_DIV, ReinforcementType.KAMPFGRUPPE]),
        ReinforcementType.KAMPFGRUPPE: 0,
        ReinforcementType.NEBELWERFER: sum(1 for u in space.units if u.type == ReinforcementType.NEBELWERFER),
        ReinforcementType.FALLSCHIRMJAGER: sum(1 for u in space.units if u.type == ReinforcementType.FALLSCHIRMJAGER),
        ReinforcementType.FLAK_88: sum(1 for u in space.units if u.type == ReinforcementType.FLAK_88),
    }

    if unit.type in [ReinforcementType.PZ_DIV, ReinforcementType.KAMPFGRUPPE]:
        return counts[ReinforcementType.PZ_DIV] < PANZER_STACKING_LIMIT
    if unit.type == ReinforcementType.NEBELWERFER:
        return counts[ReinforcementType.NEBELWERFER] < NEBELWERFER_STACKING_LIMIT
    if unit.type == ReinforcementType.FALLSCHIRMJAGER:
        return counts[ReinforcementType.FALLSCHIRMJAGER] < FALLSCHIRMJAGER_STACKING_LIMIT
    if unit.type == ReinforcementType.FLAK_88:
        return counts[ReinforcementType.FLAK_88] < FLAK_88_STACKING_LIMIT
    return True


def use_action():
    if GlobalGameState.actions_left_this_turn > 0:
        GlobalGameState.actions_left_this_turn -= 1
        return True

    if GlobalGameState.reserve_actions == 0:
        return False


    # if GlobalGameState.headless:
    #     GlobalGameState.reserve_actions -= 1
    #     return True
    if GlobalGameState.headless:
        GlobalGameState.reserve_actions -= 1
        print("AI USED 1 RESERVE ACTION")
        print(f"RESERVE ACTIONS REMAINING: {GlobalGameState.reserve_actions}")
        return True
    
    print()
    print("NO ACTIONS REMAINING")
    choice = input("USE 1 RESERVE ACTION? (Y/N): ").strip().upper()

    if choice != "Y":
        return False

    GlobalGameState.reserve_actions -= 1
    return True

def get_adjacent_german_controlled_spaces(source_space):
    german_spaces = get_german_controlled_spaces()

    if source_space.name == "FALAISE GAP":
        return [space for space in german_spaces if space.track and space.track_number == 1]

    adjacent_spaces = [space for space in german_spaces if space.track == source_space.track and abs(space.track_number - source_space.track_number) == 1]

    if source_space.track_number == 1:
        falaise_gap = next((space for space in german_spaces if space.name == "FALAISE GAP"), None)
        if falaise_gap:
            adjacent_spaces.append(falaise_gap)

    return [space for space in adjacent_spaces if not space.under_siege]

def get_german_controlled_spaces():
    spaces = [
        space
        for space in get_all_map_spaces()
        if hasattr(space, "controlling_player") and space.controlling_player == SideType.GERMAN
    ]
    unique_spaces = []
    for space in spaces:
        if space not in unique_spaces:
            unique_spaces.append(space)

    track_order = {
        "US FIRST ARMY": 1,
        "BRIT SECOND ARMY": 2,
        "CANADIAN FIRST ARMY": 3,
        "US THIRD ARMY": 4,
        "NO TRACK": 5,
    }
    unique_spaces.sort(
        key=lambda space: (
            track_order.get(space.track.value if space.track else "NO TRACK"),
            -space.track_number,
        )
    )
    return [space for space in unique_spaces if not space.under_siege]


def get_unit_location(unit: GermanUnit):
    for space in get_all_map_spaces():
        if unit in space.units:
            return space
    if unit in strategic_reserve_box.units:
        return strategic_reserve_box

    if unit in eliminated_units_box.units:
        return eliminated_units_box

    if unit in in_transit_box.units:
        return in_transit_box
    return None

def build_display_spaces(spaces):
    us_1_spaces = []
    brit_2_spaces = []
    can_1_spaces = []
    us_3_spaces = []
    falaise_gap_spaces = []

    for space in spaces:
        if space.name == "FALAISE GAP":
            falaise_gap_spaces.append(space)
        elif space.track.value == "US FIRST ARMY":
            us_1_spaces.append(space)
        elif space.track.value == "BRIT SECOND ARMY":
            brit_2_spaces.append(space)
        elif space.track.value == "CANADIAN FIRST ARMY":
            can_1_spaces.append(space)
        elif space.track.value == "US THIRD ARMY":
            us_3_spaces.append(space)

    display_spaces = []
    display_spaces.extend(us_1_spaces)
    display_spaces.extend(brit_2_spaces)
    display_spaces.extend(can_1_spaces)
    display_spaces.extend(us_3_spaces)
    display_spaces.extend(falaise_gap_spaces[:1])

    return display_spaces


def get_display_spaces(show_menu):
    german_spaces = get_german_controlled_spaces()

    if not german_spaces:
        print("No German-controlled spaces available")
        return None

    display_spaces = build_display_spaces(german_spaces)

    if show_menu:
        print_display_spaces(display_spaces)

    return display_spaces


def print_space_section(title, spaces, color, display_spaces):
    if not spaces:
        return
    print()
    print(f"{color}{title}{RESET}")
    for space in spaces:
        display_spaces.append(space)
        index = len(display_spaces)
        german_units = [unit for unit in space.units if isinstance(unit, GermanUnit)]
        marker = ""
        if space.fortified_village_modifier == 1:
            marker = " - FORTIFIED VILLAGES +1"
        elif space.fortified_village_modifier == 2:
            marker = " - FORTIFIED VILLAGES +2"
        if german_units:
            units_text = ", ".join(f"{unit.name} ({unit.combat_value})" for unit in german_units)
            print(f"{color}{index}. {space.name} (#{space.track_number}){marker} - {units_text}{RESET}")
        else:
            print(f"{color}{index}. {space.name} (#{space.track_number}){marker}{RESET}")


def print_display_spaces(display_spaces):
    us_1_spaces = [space for space in display_spaces if space.track and space.track.value == "US FIRST ARMY"]
    brit_2_spaces = [space for space in display_spaces if space.track and space.track.value == "BRIT SECOND ARMY"]
    can_1_spaces = [space for space in display_spaces if space.track and space.track.value == "CANADIAN FIRST ARMY"]
    us_3_spaces = [space for space in display_spaces if space.track and space.track.value == "US THIRD ARMY"]
    falaise_gap_spaces = [space for space in display_spaces if space.name == "FALAISE GAP"]

    numbered_spaces = []

    print()
    print("SELECT GERMAN-CONTROLLED SPACE")

    print_space_section("US 1ST ARMY", us_1_spaces, LIGHT_BROWN, numbered_spaces)
    print_space_section("BRIT 2ND ARMY", brit_2_spaces, BLUE, numbered_spaces)
    print_space_section("1st CAN ARMY", can_1_spaces, RED, numbered_spaces)
    print_space_section("US 3RD ARMY", us_3_spaces, GREEN, numbered_spaces)
    print_space_section("FALAISE GAP", falaise_gap_spaces, GREY, numbered_spaces)

    print()
    print("0. Return to main menu")


def do_panzer_transport_check(unit, die_roll):
    modified_roll = die_roll + GlobalGameState.transport_check_drm

    print(f"{unit}")
    print(f"ROLL: {die_roll}")
    print(f"DRM: {GlobalGameState.transport_check_drm:+}")
    print(f"MODIFIED ROLL: {modified_roll}")
    print(f"CHECK: {modified_roll} <= Transport {transport_track.value}")

    if modified_roll <= transport_track.value:
        in_transit_box.units.remove(unit)
        strategic_reserve_box.units.append(unit)

        print(f"\t=>{unit} MOVED TO STRATEGIC RESERVE")
        return True

    print(f"\t=>{unit} REMAINS IN TRANSIT")
    return False

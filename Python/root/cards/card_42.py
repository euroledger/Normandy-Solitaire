from core.game_constants import GREEN, RESET
from core.models import *
from core.enums import *
from core.allied_armies import US_THIRD_ARMY, BRITISH_SECOND_ARMY, CANADIAN_FIRST_ARMY, US_VIII_CORPS, US_XV_CORPS
from core.models import Effect


# =========================================================
# CARD #42
# BRADLEY HALT ORDER
# =========================================================

card = Card(card_id=42, title="Bradley Halt Order")

# =========================================================
# MILITARY
# =========================================================

def retreat_formation(army):
    if army.location is None:
        return False

    from core.allied_advances_phase import get_track_for



    current_space = army.location
    track = get_track_for(army)
    retreat_space = next((space for space in track if space.track_number == current_space.track_number + 1), None)

    if retreat_space is None:
        return False


    current_space.units.remove(army)
    retreat_space.units.append(army)
    army.location = retreat_space

    # An Allied formation occupying a space makes it Allied-controlled.
    # This is essential when Bradley's Halt Order moves an army
    # into a space that was previously German-controlled.
    retreat_space.controlling_player = SideType.ALLIED


    if army in [US_VIII_CORPS, US_XV_CORPS]:
        from core.allied_advances_phase import check_and_merge_us_third_army

        check_and_merge_us_third_army(retreat_space)

    print(f"{GREEN}*** {army.display_name} RETREATS FROM {current_space.name} TO {retreat_space.name}{RESET} ***")
    return True


def event():
    if US_THIRD_ARMY.location is not None and US_THIRD_ARMY.location.track_number <= 3:
        retreat_formation(US_THIRD_ARMY)
    elif US_XV_CORPS.location is not None and US_XV_CORPS.location.track_number <= 3:
        retreat_formation(US_XV_CORPS)
    else:
        print("BRADLEY HALT ORDER: NO EFFECT")


card.event = event

card.military.formations.extend([BRITISH_SECOND_ARMY, CANADIAN_FIRST_ARMY])

card.military.text.extend([
    "RETREAT 3rd Army.",
    "Applies if 3rd Army has reached Le Mans or further otherwise return card to draw pile."
])

# =========================================================
# AIR POWER
# =========================================================

card.air_power.effects.extend([
    Effect(modifier_type=ModifierType.AIR_POWER, value=1, target=BRITISH_SECOND_ARMY),
    Effect(modifier_type=ModifierType.AIR_POWER, value=1, target=CANADIAN_FIRST_ARMY)
])

# =========================================================
# RESOURCES
# =========================================================

card.resources.display_text = "NONE"

# =========================================================
# ACTIONS
# =========================================================

card.actions.actions_available = 2

card.actions.effects.extend([
    Effect(modifier_type=ModifierType.DEFENSE_STRENGTH, value=-2, target=BRITISH_SECOND_ARMY),
    Effect(modifier_type=ModifierType.DEFENSE_STRENGTH, value=-2, target=CANADIAN_FIRST_ARMY)
])
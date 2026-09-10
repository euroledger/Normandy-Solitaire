from core.models import *
from core.enums import *
from core.allied_armies import US_FIRST_ARMY
from core.german_units import SS_17_PZGRD


# =========================================================
# CARD #3
# CARENTAN
# =========================================================

card = Card(card_id=3, title="Carentan")

# =========================================================
# MILITARY
# =========================================================

card.military.formations.append(US_FIRST_ARMY)

# =========================================================
# AIR POWER
# =========================================================

card.air_power.effects.append(
    Effect(modifier_type=ModifierType.AIR_POWER, value=1,
           target=US_FIRST_ARMY))

# =========================================================
# RESOURCES
# =========================================================

card.resources.effects.extend([

    # 17 SS Pz Grd reinforcement
    Effect(modifier_type=ModifierType.REINFORCEMENT,
           value=1,
           target=SS_17_PZGRD),

    # Truck icon = Transport loss
    Effect(modifier_type=ModifierType.RESOURCE_LOSS,
           value=-1,
           resource_type=ResourceType.TRANSPORT),

    # Barrel icon = Supply loss
    Effect(modifier_type=ModifierType.RESOURCE_LOSS,
           value=-1,
           resource_type=ResourceType.SUPPLY)
])

# =========================================================
# RESOURCE DRMS
# =========================================================

card.resources.effects.extend([
    Effect(modifier_type=ModifierType.DRM,
           value=-1,
           resource_type=ResourceType.TRANSPORT)
])

# =========================================================
# ACTIONS
# =========================================================

card.actions.actions_available = 2

card.actions.effects.extend([
    Effect(modifier_type=ModifierType.ATTACK_STRENGTH,
           value=1,
           target=US_FIRST_ARMY)
])

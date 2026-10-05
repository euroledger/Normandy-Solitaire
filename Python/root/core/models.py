from dataclasses import dataclass, field
from enum import auto
from typing import Optional, Union
from core.enums import *
from core.conditions import *

# =========================================================
# CORE DOMAIN OBJECTS
# =========================================================


@dataclass
class AlliedArmy:
    name: str
    nation: Nation
    location: Optional["MapSpace"] = None
    reverse_name: Optional[str] = None
    _strength: int = 0
    reverse_strength: int = 0
    flipped: bool = False
    merged: bool = False

    def __str__(self):
        return self.display_name

    @property
    def display_name(self):
        if self.flipped and self.reverse_name:
            return self.reverse_name
        return self.name

    @property
    def strength(self):
        if self.flipped:
            return self.reverse_strength
        return self._strength

    @strength.setter
    def strength(self, value):
        self._strength = value

    def flip(self):
        self.flipped = not self.flipped

    def merge(self):
        self.merged = not self.merged


@dataclass
class GermanUnit:
    type: ReinforcementType
    name: Optional[str] = None
    combat_value: int = 0

    def __str__(self):
        if self.name:
            return self.name
        return self.type.value

    def is_panzer(self, incl_101=False):
        return self.type in [ReinforcementType.PZ_DIV, ReinforcementType.KAMPFGRUPPE]


# =========================================================
# EFFECTS
# =========================================================


@dataclass
class Effect:
    modifier_type: Optional[ModifierType] = None
    value: int = 0
    target: Optional[Union[AlliedArmy, GermanUnit]] = None
    resource_type: Optional[ResourceType] = None
    condition: Optional[Condition] = None
    description: Optional[str] = None
    label: Optional[str] = None

    def apply(self, game_state):
        if self.condition and not self.condition.is_met(game_state):
            return
        print(f"Applying: {self}")

    def is_active(self, game_state=None):
        if self.condition is None:
            return True
        return self.condition.is_met(game_state)

    def __str__(self):

        # =================================================
        # CONDITIONAL ACTION EFFECTS
        # =================================================

        if self.modifier_type is None:
            return f"{self.value:+}"

        parts = []

        # =================================================
        # EFFECT NAME
        # =================================================

        effect_name = self.modifier_type.value

        # Special display names

        if self.modifier_type == ModifierType.AIR_POWER:
            effect_name = "Jabos"

        elif self.modifier_type == ModifierType.REINFORCEMENT:
            effect_name = "Reinforcement"

        elif self.modifier_type == ModifierType.DRM:
            effect_name = "DRM"

        elif self.modifier_type == ModifierType.COMMANDER:
            effect_name = self.label if self.label else "Commander"

        # Example:
        # +2 Attack Strength
        # -1 DRM
        # +1 Jabos

        parts.append(f"{self.value:+} {effect_name}")

        # =================================================
        # RESOURCE TYPE
        # =================================================

        if self.resource_type:
            parts.append(self.resource_type.value)

        # =================================================
        # TARGET
        # =================================================

        if self.target:
            parts.append(str(self.target))

        return " ".join(parts)


# =========================================================
# CARD SECTIONS
# =========================================================


@dataclass
class MilitarySection:
    formations: list[AlliedArmy] = field(default_factory=list)
    effects: list[Effect] = field(default_factory=list)
    text: list[str] = field(default_factory=list)
    display_text: Optional[str] = None


@dataclass
class AirPowerSection:
    effects: list[Effect] = field(default_factory=list)
    text: list[str] = field(default_factory=list)
    display_text: Optional[str] = None

    def has_carpet_bombing(self) -> bool:
        return "Carpet Bombing" in self.text


@dataclass
class ResourceSection:
    effects: list[Effect] = field(default_factory=list)
    display_text: Optional[str] = None


@dataclass
class ActionSection:
    actions_available: int = 0
    conditional_actions: list[Effect] = field(default_factory=list)
    effects: list[Effect] = field(default_factory=list)

    def total_actions(self, game_state=None):
        total = self.actions_available
        for effect in self.conditional_actions:
            if effect.is_active(game_state):
                total += effect.value
        return total


# =========================================================
# CARD OBJECT
# =========================================================

@dataclass
class Card:
    card_id: int
    title: str
    military: MilitarySection = field(default_factory=MilitarySection)
    air_power: AirPowerSection = field(default_factory=AirPowerSection)
    resources: ResourceSection = field(default_factory=ResourceSection)
    actions: ActionSection = field(default_factory=ActionSection)
    hitler_intervention: bool = False
    hitler_intervention_target_armies: list[AlliedArmy] | None = None
    hitler_intervention_panzer_count: int = 0
    # =====================================================
    # HELPERS
    # =====================================================

    def event(self):
        pass
    
    def advancing_armies(self):
        return self.military.formations

    def is_army_advancing(self, army):
        return army in self.military.formations

    # =====================================================
    # ACTION HELPERS
    # =====================================================

    def total_actions(self, game_state=None):
        return self.actions.total_actions(game_state)

    # =====================================================
    # RESOURCE HELPERS
    # =====================================================

    def resource_changes(self):
        results = []
        for effect in self.resources.effects:
            if effect.resource_type:
                results.append((effect.resource_type, effect.value))
        return results

    # =====================================================
    # REINFORCEMENT HELPERS
    # =====================================================

    def reinforcements(self):
        results = []
        for effect in self.resources.effects:
            if effect.modifier_type == ModifierType.REINFORCEMENT and effect.target:
                results.append((effect.target, effect.value))

        return results

    # =====================================================
    # STRENGTH MODIFIERS
    # =====================================================
    def get_action_modifiers(card):
        modifiers = []
        for effect in card.actions.effects:
            modifiers.append({"modifier_type": effect.modifier_type, "value": effect.value, "target": effect.target, "label": getattr(effect, "label", None)})
        return modifiers

    # =====================================================
    # CARD SUMMARY
    # =====================================================

    def summary(self):
        print()
        print("=" * 60)
        print(f"CARD #{self.card_id} - {self.title}")
        print("=" * 60)

        # -------------------------------------------------
        # RESOURCES
        # -------------------------------------------------
        print("\nRESOURCES")
        if self.resources.display_text:
            print(f"  {self.resources.display_text}")
        elif not self.resources.effects:
            print("  NONE")
        for effect in self.resources.effects:
            print(f"  {effect}")    

        # -------------------------------------------------
        # AIR POWER
        # -------------------------------------------------

        print("\nAIR POWER")
        if not self.air_power.effects and not self.air_power.text:
            print("  N/A")
        for effect in self.air_power.effects:
            print(f"  {effect}")
        for text in self.air_power.text:
            print(f"  {text}")
        # -------------------------------------------------
        # ALLIED ADVANCES
        # -------------------------------------------------
        print("\nALLIED ADVANCES")
        if self.military.display_text:
            print(f"  {self.military.display_text}")
        else:
            if not self.military.formations and not self.military.effects and not self.military.text:
                print("  None")
            for formation in self.military.formations:
                print(f"  {formation.display_name} ({formation._strength})")
            for effect in self.military.effects:
                print(f"  {effect}")
            for text in self.military.text:
                print(f"  {text}")
 

        # -------------------------------------------------
        # ACTIONS
        # -------------------------------------------------
        print("\nACTIONS")
        print(f"  Actions Available: {self.actions.actions_available}")
        if self.actions.conditional_actions:
            print("  Conditional Actions:")
            for effect in self.actions.conditional_actions:
                print(f"    {effect}")
        if not self.actions.effects:
            print("  No Modifiers")
        for effect in self.actions.effects:
            print(f"  {effect}")


@dataclass
class UnitBox:
    name: str
    units: list = field(default_factory=list)


class Strategy(Enum):
    UNIT_TEST = "TEST"
    HUMAN = "HUMAN"
    RANDOM = "RANDOM"
    AI = "AI"

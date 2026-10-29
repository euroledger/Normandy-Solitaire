import unittest

from core.enums import ResourceType
from core.german_units import PZ_LEHR, SS_12
from core.models import GermanUnit
from core.enums import ReinforcementType

from cards.card_1 import card as card_1
from cards.card_9 import card as card_9
from cards.card_10 import card as card_10
from cards.card_19 import card as card_19
from cards.card_25 import card as card_25
from cards.card_26 import card as card_26
from cards.card_31 import card as card_31
from cards.card_34 import card as card_34


class TestResourcesAndReinforcements(unittest.TestCase):

    # =====================================================
    # RESOURCE CHANGES
    # =====================================================

    def test_card_9_resource_changes(self):
        self.assertEqual(card_9.resource_changes(),
                         [(ResourceType.HITLER_APPROVAL, 2),
                          (ResourceType.SUPPLY, 1)])

    def test_card_10_resource_changes(self):
        self.assertEqual(card_10.resource_changes(),
                         [(ResourceType.HITLER_APPROVAL, 2)])

    def test_card_25_resource_changes(self):
        self.assertEqual(card_25.resource_changes(),
                         [(ResourceType.TRANSPORT, -1),
                          (ResourceType.HITLER_APPROVAL, -1),
                          (ResourceType.SUPPLY, -1)])

    def test_card_26_resource_changes(self):
        self.assertEqual(card_26.resource_changes(),
                         [(ResourceType.HITLER_APPROVAL, 1)])

    # =====================================================
    # REINFORCEMENTS
    # =====================================================

    def test_card_1_reinforcements(self):
        self.assertEqual(card_1.reinforcements(), [(PZ_LEHR, 1), (SS_12, 1)])

    def test_card_19_reinforcements(self):
        reinforcements = card_19.reinforcements()
        self.assertEqual(len(reinforcements), 3)
        # for unit in reinforcements:
        #     self.assertEqual(unit.type, ReinforcementType.FLAK_88)
        #     self.assertEqual(unit.name, "Flak 88")
        #     self.assertEqual(unit.combat_value, 2)
        for unit, quantity in reinforcements:
            self.assertEqual(unit.type, ReinforcementType.FLAK_88)
            self.assertEqual(unit.name, "Flak 88")
            self.assertEqual(unit.combat_value, 2)
            self.assertEqual(quantity, 1)

    def test_card_31_reinforcements(self):
        reinforcements = card_31.reinforcements()
        self.assertEqual(len(reinforcements), 2)
        for unit, quantity in reinforcements:
            self.assertEqual(unit.type, ReinforcementType.FLAK_88)
            self.assertEqual(unit.name, "Flak 88")
            self.assertEqual(unit.combat_value, 2)
            self.assertEqual(quantity, 1)

    def test_card_34_reinforcements(self):
        reinforcements = card_34.reinforcements()
        self.assertEqual(len(reinforcements), 1)
        unit, quantity = reinforcements[0]
        self.assertEqual(unit.type, ReinforcementType.KAMPFGRUPPE)
        self.assertEqual(unit.name, "Kampfgruppe")
        self.assertEqual(unit.combat_value, 1)
        self.assertEqual(quantity, 1)


    def test_card_25_has_no_reinforcements(self):
        self.assertEqual(card_25.reinforcements(), [])


    def test_card_31_reinforcement_is_independent(self):
        # Check whether reinforcement objects are shared
        first = card_31.reinforcements()
        second = card_31.reinforcements()

        for (unit_1, _), (unit_2, _) in zip(first, second):
            self.assertIsNot(
                unit_1,
                unit_2,
                "Card 31 returns the same mutable reinforcement object",
            )

if __name__ == "__main__":
    unittest.main()

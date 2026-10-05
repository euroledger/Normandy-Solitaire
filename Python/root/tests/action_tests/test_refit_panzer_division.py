import unittest


from core.actions.refit_panzer_division_action import do_refit_panzer_division
from core.enums import ReinforcementType
from core.german_units import PZ_21, SS_12, TIGER_101
from core.global_game_state import GlobalGameState
from core.map.map_model import strategic_reserve_box
from core.map.map_utilities import do_opening_setup
from tests.core_mechanics.testing_utilities import reset_game_state_for_tests


class TestRefitPanzerDivision(unittest.TestCase):

    def setUp(self):
        do_opening_setup()
        GlobalGameState.actions_left_this_turn = 1

    def tearDown(self):
        reset_game_state_for_tests()

    def test_refit_reduced_panzer_division_succeeds(self):
        SS_12.combat_value = 1
        strategic_reserve_box.units.append(SS_12)

        self.assertIn(SS_12, strategic_reserve_box.units)
        self.assertEqual(SS_12.combat_value, 1)

        do_refit_panzer_division(unit_choice=1)

        self.assertIn(SS_12, strategic_reserve_box.units)
        self.assertEqual(SS_12.combat_value, 2)
        self.assertEqual(GlobalGameState.actions_left_this_turn, 0)

    def test_full_strength_panzer_division_cannot_be_refitted(self):
        PZ_21.combat_value = 2
        strategic_reserve_box.units.append(PZ_21)

        self.assertEqual(PZ_21.combat_value, 2)

        do_refit_panzer_division(unit_choice=1)

        self.assertEqual(PZ_21.combat_value, 2)
        self.assertEqual(GlobalGameState.actions_left_this_turn, 1)

    def test_non_panzer_unit_cannot_be_refitted(self):
        TIGER_101.combat_value = 3
        strategic_reserve_box.units.append(TIGER_101)

        self.assertNotEqual(TIGER_101.type, ReinforcementType.PZ_DIV)

        do_refit_panzer_division(unit_choice=1)

        self.assertEqual(TIGER_101.combat_value, 3)
        self.assertEqual(GlobalGameState.actions_left_this_turn, 1)

    def test_invalid_unit_choice_does_nothing(self):
        SS_12.combat_value = 1
        strategic_reserve_box.units.append(SS_12)

        do_refit_panzer_division(unit_choice=99)

        self.assertEqual(SS_12.combat_value, 1)
        self.assertEqual(GlobalGameState.actions_left_this_turn, 1)


if __name__ == "__main__":
    unittest.main()

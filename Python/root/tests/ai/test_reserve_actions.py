import unittest

from core.actions.actions_helper import use_action
from core.actions.fortified_villages_action import do_build_fortified_villages
from core.actions.move_action_point_to_reserve import do_move_action_point_to_strategic_reserve
from core.ai.headless_action_masks import get_action_phase_action_mask, get_total_actions
from core.enums import SideType
from core.global_game_state import GlobalGameState
from core.map.map_model import TerrainType


class TestReserveActions(unittest.TestCase):

    def setUp(self):
        GlobalGameState.headless = True
        GlobalGameState.actions_left_this_turn = 0
        GlobalGameState.reserve_actions = 0

    def get_fortifiable_space(self):
        _, german_eligible_spaces = get_total_actions()

        return next(
            space
            for space in german_eligible_spaces
            if space.controlling_player == SideType.GERMAN
            and space.terrain != TerrainType.FORTRESS
        )

    # ---------------------------------------------------------
    # SINGLE 1-AP ACTION
    # ---------------------------------------------------------

    def test_one_ap_action_uses_normal_ap_first(self):
        GlobalGameState.actions_left_this_turn = 2
        GlobalGameState.reserve_actions = 1

        result = use_action()

        self.assertTrue(result)
        self.assertEqual(GlobalGameState.actions_left_this_turn, 1)
        self.assertEqual(GlobalGameState.reserve_actions, 1)

    def test_one_ap_action_uses_last_normal_ap_before_reserve(self):
        GlobalGameState.actions_left_this_turn = 1
        GlobalGameState.reserve_actions = 2

        result = use_action()

        self.assertTrue(result)
        self.assertEqual(GlobalGameState.actions_left_this_turn, 0)
        self.assertEqual(GlobalGameState.reserve_actions, 2)

    def test_one_ap_action_uses_reserve_when_normal_ap_zero(self):
        GlobalGameState.actions_left_this_turn = 0
        GlobalGameState.reserve_actions = 2

        result = use_action()

        self.assertTrue(result)
        self.assertEqual(GlobalGameState.actions_left_this_turn, 0)
        self.assertEqual(GlobalGameState.reserve_actions, 1)

    def test_one_ap_action_uses_last_reserve_action(self):
        GlobalGameState.actions_left_this_turn = 0
        GlobalGameState.reserve_actions = 1

        result = use_action()

        self.assertTrue(result)
        self.assertEqual(GlobalGameState.actions_left_this_turn, 0)
        self.assertEqual(GlobalGameState.reserve_actions, 0)

    def test_one_ap_action_fails_when_no_ap_available(self):
        GlobalGameState.actions_left_this_turn = 0
        GlobalGameState.reserve_actions = 0

        result = use_action()

        self.assertFalse(result)
        self.assertEqual(GlobalGameState.actions_left_this_turn, 0)
        self.assertEqual(GlobalGameState.reserve_actions, 0)

    # ---------------------------------------------------------
    # MOVE NORMAL AP TO RESERVE
    # ---------------------------------------------------------

    def test_move_one_ap_to_reserve(self):
        GlobalGameState.actions_left_this_turn = 2
        GlobalGameState.reserve_actions = 0

        total_before = (
            GlobalGameState.actions_left_this_turn
            + GlobalGameState.reserve_actions
        )

        do_move_action_point_to_strategic_reserve()

        total_after = (
            GlobalGameState.actions_left_this_turn
            + GlobalGameState.reserve_actions
        )

        self.assertEqual(GlobalGameState.actions_left_this_turn, 1)
        self.assertEqual(GlobalGameState.reserve_actions, 1)
        self.assertEqual(total_after, total_before)

    def test_move_last_ap_to_reserve(self):
        GlobalGameState.actions_left_this_turn = 1
        GlobalGameState.reserve_actions = 0

        do_move_action_point_to_strategic_reserve()

        self.assertEqual(GlobalGameState.actions_left_this_turn, 0)
        self.assertEqual(GlobalGameState.reserve_actions, 1)

    def test_move_ap_to_reserve_when_one_already_stored(self):
        GlobalGameState.actions_left_this_turn = 2
        GlobalGameState.reserve_actions = 1

        do_move_action_point_to_strategic_reserve()

        self.assertEqual(GlobalGameState.actions_left_this_turn, 1)
        self.assertEqual(GlobalGameState.reserve_actions, 2)

    def test_cannot_move_ap_when_reserve_full(self):
        GlobalGameState.actions_left_this_turn = 2
        GlobalGameState.reserve_actions = 2

        do_move_action_point_to_strategic_reserve()

        self.assertEqual(GlobalGameState.actions_left_this_turn, 2)
        self.assertEqual(GlobalGameState.reserve_actions, 2)

    def test_cannot_move_ap_to_reserve_when_normal_ap_zero(self):
        GlobalGameState.actions_left_this_turn = 0
        GlobalGameState.reserve_actions = 1

        do_move_action_point_to_strategic_reserve()

        self.assertEqual(GlobalGameState.actions_left_this_turn, 0)
        self.assertEqual(GlobalGameState.reserve_actions, 1)

    # ---------------------------------------------------------
    # TRANSFER THEN SPEND
    # ---------------------------------------------------------

    def test_move_ap_to_reserve_then_use_it(self):
        GlobalGameState.actions_left_this_turn = 1
        GlobalGameState.reserve_actions = 0

        do_move_action_point_to_strategic_reserve()

        self.assertEqual(GlobalGameState.actions_left_this_turn, 0)
        self.assertEqual(GlobalGameState.reserve_actions, 1)

        result = use_action()

        self.assertTrue(result)
        self.assertEqual(GlobalGameState.actions_left_this_turn, 0)
        self.assertEqual(GlobalGameState.reserve_actions, 0)

    def test_normal_ap_is_spent_before_stored_reserve(self):
        GlobalGameState.actions_left_this_turn = 2
        GlobalGameState.reserve_actions = 1

        self.assertTrue(use_action())

        self.assertEqual(GlobalGameState.actions_left_this_turn, 1)
        self.assertEqual(GlobalGameState.reserve_actions, 1)

        self.assertTrue(use_action())

        self.assertEqual(GlobalGameState.actions_left_this_turn, 0)
        self.assertEqual(GlobalGameState.reserve_actions, 1)

        self.assertTrue(use_action())

        self.assertEqual(GlobalGameState.actions_left_this_turn, 0)
        self.assertEqual(GlobalGameState.reserve_actions, 0)

    # ---------------------------------------------------------
    # FORTIFIED VILLAGES - 3 AP
    # ---------------------------------------------------------

    def test_fortified_village_uses_three_normal_ap(self):
        space = self.get_fortifiable_space()
        original_modifier = space.fortified_village_modifier

        try:
            space.fortified_village_modifier = 0
            GlobalGameState.actions_left_this_turn = 3
            GlobalGameState.reserve_actions = 0

            do_build_fortified_villages(space=space)

            self.assertEqual(GlobalGameState.actions_left_this_turn, 0)
            self.assertEqual(GlobalGameState.reserve_actions, 0)
            self.assertEqual(space.fortified_village_modifier, 1)

        finally:
            space.fortified_village_modifier = original_modifier

    def test_fortified_village_uses_two_normal_one_reserve(self):
        space = self.get_fortifiable_space()
        original_modifier = space.fortified_village_modifier

        try:
            space.fortified_village_modifier = 0
            GlobalGameState.actions_left_this_turn = 2
            GlobalGameState.reserve_actions = 1

            do_build_fortified_villages(space=space)

            self.assertEqual(GlobalGameState.actions_left_this_turn, 0)
            self.assertEqual(GlobalGameState.reserve_actions, 0)
            self.assertEqual(space.fortified_village_modifier, 1)

        finally:
            space.fortified_village_modifier = original_modifier

    def test_fortified_village_uses_one_normal_two_reserve(self):
        space = self.get_fortifiable_space()
        original_modifier = space.fortified_village_modifier

        try:
            space.fortified_village_modifier = 0
            GlobalGameState.actions_left_this_turn = 1
            GlobalGameState.reserve_actions = 2

            do_build_fortified_villages(space=space)

            self.assertEqual(GlobalGameState.actions_left_this_turn, 0)
            self.assertEqual(GlobalGameState.reserve_actions, 0)
            self.assertEqual(space.fortified_village_modifier, 1)

        finally:
            space.fortified_village_modifier = original_modifier

    def test_fortified_village_does_not_use_reserve_when_normal_ap_sufficient(self):
        space = self.get_fortifiable_space()
        original_modifier = space.fortified_village_modifier

        try:
            space.fortified_village_modifier = 0
            GlobalGameState.actions_left_this_turn = 4
            GlobalGameState.reserve_actions = 2

            do_build_fortified_villages(space=space)

            self.assertEqual(GlobalGameState.actions_left_this_turn, 1)
            self.assertEqual(GlobalGameState.reserve_actions, 2)
            self.assertEqual(space.fortified_village_modifier, 1)

        finally:
            space.fortified_village_modifier = original_modifier

    def test_fortified_village_fails_with_two_normal_ap(self):
        space = self.get_fortifiable_space()
        original_modifier = space.fortified_village_modifier

        try:
            space.fortified_village_modifier = 0
            GlobalGameState.actions_left_this_turn = 2
            GlobalGameState.reserve_actions = 0

            do_build_fortified_villages(space=space)

            self.assertEqual(GlobalGameState.actions_left_this_turn, 2)
            self.assertEqual(GlobalGameState.reserve_actions, 0)
            self.assertEqual(space.fortified_village_modifier, 0)

        finally:
            space.fortified_village_modifier = original_modifier

    def test_fortified_village_fails_with_only_two_reserve(self):
        space = self.get_fortifiable_space()
        original_modifier = space.fortified_village_modifier

        try:
            space.fortified_village_modifier = 0
            GlobalGameState.actions_left_this_turn = 0
            GlobalGameState.reserve_actions = 2

            do_build_fortified_villages(space=space)

            self.assertEqual(GlobalGameState.actions_left_this_turn, 0)
            self.assertEqual(GlobalGameState.reserve_actions, 2)
            self.assertEqual(space.fortified_village_modifier, 0)

        finally:
            space.fortified_village_modifier = original_modifier

    # ---------------------------------------------------------
    # AP TOTAL INVARIANTS
    # ---------------------------------------------------------

    def test_one_ap_action_reduces_total_ap_by_exactly_one(self):
        combinations = [
            (3, 0),
            (2, 1),
            (1, 2),
            (0, 2),
            (0, 1),
        ]

        for normal_ap, reserve_ap in combinations:
            with self.subTest(
                normal_ap=normal_ap,
                reserve_ap=reserve_ap
            ):
                GlobalGameState.actions_left_this_turn = normal_ap
                GlobalGameState.reserve_actions = reserve_ap

                total_before = normal_ap + reserve_ap

                result = use_action()

                total_after = (
                    GlobalGameState.actions_left_this_turn
                    + GlobalGameState.reserve_actions
                )

                self.assertTrue(result)
                self.assertEqual(total_after, total_before - 1)

    def test_transfer_to_reserve_does_not_change_total_ap(self):
        combinations = [
            (3, 0),
            (2, 0),
            (1, 0),
            (3, 1),
            (2, 1),
            (1, 1),
        ]

        for normal_ap, reserve_ap in combinations:
            with self.subTest(
                normal_ap=normal_ap,
                reserve_ap=reserve_ap
            ):
                GlobalGameState.actions_left_this_turn = normal_ap
                GlobalGameState.reserve_actions = reserve_ap

                total_before = normal_ap + reserve_ap

                do_move_action_point_to_strategic_reserve()

                total_after = (
                    GlobalGameState.actions_left_this_turn
                    + GlobalGameState.reserve_actions
                )

                self.assertEqual(total_after, total_before)

    def test_pass_is_available_with_zero_normal_ap_and_one_reserve(self):
        GlobalGameState.actions_left_this_turn = 0
        GlobalGameState.reserve_actions = 1
        mask = get_action_phase_action_mask()

        self.assertEqual(mask[0], 1.0)

    def test_pass_is_available_with_zero_normal_ap_and_two_reserve(self):
        GlobalGameState.actions_left_this_turn = 0
        GlobalGameState.reserve_actions = 2
        mask = get_action_phase_action_mask()

        self.assertEqual(mask[0], 1.0)
        
if __name__ == "__main__":
    unittest.main()

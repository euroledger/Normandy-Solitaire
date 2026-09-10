# tests/test_action_mask.py
import unittest
import numpy as np
from core.global_game_state import GlobalGameState
from core.ai.headless_action_masks import get_mini_action_mask
from core.allied_armies import US_FIRST_ARMY


class TestActionMask(unittest.TestCase):

    def setUp(self):
        GlobalGameState.actions_left_this_turn = 1
        GlobalGameState.reserve_actions = 0
        GlobalGameState.counter_attacked_armies.clear() # Clear out previous entries
        GlobalGameState.headless = True
        # Force NumPy arrays to format all float values strictly with 1 decimal place
        np.set_printoptions(formatter={'float': '{:0.1f}'.format})

    def test_action_mask_baseline_pass_only(self):
        # TEST A: Verify that if you have 0 AP left, counter-attacks lock out
        GlobalGameState.actions_left_this_turn = 0
        GlobalGameState.reserve_actions = 0

        mask = get_mini_action_mask()

        print(f"\n[DEBUG] Test Baseline Mask Output: {mask}")

        # Expected output array format: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.assertEqual(mask[0], 1.0)  # Pass (ID 0) must be unlocked
        self.assertEqual(np.sum(mask), 1.0)  # All 6 remaining buttons must be locked

    def test_action_mask_blocks_already_attacked_army(self):
        # TEST B: Verify that if an army was hit, its corresponding button locks out
        # Simulate that US 1st Army was targeted earlier this phase
        GlobalGameState.counter_attacked_armies.add(US_FIRST_ARMY.name)

        mask = get_mini_action_mask()

        print(f"\n[DEBUG] Test Attacked Army Mask Output: {mask}")

        # Action ID 1 corresponds to US 1st Army (index 1 in our schema)
        self.assertEqual(mask[1], 0.0)  # Button 1 must be locked out!
        self.assertEqual(mask[0], 1.0)  # Pass must remain wide open

    def test_action_mask_output_structure(self):
        # TEST C: Verify the physical properties match PyTorch dimensional requirements
        mask = get_mini_action_mask()

        self.assertIsInstance(mask, np.ndarray)
        self.assertEqual(mask.shape, (7,))
        self.assertEqual(mask.dtype, np.float32)

from gymnasium.vector import AsyncVectorEnv, AutoresetMode
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch

from core.ai.dqn.dqn_agent import DQNAgent
from core.ai.dqn.normandy_env import NormandyEnv
from core.ai.dqn.dqn_action_adapter import get_action_phase_decision
from core.ai.dqn.dqn_game_state_encoder import OBSERVATION_SIZE
from core.ai.headless_action_mask_helper import get_move_unit_action_id
from core.ai.headless_action_masks import HeadlessDecisionState, get_total_actions

from core.ai.normandy_full_game_ai_dqn import advance_to_next_decision
from core.global_game_state import GlobalGameState
import contextlib
import os
from core.ai.dqn.dqn_game_state_encoder import (
        PANZERS_START, PANZER_FEATURES_PER_UNIT,
        GERMAN_LOCATION_INDEX, ACTIONS_LEFT_INDEX,
)
from core.actions.actions_helper import (
        get_adjacent_german_controlled_spaces,
        can_add_unit_to_space,
)
from core.german_units import PZ_21, panzer_divisions_list
from core.map.map_spaces_brit_2 import bayeux
from core.ai.dqn.dqn_action_adapter import (
    get_hitler_intervention_decision,
    execute_hitler_intervention_decision,
)
from core.ai.headless_action_masks import (
    get_hitler_intervention_targets,
)
import re


def make_silent_normandy_env():
    import os
    import sys
    import multiprocessing

    from core.ai.dqn.normandy_env import NormandyEnv

    # Suppress output only inside worker processes
    if multiprocessing.current_process().name != "MainProcess":
        sys.stdout = open(os.devnull, "w", encoding="utf-8")
        sys.stderr = open(os.devnull, "w", encoding="utf-8")

    return NormandyEnv()

class TestNormandyEnv(unittest.TestCase):
    @staticmethod
    def remove_ansi_codes(text):
        return re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text)
    
    def select_legal_destination(self):
        # Select a legal destination
        destinations = [
                space for space in get_adjacent_german_controlled_spaces(bayeux)
                if can_add_unit_to_space(space, PZ_21)
        ]
        self.assertTrue(destinations)
        return destinations[0]
        
    def create_agent_for_action(self, action_id):
        # Create a network that prefers the specified action
        network = torch.nn.Linear(
            self.env.observation_space.shape[0],
            self.env.action_space.n,
        )

        with torch.no_grad():
            network.weight.zero_()
            network.bias.zero_()
            network.bias[action_id] = 10.0

        agent = DQNAgent(network, torch.device("cpu"))
        agent.epsilon = 0.0

        return agent
    def setUp(self):
        self.env = NormandyEnv()
        self.LOG_GAME = True


    def test_observation_space_shape(self):
        self.assertEqual(
            self.env.observation_space.shape,
            (OBSERVATION_SIZE,),
        )

    def test_observation_space_dtype(self):
        self.assertEqual(
            self.env.observation_space.dtype,
            np.float32,
        )

    def test_action_space_size(self):
        total_actions, _ = get_total_actions()

        self.assertEqual(
            self.env.action_space.n,
            total_actions,
        )

    def test_pad_action_mask_same_size(self):
        mask = np.zeros(self.env.action_space.n, dtype=np.float32)
        mask[0] = 1.0
        mask[10] = 1.0

        padded_mask = self.env.pad_action_mask(mask)

        self.assertEqual(len(padded_mask), self.env.action_space.n)
        np.testing.assert_array_equal(padded_mask, mask)

    def test_pad_action_mask_shorter(self):
        mask = np.array(
            [1.0, 0.0, 1.0],
            dtype=np.float32,
        )

        padded_mask = self.env.pad_action_mask(mask)

        self.assertEqual(len(padded_mask), self.env.action_space.n)
        np.testing.assert_array_equal(
            padded_mask[:3],
            mask,
        )
        self.assertTrue(
            np.all(padded_mask[3:] == 0.0)
        )

    def test_pad_action_mask_preserves_dtype(self):
        mask = np.array(
            [1.0, 0.0, 1.0],
            dtype=np.float32,
        )

        padded_mask = self.env.pad_action_mask(mask)

        self.assertEqual(padded_mask.dtype, np.float32)

    def test_pad_action_mask_rejects_oversized_mask(self):
        mask = np.zeros(
            self.env.action_space.n + 1,
            dtype=np.float32,
        )

        with self.assertRaises(ValueError):
            self.env.pad_action_mask(mask)
            

    def test_reset(self):
        HeadlessDecisionState.hitler_intervention_stage = "TARGET"
        HeadlessDecisionState.hitler_intervention_target = object()
        HeadlessDecisionState.hitler_intervention_redeployments = 2

        observation, info = self.env.reset()

        self.assertEqual(observation.shape, (OBSERVATION_SIZE,))
        self.assertEqual(observation.dtype, np.float32)
        self.assertTrue(self.env.observation_space.contains(observation))
        # Reset returns the initial legal-action mask
        self.assertIn("action_mask", info)
        mask = info["action_mask"]
        self.assertEqual(
            mask.shape,
            (self.env.action_space.n,),
        )
        self.assertEqual(mask.dtype, np.float32)
        self.assertGreater(mask.sum(), 0)




        self.assertIsNotNone(self.env.draw_deck)
        self.assertIsNotNone(self.env.mid_deck)
        self.assertIsNotNone(self.env.late_deck)
        self.assertGreater(len(self.env.draw_deck), 0)
        self.assertGreater(len(self.env.mid_deck), 0)
        self.assertGreater(len(self.env.late_deck), 0)

        self.assertIsNone(HeadlessDecisionState.hitler_intervention_stage)
        self.assertIsNone(HeadlessDecisionState.hitler_intervention_target)
        self.assertEqual(HeadlessDecisionState.hitler_intervention_redeployments, 0)        
    
    
    def test_action(self):
        self.env.reset()

        # Advance the actual game to an Action Phase decision
        decision = advance_to_next_decision(
            self.env.draw_deck,
            self.env.mid_deck,
            self.env.late_deck,
        )

        self.assertEqual(decision["decision_type"], "action_phase")

        observation = decision["observation"]
        mask = self.env.pad_action_mask(decision["action_mask"])

        # Use a simple network; exploration does not require trained Q-values
        network = torch.nn.Linear(
            self.env.observation_space.shape[0],
            self.env.action_space.n,
        )

        agent = DQNAgent(network, torch.device("cpu"))
        agent.epsilon = 1.0


        # Select an action from the actual game mask
        legal_actions = np.flatnonzero(mask)
        self.assertGreater(len(legal_actions), 0)

        action_id = agent.select_action(observation, mask)

        self.assertIn(action_id, legal_actions)

        result = self.env.execute_action(action_id)
        self.assertTrue(result)
        

    def test_opening_panzer_movement(self):
        # Establish opening state
        self.env.reset()
        GlobalGameState.actions_left_this_turn = 1
        self.assertIn(PZ_21, bayeux.units)

        destination = self.select_legal_destination()

        # Obtain the actual observation and action mask
        observation_before, mask = get_action_phase_decision()
        mask = self.env.pad_action_mask(mask)

        # Identify the action using our helper
        action_id = get_move_unit_action_id(PZ_21, destination)
        self.assertEqual(mask[action_id], 1.0)

        # Create an agent that selects the intended action
        agent = self.create_agent_for_action(action_id)

        selected_action = agent.select_action(observation_before, mask)
        self.assertEqual(selected_action, action_id)

        # Identify the relevant observation features
        panzer_start = (
            PANZERS_START
            + panzer_divisions_list.index(PZ_21) * PANZER_FEATURES_PER_UNIT
        )
        origin_feature = panzer_start + GERMAN_LOCATION_INDEX[id(bayeux)]
        destination_feature = (
            panzer_start + GERMAN_LOCATION_INDEX[id(destination)]
        )

        # Verify the initial encoded state
        self.assertEqual(observation_before[origin_feature], 1.0)
        self.assertEqual(observation_before[destination_feature], 0.0)

        # Execute the selected action
        self.assertTrue(self.env.execute_action(selected_action))

        # Verify the actual game state
        self.assertNotIn(PZ_21, bayeux.units)
        self.assertIn(PZ_21, destination.units)
        self.assertEqual(GlobalGameState.actions_left_this_turn, 0)

        # Obtain and verify the updated observation
        observation_after, _ = get_action_phase_decision()

        self.assertEqual(observation_after[origin_feature], 0.0)
        self.assertEqual(observation_after[destination_feature], 1.0)
        self.assertEqual(observation_after[ACTIONS_LEFT_INDEX], 0.0)
        self.assertFalse(np.array_equal(observation_before, observation_after))


    def test_illegal_panzer_movement(self):
        # Establish opening state
        self.env.reset()
        GlobalGameState.actions_left_this_turn = 1
        self.assertIn(PZ_21, bayeux.units)

        destination = self.select_legal_destination()
        action_id = get_move_unit_action_id(PZ_21, destination)

        # Confirm the movement is initially legal
        observation_before, mask = get_action_phase_decision()
        self.assertEqual(mask[action_id], 1.0)

        # Remove all available action points
        GlobalGameState.actions_left_this_turn = 0

        # Verify the movement is now illegal
        observation_before, mask = get_action_phase_decision()
        self.assertEqual(mask[action_id], 0.0)

        # Attempt to execute the illegal movement
        result = self.env.execute_action(action_id)
        self.assertFalse(result)

        # Verify that the game state has not changed
        self.assertIn(PZ_21, bayeux.units)
        self.assertNotIn(PZ_21, destination.units)
        self.assertEqual(GlobalGameState.actions_left_this_turn, 0)

        # Verify that the encoded observation has not changed
        observation_after, _ = get_action_phase_decision()
        np.testing.assert_array_equal(
            observation_before,
            observation_after,
        )


    def test_step_panzer_movement(self):
        # Establish opening state
        self.env.reset()
        GlobalGameState.actions_left_this_turn = 2

        self.assertIn(PZ_21, bayeux.units)
        destination = self.select_legal_destination()

        # Identify a legal movement
        action_id = get_move_unit_action_id(PZ_21, destination)
        _, mask = get_action_phase_decision()
        self.assertEqual(mask[action_id], 1.0)

        # Execute one Gymnasium step
        observation, reward, terminated, truncated, info = (
            self.env.step(action_id)
        )

        # Verify the game state
        self.assertNotIn(PZ_21, bayeux.units)
        self.assertIn(PZ_21, destination.units)
        self.assertEqual(GlobalGameState.actions_left_this_turn, 1)

        # Verify the Gymnasium result
        self.assertEqual(observation.shape, (OBSERVATION_SIZE,))
        self.assertEqual(observation.dtype, np.float32)
        self.assertEqual(reward, 0.0)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(
            info["action_mask"].shape,
            (self.env.action_space.n,),
        )

        # Verify the returned observation is current
        expected_observation, _ = get_action_phase_decision()
        np.testing.assert_array_equal(
            observation,
            expected_observation,
        )
        

    def test_step_advances_to_next_turn(self):
        # Reach the first Action Phase
        self.env.reset()
        decision = advance_to_next_decision(
            self.env.draw_deck,
            self.env.mid_deck,
            self.env.late_deck,
        )
        self.assertEqual(decision["decision_type"], "action_phase")

        first_card = GlobalGameState.current_card
        first_cards_drawn = len(GlobalGameState.drawn_cards)

        # Leave one action and no reserve actions
        GlobalGameState.actions_left_this_turn = 1
        GlobalGameState.reserve_actions = 0

        # Select a legal Panzer movement as the final action
        self.assertIn(PZ_21, bayeux.units)
        destination = self.select_legal_destination()

        action_id = get_move_unit_action_id(PZ_21, destination)

        # Obtain the mask for the modified state
        _, mask = get_action_phase_decision()
        mask = self.env.pad_action_mask(mask)
        self.assertEqual(mask[action_id], 1.0)

        # Execute the action and advance automatically
        observation, reward, terminated, truncated, info = (
            self.env.step(action_id)
        )

        # Verify that the next turn has started
        self.assertIsNot(GlobalGameState.current_card, first_card)
        self.assertEqual(
            len(GlobalGameState.drawn_cards),
            first_cards_drawn + 1,
        )
        self.assertEqual(GlobalGameState.current_step, 6)

        # Verify the returned observation and mask
        expected_observation, expected_mask = get_action_phase_decision()

        np.testing.assert_array_equal(
            observation,
            expected_observation,
        )
        np.testing.assert_array_equal(
            info["action_mask"],
            self.env.pad_action_mask(expected_mask),
        )

        self.assertEqual(reward, 1.0)
        self.assertFalse(terminated)
        self.assertFalse(truncated)


    def test_step_game_termination(self):
        # Establish opening state
        self.env.reset()
        GlobalGameState.actions_left_this_turn = 1
        GlobalGameState.reserve_actions = 0

        destination = self.select_legal_destination()
        action_id = get_move_unit_action_id(PZ_21, destination)

        _, mask = get_action_phase_decision()
        self.assertEqual(mask[action_id], 1.0)

        # Simulate a game-ending result after the final action
        with patch(
            "core.ai.dqn.normandy_env.advance_to_next_decision",
            return_value={"result": "lost"},
        ):
            observation, reward, terminated, truncated, info = (
                self.env.step(action_id)
            )

        # Verify the action was executed
        self.assertNotIn(PZ_21, bayeux.units)
        self.assertIn(PZ_21, destination.units)

        # Verify the terminal result
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(reward, 0.0)
        self.assertEqual(info["result"], "lost")
        self.assertEqual(observation.shape, (OBSERVATION_SIZE,))
        self.assertTrue(np.all(info["action_mask"] == 0.0))


    def test_hitler_intervention_no_targets(self):
        # Establish opening game state
        self.env.reset()

        # Use Card 35
        from cards.card_35 import card
        GlobalGameState.current_card = card

        # Verify no eligible targets
        targets = get_hitler_intervention_targets(card)
        self.assertEqual(len(targets), 0)

        # Reset intervention state
        HeadlessDecisionState.reset_hitler_intervention()

        # Only action 0 should be legal
        _, mask = get_hitler_intervention_decision(card)

        self.assertEqual(mask[0], 1.0)
        self.assertEqual(mask.sum(), 1.0)

        # Execute action 0 to finish the intervention
        result = execute_hitler_intervention_decision(
            0,
            card,
            GlobalGameState.current_weather,
        )

        # Intervention is complete
        self.assertFalse(result)
        self.assertIsNone(
            HeadlessDecisionState.hitler_intervention_stage
        )
        self.assertIsNone(
            HeadlessDecisionState.hitler_intervention_target
        )
        self.assertEqual(
            HeadlessDecisionState.hitler_intervention_redeployments,
            0,
        )


    def test_hitler_intervention_decline_cancel(self):
        # Establish opening game state
        self.env.reset()

        from cards.card_35 import card
        GlobalGameState.current_card = card
        HeadlessDecisionState.reset_hitler_intervention()

        # Supply an eligible target through the existing target function
        with patch(
            "core.ai.headless_action_masks.get_hitler_intervention_targets",
            return_value=[card.hitler_intervention_target_armies[0]],
        ):
            # Initial cancellation decision
            _, mask = get_hitler_intervention_decision(card)

            self.assertEqual(
                HeadlessDecisionState.hitler_intervention_stage,
                "CANCEL",
            )
            self.assertEqual(mask[0], 1.0)
            self.assertEqual(mask[1], 1.0)

            # Action 0 declines cancellation
            result = execute_hitler_intervention_decision(
                0,
                card,
                GlobalGameState.current_weather,
            )

            self.assertTrue(result)
            self.assertEqual(
                HeadlessDecisionState.hitler_intervention_stage,
                "TARGET",
            )
            

    def test_hitler_intervention_single_target(self):
        # Establish opening game state
        self.env.reset()

        from cards.card_35 import card
        GlobalGameState.current_card = card
        HeadlessDecisionState.reset_hitler_intervention()

        # Supply exactly one eligible target
        target = card.hitler_intervention_target_armies[0]

        with patch(
            "core.ai.headless_action_masks.get_hitler_intervention_targets",
            return_value=[target],
        ):
            # Initial cancellation decision
            _, mask = get_hitler_intervention_decision(card)
            self.assertEqual(mask[0], 1.0)

            # Decline cancellation
            result = execute_hitler_intervention_decision(
                0,
                card,
                GlobalGameState.current_weather,
            )

            self.assertTrue(result)
            self.assertEqual(
                HeadlessDecisionState.hitler_intervention_stage,
                "TARGET",
            )

            # Request the next decision
            _, next_mask = get_hitler_intervention_decision(card)

            # Single target is selected automatically
            self.assertEqual(
                HeadlessDecisionState.hitler_intervention_stage,
                "REDEPLOY",
            )
            self.assertIs(
                HeadlessDecisionState.hitler_intervention_target,
                target,
            )

            # Redeployment stage must expose a legal action
            self.assertGreater(next_mask.sum(), 0)


    def test_hitler_intervention_multiple_targets(self):
        # Establish opening game state
        self.env.reset()

        from cards.card_35 import card
        from core.allied_armies import US_FIRST_ARMY, BRITISH_SECOND_ARMY
        from core.ai.headless_action_masks import armies_list

        GlobalGameState.current_card = card
        HeadlessDecisionState.reset_hitler_intervention()

        targets = [US_FIRST_ARMY, BRITISH_SECOND_ARMY]

        with patch(
            "core.ai.headless_action_masks.get_hitler_intervention_targets",
            return_value=targets,
        ):
            # Start cancellation stage
            _, _ = get_hitler_intervention_decision(card)

            self.assertEqual(
                HeadlessDecisionState.hitler_intervention_stage,
                "CANCEL",
            )

            # Decline cancellation
            result = execute_hitler_intervention_decision(
                0,
                card,
                GlobalGameState.current_weather,
            )

            self.assertTrue(result)

            # Obtain target-selection decision
            _, target_mask = get_hitler_intervention_decision(card)

            self.assertEqual(
                HeadlessDecisionState.hitler_intervention_stage,
                "TARGET",
            )

            # Target action IDs use the fixed armies_list order
            first_action = 2 + armies_list.index(US_FIRST_ARMY)
            british_action = 2 + armies_list.index(BRITISH_SECOND_ARMY)

            self.assertEqual(target_mask[first_action], 1.0)
            self.assertEqual(target_mask[british_action], 1.0)
            self.assertEqual(target_mask.sum(), 2.0)

            # Select British Second Army
            result = execute_hitler_intervention_decision(
                british_action,
                card,
                GlobalGameState.current_weather,
            )

            self.assertTrue(result)

            # Selection advances to redeployment
            self.assertEqual(
                HeadlessDecisionState.hitler_intervention_stage,
                "REDEPLOY",
            )
            self.assertIs(
                HeadlessDecisionState.hitler_intervention_target,
                BRITISH_SECOND_ARMY,
            )
            

    def test_hitler_intervention_completion(self):
        # Establish opening game state
        self.env.reset()

        from cards.card_35 import card
        from core.allied_armies import BRITISH_SECOND_ARMY

        GlobalGameState.current_card = card
        HeadlessDecisionState.reset_hitler_intervention()

        # Establish redeployment stage
        HeadlessDecisionState.hitler_intervention_stage = "REDEPLOY"
        HeadlessDecisionState.hitler_intervention_target = (
            BRITISH_SECOND_ARMY
        )


        # Both permitted redeployments have been completed
        HeadlessDecisionState.hitler_intervention_redeployments = (
            card.hitler_intervention_panzer_count
        )
        
        with patch(
            "core.ai.headless_action_masks.get_hitler_intervention_targets",
            return_value=[BRITISH_SECOND_ARMY],
        ), patch(
            "core.ai.headless_action_masks.get_german_space_facing_front_line",
            return_value=SimpleNamespace(units=[]),
        ), patch(
            "core.ai.headless_action_masks.do_hitler_intervention_attack",
        ) as mock_attack:

            # Finish redeployment and execute intervention attack
            result = execute_hitler_intervention_decision(
                0,
                card,
                GlobalGameState.current_weather,
            )

        # False indicates that intervention is complete
        self.assertFalse(result)

        # Attack was executed exactly once
        mock_attack.assert_called_once()

        # Intervention state has been reset
        self.assertIsNone(
            HeadlessDecisionState.hitler_intervention_stage
        )
        self.assertIsNone(
            HeadlessDecisionState.hitler_intervention_target
        )
        self.assertEqual(
            HeadlessDecisionState.hitler_intervention_redeployments,
            0,
        )


    def test_step_hitler_intervention_cancel(self):
        # Establish opening game state
        self.env.reset()

        from cards.card_35 import card
        from core.allied_armies import US_FIRST_ARMY, BRITISH_SECOND_ARMY
        from core.ai.headless_action_masks import armies_list

        GlobalGameState.current_card = card
        self.env.current_decision_type = "hitler_intervention"
        HeadlessDecisionState.reset_hitler_intervention()

        targets = [US_FIRST_ARMY, BRITISH_SECOND_ARMY]

        with patch(
            "core.ai.headless_action_masks.get_hitler_intervention_targets",
            return_value=targets,
        ):
            # Initialise cancellation stage
            _, mask = get_hitler_intervention_decision(card)
            self.assertEqual(
                HeadlessDecisionState.hitler_intervention_stage,
                "CANCEL",
            )
            self.assertEqual(mask[0], 1.0)

            # Decline cancellation through the environment
            observation, reward, terminated, truncated, info = (
                self.env.step(0)
            )

            # Intervention continues at target selection
            self.assertEqual(
                self.env.current_decision_type,
                "hitler_intervention",
            )
            self.assertEqual(
                HeadlessDecisionState.hitler_intervention_stage,
                "TARGET",
            )

            # Verify returned target-selection mask
            target_mask = info["action_mask"]

            first_action = 2 + armies_list.index(US_FIRST_ARMY)
            british_action = 2 + armies_list.index(BRITISH_SECOND_ARMY)

            self.assertEqual(target_mask[first_action], 1.0)
            self.assertEqual(target_mask[british_action], 1.0)
            self.assertEqual(target_mask.sum(), 2.0)

            # Verify Gymnasium step result
            self.assertEqual(
                observation.shape,
                self.env.observation_space.shape,
            )
            self.assertEqual(
                target_mask.shape,
                (self.env.action_space.n,),
            )
            self.assertEqual(reward, 0.0)
            self.assertFalse(terminated)
            self.assertFalse(truncated)


    def test_step_hitler_intervention_target_selection(self):
        # Establish opening game state
        self.env.reset()

        from cards.card_35 import card
        from core.allied_armies import US_FIRST_ARMY, BRITISH_SECOND_ARMY
        from core.ai.headless_action_masks import armies_list

        GlobalGameState.current_card = card
        self.env.current_decision_type = "hitler_intervention"
        HeadlessDecisionState.reset_hitler_intervention()

        targets = [US_FIRST_ARMY, BRITISH_SECOND_ARMY]

        with patch(
            "core.ai.headless_action_masks.get_hitler_intervention_targets",
            return_value=targets,
        ):
            # Initialise cancellation stage
            get_hitler_intervention_decision(card)

            # Decline cancellation
            self.env.step(0)

            self.assertEqual(
                HeadlessDecisionState.hitler_intervention_stage,
                "TARGET",
            )

            # Select British Second Army
            action_id = 2 + armies_list.index(BRITISH_SECOND_ARMY)

            # Mock redeployment space to isolate target selection
            with patch(
                "core.ai.headless_action_masks.get_german_space_facing_front_line",
                return_value=SimpleNamespace(units=[]),
            ):
                observation, reward, terminated, truncated, info = (
                    self.env.step(action_id)
                )

            # Verify target selection
            self.assertEqual(
                HeadlessDecisionState.hitler_intervention_stage,
                "REDEPLOY",
            )
            self.assertIs(
                HeadlessDecisionState.hitler_intervention_target,
                BRITISH_SECOND_ARMY,
            )
            self.assertEqual(
                self.env.current_decision_type,
                "hitler_intervention",
            )

            # Verify Gymnasium result
            self.assertEqual(
                observation.shape,
                self.env.observation_space.shape,
            )
            self.assertEqual(
                info["action_mask"].shape,
                (self.env.action_space.n,),
            )
            self.assertGreater(info["action_mask"].sum(), 0)
            self.assertEqual(reward, 0.0)
            self.assertFalse(terminated)
            self.assertFalse(truncated)


    def test_step_hitler_intervention_completion(self):
        # Establish opening game state
        self.env.reset()

        from cards.card_35 import card
        from core.allied_armies import BRITISH_SECOND_ARMY

        GlobalGameState.current_card = card
        self.env.current_decision_type = "hitler_intervention"

        # Establish completed redeployment stage
        HeadlessDecisionState.reset_hitler_intervention()
        HeadlessDecisionState.hitler_intervention_stage = "REDEPLOY"
        HeadlessDecisionState.hitler_intervention_target = (
            BRITISH_SECOND_ARMY
        )
        HeadlessDecisionState.hitler_intervention_redeployments = (
            card.hitler_intervention_panzer_count
        )

        with patch(
            "core.ai.headless_action_masks.get_hitler_intervention_targets",
            return_value=[BRITISH_SECOND_ARMY],
        ), patch(
            "core.ai.headless_action_masks.get_german_space_facing_front_line",
            return_value=SimpleNamespace(units=[]),
        ), patch(
            "core.ai.headless_action_masks.do_hitler_intervention_attack",
        ) as mock_attack:

            # Finish intervention through the environment
            observation, reward, terminated, truncated, info = (
                self.env.step(0)
            )

        # Intervention attack executes once
        mock_attack.assert_called_once()

        # Intervention state is cleared
        self.assertIsNone(
            HeadlessDecisionState.hitler_intervention_stage
        )
        self.assertIsNone(
            HeadlessDecisionState.hitler_intervention_target
        )
        self.assertEqual(
            HeadlessDecisionState.hitler_intervention_redeployments,
            0,
        )

        # Environment resumes normal Action Phase
        self.assertEqual(
            self.env.current_decision_type,
            "action_phase",
        )

        # Verify Gymnasium result
        self.assertEqual(
            observation.shape,
            self.env.observation_space.shape,
        )
        self.assertEqual(
            info["action_mask"].shape,
            (self.env.action_space.n,),
        )
        self.assertEqual(reward, 0.0)
        self.assertFalse(terminated)
        self.assertFalse(truncated)


    def test_step_hitler_intervention_no_targets(self):
        # Establish opening game state
        self.env.reset()

        from cards.card_35 import card

        GlobalGameState.current_card = card
        self.env.current_decision_type = "hitler_intervention"
        HeadlessDecisionState.reset_hitler_intervention()

        # Confirm no eligible targets
        targets = get_hitler_intervention_targets(card)
        self.assertEqual(len(targets), 0)

        with patch(
            "core.ai.headless_action_masks.do_hitler_intervention_attack",
        ) as mock_attack:

            # Action 0 closes the intervention
            observation, reward, terminated, truncated, info = (
                self.env.step(0)
            )

        # No attack should occur
        mock_attack.assert_not_called()

        # Intervention state is reset
        self.assertIsNone(
            HeadlessDecisionState.hitler_intervention_stage
        )
        self.assertIsNone(
            HeadlessDecisionState.hitler_intervention_target
        )

        # The game records that intervention had no effect
        self.assertTrue(
            GlobalGameState.hitler_intervention_no_effect
        )

        # Environment resumes normal Action Phase
        self.assertEqual(
            self.env.current_decision_type,
            "action_phase",
        )

        # Verify Gymnasium result
        self.assertEqual(
            observation.shape,
            self.env.observation_space.shape,
        )
        self.assertEqual(
            info["action_mask"].shape,
            (self.env.action_space.n,),
        )
        self.assertEqual(reward, 0.0)
        self.assertFalse(terminated)
        self.assertFalse(truncated)

    def test_advance_to_hitler_intervention(self):
        # Establish opening game state
        self.env.reset()

        from cards.card_35 import card

        GlobalGameState.current_card = card
        GlobalGameState.current_step = 5

        with patch(
            "core.ai.normandy_full_game_ai_dqn.common_pre_action_phase",
            return_value=False,
        ):
            decision = advance_to_next_decision(
                self.env.draw_deck,
                self.env.mid_deck,
                self.env.late_deck,
            )

        # Step 5 must return an intervention decision
        self.assertEqual(
            decision["decision_type"],
            "hitler_intervention",
        )

        # Verify observation and action mask
        self.assertEqual(
            decision["observation"].shape,
            self.env.observation_space.shape,
        )
        self.assertEqual(
            self.env.pad_action_mask(
                decision["action_mask"]
            ).shape,
            (self.env.action_space.n,),
        )

        # Game must be positioned at Step 6
        self.assertEqual(
            GlobalGameState.current_step,
            6,
        )
        

    def test_step_action_phase_to_hitler_intervention(self):
        # Establish opening game state
        self.env.reset()

        from cards.card_35 import card

        GlobalGameState.current_card = card
        self.env.current_decision_type = "action_phase"


        # Simulate the final action of the turn
        GlobalGameState.actions_left_this_turn = 1
        GlobalGameState.reserve_actions = 0

        # Select an action legal in this state
        _, mask = get_action_phase_decision()
        legal_actions = np.flatnonzero(mask)

        self.assertGreater(len(legal_actions), 0)
        action_id = int(legal_actions[0])

        # Prepare the next decision
        intervention_observation, intervention_mask = (
            get_hitler_intervention_decision(card)
        )

        decision = {
            "decision_type": "hitler_intervention",
            "observation": intervention_observation,
            "action_mask": intervention_mask,
        }

        with patch.object(
            self.env,
            "execute_action",
            return_value=True,
        ) as mock_execute, patch(
            "core.ai.dqn.normandy_env.advance_to_next_decision",
            return_value=decision,
        ) as mock_advance:

            # Simulate action consumption
            def execute_final_action(action):
                GlobalGameState.actions_left_this_turn = 0
                return True

            mock_execute.side_effect = execute_final_action

            observation, reward, terminated, truncated, info = (
                self.env.step(action_id)
            )

        # Final action executed
        mock_execute.assert_called_once_with(action_id)

        # Game progression invoked
        mock_advance.assert_called_once()

        # Environment switches to Hitler Intervention
        self.assertEqual(
            self.env.current_decision_type,
            "hitler_intervention",
        )

        # Correct decision data returned
        np.testing.assert_array_equal(
            observation,
            intervention_observation,
        )
        np.testing.assert_array_equal(
            info["action_mask"],
            self.env.pad_action_mask(intervention_mask),
        )

        # Game continues        
        # The final action completes the turn and earns +1
        self.assertEqual(reward, 0.0)
        self.assertFalse(terminated)
        self.assertFalse(truncated)


    def test_step_real_transition_to_hitler_intervention(self):
        # Establish opening game state
        self.env.reset()

        from cards.card_35 import card

        GlobalGameState.current_card = card
        GlobalGameState.current_step = 6
        self.env.current_decision_type = "action_phase"

        # Set up the final action of the turn
        GlobalGameState.actions_left_this_turn = 1
        GlobalGameState.reserve_actions = 0

        # Find a legal action for the real executor
        _, mask = get_action_phase_decision()
        legal_actions = np.flatnonzero(mask)

        self.assertGreater(len(legal_actions), 0)
        action_id = int(legal_actions[0])

        def trigger_hitler_intervention(*args, **kwargs):
            # Ensure the next turn uses Card 35
            GlobalGameState.current_card = card
            return False

        with patch(
            "core.ai.normandy_full_game_ai_dqn.common_pre_action_phase",
            side_effect=trigger_hitler_intervention,
        ):
            # Execute real action and advance real phases
            observation, reward, terminated, truncated, info = (
                self.env.step(action_id)
            )

        # Environment reaches Hitler Intervention
        self.assertEqual(
            self.env.current_decision_type,
            "hitler_intervention",
        )

        # Verify Gymnasium result
        self.assertEqual(
            observation.shape,
            self.env.observation_space.shape,
        )
        self.assertEqual(
            info["action_mask"].shape,
            (self.env.action_space.n,),
        )
        self.assertGreater(info["action_mask"].sum(), 0)

        self.assertEqual(reward, 1.0)
        self.assertFalse(terminated)
        self.assertFalse(truncated)


    def test_step_hitler_intervention_completion_advances_game(self):
        # Establish opening game state
        self.env.reset()

        from cards.card_35 import card

        GlobalGameState.current_card = card
        self.env.current_decision_type = "hitler_intervention"

        # No Action Phase actions remain
        GlobalGameState.actions_left_this_turn = 0
        GlobalGameState.reserve_actions = 0

        HeadlessDecisionState.reset_hitler_intervention()

        # Card 35 has no eligible targets in the opening state
        self.assertEqual(
            get_hitler_intervention_targets(card),
            [],
        )

        # Complete intervention without an attack
        observation, reward, _, truncated, info = (
            self.env.step(0)
        )

        # Intervention state is cleared
        self.assertIsNone(
            HeadlessDecisionState.hitler_intervention_stage
        )


        # Game has advanced to the next turn's Action Phase
        self.assertEqual(
            self.env.current_decision_type,
            "action_phase",
        )

        self.assertEqual(
            GlobalGameState.current_step,
            6,
        )

        # The new Action Phase must have available actions
        self.assertGreater(
            GlobalGameState.actions_left_this_turn
            + GlobalGameState.reserve_actions,
            0,
        )

        # Verify Gymnasium result
        self.assertEqual(
            observation.shape,
            self.env.observation_space.shape,
        )
        self.assertEqual(
            info["action_mask"].shape,
            (self.env.action_space.n,),
        )
        self.assertEqual(reward, 0.0)
        self.assertFalse(truncated)


    def test_complete_random_game(self):
        # Start a new game
        _, info = self.env.reset()

        rng = np.random.default_rng(42)
        terminated = False
        truncated = False
        decisions = 0

        # Log game output or suppress it
        log_path = Path("normandy_game_log.txt")

        with (
            open(log_path, "w", encoding="utf-8")
            if self.LOG_GAME
            else open(os.devnull, "w")
        ) as output_file, contextlib.redirect_stdout(output_file):

            # Play until the game ends
            while not terminated and not truncated:
                mask = info["action_mask"]

                # Select a random legal action
                legal_actions = np.flatnonzero(mask)

                self.assertGreater(
                    len(legal_actions),
                    0,
                    f"No legal actions at decision {decisions}",
                )

                action_id = int(rng.choice(legal_actions))

                # Log the decision made by the random agent
                print(
                    f"DQN DECISION | "
                    f"Turn: {len(GlobalGameState.drawn_cards)+1} | "
                    f"Decision: {decisions + 1} | "
                    f"Type: {self.env.current_decision_type} | "
                    f"Action ID: {action_id} | "
                    f"Actions remaining: {GlobalGameState.actions_left_this_turn}"
                )
                _, _, terminated, truncated, info = (
                    self.env.step(action_id)
                )

                decisions += 1

                # Detect a game that fails to terminate
                self.assertLess(
                    decisions,
                    10000,
                    "Game exceeded 10,000 decisions",
                )

        # Remove ANSI colour codes from the log
        if self.LOG_GAME:
            log_path.write_text(
                self.remove_ansi_codes(
                    log_path.read_text(encoding="utf-8")
                ),
                encoding="utf-8",
            )

        # Verify the game completed
        self.assertTrue(terminated)
        self.assertIn(info["result"], ("won", "lost"))

        print(
            f"\nComplete game: "
            f"{info['result']}, "
            f"{decisions} decisions"
        )

        if self.LOG_GAME:
            print(f"Game log saved to: {log_path.resolve()}")
            

    @unittest.skip("")
    def test_100_complete_random_games(self):
        import os
        import time
        from contextlib import redirect_stdout

        import numpy as np

        from core.ai.dqn.normandy_env import NormandyEnv
        from core.global_game_state import GlobalGameState

        num_games = 100
        max_decisions = 10000

        wins = 0
        losses = 0
        total_decisions = 0
        total_turns = 0
        failures = []

        rng = np.random.default_rng(42)

        start_time = time.perf_counter()


        # Support Unicode characters in suppressed game output
        with open(os.devnull, "w", encoding="utf-8") as devnull, redirect_stdout(devnull):

            for game in range(num_games):
                try:
                    env = NormandyEnv()
                    observation, info = env.reset()

                    decisions = 0
                    terminated = False
                    truncated = False

                    while not terminated and not truncated:
                        legal_actions = np.flatnonzero(
                            info["action_mask"]
                        )

                        if len(legal_actions) == 0:
                            raise RuntimeError("No legal actions available")

                        action_id = int(rng.choice(legal_actions))

                        observation, reward, terminated, truncated, info = (
                            env.step(action_id)
                        )

                        decisions += 1

                        if decisions >= max_decisions:
                            raise RuntimeError("Decision limit exceeded")

                    total_decisions += decisions
                    # Count turns played
                    total_turns += len(GlobalGameState.drawn_cards)
                    if reward > 0:
                        wins += 1
                    else:
                        losses += 1

                except Exception as error:  # noqa: BLE001
                    failures.append((game + 1, str(error)))

                finally:
                    if "env" in locals():
                        env.close()

        elapsed = time.perf_counter() - start_time
        completed = wins + losses

        print("\n100 RANDOM GAMES BENCHMARK")
        print(f"Games completed:     {completed}/{num_games}")
        print(f"German wins:         {wins}")
        print(f"German losses:       {losses}")
        print(f"Failures:            {len(failures)}")
        print(f"Total time:          {elapsed:.2f} seconds")
        print(f"Games/second:        {completed / elapsed:.2f}")
        print(f"Decisions/second:    {total_decisions / elapsed:.2f}")

        if completed:
            print(f"Average turns:       {total_turns / completed:.1f}")
            print(f"Average decisions:   {total_decisions / completed:.1f}")

        for game_number, error in failures:
            print(f"Game {game_number} failed: {error}")

        self.assertEqual(len(failures), 0)
        self.assertEqual(completed, num_games)

    @unittest.skip("")
    def test_100_random_games_async_two_workers(self):
        import time

        import numpy as np

        num_games = 100
        num_workers = 4
        max_decisions = 10000

        rng = np.random.default_rng(42)

        completed = 0
        total_decisions = 0
        worker_decisions = np.zeros(num_workers, dtype=np.int32)
        needs_reset = np.zeros(num_workers, dtype=bool)

        start_time = time.perf_counter()

        # Each worker suppresses its own game output
        env = AsyncVectorEnv(
            [make_silent_normandy_env for _ in range(num_workers)],
            autoreset_mode=AutoresetMode.NEXT_STEP,
        )

        try:
            observations, info = env.reset()

            while completed < num_games:

                masks = info["action_mask"]

                # Select actions independently for each worker
                actions = np.zeros(num_workers, dtype=np.int64)

                for i in range(num_workers):

                    # NEXT_STEP ignores the action during autoreset
                    if needs_reset[i]:
                        continue

                    legal_actions = np.flatnonzero(masks[i])

                    if len(legal_actions) == 0:
                        raise RuntimeError(
                            f"Worker {i} has no legal actions "
                            f"without a pending reset"
                        )

                    actions[i] = int(rng.choice(legal_actions))

                    worker_decisions[i] += 1
                    total_decisions += 1

                    if worker_decisions[i] > max_decisions:
                        raise RuntimeError(
                            f"Worker {i} exceeded decision limit"
                        )

                # Advance both worker processes
                observations, rewards, terminated, truncated, info = env.step(actions)

                # Track completed games
                for i in range(num_workers):

                    if needs_reset[i]:
                        # This step performed an automatic reset
                        needs_reset[i] = False
                        worker_decisions[i] = 0
                        continue

                    if terminated[i] or truncated[i]:
                        completed += 1
                        needs_reset[i] = True

                        worker_decisions[i] = 0

                        if truncated[i]:
                            raise RuntimeError(
                                f"Worker {i} truncated unexpectedly"
                            )

        finally:
            env.close()

        elapsed = time.perf_counter() - start_time

        print("\n100 RANDOM GAMES - ASYNC VECTOR ENV")
        print(f"Workers:            {num_workers}")
        print(f"Games completed:    {completed}/{num_games}")
        print(f"Total time:         {elapsed:.2f} seconds")
        print(f"Games/second:       {completed / elapsed:.2f}")
        print(f"Decisions/second:   {total_decisions / elapsed:.2f}")
        print(f"Average decisions:  {total_decisions / completed:.1f}")

        self.assertEqual(completed, num_games)

    @unittest.skip("")
    def test_100_games_survival_rewards(self):
        import contextlib
        import os

        rng = np.random.default_rng(42)

        games = 100
        total_rewards = 0
        total_turns = 0
        wins = 0

        with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):

            for game in range(games):
                env = NormandyEnv()
                _, info = env.reset()

                episode_reward = 0.0
                terminated = False
                truncated = False

                while not terminated and not truncated:
                    legal_actions = np.flatnonzero(
                        info["action_mask"]
                    )

                    action_id = int(rng.choice(legal_actions))

                    cards_before = GlobalGameState.cards_drawn

                    _, reward, terminated, truncated, info = (
                        env.step(action_id)
                    )

                    cards_after = GlobalGameState.cards_drawn

                    # Check reward against completed turns
                    expected_reward = float(
                        cards_after - cards_before
                    )

                    if terminated and info["result"] == "won":
                        expected_reward += 1.0

                    self.assertEqual(
                        reward,
                        expected_reward,
                        f"Incorrect reward in game {game + 1}",
                    )

                    episode_reward += reward

                completed_turns = GlobalGameState.cards_drawn
                cards_drawn = len(GlobalGameState.drawn_cards)

                # Check cumulative reward
                expected_total = float(completed_turns)

                if info["result"] == "won":
                    expected_total += 1.0
                    wins += 1

                self.assertEqual(
                    episode_reward,
                    expected_total,
                    f"Incorrect total reward in game {game + 1}",
                )

                total_rewards += episode_reward
                total_turns += completed_turns

                env.close()

        print("\nSURVIVAL REWARD TEST")
        print(f"Games completed: {games}")
        print(f"German victories: {wins}")
        print(f"Average reward: {total_rewards / games:.2f}")
        print(f"Average completed turns: {total_turns / games:.2f}")
        print("All reward checks passed")


    def test_single_game_survival_rewards(self):
        import contextlib
        import os

        rng = np.random.default_rng(42)
        with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
            observation, info = self.env.reset()

            total_reward = 0.0
            terminated = False
            truncated = False

            while not terminated and not truncated:
                legal_actions = np.flatnonzero(info["action_mask"])
                action_id = int(rng.choice(legal_actions))

                observation, reward, terminated, truncated, info = (
                    self.env.step(action_id)
                )

                total_reward += reward

        turns_completed = GlobalGameState.cards_drawn
        cards_drawn = len(GlobalGameState.drawn_cards)

        print(f"\nGame result: {info['result']}")
        print(f"Cards drawn: {cards_drawn}")
        print(f"Turns completed: {turns_completed}")
        print(f"Total reward: {total_reward}")

        # Every completed turn earns one point
        self.assertEqual(total_reward, float(turns_completed))

if __name__ == "__main__":
    unittest.main()

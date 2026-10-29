import gymnasium as gym
import numpy as np
from gymnasium import spaces

from core.ai.dqn.dqn_action_adapter import execute_action_phase_decision, get_action_phase_decision, get_hitler_intervention_decision
from core.ai.dqn.dqn_game_state_encoder import OBSERVATION_SIZE, encode_game_state
from core.ai.headless_action_masks import HeadlessDecisionState, get_total_actions
from reset import reset_game
from core.global_game_state import GlobalGameState
from core.ai.normandy_full_game_ai_dqn import advance_to_next_decision
from core.ai.dqn.dqn_action_adapter import (
    execute_hitler_intervention_decision,
)


from core.ai.headless_action_masks import (
    german_eligible_spaces,
    panzer_divisions_list,
)
from core.allied_armies import armies_list


def debug_action_mask(decision_type, mask):
    legal_ids = np.flatnonzero(mask == 1.0).tolist()

    # debug_pants(
    #     f"ACTION MASK DEBUG | "
    #     f"Decision type: {decision_type} | "
    #     f"Legal actions: {legal_ids}"
    # )

    for action_id in legal_ids:

        if decision_type == "action_phase":

            if action_id == 0:
                description = "END TURN / PASS"

            elif 1 <= action_id <= len(armies_list):
                army = armies_list[action_id - 1]
                description = f"COUNTERATTACK: {army.name}"

            elif 7 <= action_id < (
                7 + len(panzer_divisions_list)
                * len(german_eligible_spaces)
            ):
                relative_id = action_id - 7

                panzer_index = (
                    relative_id // len(german_eligible_spaces)
                )
                space_index = (
                    relative_id % len(german_eligible_spaces)
                )

                panzer = panzer_divisions_list[panzer_index]
                space = german_eligible_spaces[space_index]

                description = (
                    f"DEPLOY PANZER: {panzer.name} -> {space.name}"
                )

            else:
                description = "OTHER ACTION"

        elif decision_type == "hitler_intervention":
            description = (
                "HITLER INTERVENTION ACTION "
                "(interpretation depends on intervention stage)"
            )

        else:
            description = "UNKNOWN DECISION TYPE"

        # debug_pants(
        #     f"ACTION MASK DEBUG | "
        #     f"ID {action_id}: {description}"
        # )
class NormandyEnv(gym.Env):

    def __init__(self):
        super().__init__()

        total_actions, _ = get_total_actions()

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(OBSERVATION_SIZE,),
            dtype=np.float32,
        )

        self.action_space = spaces.Discrete(total_actions)
        self.current_decision_type = "action_phase"

    def pad_action_mask(self, mask):
        if len(mask) > self.action_space.n:
            raise ValueError(
                f"Action mask has {len(mask)} actions, "
                f"but environment supports only {self.action_space.n}"
            )

        padded_mask = np.zeros(self.action_space.n, dtype=np.float32)
        padded_mask[:len(mask)] = mask

        return padded_mask

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.draw_deck, self.mid_deck, self.late_deck = reset_game()
        HeadlessDecisionState.reset_hitler_intervention()
        self.current_decision_type = "action_phase"

        observation = encode_game_state()

        # return observation, {}
        # Return legal actions for the initial decision
        _, mask = get_action_phase_decision()

        return observation, {
            "action_mask": self.pad_action_mask(mask),
        }
    

    def execute_action(self, action_id):
        if self.current_decision_type == "hitler_intervention":
            return execute_hitler_intervention_decision(
                action_id,
                GlobalGameState.current_card,
                GlobalGameState.current_weather,
            )

        return execute_action_phase_decision(action_id)

    def step(self, action_id):
        reward = 0.0
        terminated = False
        truncated = False

        # Record the number of cards before the action
        cards_before = GlobalGameState.cards_drawn

        if self.current_decision_type == "hitler_intervention":
            _, mask = get_hitler_intervention_decision(
                GlobalGameState.current_card
            )
        else:
            _, mask = get_action_phase_decision()


        # debug_action_mask(self.current_decision_type, mask)
        
        
        if (
            action_id < 0
            or action_id >= len(mask)
            or mask[action_id] != 1.0
        ):
            raise ValueError(f"Illegal action: {action_id}")

        result = self.execute_action(action_id)

        if self.current_decision_type == "hitler_intervention":
            if result:
                observation, mask = get_hitler_intervention_decision(
                    GlobalGameState.current_card
                )
            else:
                self.current_decision_type = "action_phase"

                if (
                    GlobalGameState.actions_left_this_turn == 0
                    and GlobalGameState.reserve_actions == 0
                ):
                    GlobalGameState.current_step = 1

                    decision = advance_to_next_decision(
                        self.draw_deck,
                        self.mid_deck,
                        self.late_deck,
                    )

                    if "result" in decision:
                        terminated = True

                        # Reward survived cards
                        reward = float(
                            GlobalGameState.cards_drawn - cards_before
                        )

                        # Include the final card on victory
                        if decision["result"] == "won":
                            reward += 101.0

                        self.current_decision_type = None
                        observation, _ = get_action_phase_decision()
                        mask = np.zeros(
                            self.action_space.n,
                            dtype=np.float32,
                        )

                        return (
                            observation,
                            reward,
                            terminated,
                            truncated,
                            {
                                "result": decision["result"],
                                "action_mask": mask,
                                "turn_reached": int(GlobalGameState.cards_drawn),
                            },
                        )

                    self.current_decision_type = decision["decision_type"]
                    observation = decision["observation"]
                    mask = decision["action_mask"]

                else:
                    observation, mask = get_action_phase_decision()

        elif (
            GlobalGameState.actions_left_this_turn == 0
            and GlobalGameState.reserve_actions == 0
        ):
            decision = advance_to_next_decision(
                self.draw_deck,
                self.mid_deck,
                self.late_deck,
            )

            if "result" in decision:
                terminated = True

                # Reward survived cards
                reward = float(
                    GlobalGameState.cards_drawn - cards_before
                )

                # Include the final card on victory
                if decision["result"] == "won":
                    reward += 101.0

                self.current_decision_type = None
                observation, _ = get_action_phase_decision()
                mask = np.zeros(
                    self.action_space.n,
                    dtype=np.float32,
                )

                return (
                    observation,
                    reward,
                    terminated,
                    truncated,
                    {
                        "result": decision["result"],
                        "action_mask": mask,
                        "turn_reached": int(GlobalGameState.cards_drawn),
                    },
                )

            self.current_decision_type = decision["decision_type"]
            observation = decision["observation"]
            mask = decision["action_mask"]

        else:
            observation, mask = get_action_phase_decision()

        # Reward survival when the next card is reached
        reward = float(
            GlobalGameState.cards_drawn - cards_before
        )

        return (
            observation,
            reward,
            terminated,
            truncated,
            {
                "action_mask": self.pad_action_mask(mask),
            },
        )

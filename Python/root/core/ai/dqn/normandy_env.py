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

        # Check legality before executing the action
        if self.current_decision_type == "hitler_intervention":
            _, mask = get_hitler_intervention_decision(
                GlobalGameState.current_card
            )
        else:
            _, mask = get_action_phase_decision()

        if (
            action_id < 0
            or action_id >= len(mask)
            or mask[action_id] != 1.0
        ):
            raise ValueError(f"Illegal action: {action_id}")

        # Execute the action
        result = self.execute_action(action_id)

        if self.current_decision_type == "hitler_intervention":
            if result:
                # Same intervention, next decision
                observation, mask = get_hitler_intervention_decision(
                    GlobalGameState.current_card
                )
            else:
                # Intervention complete; resume game progression
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
            # Advance to the next decision or game result
            decision = advance_to_next_decision(
                self.draw_deck,
                self.mid_deck,
                self.late_deck,
            )

            if "result" in decision:
                terminated = True
                self.current_decision_type = None
                observation, _ = get_action_phase_decision()
                mask = np.zeros(self.action_space.n, dtype=np.float32)

                return (
                    observation,
                    reward,
                    terminated,
                    truncated,
                    {
                        "result": decision["result"],
                        "action_mask": mask,
                    },
                )

            self.current_decision_type = decision["decision_type"]
            observation = decision["observation"]
            mask = decision["action_mask"]

        else:
            # Normal Action Phase continues
            observation, mask = get_action_phase_decision()

        return (
            observation,
            reward,
            terminated,
            truncated,
            {
                "action_mask": self.pad_action_mask(mask),
            },
        )

import numpy as np
import torch


from pathlib import Path
import yaml


def load_hyperparameters(config_name="normandy_v1"):
    config_path = Path(__file__).with_name("hyperparameters.yaml")

    with config_path.open("r", encoding="utf-8") as file:
        configurations = yaml.safe_load(file)

    return configurations[config_name]

class DQNAgent:
    def __init__(self, policy_net, device):
        self.policy_net = policy_net
        self.device = device

        params = load_hyperparameters("normandy_v1")

        # Replay memory and training
        self.replay_memory_size = params["replay_memory_size"]
        self.mini_batch_size = params["mini_batch_size"]
        self.learning_rate = params["learning_rate"]
        self.discount_factor = params["discount_factor"]

        # Exploration
        self.epsilon = params["epsilon_init"]
        self.epsilon_decay = params["epsilon_decay"]
        self.epsilon_min = params["epsilon_min"]

        # Target network
        self.network_sync_rate = params["network_sync_rate"]

        # Network architecture
        self.hidden_layers = params["hidden_layers"]

        # DQN variants
        self.enable_double_dqn = params["enable_double_dqn"]
        self.enable_dueling_dqn = params["enable_dueling_dqn"]
        
    def select_action(self, observation, action_mask):
        legal_actions = np.flatnonzero(action_mask)

        if len(legal_actions) == 0:
            raise ValueError("No legal actions available")

        # Exploration
        if np.random.random() < self.epsilon:
            return int(np.random.choice(legal_actions))

        # Exploitation
        with torch.no_grad():
            state = torch.as_tensor(
                observation, dtype=torch.float32, device=self.device
            ).unsqueeze(0)

            q_values = self.policy_net(state).squeeze(0)
            q_values = q_values.masked_fill(
                torch.as_tensor(action_mask, device=self.device) == 0,
                float("-inf"),
            )

            return int(torch.argmax(q_values).item())

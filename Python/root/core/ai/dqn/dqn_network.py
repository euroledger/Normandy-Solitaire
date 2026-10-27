
from torch import nn

from core.ai.dqn.dqn_agent import load_hyperparameters


class DQNNetwork(nn.Module):
    def __init__(self, observation_size, action_size):
        super().__init__()

        params = load_hyperparameters()
        hidden_layers = params["hidden_layers"]

        layers = []
        input_size = observation_size

        for hidden_size in hidden_layers:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size

        layers.append(nn.Linear(input_size, action_size))

        self.network = nn.Sequential(*layers)
        
        print(self.network)

    def forward(self, observation):
        return self.network(observation)

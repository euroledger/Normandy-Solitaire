
from torch import nn

from core.ai.dqn.dqn_agent import load_hyperparameters
from core.ai.dqn.noisy.dqn_noisy_linear import NoisyLinear


class DQNNoisyNetwork(nn.Module):

    def __init__(self, observation_size, action_size):
        super().__init__()

        params = load_hyperparameters()

        hidden_layers = params["hidden_layers"]
        sigma0 = params.get("noisy_sigma0", 0.5)

        layers = []
        input_size = observation_size

        for index, hidden_size in enumerate(hidden_layers):

            # Use NoisyLinear only in the final hidden layer.
            if index == len(hidden_layers) - 1:
                layer = NoisyLinear(
                    input_size,
                    hidden_size,
                    sigma0=sigma0,
                )
            else:
                layer = nn.Linear(
                    input_size,
                    hidden_size,
                )

            layers.append(layer)
            layers.append(nn.ReLU())

            input_size = hidden_size

        # Use NoisyLinear for the output layer.
        layers.append(
            NoisyLinear(
                input_size,
                action_size,
                sigma0=sigma0,
            )
        )

        self.network = nn.Sequential(*layers)

        print(self.network)

    def forward(self, observation):
        return self.network(observation)

    def reset_noise(self):
        # Resample noise in every NoisyLinear layer.
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()

    def enable_noise(self):
        # Enable parameter noise.
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.enable_noise()

    def disable_noise(self):
        # Disable parameter noise.
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.disable_noise()

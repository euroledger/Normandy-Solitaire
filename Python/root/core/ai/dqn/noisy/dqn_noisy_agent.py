
import numpy as np
import torch

from core.ai.dqn.dqn_agent import DQNAgent
from core.ai.dqn.noisy.dqn_noisy_network import DQNNoisyNetwork


class NoisyDQNAgent(DQNAgent):

    def __init__(self, policy_net, device):
        if not isinstance(policy_net, DQNNoisyNetwork):
            raise TypeError(
                "NoisyDQNAgent requires DQNNoisyNetwork"
            )

        # Reuse the existing DQN initialization.
        super().__init__(policy_net, device)

        # Enable parameter noise for exploration.
        self.policy_net.enable_noise()
        self.target_net.enable_noise()


    def report_noise_parameters(self):
        print("\n=== NOISY LAYER DIAGNOSTICS ===")

        optimizer_ids = {
            id(param)
            for group in self.optimizer.param_groups
            for param in group["params"]
        }

        for name, module in self.policy_net.named_modules():
            if not hasattr(module, "weight_sigma"):
                continue

            print(f"\nLayer: {name}")

            for param_name in ("weight_sigma", "bias_sigma"):
                param = getattr(module, param_name)

                with torch.no_grad():
                    mean_abs = param.abs().mean().item()
                    maximum = param.abs().max().item()

                print(
                    f"  {param_name}: "
                    f"mean_abs={mean_abs:.8f}, "
                    f"max_abs={maximum:.8f}, "
                    f"in_optimizer={id(param) in optimizer_ids}, "
                    f"requires_grad={param.requires_grad}"
                )

    def select_action(self, observation, action_mask):
        # Select the highest-valued legal action
        # after sampling new network noise.

        legal_actions = np.flatnonzero(action_mask)

        if len(legal_actions) == 0:
            raise ValueError("No legal actions available")

        self.policy_net.reset_noise()

        with torch.no_grad():
            state = torch.as_tensor(
                observation,
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)

            q_values = self.policy_net(state).squeeze(0)

            mask = torch.as_tensor(
                action_mask,
                dtype=torch.bool,
                device=self.device,
            )

            q_values = q_values.masked_fill(
                ~mask,
                float("-inf"),
            )

            return int(q_values.argmax().item())

    def select_actions(
        self,
        observations,
        action_masks,
        needs_reset=None,
    ):
        # Resample once for the entire decision batch.
        # All active workers share this parameter sample.

        self.policy_net.reset_noise()

        if needs_reset is None:
            needs_reset = np.zeros(
                len(observations),
                dtype=bool,
            )

        actions = np.zeros(
            len(observations),
            dtype=np.int64,
        )

        active_indices = np.flatnonzero(~needs_reset)

        if len(active_indices) == 0:
            return actions

        active_observations = np.asarray(
            [observations[i] for i in active_indices],
            dtype=np.float32,
        )

        active_masks = np.asarray(
            [action_masks[i] for i in active_indices],
            dtype=bool,
        )

        if not active_masks.any(axis=1).all():
            raise ValueError(
                "An active worker has no legal actions"
            )

        with torch.no_grad():
            states = torch.as_tensor(
                active_observations,
                dtype=torch.float32,
                device=self.device,
            )

            masks = torch.as_tensor(
                active_masks,
                dtype=torch.bool,
                device=self.device,
            )

            q_values = self.policy_net(states)

            q_values = q_values.masked_fill(
                ~masks,
                float("-inf"),
            )

            selected = (
                q_values.argmax(dim=1).cpu().numpy()
            )

        actions[active_indices] = selected
        for i in active_indices:
            if np.random.random() < self.epsilon:
                legal_actions = np.flatnonzero(action_masks[i])
                actions[i] = int(np.random.choice(legal_actions))

        return actions

    def train_step(self):
        # Do not change the existing Double DQN,
        # six-step TD, replay or loss calculations.

        if len(self.replay_buffer) < self.mini_batch_size:
            return None

        # Use independently sampled noise in the
        # online and target networks.
        self.policy_net.reset_noise()
        self.target_net.reset_noise()

        # Keep these samples fixed throughout this
        # gradient update, including Double DQN
        # action selection and target evaluation.
        return super().train_step()

    def set_deterministic(self):
        # Evaluate the learned mean policy.
        self.policy_net.disable_noise()
        self.target_net.disable_noise()

    def set_noisy(self):
        # Restore parameter-noise exploration.
        self.policy_net.enable_noise()
        self.target_net.enable_noise()

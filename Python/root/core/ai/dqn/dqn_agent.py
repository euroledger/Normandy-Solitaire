import numpy as np
import torch


from pathlib import Path
import yaml

from core.ai.dqn.dqn_replay_buffer import DQNReplayBuffer
from core.ai.dqn.dqn_priority_replay_buffer import DQNPriorityReplayBuffer
import copy
import torch.nn.functional as F


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


        self.gradient_clip_norm = params.get("gradient_clip_norm", None)
        self.replay_memory_size = params["replay_memory_size"]
        self.mini_batch_size = params["mini_batch_size"]
        self.learning_rate = params["learning_rate"]
        self.discount_factor = params["discount_factor"]

        self.epsilon = params["epsilon_init"]
        self.epsilon_decay = params["epsilon_decay"]
        self.epsilon_min = params["epsilon_min"]
        
        self.n_step = int(params.get("n_step", 1))
        if self.n_step < 1:
            raise ValueError("n_step must be at least 1")

        self.network_sync_rate = params["network_sync_rate"]
        self.hidden_layers = params["hidden_layers"]

        self.enable_double_dqn = params["enable_double_dqn"]
        self.enable_dueling_dqn = params["enable_dueling_dqn"]

        # Create replay buffer based on configuration
        self.replay_buffer_type = params.get("replay_buffer_type", "uniform")
        
        # PER hyperparameters
        self.per_alpha = params.get("per_alpha", 0.6)
        self.per_beta_initial = params.get("per_beta_initial", 0.4)
        self.per_beta_final = params.get("per_beta_final", 1.0)
        self.per_beta_anneal_steps = params.get("per_beta_anneal_steps", 500000)

        # Track annealing independently of historical training updates
        self.per_beta_steps = 0

        if self.replay_buffer_type == "priority":
            self.replay_buffer = DQNPriorityReplayBuffer(
                capacity=self.replay_memory_size,
                alpha=self.per_alpha,
            )
        elif self.replay_buffer_type == "uniform":
            self.replay_buffer = DQNReplayBuffer(
                capacity=self.replay_memory_size
            )
        else:
            raise ValueError(
                f"Unknown replay_buffer_type: {self.replay_buffer_type}"
            )

        # Create target network with identical initial weights
        self.target_net = copy.deepcopy(self.policy_net)
        self.target_net.to(self.device)
        self.target_net.eval()

        # Create optimizer for the policy network
        self.optimizer = torch.optim.Adam(
            self.policy_net.parameters(),
            lr=self.learning_rate,
        )

        # Count training updates
        self.training_steps = 0

        # Frequency of gradient updates
        self.train_frequency = params["train_frequency"]

        if self.train_frequency < 1:
            raise ValueError("train_frequency must be at least 1")
        
        self.grad_norm_sum = 0.0
        self.grad_norm_max = 0.0
        self.grad_norm_count = 0
        self.grad_norm_clipped_count = 0


    def restore_priority_replay_buffer(self, state):
        self.replay_buffer = DQNPriorityReplayBuffer(
            capacity=self.replay_memory_size,
            alpha=self.per_alpha,
        )
        self.replay_buffer.load_state_dict(state)
        
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


    def restore_replay_buffer(self, experiences):
        # Restore experiences using the configured replay buffer type
        if self.replay_buffer_type == "priority":
            self.replay_buffer = DQNPriorityReplayBuffer(
                capacity=self.replay_memory_size
            )

            for experience in experiences:
                self.replay_buffer.add(*experience)

        else:
            self.replay_buffer = DQNReplayBuffer(
                capacity=self.replay_memory_size
            )

            for experience in experiences:
                self.replay_buffer.add(*experience)
         
                
    def current_per_beta(self):
        # Linearly increase beta from its initial to final value.
        progress = min(
            1.0,
            self.per_beta_steps / self.per_beta_anneal_steps,
        )

        return (
            self.per_beta_initial
            + progress * (self.per_beta_final - self.per_beta_initial)
        )
        
    def train_step(self):
        # Wait until enough experiences are available
        if len(self.replay_buffer) < self.mini_batch_size:
            return None

        # Sample experiences
        if self.replay_buffer_type == "priority":
            (
                observations,
                actions,
                rewards,
                next_observations,
                terminated,
                next_masks,
                indices,
                weights,
                n_steps
            ) = self.replay_buffer.sample(
                self.mini_batch_size,
                beta=self.current_per_beta(),
            )
        else:
            (
                observations,
                actions,
                rewards,
                next_observations,
                terminated,
                next_masks,
                n_steps,
            ) = self.replay_buffer.sample(self.mini_batch_size)

            indices = None
            weights = None

        # Convert batch to tensors
        states = torch.as_tensor(
            observations, dtype=torch.float32, device=self.device
        )
        
        
        actions = torch.as_tensor(
            actions, dtype=torch.long, device=self.device
        )
        rewards = torch.as_tensor(
            rewards, dtype=torch.float32, device=self.device
        )
        n_steps = torch.as_tensor(
            n_steps, dtype=torch.float32, device=self.device
        )
        next_states = torch.as_tensor(
            next_observations, dtype=torch.float32, device=self.device
        )
        dones = torch.as_tensor(
            terminated, dtype=torch.bool, device=self.device
        )
        next_masks = torch.as_tensor(
            next_masks, dtype=torch.bool, device=self.device
        )

        # Calculate Q-values for actions taken
        current_q = self.policy_net(states).gather(
            1, actions.unsqueeze(1)
        ).squeeze(1)

        with torch.no_grad():
            # Identify states with at least one legal next action
            valid_next = (~dones) & next_masks.any(dim=1)

            next_q = torch.zeros_like(rewards)

            if valid_next.any():
                valid_states = next_states[valid_next]
                valid_masks = next_masks[valid_next]

                if self.enable_double_dqn:
                    # Policy network selects the best legal action
                    policy_q = self.policy_net(valid_states)
                    policy_q = policy_q.masked_fill(
                        ~valid_masks, float("-inf")
                    )
                    best_actions = policy_q.argmax(dim=1)

                    # Target network evaluates that action
                    target_q = self.target_net(valid_states)
                    next_q[valid_next] = target_q.gather(
                        1, best_actions.unsqueeze(1)
                    ).squeeze(1)

                else:
                    # Standard DQN uses target network for selection
                    target_q = self.target_net(valid_states)
                    target_q = target_q.masked_fill(
                        ~valid_masks, float("-inf")
                    )
                    next_q[valid_next] = target_q.max(dim=1).values

            # Apply the correct bootstrap discount for each experience
            discounts = self.discount_factor ** n_steps

            target_values = (
                rewards + discounts * next_q
            )

        # Calculate TD errors
        td_errors = target_values - current_q


        if isinstance(self.replay_buffer, DQNPriorityReplayBuffer):            # Calculate individual Huber losses
            individual_losses = F.smooth_l1_loss(
                current_q,
                target_values,
                reduction="none",
            )

            # Apply importance-sampling weights
            importance_weights = torch.as_tensor(
                weights,
                dtype=torch.float32,
                device=self.device,
            )

            loss = (individual_losses * importance_weights).mean()

            # Update replay priorities using TD errors
            self.replay_buffer.update_priorities(
                indices,
                td_errors.detach().abs().cpu().numpy(),
            )
        else:
            # Preserve original uniform replay behaviour
            loss = F.smooth_l1_loss(current_q, target_values)


        # Update policy network
        # self.optimizer.zero_grad()
        # loss.backward()
        # self.optimizer.step()
        self.optimizer.zero_grad()
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.policy_net.parameters(),
            max_norm=(
                self.gradient_clip_norm
                if self.gradient_clip_norm is not None
                else float("inf")
            ),
        )
        
    
        
        grad_norm_value = float(grad_norm.item())
        self.grad_norm_sum += grad_norm_value
        self.grad_norm_max = max(self.grad_norm_max, grad_norm_value)
        self.grad_norm_count += 1


        if (
            self.gradient_clip_norm is not None
            and grad_norm_value > self.gradient_clip_norm
        ):
            self.grad_norm_clipped_count += 1

        self.optimizer.step()
        self.training_steps += 1
        if self.replay_buffer_type == "priority":
            self.per_beta_steps += 1

        # Periodically synchronize target network
        if self.training_steps % self.network_sync_rate == 0:
            self.target_net.load_state_dict(
                self.policy_net.state_dict()
            )

        return float(loss.item())

    def select_actions(self, observations, action_masks, needs_reset=None):
        """
        Select epsilon-greedy actions for multiple environments.

        Neural-network inference is batched into a single forward pass.
        Each environment makes an independent exploration decision.

        Workers awaiting automatic reset are skipped.
        """

        num_envs = len(observations)

        if needs_reset is None:
            needs_reset = np.zeros(num_envs, dtype=bool)

        actions = np.zeros(num_envs, dtype=np.int64)

        greedy_indices = []
        greedy_observations = []
        greedy_masks = []

        # --------------------------------------------------
        # EPSILON-GREEDY SELECTION
        # --------------------------------------------------

        for i in range(num_envs):

            if needs_reset[i]:
                continue

            mask = action_masks[i]

            legal_actions = np.flatnonzero(mask)

            if len(legal_actions) == 0:
                raise ValueError(
                    f"No legal actions available for worker {i}"
                )

            # Independent exploration decision per worker
            if np.random.random() < self.epsilon:

                actions[i] = int(
                    np.random.choice(legal_actions)
                )

            else:

                greedy_indices.append(i)
                greedy_observations.append(observations[i])
                greedy_masks.append(mask)

        # --------------------------------------------------
        # BATCHED NETWORK INFERENCE
        # --------------------------------------------------

        if greedy_indices:

            with torch.no_grad():

                states = torch.as_tensor(
                    np.asarray(greedy_observations),
                    dtype=torch.float32,
                    device=self.device,
                )

                q_values = self.policy_net(states)

                masks = torch.as_tensor(
                    np.asarray(greedy_masks),
                    device=self.device,
                )

                q_values = q_values.masked_fill(
                    masks == 0,
                    float("-inf"),
                )

                selected_actions = (
                    torch.argmax(q_values, dim=1)
                    .cpu()
                    .numpy()
                )

            for index, action in zip(
                greedy_indices,
                selected_actions,
            ):

                actions[index] = int(action)

        return actions

# Independent proportional Prioritized Experience Replay buffer.

# Does not modify DQNReplayBuffer. sample() returns the original six arrays
# plus sampled indices and normalized importance-sampling weights.

import random
import numpy as np


class DQNPriorityReplayBuffer:
    def __init__(self, capacity, alpha=0.6, priority_epsilon=1e-6):
        if capacity < 1 or not 0 <= alpha <= 1 or priority_epsilon <= 0:
            raise ValueError('Invalid PER configuration')
        self.capacity = int(capacity)
        self.alpha = float(alpha)
        self.priority_epsilon = float(priority_epsilon)
        self.buffer = [None] * self.capacity
        self.position = 0
        self.size = 0
        self.max_priority = 1.0
        self.tree_capacity = 1 << (self.capacity - 1).bit_length()
        self.sum_tree = np.zeros(2 * self.tree_capacity, dtype=np.float64)
        self.min_tree = np.full(2 * self.tree_capacity, np.inf, dtype=np.float64)

    def __len__(self):
        return self.size

    def _set_priority(self, index, raw_priority):
        value = float(raw_priority) ** self.alpha
        node = self.tree_capacity + int(index)
        self.sum_tree[node] = value
        self.min_tree[node] = value
        node //= 2
        while node:
            self.sum_tree[node] = self.sum_tree[2 * node] + self.sum_tree[2 * node + 1]
            self.min_tree[node] = min(self.min_tree[2 * node], self.min_tree[2 * node + 1])
            node //= 2


    def state_dict(self):
        return {
            "buffer": list(self.buffer),
            "sum_tree": self.sum_tree.copy(),
            "min_tree": self.min_tree.copy(),
            "position": self.position,
            "size": self.size,
            "max_priority": self.max_priority,
        }


    def load_state_dict(self, state):
        if len(state["buffer"]) > self.capacity:
            raise ValueError("Saved replay buffer exceeds capacity")

        if len(state["sum_tree"]) != len(self.sum_tree):
            raise ValueError("Incompatible sum tree size")

        if len(state["min_tree"]) != len(self.min_tree):
            raise ValueError("Incompatible min tree size")

        self.buffer = list(state["buffer"])
        self.sum_tree[:] = state["sum_tree"]
        self.min_tree[:] = state["min_tree"]
        self.position = state["position"]
        self.size = state["size"]
        self.max_priority = state["max_priority"]
        
    def add(self, observation, action_id, reward, next_observation, terminated, next_action_mask, n_steps=1):
        experience = (
            np.array(observation, dtype=np.float32, copy=True),
            int(action_id),
            float(reward),
            np.array(next_observation, dtype=np.float32, copy=True),
            bool(terminated),
            np.array(next_action_mask, dtype=np.float32, copy=True),
            int(n_steps),
        )
        self.buffer[self.position] = experience
        
        # Give each new experience a neutral initial priority.
        # Its priority will be updated after its first training sample.
        self._set_priority(self.position, 1.0)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _find_index(self, mass):
        node = 1
        while node < self.tree_capacity:
            left = node * 2
            if mass < self.sum_tree[left]:
                node = left
            else:
                mass -= self.sum_tree[left]
                node = left + 1
        return node - self.tree_capacity

    def sample(self, batch_size, beta=0.4):
        if batch_size < 1 or self.size < batch_size:
            raise ValueError('Not enough transitions to sample')
        if not 0 <= beta <= 1:
            raise ValueError('beta must be in [0, 1]')
        total = self.sum_tree[1]
        if not np.isfinite(total) or total <= 0:
            raise RuntimeError('Invalid priority sum')
        # Stratified proportional sampling, with replacement.
        segment = total / batch_size
        indices = np.array([
            self._find_index(min((j + random.random()) * segment, np.nextafter(total, 0)))
            for j in range(batch_size)
        ], dtype=np.int64)
        probabilities = self.sum_tree[self.tree_capacity + indices] / total
        min_probability = self.min_tree[1] / total
        weights = (probabilities / min_probability) ** (-beta)
        experiences = [self.buffer[int(i)] for i in indices]

        # Support both legacy six-field and new seven-field experiences
        normalized = [
            experience if len(experience) == 7 else (*experience, 1)
            for experience in experiences
        ]

        observations, actions, rewards, next_observations, terminated, next_masks, n_steps = zip(*normalized)
                
        return (
            np.stack(observations),
            np.asarray(actions, dtype=np.int64),
            np.asarray(rewards, dtype=np.float32),
            np.stack(next_observations),
            np.asarray(terminated, dtype=np.bool_),
            np.stack(next_masks),
            indices,
            np.asarray(weights, dtype=np.float32),
            np.asarray(n_steps, dtype=np.int64),
        )
        
    def update_priorities(self, indices, td_errors):
        for index, error in zip(indices, td_errors):
            index = int(index)
            if not 0 <= index < self.size:
                raise IndexError(index)
            priority = abs(float(error)) + self.priority_epsilon
            if not np.isfinite(priority):
                raise ValueError('Non-finite TD error')
            self.max_priority = max(self.max_priority, priority)
            self._set_priority(index, priority)

    def load_from_uniform(self, uniform_buffer):
        for experience in uniform_buffer.buffer:
            # Old checkpoints contain six fields; new experiences contain seven
            if len(experience) == 7:
                observation, action, reward, next_observation, done, next_mask, n_steps = experience
            else:
                observation, action, reward, next_observation, done, next_mask = experience
                n_steps = 1

            self.add(
                observation,
                action,
                reward,
                next_observation,
                done,
                next_mask,
                n_steps=n_steps,
            )

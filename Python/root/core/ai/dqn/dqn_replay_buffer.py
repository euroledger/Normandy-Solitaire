import random
from collections import deque

import numpy as np


class DQNReplayBuffer:

    def __init__(self, capacity):
        # Automatically discard oldest experiences when full
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def add(
        self,
        observation,
        action_id,
        reward,
        next_observation,
        terminated,
        next_action_mask,
        n_steps=1,
    ):
        # Store independent copies of observations and masks
        experience = (
            np.array(observation, dtype=np.float32, copy=True),
            int(action_id),
            float(reward),
            np.array(next_observation, dtype=np.float32, copy=True),
            bool(terminated),
            np.array(next_action_mask, dtype=np.float32, copy=True),
            int(n_steps)
        )

        self.buffer.append(experience)


    def sample(self, batch_size):
        experiences = random.sample(self.buffer, batch_size)

        # Support both existing six-field and new seven-field experiences
        normalized = [
            experience if len(experience) == 7 else (*experience, 1)
            for experience in experiences
        ]

        observations, actions, rewards, next_observations, terminated, next_masks, n_steps = zip(*normalized)


        return (
            np.stack(observations),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.stack(next_observations),
            np.array(terminated, dtype=np.bool_),
            np.stack(next_masks),
            np.array(n_steps, dtype=np.int64),
        )

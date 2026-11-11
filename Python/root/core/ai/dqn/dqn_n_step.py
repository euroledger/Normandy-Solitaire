
from collections import deque


class NStepTransitionCollector:
    def __init__(self, n_step=1, gamma=0.99):
        if n_step < 1:
            raise ValueError("n_step must be at least 1")

        self.n_step = n_step
        self.gamma = gamma
        self.pending = deque()

    def add(
        self,
        observation,
        action_id,
        reward,
        next_observation,
        terminated,
        next_action_mask,
    ):
        # Add one decision transition
        self.pending.append((
            observation,
            action_id,
            reward,
            next_observation,
            terminated,
            next_action_mask,
        ))

        completed = []

        # Produce a transition once enough decisions are available
        if len(self.pending) >= self.n_step:
            completed.append(self._build_transition())
            self.pending.popleft()

        # At game termination, flush all remaining transitions
        if terminated:
            while self.pending:
                completed.append(self._build_transition())
                self.pending.popleft()

        return completed

    def _build_transition(self):
        total_reward = 0.0
        steps = 0

        for transition in list(self.pending)[:self.n_step]:
            (
                _,
                _,
                reward,
                next_observation,
                terminated,
                next_action_mask,
            ) = transition

            total_reward += (self.gamma ** steps) * reward
            steps += 1

            if terminated:
                break

        first = self.pending[0]

        return (
            first[0],          # Initial observation
            first[1],          # Initial action
            total_reward,      # Discounted cumulative reward
            next_observation,  # Observation after n steps
            terminated,        # Terminal status
            next_action_mask,  # Action mask after n steps
            steps,             # Actual number of steps
        )

    def reset(self):
        self.pending.clear()

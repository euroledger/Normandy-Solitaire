
from pathlib import Path

import numpy as np
import torch

from core.ai.dqn.noisy.dqn_noisy_network import DQNNoisyNetwork


# Configuration
CHECKPOINT_DIR = Path("core/ai/dqn/checkpoints/noisy")
INITIAL = CHECKPOINT_DIR / "noisy_initial.pt"
FINAL = CHECKPOINT_DIR / "noisy_6.pt"

NUM_STATES = 1000
NUM_NOISE_SAMPLES = 20
BATCH_SIZE = 128
SEED = 42


def load_checkpoint(path):
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )
    return checkpoint


def make_network(checkpoint):
    state = checkpoint["policy_net"]

    observation_size = state["network.0.weight"].shape[1]
    action_size = state["network.6.weight_mu"].shape[0]

    network = DQNNoisyNetwork(
        observation_size,
        action_size,
    )
    network.load_state_dict(state, strict=True)
    network.eval()

    return network


def get_test_states(checkpoint):
    rng = np.random.default_rng(SEED)
    replay = checkpoint["replay_buffer"]

    eligible = [
        item for item in replay
        if not bool(item[4]) and np.any(item[5])
    ]

    if len(eligible) < NUM_STATES:
        raise ValueError("Not enough eligible replay states")

    indices = rng.choice(
        len(eligible),
        size=NUM_STATES,
        replace=False,
    )

    selected = [eligible[int(i)] for i in indices]

    observations = np.stack(
        [item[3] for item in selected]
    ).astype(np.float32)

    masks = np.stack(
        [item[5] for item in selected]
    ).astype(bool)

    return observations, masks


@torch.no_grad()
def predict(network, observations, masks):
    actions = []
    chosen_values = []

    for start in range(0, len(observations), BATCH_SIZE):
        end = start + BATCH_SIZE

        states = torch.from_numpy(
            observations[start:end]
        )

        legal = torch.from_numpy(
            masks[start:end]
        )

        q_values = network(states)
        masked_q = q_values.masked_fill(
            ~legal,
            float("-inf"),
        )

        chosen = masked_q.argmax(dim=1)

        actions.append(chosen.numpy())

        chosen_values.append(
            q_values.gather(
                1, chosen.unsqueeze(1)
            ).squeeze(1).numpy()
        )

    return (
        np.concatenate(actions),
        np.concatenate(chosen_values),
    )


def diagnose(label, checkpoint, observations, masks):
    network = make_network(checkpoint)

    network.disable_noise()

    baseline_actions, baseline_q = predict(
        network, observations, masks
    )

    network.enable_noise()

    disagreements = []
    q_changes = []
    all_actions = []

    for sample in range(NUM_NOISE_SAMPLES):
        network.reset_noise()

        noisy_actions, noisy_q = predict(
            network, observations, masks
        )

        disagreements.append(
            noisy_actions != baseline_actions
        )

        q_changes.append(
            np.abs(noisy_q - baseline_q)
        )

        all_actions.append(noisy_actions)

    disagreements = np.stack(disagreements)
    q_changes = np.stack(q_changes)
    all_actions = np.stack(all_actions)

    distinct_counts = np.array([
        len(set(all_actions[:, i]))
        for i in range(NUM_STATES)
    ])

    changed_at_least_once = disagreements.any(axis=0)

    print(f"\n=== {label} ===")

    print(
        "Action disagreement: "
        f"{100 * disagreements.mean():.2f}%"
    )

    print(
        "States with at least one changed action: "
        f"{100 * changed_at_least_once.mean():.2f}%"
    )

    print(
        "Mean distinct noisy actions per state: "
        f"{distinct_counts.mean():.3f}"
    )

    print(
        "Mean absolute change in selected Q-value: "
        f"{q_changes.mean():.5f}"
    )

    print(
        "Maximum absolute change in selected Q-value: "
        f"{q_changes.max():.5f}"
    )

    return baseline_actions


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("Loading checkpoints...")

    initial = load_checkpoint(INITIAL)
    final = load_checkpoint(FINAL)

    # Use identical states for both checkpoints
    observations, masks = get_test_states(final)

    print(f"Test states: {len(observations)}")
    print(f"Noise samples: {NUM_NOISE_SAMPLES}")

    initial_actions = diagnose(
        "INITIAL CHECKPOINT",
        initial,
        observations,
        masks,
    )

    final_actions = diagnose(
        "FINAL CHECKPOINT",
        final,
        observations,
        masks,
    )

    policy_change = np.mean(
        initial_actions != final_actions
    )

    print("\n=== DETERMINISTIC POLICY CHANGE ===")
    print(
        "Different actions between checkpoints: "
        f"{100 * policy_change:.2f}%"
    )


if __name__ == "__main__":
    main()

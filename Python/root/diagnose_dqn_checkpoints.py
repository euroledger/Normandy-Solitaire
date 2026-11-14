
from pathlib import Path
import gc
import random
import contextlib
import os

import numpy as np
import torch

from core.ai.dqn.normandy_env import NormandyEnv
from core.ai.dqn.dqn_network import DQNNetwork
from core.ai.dqn.noisy.dqn_noisy_network import DQNNoisyNetwork


ROOT = Path("core/ai/dqn/checkpoints")
NUM_GAMES = 1000
BASE_SEED = 42000

CHECKPOINTS = [
    ("Point A", ROOT / "normandy_21.pt", False),
    ("Pre-conversion", ROOT / "normandy_25.pt", False),
    ("Noisy initial", ROOT / "noisy/noisy_initial.pt", True),
    ("Noisy final", ROOT / "noisy/noisy_6.pt", True),
]


def load_network(path, noisy, observation_size, action_size):
    print(f"Loading {path.name}...", flush=True)

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )

    if noisy:
        network = DQNNoisyNetwork(
            observation_size, action_size
        )
    else:
        network = DQNNetwork(
            observation_size, action_size
        )

    network.load_state_dict(
        checkpoint["policy_net"], strict=True
    )

    del checkpoint
    gc.collect()

    network.eval()

    if noisy:
        network.disable_noise()

    return network


@torch.inference_mode()
def choose_action(network, observation, action_mask):
    state = torch.as_tensor(
        np.asarray(observation, dtype=np.float32)
    ).unsqueeze(0)

    mask = torch.as_tensor(
        np.asarray(action_mask, dtype=bool)
    )

    q_values = network(state).squeeze(0)
    q_values = q_values.masked_fill(
        ~mask, float("-inf")
    )

    return int(q_values.argmax().item())


def evaluate(label, path, noisy, observation_size, action_size):
    network = load_network(
        path, noisy, observation_size, action_size
    )

    turns = []
    wins = 0

    for game in range(NUM_GAMES):
        seed = BASE_SEED + game

        # Seed the random generators used by the game
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        env = NormandyEnv()

        try:
            with open(os.devnull, "w") as null:
                with contextlib.redirect_stdout(null):
                    observation, info = env.reset(seed=seed)

                    while True:
                        action = choose_action(
                            network,
                            observation,
                            info["action_mask"],
                        )

                        observation, reward, terminated, truncated, info = (
                            env.step(action)
                        )

                        if terminated or truncated:
                            break

            turn = int(info["turn_reached"])
            won = info["result"] == "won"

            turns.append(turn)
            wins += int(won)

        finally:
            env.close()

        if (game + 1) % 200 == 0:
            print(
                f"{label}: {game + 1}/{NUM_GAMES} "
                f"games, wins={wins}",
                flush=True,
            )

    turns = np.asarray(turns)

    result = {
        "name": label,
        "games": NUM_GAMES,
        "wins": wins,
        "win_rate": 100 * wins / NUM_GAMES,
        "avg_turn": float(turns.mean()),
        "turn30": 100 * float(np.mean(turns >= 30)),
        "turn40": 100 * float(np.mean(turns >= 40)),
        "early": 100 * float(np.mean(turns < 10)),
    }

    del network
    gc.collect()

    return result


def main():
    torch.set_num_threads(1)

    # Silence network construction and environment setup
    with open(os.devnull, "w") as null:
        with contextlib.redirect_stdout(null):
            env = NormandyEnv()
            observation_size = env.observation_space.shape[0]
            action_size = env.action_space.n
            env.close()

    results = []

    for label, path, noisy in CHECKPOINTS:
        if not path.exists():
            raise FileNotFoundError(path)

        result = evaluate(
            label,
            path,
            noisy,
            observation_size,
            action_size,
        )
        results.append(result)

    print("\n=== CHECKPOINT COMPARISON ===")
    print(
        f"{'Checkpoint':<18}"
        f"{'Wins':>7}"
        f"{'Win %':>10}"
        f"{'Avg turn':>12}"
        f"{'30+ %':>10}"
        f"{'40+ %':>10}"
        f"{'<10 %':>10}"
    )

    for r in results:
        print(
            f"{r['name']:<18}"
            f"{r['wins']:>7}"
            f"{r['win_rate']:>10.2f}"
            f"{r['avg_turn']:>12.2f}"
            f"{r['turn30']:>10.2f}"
            f"{r['turn40']:>10.2f}"
            f"{r['early']:>10.2f}"
        )


if __name__ == "__main__":
    main()

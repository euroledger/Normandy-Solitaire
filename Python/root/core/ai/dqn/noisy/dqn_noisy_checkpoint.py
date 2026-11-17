
from pathlib import Path

import torch


def load_noisy_checkpoint(
    trainer,
    path="core/ai/dqn/checkpoints/noisy/noisy_initial.pt",
):
    # Restore a Noisy DQN checkpoint without changing
    # the original DQN checkpoint-loading implementation.

    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(path)

    # Load only checkpoints created by our own trusted code.
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    if checkpoint.get("version") != 1:
        raise ValueError("Unsupported checkpoint version")

    if checkpoint.get("network_architecture") != "noisy":
        raise ValueError("Not a Noisy DQN checkpoint")

    agent = trainer.agent

    if checkpoint.get("n_step") != agent.n_step:
        raise ValueError(
            "N-step mismatch. Refusing to discard replay data."
        )

    if checkpoint.get("train_frequency") != agent.train_frequency:
        raise ValueError(
            "Training frequency mismatch."
        )

    saved_buffer = checkpoint.get("replay_buffer")

    if agent.replay_buffer_type != "uniform":
        raise ValueError(
            "This loader currently supports uniform replay only."
        )

    if not isinstance(saved_buffer, list):
        raise ValueError(
            "Checkpoint does not contain a uniform replay buffer."
        )

    if len(saved_buffer) > agent.replay_memory_size:
        raise ValueError(
            "Replay buffer exceeds configured capacity."
        )

    # Restore the two networks independently.
    agent.policy_net.load_state_dict(
        checkpoint["policy_net"],
        strict=True,
    )

    agent.target_net.load_state_dict(
        checkpoint["target_net"],
        strict=True,
    )

    agent.policy_net.to(agent.device)
    agent.target_net.to(agent.device)

    agent.target_net.eval()

    # Restore all saved replay experiences.
    agent.restore_replay_buffer(saved_buffer)

    # Restore training progress.
    agent.training_steps = checkpoint["training_steps"]
    agent.per_beta_steps = checkpoint.get("per_beta_steps", 0)

    trainer.games_completed = checkpoint["games_completed"]
    trainer.total_decisions = checkpoint["total_decisions"]
    trainer.total_training_updates = checkpoint[
        "total_training_updates"
    ]

    # The converted checkpoint has no compatible
    # optimizer state. Keep the newly initialised Adam.
    # optimizer_state = checkpoint.get("optimizer")

    # if optimizer_state is not None:
    #     agent.optimizer.load_state_dict(optimizer_state)
    # SUNDAY 15 NOV
    optimizer_state = checkpoint.get("optimizer")
    if optimizer_state is not None:
        agent.optimizer.load_state_dict(optimizer_state)

    # Apply the configured learning rate after restoring Adam's state.
    for param_group in agent.optimizer.param_groups:
        param_group["lr"] = agent.learning_rate

    print(f"Learning rate: {agent.optimizer.param_groups[0]['lr']}")
    print(f"Mini-batch size: {agent.mini_batch_size}")
    print(f"Target sync interval: {agent.network_sync_rate}")





    # No epsilon-greedy exploration in Noisy DQN.
    agent.epsilon = 0.0
    agent.epsilon_min = 0.0
    agent.epsilon_decay = 1.0

    agent.set_noisy()

    print(f"Noisy checkpoint loaded: {path}")
    print(f"Games completed: {trainer.games_completed:,}")
    print(f"Replay entries: {len(agent.replay_buffer):,}")
    print(f"Training steps: {agent.training_steps:,}")
    print(f"Exploration: NoisyNet (epsilon={agent.epsilon})")

    return path

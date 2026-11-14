
from pathlib import Path

import torch

from core.ai.dqn.dqn_network import DQNNetwork
from core.ai.dqn.noisy.dqn_noisy_network import DQNNoisyNetwork
from core.ai.dqn.noisy.dqn_noisy_linear import NoisyLinear


def transfer_weights(source_network, destination_network):
    # Transfer standard linear weights into NoisyLinear means.
    source_layers = list(source_network.network)
    destination_layers = list(destination_network.network)

    if len(source_layers) != len(destination_layers):
        raise ValueError("Network architectures do not match")

    with torch.no_grad():
        for source, destination in zip(
            source_layers, destination_layers
        ):
            if isinstance(destination, NoisyLinear):
                destination.weight_mu.copy_(source.weight)
                destination.bias_mu.copy_(source.bias)

            elif isinstance(destination, torch.nn.Linear):
                destination.weight.copy_(source.weight)
                destination.bias.copy_(source.bias)

    destination_network.disable_noise()


def verify_conversion(source_network, destination_network):
    # Check exact agreement with noise disabled.
    source_network.eval()
    destination_network.eval()
    destination_network.disable_noise()

    input_size = source_network.network[0].in_features

    torch.manual_seed(12345)
    observations = torch.randn(100, input_size)

    with torch.no_grad():
        original_q = source_network(observations)
        noisy_q = destination_network(observations)

    max_difference = (
        original_q - noisy_q
    ).abs().max().item()

    print(f"Maximum Q-value difference: {max_difference:.10f}")

    if not torch.allclose(
        original_q, noisy_q, rtol=1e-6, atol=1e-5
    ):
        raise RuntimeError("Checkpoint conversion verification failed")

    print("Network conversion verified")


def convert_checkpoint(source_path, destination_path):
    source_path = Path(source_path)
    destination_path = Path(destination_path)

    if not source_path.is_file():
        raise FileNotFoundError(source_path)

    if destination_path.exists():
        raise FileExistsError(
            f"Destination already exists: {destination_path}"
        )

    # Only load checkpoints from a trusted local source.
    checkpoint = torch.load(
        source_path,
        map_location="cpu",
        weights_only=False,
    )

    if checkpoint.get("version") != 1:
        raise ValueError("Unsupported source checkpoint version")

    policy_state = checkpoint["policy_net"]
    target_state = checkpoint["target_net"]

    input_size = policy_state["network.0.weight"].shape[1]
    action_size = policy_state["network.6.weight"].shape[0]

    # Construct original networks.
    original_policy = DQNNetwork(input_size, action_size)
    original_target = DQNNetwork(input_size, action_size)

    original_policy.load_state_dict(policy_state)
    original_target.load_state_dict(target_state)

    # Construct separate noisy networks.
    noisy_policy = DQNNoisyNetwork(input_size, action_size)
    noisy_target = DQNNoisyNetwork(input_size, action_size)

    # Transfer and verify policy network.
    transfer_weights(original_policy, noisy_policy)
    print("Policy network:")
    verify_conversion(original_policy, noisy_policy)

    # Transfer and verify target network independently.
    transfer_weights(original_target, noisy_target)
    print("Target network:")
    verify_conversion(original_target, noisy_target)

    # Preserve checkpoint metadata and replay experiences.
    checkpoint["policy_net"] = noisy_policy.state_dict()
    checkpoint["target_net"] = noisy_target.state_dict()

    # The optimizer must be reconstructed for the new parameters.
    checkpoint["optimizer"] = None

    # Identify the converted architecture.
    checkpoint["network_architecture"] = "noisy"
    checkpoint["noisy_layers"] = "last_two"
    checkpoint["noisy_conversion_source"] = source_path.name

    # Preserve all other fields, including the replay buffer.
    destination_path.parent.mkdir(
        parents=True, exist_ok=True
    )

    # Exclusive creation prevents accidental overwriting.
    with destination_path.open("xb") as output:
        torch.save(checkpoint, output)

    print()
    print("Conversion completed successfully")
    print(f"Games completed: {checkpoint['games_completed']}")
    print(f"Replay entries: {len(checkpoint['replay_buffer'])}")
    print(f"Saved to: {destination_path}")


if __name__ == "__main__":
    convert_checkpoint(
        source_path=(
            "core/ai/dqn/checkpoints/normandy_25.pt"
        ),
        destination_path=(
            "core/ai/dqn/checkpoints/noisy/noisy_initial.pt"
        ),
    )

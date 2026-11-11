
import numpy as np
import torch

from core.ai.dqn.dqn_trainer import DQNTrainer

# Change this to the actual checkpoint saved by your current run.
CHECKPOINT = "core/ai/dqn/checkpoints/normandy_8.pt"


def main():
    # Load the saved checkpoint directly for independent inspection.
    saved = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    assert isinstance(saved["replay_buffer"], dict), (
        "Checkpoint does not contain the new PER buffer format"
    )

    # Create a fresh trainer and restore the checkpoint's components.
    trainer = DQNTrainer()
    agent = trainer.agent

    agent.policy_net.load_state_dict(saved["policy_net"])
    agent.target_net.load_state_dict(saved["target_net"])
    agent.optimizer.load_state_dict(saved["optimizer"])
    agent.restore_priority_replay_buffer(saved["replay_buffer"])

    agent.epsilon = saved["epsilon"]
    agent.training_steps = saved["training_steps"]
    agent.per_beta_steps = saved["per_beta_steps"]

    restored = agent.replay_buffer
    original = saved["replay_buffer"]

    # Verify that PER state survived serialization.
    assert len(restored) == original["size"]
    assert restored.position == original["position"]

    np.testing.assert_array_equal(
        restored.sum_tree,
        original["sum_tree"],
    )
    np.testing.assert_array_equal(
        restored.min_tree,
        original["min_tree"],
    )

    assert agent.per_beta_steps == saved["per_beta_steps"]

    # Verify that the annealing schedule resumes at the saved value.
    expected_beta = (
        agent.per_beta_initial
        + min(1.0, agent.per_beta_steps / agent.per_beta_anneal_steps)
        * (agent.per_beta_final - agent.per_beta_initial)
    )
    assert abs(agent.current_per_beta() - expected_beta) < 1e-12

    print("PER CHECKPOINT ROUND-TRIP PASSED")
    print(f"Saved game:        {saved['games_completed']:,}")
    print(f"Replay buffer:     {len(restored):,} experiences")
    print(f"Buffer position:   {restored.position:,}")
    print(f"PER alpha:         {agent.per_alpha:.3f}")
    print(f"PER beta:          {agent.current_per_beta():.6f}")
    print(f"PER beta steps:    {agent.per_beta_steps:,}")


if __name__ == "__main__":
    main()

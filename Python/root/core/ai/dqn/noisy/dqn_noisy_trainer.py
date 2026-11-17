# Noisy DQN training variant for Normandy: Solitaire.
# Based on the existing DQNTrainer loop, with isolated checkpoints and logs.

import re
from pathlib import Path
import contextlib
import os
from time import perf_counter
from functools import partial
import multiprocessing
import numpy as np
import torch
from gymnasium.vector import AsyncVectorEnv, AutoresetMode

from core.ai.dqn.dqn_trainer import DQNTrainer, make_silent_normandy_env
from core.ai.dqn.dqn_n_step import NStepTransitionCollector
from core.ai.dqn.normandy_env import NormandyEnv
from core.ai.dqn.dqn_agent import load_hyperparameters
from core.ai.dqn.midgame_config import MIDGAME_CONFIG
from core.ai.dqn.noisy.dqn_noisy_network import DQNNoisyNetwork
from core.ai.dqn.noisy.dqn_noisy_agent import NoisyDQNAgent
from core.ai.dqn.noisy.dqn_noisy_checkpoint import load_noisy_checkpoint
import json
from pathlib import Path


def latest_noisy_checkpoint():
    checkpoint_dir = Path("core/ai/dqn/checkpoints/noisy")

    checkpoints = list(checkpoint_dir.glob("noisy_*.pt"))

    if not checkpoints:
        raise FileNotFoundError(
            f"No Noisy DQN checkpoints found in {checkpoint_dir}"
        )

    return max(
        checkpoints,
        key=lambda p: p.stat().st_mtime
    )
class NoisyDQNTrainer(DQNTrainer):
    CHECKPOINT_DIR = Path("core/ai/dqn/checkpoints/noisy")
    LOG_DIR = Path("core/ai/dqn/logs/noisy")
    INITIAL_CHECKPOINT = CHECKPOINT_DIR / "noisy_initial.pt"

    def __init__(self, env=None, agent=None):
        if env is None:
            env = NormandyEnv()
        if agent is None:
            observation_size = env.observation_space.shape[0]
            action_size = env.action_space.n
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            network = DQNNoisyNetwork(observation_size, action_size).to(device)
            agent = NoisyDQNAgent(network, device)
        if not isinstance(agent, NoisyDQNAgent):
            raise TypeError("NoisyDQNTrainer requires NoisyDQNAgent")
        super().__init__(env=env, agent=agent)
        self.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        self.LOG_DIR.mkdir(parents=True, exist_ok=True)

    def _safe_checkpoint_path(self, path):
        path = Path(path).resolve()
        if path.parent != self.CHECKPOINT_DIR.resolve():
            raise ValueError("Checkpoint must be inside the Noisy directory")
        return path

    def load_checkpoint(self, path=None):
        # Explicit file loads exactly that file; no silent fallback.
        # With no path, resume the latest numbered checkpoint or initial.
        if path is None:
            numbered = []
            for candidate in self.CHECKPOINT_DIR.glob("noisy_*.pt"):
                match = re.fullmatch(r"noisy_(\d+)\.pt", candidate.name)
                if match:
                    numbered.append((int(match.group(1)), candidate))
            if numbered:
                path = max(numbered, key=lambda pair: pair[0])[1]
            else:
                path = self.INITIAL_CHECKPOINT
        path = self._safe_checkpoint_path(path)
        return load_noisy_checkpoint(self, path)

    def save_checkpoint(self, path=None):
        # The argument may designate the directory, but not an overwrite target.
        if path is not None:
            self._safe_checkpoint_path(path)
        if self.agent.replay_buffer_type != "uniform":
            raise ValueError("Noisy checkpoint loader supports uniform replay only")
        checkpoint = {
            "version": 1,
            "network_architecture": "noisy",
            "noisy_layers": "last_two",
            "n_step": self.agent.n_step,
            "train_frequency": self.agent.train_frequency,
            "policy_net": self.agent.policy_net.state_dict(),
            "target_net": self.agent.target_net.state_dict(),
            "optimizer": self.agent.optimizer.state_dict(),
            "replay_buffer": list(self.agent.replay_buffer.buffer),
            "epsilon": 0.0,
            "training_steps": self.agent.training_steps,
            "per_beta_steps": self.agent.per_beta_steps,
            "games_completed": self.games_completed,
            "total_decisions": self.total_decisions,
            "total_training_updates": self.total_training_updates,
        }
        existing_numbers = []
        for file in self.CHECKPOINT_DIR.glob("noisy_*.pt"):
            match = re.fullmatch(r"noisy_(\d+)\.pt", file.name)
            if match:
                existing_numbers.append(int(match.group(1)))
        number = max(existing_numbers, default=0) + 1
        while True:
            destination = self.CHECKPOINT_DIR / f"noisy_{number}.pt"
            try:
                with destination.open("xb") as stream:
                    try:
                        torch.save(checkpoint, stream)
                    except Exception:
                        stream.close()
                        destination.unlink(missing_ok=True)
                        raise
                break
            except FileExistsError:
                number += 1
        print(f"Noisy checkpoint saved: {destination} (game {self.games_completed})")
        return destination

    def cleanup_old_checkpoints(self, path=None):
        # Deliberately retain all numbered Noisy checkpoints.
        return None

    def log_game(self, result, path=None):
        # Ignore caller-supplied paths to prevent writing standard DQN logs.
        return super().log_game(
            result, path=self.LOG_DIR / "training_history.csv"
        )

    def log_evaluation(self, training_games, result, path=None):
        return super().log_evaluation(
            training_games,
            result,
            path=self.LOG_DIR / "evaluation_history.csv",
        )

    def evaluate_policy(self):
        # Run one deterministic game without changing the training policy.
        # This uses a separate environment instance in the main process.
        # Game state is process-local: vector workers run in child processes.
        self.agent.set_deterministic()
        evaluation_env = NormandyEnv()
        try:
            with open(os.devnull, "w", encoding="utf-8") as null_output:
                with contextlib.redirect_stdout(null_output):
                    observation, info = evaluation_env.reset()
                    while True:
                        action = self.agent.select_action(
                            observation, info["action_mask"]
                        )
                        observation, _, terminated, truncated, info = (
                            evaluation_env.step(action)
                        )
                        if terminated or truncated:
                            return {
                                "turn_reached": int(info["turn_reached"]),
                                "result": info["result"],
                            }
        finally:
            evaluation_env.close()
            self.agent.set_noisy()

    def train(
        self,
        num_games=10,
        resume=False,
        checkpoint_path=None,
        save_interval=5000,
        overwrite=False,
        num_workers=4,
        progress_interval=200,
        evaluation_interval=500,
        evaluation_games=1,
        greedy_mode=False,
        save_checkpoints=True
    ):
        if num_games < 1:
            raise ValueError("num_games must be at least 1")
        if num_workers < 1:
            raise ValueError("num_workers must be at least 1")
        if progress_interval < 1:
            raise ValueError("progress_interval must be at least 1")
        if evaluation_interval < 1:
            raise ValueError("evaluation_interval must be at least 1")
        if evaluation_games < 1:
            raise ValueError("evaluation_games must be at least 1")
        # A Noisy run always writes to its own checkpoint directory.
        # Explicit checkpoint_path is used only as a restoration source.
        checkpointing = bool(save_checkpoints)
        if resume:
            self.load_checkpoint(checkpoint_path)
            print(f"Replay buffer type: {type(self.agent.replay_buffer).__name__}")
            print(f"Replay buffer size: {len(self.agent.replay_buffer):,}")
            
            # self.agent.report_noise_parameters()
            
        elif checkpoint_path is not None:
            raise ValueError("checkpoint_path requires resume=True")
        if overwrite:
            raise ValueError("Noisy checkpoints are never overwritten")

        # self.agent.epsilon_decay = 1.0
        self.agent.set_noisy()
         # Override checkpoint epsilon TEMP
        # self.agent.epsilon = 0.15
        print(f"Epsilon restarted at: {self.agent.epsilon:.4f}")

        start_turn = 1
        previous_end_turn = None
        if MIDGAME_CONFIG["enabled"]:
            snapshot_path = Path(MIDGAME_CONFIG["fixed_snapshot"])
            with snapshot_path.open("r", encoding="utf-8") as f:
                snapshot_metadata = json.load(f)["metadata"]
            start_turn = int(snapshot_metadata["start_turn"])  # noqa: F841
            previous_end_turn = int(snapshot_metadata["original_final_turn"])
            self.previous_end_turn = previous_end_turn
        _ = self.games_completed
        if greedy_mode:
            print("NOISY MODE: epsilon = 0; noisy exploration remains active")
        results = []
        evaluation_results = []
        turns_in_window = []
        window_start_game = self.games_completed + 1
        run_start = perf_counter()
        
        # QUACK Training performance profiling
        profile_train_seconds = 0.0
        profile_train_calls = 0
        # QUACK END
        
        last_save_game = 0
        updates_before = self.total_training_updates
        decisions_before = self.total_decisions
        completed = 0
        next_evaluation = evaluation_interval
        episode_rewards = np.zeros(
            num_workers, dtype=np.float64
        )
        episode_decisions = np.zeros(
            num_workers, dtype=np.int64
        )
        episode_losses = [
            [] for _ in range(num_workers)
        ]
        needs_reset = np.zeros(
            num_workers, dtype=bool
        )
        n_step_collectors = [
            NStepTransitionCollector(
                n_step=self.agent.n_step,
                gamma=self.agent.discount_factor,
            )
            for _ in range(num_workers)
        ]
        hyperparameters = load_hyperparameters("normandy_v1")
        game_logging_config = hyperparameters.get(
            "game_logging", {"enabled": False}
        )
        error_logging_config = hyperparameters.get(
            "error_logging", {"enabled": False}
        )
        log_manager = multiprocessing.Manager()
        shared_count = log_manager.Value("i", 0)
        shared_lock = log_manager.Lock()
        env_factory = partial(
            make_silent_normandy_env,
            game_config=game_logging_config,
            error_config=error_logging_config,
            shared_count=shared_count,
            shared_lock=shared_lock,
        )
        env = AsyncVectorEnv(
            [env_factory for _ in range(num_workers)],
            autoreset_mode=AutoresetMode.NEXT_STEP,
        )
        try:
            observations, info = env.reset()
            while completed < num_games:
                masks = info["action_mask"]
                actions = self.agent.select_actions(
                    observations,
                    masks,
                    needs_reset,
                )
                (
                    next_observations,
                    rewards,
                    terminated,
                    truncated,
                    next_info,
                ) = env.step(actions)
                next_masks = next_info["action_mask"]
                for i in range(num_workers):
                    if needs_reset[i]:
                        needs_reset[i] = False
                        episode_rewards[i] = 0.0
                        episode_decisions[i] = 0
                        episode_losses[i] = []
                        continue
                    done = bool(
                        terminated[i] or truncated[i]
                    )
                    completed_transitions = n_step_collectors[i].add(
                        observation=observations[i],
                        action_id=int(actions[i]),
                        reward=float(rewards[i]),
                        next_observation=next_observations[i],
                        terminated=done,
                        next_action_mask=next_masks[i],
                    )
                    for transition in completed_transitions:
                        (
                            observation,
                            action_id,
                            reward,
                            next_observation,
                            transition_done,
                            next_action_mask,
                            n_steps,
                        ) = transition
                        self.agent.replay_buffer.add(
                            observation=observation,
                            action_id=action_id,
                            reward=reward,
                            next_observation=next_observation,
                            terminated=transition_done,
                            next_action_mask=next_action_mask,
                            n_steps=n_steps,
                        )
                    # loss = None
                    # if (
                    #     (self.total_decisions + 1)
                    #     % self.agent.train_frequency
                    #     == 0
                    # ):
                    #     loss = self.agent.train_step()
                    # if loss is not None:
                    #     episode_losses[i].append(loss)
                    #     self.total_training_updates += 1
                    
                    loss = None
                    if (
                        (self.total_decisions + 1)
                        % self.agent.train_frequency
                        == 0
                    ):
                        # QUACK Training update timer
                        train_start = perf_counter()
                        # QUACK END

                        loss = self.agent.train_step()

                        # QUACK Training update timer
                        profile_train_seconds += perf_counter() - train_start
                        profile_train_calls += 1
                        # QUACK END

                    if loss is not None:
                        episode_losses[i].append(loss)
                        self.total_training_updates += 1
                    # No epsilon-greedy decay: exploration is parameter noise.
                    episode_rewards[i] += float(rewards[i])
                    episode_decisions[i] += 1
                    self.total_decisions += 1
                    if done:
                        needs_reset[i] = True
                        if completed >= num_games:
                            continue
                        completed += 1
                        self.games_completed += 1
                        game_result = None
                        if "result" in next_info:
                            game_result = next_info["result"][i]
                            turn_reached = int(next_info["turn_reached"][i])
                        cards_in_drawn_pile = int(
                            next_info.get(
                                "cards_in_drawn_pile",
                                next_info["turn_reached"]
                            )[i]
                        )
                        result = {
                            "game": self.games_completed,
                            "reward": float(
                                episode_rewards[i]
                            ),
                            "decisions": int(
                                episode_decisions[i]
                            ),
                            "result": game_result,
                            "epsilon": self.agent.epsilon,
                            "average_loss": (
                                float(np.mean(episode_losses[i]))
                                if episode_losses[i]
                                else None
                            ),
                            "turn_reached": turn_reached,
                            "cards_in_drawn_pile": cards_in_drawn_pile,
                        }
                        results.append(result)
                        turns_in_window.append(turn_reached)
                        if len(turns_in_window) == progress_interval:
                            print()
                            print("========== DQN LEARNING PROGRESS ==========")
                            print(
                                f"Games:               "
                                f"{window_start_game}–{self.games_completed}"
                            )
                            print(
                                f"Average turn reached: "
                                f"{np.mean(turns_in_window):.2f}"
                            )
                            early_defeat_pct = 100.0 * sum(
                                r["result"] == "lost" and r["turn_reached"] < 10
                                for r in results[-len(turns_in_window):]
                            ) / len(turns_in_window)
                            print(f"Loss before turn 10: {early_defeat_pct:.1f}%")
                            n = len(turns_in_window)
                            pct_20 = 100 * sum(t >= 20 for t in turns_in_window) / n
                            pct_30 = 100 * sum(t >= 30 for t in turns_in_window) / n
                            pct_40 = 100 * sum(t >= 40 for t in turns_in_window) / n
                            print(f"Reached turn 20+:     {pct_20:.1f}%")
                            print(f"Reached turn 30+:     {pct_30:.1f}%")
                            print(f"Reached turn 40+:     {pct_40:.1f}%")
                            
                            # Victory percentage in the current progress window.
                            window_results = results[-n:]
                            wins = sum(r["result"] == "won" for r in window_results)
                            win_pct = 100.0 * wins / n
                            print(f"Games won:            {wins} ({win_pct:.1f}%)")

                            if self.agent.grad_norm_count > 0:
                                avg_grad_norm = (
                                    self.agent.grad_norm_sum /
                                    self.agent.grad_norm_count
                                )
                                clipped_pct = (
                                    100.0 * self.agent.grad_norm_clipped_count
                                    / self.agent.grad_norm_count
                                )
                                print(f"Gradient norm AVG:    {avg_grad_norm:.2f}")
                                print(f"Gradient norm MAX:    {self.agent.grad_norm_max:.2f}")
                                print(f"Gradient samples:     {self.agent.grad_norm_count:,}")
                                print(f"Gradient clipped:     {clipped_pct:.1f}%")
                                self.agent.grad_norm_sum = 0.0
                                self.agent.grad_norm_max = 0.0
                                self.agent.grad_norm_count = 0
                                self.agent.grad_norm_clipped_count = 0
                            if self.agent.replay_buffer_type == "priority":
                                print(f"PER alpha:            {self.agent.per_alpha:.3f}")
                                print(f"PER beta:             {self.agent.current_per_beta():.6f}")
                                print(f"PER beta steps:       {self.agent.per_beta_steps:,}")
                            if MIDGAME_CONFIG["enabled"]:
                                original_end = self.previous_end_turn
                                pct_exceeded = 100 * sum(
                                    t > original_end for t in turns_in_window
                                ) / n
                                print(
                                    f"Exceeded original end ({original_end}): "
                                    f"{pct_exceeded:.1f}%"
                                )
                            print("===========================================")
                            print(
                                f"Maximum turn reached: "
                                f"{max(turns_in_window)}"
                            )
                            print("===========================================")
                            print()
                            turns_in_window.clear()
                            window_start_game = self.games_completed + 1
                        self.log_game(result)
                        WHITE_ON_RED = "\033[97;41m"
                        WHITE_ON_GREEN = "\033[97;42m"
                        WHITE_ON_BLUE = "\033[97;44m"
                        RESET = "\033[0m"
                        if game_result == "won":
                            row_color = WHITE_ON_BLUE
                        elif turn_reached < 10:
                            row_color = WHITE_ON_RED
                        elif turn_reached >= 40:
                            row_color = WHITE_ON_GREEN
                        else:
                            row_color = RESET
                        print(
                            row_color
                            + f"Game {self.games_completed:6d} | "
                            f"Worker: {i} | "
                            f"Result: {game_result!s:6s} | "
                            f"Turns: {turn_reached:2d} | "
                            f"Previous End: {previous_end_turn if previous_end_turn is not None else '-'} | "
                            f"Cards Drawn: {cards_in_drawn_pile:2d} | "
                            f"Reward: {episode_rewards[i]:6.1f} | "
                            f"Decisions: {episode_decisions[i]:4d} | "
                            f"Epsilon: {self.agent.epsilon:.4f}"
                            + RESET,
                            flush=True,
                        )
                observations = next_observations
                info = next_info
                if completed >= next_evaluation:
                    next_evaluation += evaluation_interval
                    for _ in range(evaluation_games):
                        evaluation = self.evaluate_policy()
                        evaluation["training_games"] = self.games_completed
                        evaluation_results.append(evaluation)
                        self.log_evaluation(
                            self.games_completed,
                            evaluation,
                        )
                        print()
                        print("========== DQN POLICY EVALUATION ==========")
                        print(f"Training games:  {self.games_completed}")
                        print(f"Evaluation turn: {evaluation['turn_reached']}")
                        print(f"Result:          {evaluation['result']}")
                        print("Epsilon:         0.0")
                        print("===========================================")
                if (
                    checkpointing
                    and completed - last_save_game >= save_interval
                    and save_checkpoints
                ):
                    self.save_checkpoint()
                    last_save_game = completed
                    self.print_checkpoint_performance(
                        results=results,
                        progress_interval=progress_interval,
                        run_start=run_start,
                        decisions_before=decisions_before,
                        updates_before=updates_before,
                        num_workers=num_workers,
                    )
        finally:
            try:
                env.close()
            finally:
                try:
                    self.close_evaluation_pool(force=True)
                finally:
                    if log_manager is not None:
                        log_manager.shutdown()
        # elapsed = perf_counter() - run_start
        # games_won = sum(
        #     1 for result in results
        #     if result["result"] == "won"
        # )
        
        elapsed = perf_counter() - run_start

        # QUACK Training profiling report
        print()
        print("========== TRAINING UPDATE PROFILE ==========")
        print(f"Total elapsed:  {elapsed:.2f} seconds")
        print(f"Training time:  {profile_train_seconds:.2f} seconds")
        print(f"Training calls: {profile_train_calls:,}")

        if profile_train_calls > 0:
            print(
                f"Average update: "
                f"{1000 * profile_train_seconds / profile_train_calls:.2f} ms"
            )

        if elapsed > 0:
            print(
                f"Training share: "
                f"{100 * profile_train_seconds / elapsed:.1f}%"
            )

        print("=============================================")
        # QUACK END

        games_won = sum(
            1 for result in results
            if result["result"] == "won"
        )
        win_rate = 100 * games_won / len(results) if results else 0.0
        games_per_second = (
            len(results) / elapsed
            if elapsed > 0
            else 0.0
        )
        decisions_this_run = (
            self.total_decisions - decisions_before
        )
        decisions_per_second = (
            decisions_this_run / elapsed
            if elapsed > 0
            else 0.0
        )
        updates_this_run = (
            self.total_training_updates - updates_before
        )
        updates_per_second = (
            updates_this_run / elapsed
            if elapsed > 0
            else 0.0
        )
        print()
        print("=" * 65)
        print("DQN TRAINING PERFORMANCE")
        print("=" * 65)
        print(f"Parallel workers:      {num_workers}")
        print(f"Train frequency:       {self.agent.train_frequency}")
        print(f"Games completed:       {len(results)}")
        print(f"Games won:             {games_won}")
        print(f"Win rate:              {win_rate:.3f}%")
        print(f"Cumulative games:      {self.games_completed}")
        print(f"Cumulative decisions:  {self.total_decisions}")
        print(f"Cumulative updates:    {self.total_training_updates}")
        print(f"Elapsed time:          {elapsed:.2f} seconds")
        print(f"Games/second:          {games_per_second:.2f}")
        print(f"Decisions/second:      {decisions_per_second:.2f}")
        print(f"Training updates:      {updates_this_run}")
        print(f"Updates/second:        {updates_per_second:.2f}")
        print(f"Final epsilon:         {self.agent.epsilon:.4f}")
        print(f"PyTorch CPU threads: {torch.get_num_threads()}")
        print("=" * 65)
        if evaluation_results:
            print()
            print("========== DQN POLICY EVALUATION SUMMARY ==========")
            print()
            print(f"{'Training Games':<20} {'Turn Reached':>15} {'Result':>10}")
            for evaluation in evaluation_results:
                print(
                    f"{evaluation['training_games']:<20} "
                    f"{evaluation['turn_reached']:>15} "
                    f"{evaluation['result']:>10}"
                )
            turns = [
                evaluation["turn_reached"]
                for evaluation in evaluation_results
            ]
            print()
            print(f"Evaluation games:   {len(turns)}")
            print(f"Average turns:      {np.mean(turns):.2f}")
            print(f"First evaluation:   {turns[0]}")
            print(f"Last evaluation:    {turns[-1]}")
            print(f"Change:             {turns[-1] - turns[0]:+d} turns")
            print("===================================================")
        if checkpointing and save_checkpoints and last_save_game != completed:
            self.save_checkpoint()
            self.print_checkpoint_performance(
                results=results,
                progress_interval=progress_interval,
                run_start=run_start,
                decisions_before=decisions_before,
                updates_before=updates_before,
                num_workers=num_workers,
            )
            # Preserve all numbered Noisy checkpoints for rollback.
        return results


if __name__ == "__main__":
    torch.set_num_threads(4)
    trainer = NoisyDQNTrainer()
    trainer.train(
        num_games=10000,
        resume=True,
        save_interval=10000,
        num_workers=8,
        progress_interval=200,
        evaluation_interval=999999999,
        evaluation_games=1,
        save_checkpoints=True,
        checkpoint_path=latest_noisy_checkpoint(),
        # checkpoint_path=(
        #     "core/ai/dqn/checkpoints/noisy/noisy_30.pt"
        # ),
    )

# if __name__ == "__main__":
#     torch.set_num_threads(3)

#     trainer = NoisyDQNTrainer()

#     trainer.train(
#         num_games=10,
#         resume=True,
#         checkpoint_path=(
#             "core/ai/dqn/checkpoints/noisy/noisy_initial.pt"
#         ),
#         save_interval=10,
#         num_workers=8,
#         progress_interval=10,
#         evaluation_interval=999999999,
#         evaluation_games=1,
#         save_checkpoints=True,
#     )

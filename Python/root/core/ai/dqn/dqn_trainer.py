import csv
from pathlib import Path
import os
import contextlib
from time import perf_counter
from core.ai.dqn.normandy_env import NormandyEnv
from core.ai.dqn.dqn_network import DQNNetwork
from core.ai.dqn.dqn_agent import DQNAgent
from core.ai.dqn.debug import debug_pants
import numpy as np
import torch
from gymnasium.vector import AsyncVectorEnv, AutoresetMode
import multiprocessing
import sys
import pickle


def make_silent_normandy_env():

    if multiprocessing.current_process().name != "MainProcess":
        sys.stdout = open(os.devnull, "w", encoding="utf-8")
        sys.stderr = open(os.devnull, "w", encoding="utf-8")

    return NormandyEnv()


def run_greedy_evaluation(policy_state, observation_size, action_size):
    """
    Play one Normandy game with epsilon = 0.

    Runs in a separate process to isolate GlobalGameState.
    Does not train, update replay memory or modify training counters.
    """

    import torch
    import numpy as np

    sys.stdout = open(os.devnull, "w", encoding="utf-8")
    torch.set_num_threads(1)

    network = DQNNetwork(
        observation_size,
        action_size,
    )

    network.load_state_dict(policy_state)
    network.eval()

    env = NormandyEnv()

    try:
        with torch.no_grad():

            observation, info = env.reset()

            done = False

            while not done:

                mask = info["action_mask"]

                legal_actions = np.flatnonzero(mask)

                if len(legal_actions) == 0:
                    raise RuntimeError(
                        "No legal actions during evaluation"
                    )

                state = torch.as_tensor(
                    observation,
                    dtype=torch.float32,
                ).unsqueeze(0)

                q_values = network(state).squeeze(0).numpy()

                q_values[mask == 0] = -np.inf

                action = int(np.argmax(q_values))

                observation, reward, terminated, truncated, info = (
                    env.step(action)
                )

                done = terminated or truncated

        return {
            "turn_reached": int(info["turn_reached"]),
            "result": info["result"],
        }

    finally:
        env.close()


class DQNTrainer:

    def __init__(self, env=None, agent=None):
        # Create environment
        self.env = env if env is not None else NormandyEnv()

        # Create agent using environment dimensions
        if agent is None:
            observation_size = self.env.observation_space.shape[0]
            action_size = self.env.action_space.n

            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )

            network = DQNNetwork(
                observation_size,
                action_size,
            ).to(device)

            agent = DQNAgent(network, device)

        self.agent = agent

        # Training statistics
        self.games_completed = 0
        self.total_decisions = 0
        self.total_training_updates = 0
        self._evaluation_pool = None

    def evaluate_policy(self):
        """
        Run one greedy evaluation game in a separate process.
        """

        observation_size = self.env.observation_space.shape[0]
        action_size = self.env.action_space.n

        policy_state = {
            key: value.detach().cpu().clone()
            for key, value in self.agent.policy_net.state_dict().items()
        }

        # Create one persistent evaluation worker, not a new pool per game.
        if self._evaluation_pool is None:
            self._evaluation_pool = multiprocessing.get_context("spawn").Pool(
                processes=1
            )

        pending = self._evaluation_pool.apply_async(
            run_greedy_evaluation,
            (policy_state, observation_size, action_size),
        )

        try:
            # Avoid an indefinite wait if the evaluation worker fails.
            return pending.get(timeout=180)
        except multiprocessing.TimeoutError as exc:
            self.close_evaluation_pool(force=True)
            raise RuntimeError(
                "Greedy evaluation exceeded 180 seconds; worker terminated"
            ) from exc

    def close_evaluation_pool(self, force=False):
        """Shut down the persistent evaluation process."""
        pool = self._evaluation_pool
        self._evaluation_pool = None
        if pool is None:
            return
        if force:
            pool.terminate()
        else:
            pool.close()
        pool.join()

    def log_evaluation(
        self,
        training_games,
        result,
        path="core/ai/dqn/logs/evaluation_history.csv",
    ):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        new_file = not path.exists() or path.stat().st_size == 0

        with path.open("a", newline="", encoding="utf-8") as file:

            writer = csv.writer(file)

            if new_file:
                writer.writerow([
                    "training_games",
                    "epsilon",
                    "turn_reached",
                    "result",
                ])

            writer.writerow([
                training_games,
                0.0,
                result["turn_reached"],
                result["result"],
            ])

    def run_silently(self, function, *args, **kwargs):
        # Suppress game-engine output
        with open(os.devnull, "w") as null_output, contextlib.redirect_stdout(null_output):
            return function(*args, **kwargs)


    def print_learning_summary(self, results, interval=100):

        if not results:
            return

        print()
        print("===================== DQN LEARNING PROGRESS SUMMARY =====================")
        print()
        print(
            f"{'Games':<20}"
            f"{'Average Turn':>14}"
            f"{'Turn 20+':>12}"
            f"{'Turn 30+':>12}"
            f"{'Turn 40+':>12}"
        )

        print("-" * 70)

        averages = []
        batch_stats = []

        for start in range(0, len(results), interval):

            batch = results[start:start + interval]

            turns = [r["turn_reached"] for r in batch]
            n = len(turns)

            avg_turn = sum(turns) / n

            pct_20 = 100 * sum(t >= 20 for t in turns) / n
            pct_30 = 100 * sum(t >= 30 for t in turns) / n
            pct_40 = 100 * sum(t >= 40 for t in turns) / n

            averages.append(avg_turn)

            batch_stats.append({
                "avg": avg_turn,
                "pct_20": pct_20,
                "pct_30": pct_30,
                "pct_40": pct_40,
                "n": n,
            })

            first_game = batch[0]["game"]
            last_game = batch[-1]["game"]

            game_range = f"{first_game}–{last_game}"

            print(
                f"{game_range:<20}"
                f"{avg_turn:>14.2f}"
                f"{pct_20:>11.1f}%"
                f"{pct_30:>11.1f}%"
                f"{pct_40:>11.1f}%"
            )

        # Overall statistics
        all_turns = [r["turn_reached"] for r in results]
        total_games = len(all_turns)

        overall_avg = sum(all_turns) / total_games
        overall_20 = 100 * sum(t >= 20 for t in all_turns) / total_games
        overall_30 = 100 * sum(t >= 30 for t in all_turns) / total_games
        overall_40 = 100 * sum(t >= 40 for t in all_turns) / total_games

        print("-" * 70)

        print(
            f"{'OVERALL':<20}"
            f"{overall_avg:>14.2f}"
            f"{overall_20:>11.1f}%"
            f"{overall_30:>11.1f}%"
            f"{overall_40:>11.1f}%"
        )

        # Compare first and last batches
        first = batch_stats[0]
        last = batch_stats[-1]

        avg_change = last["avg"] - first["avg"]

        avg_pct_change = (
            100 * avg_change / first["avg"]
            if first["avg"] != 0
            else 0
        )

        change_20 = last["pct_20"] - first["pct_20"]
        change_30 = last["pct_30"] - first["pct_30"]
        change_40 = last["pct_40"] - first["pct_40"]

        print()
        print("======================= CHANGE OVER RUN =======================")
        print()

        print(
            f"{'Metric':<25}"
            f"{'First Batch':>14}"
            f"{'Last Batch':>14}"
            f"{'Change':>14}"
        )

        print("-" * 67)

        print(
            f"{'Average turn':<25}"
            f"{first['avg']:>14.2f}"
            f"{last['avg']:>14.2f}"
            f"{avg_change:>+14.2f}"
        )

        print(
            f"{'Games reaching turn 20':<25}"
            f"{first['pct_20']:>13.1f}%"
            f"{last['pct_20']:>13.1f}%"
            f"{change_20:>+13.1f}pp"
        )

        print(
            f"{'Games reaching turn 30':<25}"
            f"{first['pct_30']:>13.1f}%"
            f"{last['pct_30']:>13.1f}%"
            f"{change_30:>+13.1f}pp"
        )

        print(
            f"{'Games reaching turn 40':<25}"
            f"{first['pct_40']:>13.1f}%"
            f"{last['pct_40']:>13.1f}%"
            f"{change_40:>+13.1f}pp"
        )

        print()
        print(
            f"Average turn relative change: {avg_pct_change:+.1f}%"
        )

        print("===============================================================") 
      
    def log_game(self, result, path="core/ai/dqn/logs/training_history.csv"):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        fields = [
            "game",
            "reward",
            "decisions",
            "result",
            "epsilon",
            "average_loss",
            "turn_reached"
        ]

        new_file = not path.exists() or path.stat().st_size == 0

        with path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fields)

            if new_file:
                writer.writeheader()

            writer.writerow({field: result.get(field) for field in fields})

    def cleanup_old_checkpoints(
        self,
        path="core/ai/dqn/checkpoints/normandy_latest.pt"
    ):
        import re

        checkpoint_dir = Path(path).parent

        numbered_checkpoints = []

        for file in checkpoint_dir.glob("normandy_*.pt"):
            match = re.fullmatch(r"normandy_(\d+)\.pt", file.name)

            if match:
                numbered_checkpoints.append(
                    (int(match.group(1)), file)
                )

        if len(numbered_checkpoints) <= 1:
            return

        numbered_checkpoints.sort(reverse=True)

        # Keep the newest numbered checkpoint.
        latest_number, latest_file = numbered_checkpoints[0]

        print(f"Keeping checkpoint: {latest_file.name}")

        # Delete all older numbered checkpoints.
        for number, file in numbered_checkpoints[1:]:
            try:
                file.unlink()
                print(f"Deleted old checkpoint: {file.name}")

            except OSError as error:
                print(
                    f"WARNING: Could not delete {file.name}: {error}"
                )


    def save_checkpoint(self, path="core/ai/dqn/checkpoints/normandy_latest.pt"):
        import re

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Use the supplied path to determine the checkpoint directory.
        # All new checkpoints use sequential filenames.
        checkpoint_dir = path.parent

        existing_numbers = []

        for file in checkpoint_dir.glob("normandy_*.pt"):
            match = re.fullmatch(r"normandy_(\d+)\.pt", file.name)

            if match:
                existing_numbers.append(int(match.group(1)))

        next_number = max(existing_numbers, default=0) + 1

        # Never overwrite an existing checkpoint.
        while True:
            checkpoint_path = checkpoint_dir / f"normandy_{next_number}.pt"

            try:
                with checkpoint_path.open("xb") as checkpoint_file:

                    checkpoint = {
                        "version": 1,
                        "policy_net": self.agent.policy_net.state_dict(),
                        "target_net": self.agent.target_net.state_dict(),
                        "optimizer": self.agent.optimizer.state_dict(),
                        "replay_buffer": list(self.agent.replay_buffer.buffer),
                        "epsilon": self.agent.epsilon,
                        "training_steps": self.agent.training_steps,
                        "games_completed": self.games_completed,
                        "total_decisions": self.total_decisions,
                        "total_training_updates": self.total_training_updates,
                        "train_frequency": self.agent.train_frequency,
                    }

                    torch.save(checkpoint, checkpoint_file)

                break

            except FileExistsError:
                next_number += 1

            except Exception:
                # Remove an incomplete checkpoint if writing failed.
                try:
                    checkpoint_path.unlink(missing_ok=True)
                except OSError:
                    pass

                raise

        print(
            f"Checkpoint saved: {checkpoint_path} "
            f"(game {self.games_completed})"
        )

        return checkpoint_path

    def load_checkpoint(self, path="core/ai/dqn/checkpoints/normandy_latest.pt"):
        import re

        path = Path(path)
        checkpoint_dir = path.parent

        # Find numbered checkpoints, newest first.
        numbered_checkpoints = []

        for file in checkpoint_dir.glob("normandy_*.pt"):
            match = re.fullmatch(r"normandy_(\d+)\.pt", file.name)

            if match:
                numbered_checkpoints.append(
                    (int(match.group(1)), file)
                )

        numbered_checkpoints.sort(reverse=True)

        candidates = [file for _, file in numbered_checkpoints]

        # Support the original checkpoint during migration.
        legacy_path = checkpoint_dir / "normandy_latest.pt"

        if legacy_path.is_file() and legacy_path not in candidates:
            candidates.append(legacy_path)

        if not candidates:
            raise FileNotFoundError(
                f"No checkpoints found in: {checkpoint_dir}"
            )

        # Try newest checkpoint first.
        for checkpoint_path in candidates:

            try:
                checkpoint = torch.load(
                    checkpoint_path,
                    map_location=self.agent.device,
                    weights_only=False,
                )

                if checkpoint["version"] != 1:
                    raise ValueError("Unsupported checkpoint version")

                if checkpoint["train_frequency"] != self.agent.train_frequency:
                    raise ValueError(
                        "Checkpoint train_frequency does not match configuration"
                    )

                # Restore neural networks.
                self.agent.policy_net.load_state_dict(
                    checkpoint["policy_net"]
                )

                self.agent.target_net.load_state_dict(
                    checkpoint["target_net"]
                )

                self.agent.target_net.eval()

                # Restore optimizer.
                self.agent.optimizer.load_state_dict(
                    checkpoint["optimizer"]
                )

                # Restore replay buffer.
                self.agent.replay_buffer.buffer.clear()
                self.agent.replay_buffer.buffer.extend(
                    checkpoint["replay_buffer"]
                )

                # Restore training state.
                self.agent.epsilon = checkpoint["epsilon"]
                self.agent.training_steps = checkpoint["training_steps"]

                self.games_completed = checkpoint["games_completed"]
                self.total_decisions = checkpoint["total_decisions"]
                self.total_training_updates = checkpoint["total_training_updates"]

                print(
                    f"Checkpoint loaded: {checkpoint_path} "
                    f"(game {self.games_completed}, "
                    f"epsilon {self.agent.epsilon:.4f})"
                )

                return checkpoint_path

            except (
                OSError,
                EOFError,
                RuntimeError,
                ValueError,
                KeyError,
                TypeError,
                pickle.UnpicklingError,
            ) as error:

                print(
                    f"WARNING: Could not load checkpoint "
                    f"{checkpoint_path}: {error}"
                )
                print("Trying previous checkpoint...")

        raise RuntimeError(
            "No valid checkpoint could be loaded."
        )
    
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
        evaluation_games=1
    ):
        # --------------------------------------------------
        # CHECKPOINT INITIALISATION
        # --------------------------------------------------

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
        checkpointing = checkpoint_path is not None

        if checkpointing:
            checkpoint_path = Path(checkpoint_path)

            if resume:
                try:
                    self.load_checkpoint(checkpoint_path)
                except FileNotFoundError:
                    print("No checkpoint found. Starting fresh.")

            elif checkpoint_path.exists() and not overwrite:
                raise FileExistsError(
                    f"Checkpoint already exists: {checkpoint_path}. "
                    "Use resume=True or overwrite=True."
                )

        # --------------------------------------------------
        # INITIALISE TRAINING
        # --------------------------------------------------

        results = []
        evaluation_results = []

        turns_in_window = []
        window_start_game = self.games_completed + 1

        run_start = perf_counter()
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

        # --------------------------------------------------
        # CREATE PARALLEL ENVIRONMENTS
        # --------------------------------------------------

        env = AsyncVectorEnv(
            [
                make_silent_normandy_env
                for _ in range(num_workers)
            ],
            autoreset_mode=AutoresetMode.NEXT_STEP,
        )

        try:

            observations, info = env.reset()

            # --------------------------------------------------
            # MAIN TRAINING LOOP
            # --------------------------------------------------

            while completed < num_games:

                masks = info["action_mask"]

                actions = self.agent.select_actions(
                    observations,
                    masks,
                    needs_reset,
                )

                # ----------------------------------------------
                # EXECUTE PARALLEL ENVIRONMENT STEPS
                # ----------------------------------------------

                (
                    next_observations,
                    rewards,
                    terminated,
                    truncated,
                    next_info,
                ) = env.step(actions)

                next_masks = next_info["action_mask"]

                # ----------------------------------------------
                # PROCESS EACH WORKER
                # ----------------------------------------------

                for i in range(num_workers):

                    # ------------------------------------------
                    # HANDLE AUTOMATIC RESET
                    # ------------------------------------------

                    if needs_reset[i]:

                        needs_reset[i] = False

                        episode_rewards[i] = 0.0
                        episode_decisions[i] = 0
                        episode_losses[i] = []

                        continue

                    # ------------------------------------------
                    # STORE TRANSITION
                    # ------------------------------------------

                    done = bool(
                        terminated[i] or truncated[i]
                    )

                    self.agent.replay_buffer.add(
                        observation=observations[i],
                        action_id=int(actions[i]),
                        reward=float(rewards[i]),
                        next_observation=next_observations[i],
                        terminated=done,
                        next_action_mask=next_masks[i],
                    )

                    # ------------------------------------------
                    # TRAIN DQN
                    # ------------------------------------------

                    loss = None

                    if (
                        (self.total_decisions + 1)
                        % self.agent.train_frequency
                        == 0
                    ):

                        start = perf_counter()

                        loss = self.agent.train_step()

                        duration = perf_counter() - start

                        if duration > 1.0:
                            print(
                                f"SLOW TRAINING UPDATE: "
                                f"Worker {i}, "
                                f"Decision {episode_decisions[i] + 1}, "
                                f"Duration {duration:.2f}s"
                            )

                    if loss is not None:

                        episode_losses[i].append(loss)

                        self.total_training_updates += 1

                    # ------------------------------------------
                    # UPDATE EPSILON
                    # ------------------------------------------

                    self.agent.epsilon = max(
                        self.agent.epsilon_min,
                        self.agent.epsilon
                        * self.agent.epsilon_decay,
                    )

                    # ------------------------------------------
                    # UPDATE DECISION STATISTICS
                    # ------------------------------------------

                    episode_rewards[i] += float(rewards[i])

                    episode_decisions[i] += 1

                    self.total_decisions += 1

                    # ------------------------------------------
                    # GAME COMPLETED
                    # ------------------------------------------

                    if done:

                        needs_reset[i] = True

                        # Do not count more than the requested
                        # number of completed games.

                        if completed >= num_games:
                            continue

                        completed += 1

                        self.games_completed += 1

                        # Gymnasium vector info stores results
                        # as arrays indexed by worker.

                        game_result = None

                        if "result" in next_info:
                            game_result = next_info["result"][i]
                            turn_reached = int(next_info["turn_reached"][i])

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
                            n = len(turns_in_window)

                            pct_20 = 100 * sum(t >= 20 for t in turns_in_window) / n
                            pct_30 = 100 * sum(t >= 30 for t in turns_in_window) / n
                            pct_40 = 100 * sum(t >= 40 for t in turns_in_window) / n

                            print(f"Reached turn 20+:     {pct_20:.1f}%")
                            print(f"Reached turn 30+:     {pct_30:.1f}%")
                            print(f"Reached turn 40+:     {pct_40:.1f}%")
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

                        loss_text = (
                            f"{result['average_loss']:.4f}"
                            if result["average_loss"] is not None
                            else "N/A"
                        )

                        print(
                            f"Game {self.games_completed:6d} | "
                            f"Worker: {i} | "
                            f"Result: {str(game_result):6s} | "
                            f"Reward: {episode_rewards[i]:5.1f} | "
                            f"Decisions: {episode_decisions[i]:4d} | "
                            f"Epsilon: {self.agent.epsilon:.4f} | "
                            f"Loss: {loss_text}"
                        )

                # ----------------------------------------------
                # ADVANCE VECTOR OBSERVATIONS
                # ----------------------------------------------

                observations = next_observations
                info = next_info

                # --------------------------------------------------
                # PERIODIC GREEDY POLICY EVALUATION
                # --------------------------------------------------
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

                # ----------------------------------------------
                # PERIODIC CHECKPOINT
                # ----------------------------------------------


                if checkpointing and completed - last_save_game >= save_interval:
                    self.save_checkpoint(checkpoint_path)
                    last_save_game = completed

        finally:

            try:
                env.close()
            finally:
                # Always release the evaluation worker, including on errors.
                self.close_evaluation_pool(force=True)

        # --------------------------------------------------
        # PERFORMANCE STATISTICS
        # --------------------------------------------------

        elapsed = perf_counter() - run_start

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

        self.print_learning_summary(
            results,
            interval=progress_interval,
        )

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

        # --------------------------------------------------
        # FINAL CHECKPOINT
        # --------------------------------------------------

        if checkpointing:

            # Avoid saving twice if the final game already
            # triggered a periodic checkpoint.
            if last_save_game != completed:
                self.save_checkpoint(checkpoint_path)

            # Training completed successfully.
            # Keep only the newest numbered checkpoint.
            self.cleanup_old_checkpoints(checkpoint_path)

        return results


if __name__ == "__main__":

    torch.set_num_threads(3)

    trainer = DQNTrainer()

    trainer.train(
        num_games=10000,
        num_workers=8,
        progress_interval=200,
        evaluation_interval=999999999,
        evaluation_games=1,
        resume=True,
        checkpoint_path="core/ai/dqn/checkpoints/normandy_latest.pt",
        save_interval=5000,
    )
    
# if __name__ == "__main__":
#     torch.set_num_threads(3)

#     trainer = DQNTrainer()

#     trainer.train(
#         num_games=10,
#         num_workers=8,
#         progress_interval=10,
#         evaluation_interval=5,
#         evaluation_games=1,
#         resume=True,
#         checkpoint_path="core/ai/dqn/checkpoints/normandy_latest.pt",
#         save_interval=900,
#     )

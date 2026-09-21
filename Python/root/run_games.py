from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import time
import traceback

from core.global_game_state import GlobalGameState
from normandy_full_game_manual import execute_single_game_run

NUM_EPISODES = 100

def run_game_worker(i):
    game_start = time.perf_counter()

    try:
        game_data = execute_single_game_run(i)

        return {
            "game_number": i,
            "game_data": game_data,
            "cards_drawn": GlobalGameState.cards_drawn,
            "duration": time.perf_counter() - game_start,
            "error": None,
        }

    except Exception:
        return {
            "game_number": i,
            "game_data": None,
            "cards_drawn": GlobalGameState.cards_drawn,
            "duration": time.perf_counter() - game_start,
            "error": traceback.format_exc(),
        }


def run_parallel_games(num_episodes, max_workers=None):
    if max_workers is None:
        max_workers = os.cpu_count()

    results = []

    with ProcessPoolExecutor(
        max_workers=max_workers
    ) as executor:

        futures = {
            executor.submit(run_game_worker, i): i
            for i in range(num_episodes)
        }

        for future in as_completed(futures):
            result = future.result()
            results.append(result)

    return results


if __name__ == "__main__":
    results = run_parallel_games(
        NUM_EPISODES,
        max_workers=8,
    )

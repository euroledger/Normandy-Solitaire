
from pathlib import Path
import numpy as np


SAVE_DIR = Path("core/ai/dqn/saved_games")


def save_game(
    observation,
    turn,
    game_id,
    action_mask=None,
    original_final_turn=None,
):
    state = np.asarray(observation).copy()

    if state.shape != (967,):
        raise ValueError(
            f"Expected 967 state elements, got {state.shape}"
        )

    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    filename = SAVE_DIR / f"game_{game_id}_turn_{turn}.npz"

    data = {
        "state": state,
        "turn": np.int32(turn),
        "game_id": str(game_id),
    }

    if action_mask is not None:
        mask = np.asarray(action_mask).copy()

        if mask.shape != (1032,):
            raise ValueError(
                f"Expected 1032 mask elements, got {mask.shape}"
            )

        data["action_mask"] = mask

    if original_final_turn is not None:
        data["original_final_turn"] = np.int32(
            original_final_turn
        )

    np.savez_compressed(filename, **data)
    return filename


def load_saved_observation(filename):
    """Read a saved observation; does not restore the game engine."""

    with np.load(filename, allow_pickle=False) as data:
        return {
            key: data[key].copy()
            for key in data.files
        }

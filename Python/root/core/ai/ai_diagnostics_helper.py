import os
import traceback

from core.ai.headless_env_manager import (
    _REAL_PRINT_,
    save_log_to_disk,
)


def save_winning_game(
    game_number,
    game_data,
):
    save_log_to_disk(
        game_number=game_number
    )

    _REAL_PRINT_(
        f"AI WIN: game {game_number}"
    )

    card_28_position = game_data.get(
        "card_28_draw_position"
    )

    us_3_position = game_data.get(
        "us_3_activation_position"
    )

    us_3_location = game_data.get(
        "us_3_activation_location"
    )

    _REAL_PRINT_(
        f"  Card 28 drawn: "
        f"{card_28_position}"
    )

    if us_3_position is None:
        _REAL_PRINT_(
            "  US 3rd Army activated: NEVER"
        )
    else:
        _REAL_PRINT_(
            f"  US 3rd Army activated: "
            f"turn {us_3_position} "
            f"at {us_3_location}"
        )


def save_exception_game(
    game_number,
    exception,
):
    _REAL_PRINT_()
    _REAL_PRINT_("!" * 60)

    _REAL_PRINT_(
        f"EXCEPTION IN AI GAME {game_number}"
    )

    _REAL_PRINT_(
        f"Worker PID: {os.getpid()}"
    )

    _REAL_PRINT_("-" * 60)

    traceback.print_exception(
        type(exception),
        exception,
        exception.__traceback__,
    )

    _REAL_PRINT_("!" * 60)

    save_log_to_disk(
        game_number=game_number
    )

import io
import builtins
import os
from core.global_game_state import GlobalGameState

_REAL_print = builtins.print
_REAL_INPUT = builtins.input
_current_ai_arguments = None


# Create a fast memory buffer to hold the log text during execution
_LOG_BUFFER = io.StringIO()


def configure_headless_printing(game_number=1):
    import builtins
    # Clear out any text remaining from a previous run
    _LOG_BUFFER.seek(0)
    _LOG_BUFFER.truncate(0)

    # 🎯 FORCE BINDING: Overwrite the print hook right here
    builtins.print = headless_file_logger

    _LOG_BUFFER.write(f"=== START OF GAME {game_number} LOG ===\n\n")

def headless_file_logger(*args, **kwargs):
    # 🎯 RECORD THE GAMEPLAY: Construct the clean string line
    message = " ".join(str(arg) for arg in args)

    # Strip out color escape codes so the written text file is perfectly clean
    for color_code in ["\033[94m", "\033[91m", "\033[92m", "\033[0m", "\033[96m", "\38;5;180m"]:
        message = message.replace(color_code, "")

    # Write the clean text line straight into our RAM buffer
    _LOG_BUFFER.write(message + "\n")


builtins.print = headless_file_logger


def save_log_to_disk(game_number):
    # Writes the memory buffer to a physical file only when a win occurs
    with open(f"game_{game_number}_log.txt", "w", encoding="utf-8") as f:
        f.write(_LOG_BUFFER.getvalue())



def configure_automated_inputs(auto_enter=True):
    if auto_enter:
        builtins.input = lambda *args, **kwargs: ""
    else:
        builtins.input = _REAL_INPUT


def inject_ai_turn_arguments(action_vector):
    global _current_ai_arguments
    _current_ai_arguments = action_vector


def get_automated_choice(prompt_text, ai_slot=None, default_fallback="0"):
    if getattr(GlobalGameState, "headless", False):
        if _current_ai_arguments is not None and ai_slot is not None:
            return str(_current_ai_arguments[ai_slot])
        return str(default_fallback)
    return _REAL_INPUT(prompt_text).strip()

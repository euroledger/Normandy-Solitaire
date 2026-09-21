import io
import builtins
import re
from core.global_game_state import GlobalGameState

_REAL_PRINT_ = builtins.print
_REAL_INPUT = builtins.input
_current_ai_arguments = None

_LOG_BUFFER = io.StringIO()
CAPTURE_FULL_LOG = True


def configure_headless_printing(game_number=1):
    _LOG_BUFFER.seek(0)
    _LOG_BUFFER.truncate(0)

    if CAPTURE_FULL_LOG:
        _LOG_BUFFER.write(f"=== START OF GAME {game_number} LOG ===\n\n")
        builtins.print = headless_file_logger
    else:
        builtins.print = lambda *args, **kwargs: None



def headless_file_logger(*args, **kwargs):
    if not CAPTURE_FULL_LOG:
        return

    message = " ".join(str(arg) for arg in args)
    _LOG_BUFFER.write(message + "\n")


# def save_log_to_disk(game_number):
#     with open(f"game_{game_number}_log.txt", "w", encoding="utf-8") as f:
#         f.write(_LOG_BUFFER.getvalue())
def save_log_to_disk(game_number):
    if not CAPTURE_FULL_LOG:
        return

    log_text = _LOG_BUFFER.getvalue()
    log_text = re.sub(r"\x1b\[[0-9;]*m", "", log_text)

    with open(
        f"game_{game_number}_log.txt",
        "w",
        encoding="utf-8"
    ) as f:
        f.write(log_text)

def restore_normal_printing():
    builtins.print = _REAL_PRINT_


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

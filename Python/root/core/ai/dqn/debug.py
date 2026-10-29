from pathlib import Path

from datetime import datetime




DEBUG_PATH = Path("core/ai/dqn/logs/debug_trace.log")


def debug_pants(message):
    DEBUG_PATH.parent.mkdir(parents=True, exist_ok=True)

    with DEBUG_PATH.open("a", encoding="utf-8") as f:
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        f.write(f"{timestamp} | {message}\n")
        f.flush()

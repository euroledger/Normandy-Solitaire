
import contextlib
import io
import os
from pathlib import Path
import re

from core.ai.dqn.normandy_env import NormandyEnv
from core.global_game_state import GlobalGameState


class NormandyLoggingEnv(NormandyEnv):
    def __init__(self, config, shared_count, shared_lock):
        super().__init__()
        self.log_config = config
        self.shared_count = shared_count
        self.shared_lock = shared_lock
        self.log_buffer = None

    def _limit_reached(self):
        with self.shared_lock:
            return self.shared_count.value >= self.log_config["max_files"]

    def _discard(self):
        if self.log_buffer is not None:
            self.log_buffer.close()
            self.log_buffer = None
            
    def _target_turns(self):
        target = self.log_config["target_turn"]
        if target is None:
            return None
        if isinstance(target, (list, tuple)):
            return target
        return [target]

    def reset(self, *, seed=None, options=None):
        self._discard()

        if self._limit_reached():
            return super().reset(seed=seed, options=options)

        self.log_buffer = io.StringIO()

        with contextlib.redirect_stdout(self.log_buffer):
            return super().reset(seed=seed, options=options)

    def step(self, action):
        if self.log_buffer is None:
            return super().step(action)

        if self._limit_reached():
            self._discard()
            return super().step(action)

        targets = self._target_turns()
        max_target = max(targets) if targets else None

        if (
            max_target is not None
            and GlobalGameState.cards_drawn > max_target
        ):
            self._discard()
            return super().step(action)

        with contextlib.redirect_stdout(self.log_buffer):
            result = super().step(action)

        _, _, terminated, truncated, info = result

        if terminated or truncated:
            self._finish_log(info)
        elif (
            max_target is not None
            and GlobalGameState.cards_drawn > max_target
        ):
            self._discard()

        return result

    def _finish_log(self, info):
        wanted_result = self.log_config["result"]

        actual_result = info.get("result")
        turn_reached = info.get("turn_reached")

        targets = self._target_turns()

        matches = (
            (wanted_result == "any" or actual_result == wanted_result)
            and (targets is None or turn_reached in targets)
        )
        
        import sys


        print(
            f"LOG DEBUG: result={actual_result!r}, "
            f"wanted={wanted_result!r}, "
            f"turn={turn_reached!r}, "
            f"targets={targets!r}, "
            f"matches={matches!r}, "
            f"count={self.shared_count.value}, "
            f"buffer={self.log_buffer is not None}",
            file=sys.stderr,
        )

        if matches:
            with self.shared_lock:
                if self.shared_count.value < self.log_config["max_files"]:
                    output_dir = Path(self.log_config["output_dir"])
                    output_dir.mkdir(parents=True, exist_ok=True)

                    file_number = self.shared_count.value + 1
                    filename = (
                        f"game_{file_number:03d}_"
                        f"{actual_result}_turn_{turn_reached}_"
                        f"pid_{os.getpid()}.txt"
                    )

                    path = output_dir / filename

                    # Remove ANSI terminal formatting sequences from saved logs.
                    log_text = self.log_buffer.getvalue()
                    log_text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", log_text)

                    path.write_text(
                        log_text,
                        encoding="utf-8",
                    )

                    self.shared_count.value = file_number

        self._discard()

    def close(self):
        self._discard()
        return super().close()

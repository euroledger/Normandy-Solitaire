
import contextlib
import io
import os
import re
import traceback
import uuid
from datetime import datetime
from pathlib import Path

from core.ai.dqn.midgame_config import MIDGAME_CONFIG
from core.ai.dqn.normandy_env import NormandyEnv
from core.global_game_state import GlobalGameState


class NormandyLoggingEnv(NormandyEnv):
    def __init__(
        self,
        config,
        shared_count=None,
        shared_lock=None,
        error_config=None,
    ):
        super().__init__()

        self.log_config = config or {}
        self.error_config = error_config or {}
        self.shared_count = shared_count
        self.shared_lock = shared_lock

        self.log_buffer = None
        self.game_number = 0
        self.last_action = None

    def _normal_enabled(self):
        return bool(self.log_config.get("enabled", False))

    def _error_enabled(self):
        return bool(self.error_config.get("enabled", False))

    def _limit_reached(self):
        if not self._normal_enabled():
            return True

        if self.shared_count is None:
            return False

        with self.shared_lock:
            return (
                self.shared_count.value
                >= self.log_config.get("max_files", 1)
            )

    def _discard(self):
        if self.log_buffer is not None:
            self.log_buffer.close()
            self.log_buffer = None

    def _target_turns(self):
        target = self.log_config.get("target_turn")

        if target is None:
            return None

        if isinstance(target, (list, tuple)):
            return target

        return [target]

    def _capture(self):
        return self.log_buffer is not None


    def reset(self, *, seed=None, options=None):
        self._discard()
        self.game_number += 1
        self.last_action = None

        # Request the configured midgame snapshot on EVERY reset,
        # including Gymnasium's automatic worker resets.
        options = dict(options or {})
        options["midgame"] = MIDGAME_CONFIG["enabled"]

        capture_needed = (
            self._error_enabled()
            or (
                self._normal_enabled()
                and not self._limit_reached()
            )
        )

        if capture_needed:
            self.log_buffer = io.StringIO()

        try:
            if self._capture():
                with contextlib.redirect_stdout(self.log_buffer):
                    result = super().reset(
                        seed=seed,
                        options=options,
                    )
            else:
                result = super().reset(
                    seed=seed,
                    options=options,
                )

            if result is None:
                raise RuntimeError(
                    "NormandyEnv.reset() returned None"
                )

            return result

        except BaseException as error:
            self._handle_error(error, "reset")
            raise

    def step(self, action):
        self.last_action = action

        try:
            if not self._capture():
                return super().step(action)

            with contextlib.redirect_stdout(self.log_buffer):
                result = super().step(action)

            _, _, terminated, truncated, info = result

            if terminated or truncated:
                self._finish_log(info)
                self._discard()

            elif not self._error_enabled():
                # Retain the original early-discard optimisation
                # when only ordinary game logging is active.
                targets = self._target_turns()
                max_target = max(targets) if targets else None

                if self._limit_reached() or (
                    max_target is not None
                    and GlobalGameState.cards_drawn > max_target
                ):
                    self._discard()

            return result

        except BaseException as error:
            self._handle_error(error, "step")
            raise

    def _handle_error(self, error, operation):
        # Never replace the original game exception with
        # a secondary logging exception.
        try:
            if self._error_enabled():
                self._save_error(error, operation)
        except Exception as logging_error:
            # stderr is preferable here because stdout may
            # be redirected or suppressed in the worker.
            import sys
            print(
                f"ERROR: Could not save crash log: "
                f"{logging_error}",
                file=sys.__stderr__,
            )
        finally:
            self._discard()

    def _save_error(self, error, operation):
        output_dir = Path(
            self.error_config.get(
                "output_dir",
                "core/ai/dqn/logs/errors",
            )
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S_%f"
        )

        filename = (
            f"error_{timestamp}_"
            f"pid_{os.getpid()}_"
            f"{uuid.uuid4().hex[:8]}.txt"
        )

        path = output_dir / filename

        log_text = (
            self.log_buffer.getvalue()
            if self.log_buffer is not None
            else "[No game history captured]\n"
        )

        log_text = re.sub(
            r"\x1b\[[0-9;]*m",
            "",
            log_text,
        )

        card = getattr(
            GlobalGameState,
            "current_card",
            None,
        )

        card_id = getattr(card, "card_id", None)

        turn = getattr(
            GlobalGameState,
            "cards_drawn",
            None,
        )

        exception_text = "".join(
            traceback.format_exception(
                type(error),
                error,
                error.__traceback__,
            )
        )

        with path.open("w", encoding="utf-8") as file:
            file.write("NORMANDY DQN ERROR REPORT\n")
            file.write("=" * 70 + "\n")
            file.write(f"Timestamp: {timestamp}\n")
            file.write(f"Worker PID: {os.getpid()}\n")
            file.write(f"Worker-local game: {self.game_number}\n")
            file.write(f"Operation: {operation}\n")
            file.write(f"Turn: {turn}\n")
            file.write(f"Card: {card_id}\n")
            file.write(f"Last action: {self.last_action}\n")
            file.write(
                f"Decision type: "
                f"{self.current_decision_type}\n"
            )

            file.write("\nFULL GAME HISTORY\n")
            file.write("=" * 70 + "\n")
            file.write(log_text)

            file.write("\n\nEXCEPTION TRACEBACK\n")
            file.write("=" * 70 + "\n")
            file.write(exception_text)

    def _finish_log(self, info):
        if not self._normal_enabled():
            return

        if self._limit_reached():
            return

        targets = self._target_turns()
        wanted_result = self.log_config.get("result", "any")
        actual_result = info.get("result")
        turn_reached = info.get("turn_reached")

        matches = (
            (
                wanted_result == "any"
                or actual_result == wanted_result
            )
            and (
                targets is None
                or turn_reached in targets
            )
        )

        if not matches:
            return

        if self.shared_count is None or self.shared_lock is None:
            raise RuntimeError(
                "Normal game logging requires a shared "
                "counter and lock."
            )

        with self.shared_lock:
            max_files = self.log_config.get("max_files", 1)

            if self.shared_count.value >= max_files:
                return

            output_dir = Path(
                self.log_config.get(
                    "output_dir",
                    "core/ai/dqn/logs/games",
                )
            )
            output_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            file_number = self.shared_count.value + 1

            filename = (
                f"game_{file_number:03d}_"
                f"{actual_result}_turn_{turn_reached}_"
                f"pid_{os.getpid()}.txt"
            )

            log_text = self.log_buffer.getvalue()
            log_text = re.sub(
                r"\x1b\[[0-9;]*m",
                "",
                log_text,
            )

            (output_dir / filename).write_text(
                log_text,
                encoding="utf-8",
            )

            self.shared_count.value = file_number

    def close(self):
        self._discard()
        return super().close()

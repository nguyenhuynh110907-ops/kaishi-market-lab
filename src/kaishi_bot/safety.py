from __future__ import annotations

import secrets
import time
from collections.abc import Callable
from decimal import Decimal

from kaishi_bot.dashboard_models import RuntimeMode


class SafetyError(RuntimeError):
    pass


class SafetyGate:
    def __init__(
        self, *, credentials_available: bool,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.credentials_available = credentials_available
        self.clock = clock
        self.mode = RuntimeMode.PAPER
        self.armed = False
        self._challenge: str | None = None
        self._challenge_expires = 0.0

    def set_mode(self, mode: RuntimeMode) -> None:
        self.mode = mode
        self.disarm()

    def create_challenge(self) -> str:
        if self.mode is not RuntimeMode.LIVE:
            raise SafetyError("Live mode is not selected")
        if not self.credentials_available:
            raise SafetyError("Kalshi credentials are not available")
        self._challenge = f"ARM LIVE {secrets.token_hex(3).upper()}"
        self._challenge_expires = self.clock() + 60.0
        return self._challenge

    def arm(self, phrase: str) -> None:
        if self.clock() > self._challenge_expires:
            self.disarm()
            raise SafetyError("Live challenge expired")
        if not self._challenge or not secrets.compare_digest(phrase, self._challenge):
            raise SafetyError("Live confirmation phrase does not match")
        if self.mode is not RuntimeMode.LIVE or not self.credentials_available:
            raise SafetyError("Live mode and credentials are required")
        self.armed = True
        self._challenge = None

    def disarm(self) -> None:
        self.armed = False
        self._challenge = None
        self._challenge_expires = 0.0

    def mark_data_stale(self) -> None:
        self.disarm()

    def authorize_write(
        self, entry_cost: Decimal, daily_used: Decimal, *, reduce_only: bool = False
    ) -> None:
        if self.mode is not RuntimeMode.LIVE:
            raise SafetyError("Live mode is not selected")
        if not self.credentials_available:
            self.disarm()
            raise SafetyError("Kalshi credentials are not available")
        if not reduce_only and not self.armed:
            raise SafetyError("Live mode is not armed")
        if entry_cost > Decimal("200.00"):
            raise SafetyError("per-entry limit is $200.00")
        if entry_cost > 0 and daily_used + entry_cost > Decimal("10000.00"):
            raise SafetyError("daily entry limit is $10000.00")

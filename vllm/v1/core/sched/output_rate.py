# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-threaded per-user output pacing with GCRA (Generic Cell Rate Algorithm)."""

import time
from collections.abc import Callable, Mapping

_SECOND_NS = 1_000_000_000
# Must cover at least one engine step, or credit lapses between steps and a user
# gets less than its rate. Any window of W seconds admits at most
# (W + _BURST_NS / 1e9) * rate + 1 outputs.
_BURST_NS = 100_000_000


class OutputRateLimiter:
    """Charge one output per sampling step at dispatch time.

    Each user keeps one theoretical arrival time (TAT). A take succeeds when the
    TAT is at most ``burst_ns`` ahead of now, then advances it by ``1 / rate``.
    Outputs are spaced evenly instead of being spent at the start of a window.
    Times are stored multiplied by the user's rate, so the arithmetic is exact.

    Args:
        default_limit: Positive outputs per second for users without an override.
        per_user_limits: Positive per-user limits, copied at construction.
        burst_ns: How far a user's TAT may run ahead of now.
        clock: Nondecreasing clock returning integer nanoseconds.

    """

    def __init__(
        self,
        default_limit: int,
        per_user_limits: Mapping[str, int] | None = None,
        *,
        burst_ns: int = _BURST_NS,
        clock: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        self._validate_limit(default_limit)
        self._default_limit = default_limit
        self._per_user_limits = dict(per_user_limits or {})
        for limit in self._per_user_limits.values():
            self._validate_limit(limit)
        self._burst_ns = burst_ns
        self._clock = clock
        # Scaled TAT: nanoseconds multiplied by the user's limit.
        self._tat: dict[str, int] = {}

    def try_take(self, user: str) -> bool:
        """Charge one output, or return False without charging."""
        limit = self._limit_for(user)
        now = self._clock() * limit
        tat = max(self._tat.get(user, now), now)
        if tat - now > self._burst_ns * limit:
            return False
        self._tat[user] = tat + _SECOND_NS
        return True

    def refund(self, user: str) -> None:
        """Return one output charged in this step that was never dispatched."""
        self._tat[user] -= _SECOND_NS

    def ready_at_ns(self, user: str) -> int:
        """Return the earliest clock value at which ``try_take`` can succeed."""
        limit = self._limit_for(user)
        now_ns = self._clock()
        tat = self._tat.get(user, now_ns * limit)
        # Ceiling division keeps the result at or after the admission boundary.
        return max(now_ns, -(-(tat - self._burst_ns * limit) // limit))

    def cleanup_idle_users(self) -> int:
        """Drop users whose TAT has passed; a fresh state behaves identically."""
        now = self._clock()
        idle = [
            user
            for user, tat in self._tat.items()
            if tat <= now * self._limit_for(user)
        ]
        for user in idle:
            del self._tat[user]
        return len(idle)

    def _limit_for(self, user: str) -> int:
        return self._per_user_limits.get(user, self._default_limit)

    @staticmethod
    def _validate_limit(limit: int) -> None:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("output limits must be positive integers")

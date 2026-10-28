# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-threaded per-user output accounting over a rolling one-second window."""

import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

_WINDOW_NS = 1_000_000_000


class DuplicateReservationError(ValueError):
    """A request already has an in-flight output reservation."""


class UnknownReservationError(KeyError):
    """A completion has no corresponding in-flight reservation."""


class OutputOverrunError(RuntimeError):
    """The full actual output was charged, but exceeded the reserved slot."""

    def __init__(self, request_id: str, user: str, actual_count: int) -> None:
        self.request_id = request_id
        self.user = user
        self.actual_count = actual_count
        self.reserved_count = 1
        super().__init__(
            f"Request {request_id!r} for user {user!r} produced {actual_count} "
            "outputs with only 1 reserved; all actual outputs remain charged"
        )


@dataclass(frozen=True, slots=True)
class OutputBudgetStatistics:
    limit: int
    inflight_count: int
    completed_count: int
    available: int
    ready_at_ns: int | None


@dataclass(slots=True)
class _UserBudget:
    inflight_count: int = 0
    completed_count: int = 0
    completions: deque[tuple[int, int]] = field(default_factory=deque)


class RollingOutputBudget:
    """Reserve at most one output per request before dispatching model work.

    In-flight slots never expire. A completed output is charged until exactly
    one second after ``finish``. All requests for one user share the same budget.
    This ledger provides no queueing or fairness policy.

    Args:
        default_limit: Positive per-second limit for users without an override.
        per_user_limits: Positive per-user limits, copied at construction.
        clock: Nondecreasing clock returning integer nanoseconds.

    """

    def __init__(
        self,
        default_limit: int,
        per_user_limits: Mapping[str, int] | None = None,
        *,
        clock: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        self._validate_limit(default_limit)
        self._default_limit = default_limit
        self._per_user_limits = dict(per_user_limits or {})
        for limit in self._per_user_limits.values():
            self._validate_limit(limit)
        self._clock = clock
        self._last_now_ns: int | None = None
        self._users: dict[str, _UserBudget] = {}
        self._inflight: dict[str, str] = {}

    def reserve(self, request_id: str, user: str) -> bool:
        """Reserve one slot, or return False without reserving when full.

        Request IDs must be unique across in-flight work, including across users.
        An ID may be reused for the next decode step after its previous finish.
        """
        if request_id in self._inflight:
            raise DuplicateReservationError(
                f"Request {request_id!r} already has an in-flight reservation"
            )
        now_ns = self._now_ns()
        state = self._users.get(user)
        if state is None:
            state = _UserBudget()
            self._users[user] = state
        self._prune(state, now_ns)
        if state.inflight_count + state.completed_count >= self._limit_for(user):
            return False
        state.inflight_count += 1
        self._inflight[request_id] = user
        return True

    def finish(self, request_id: str, actual_count: int) -> None:
        """Confirm safe completion, replacing a reservation with actual output.

        Call only once the backend can no longer produce output for this slot.
        A cancellation signal alone is insufficient: keep the reservation until
        safe completion, then report zero or the actual generated count.

        Raises:
            ValueError: The count is not a nonnegative integer; the slot is kept.
            UnknownReservationError: The request has no in-flight reservation.
            OutputOverrunError: The count exceeds one. The reservation is closed
                and the entire count is charged for a full second before raising.

        """
        if (
            isinstance(actual_count, bool)
            or not isinstance(actual_count, int)
            or actual_count < 0
        ):
            raise ValueError("actual_count must be a nonnegative integer")
        if request_id not in self._inflight:
            raise UnknownReservationError(
                f"Request {request_id!r} has no in-flight reservation"
            )
        now_ns = self._now_ns()
        user = self._inflight.pop(request_id)
        state = self._users[user]
        self._prune(state, now_ns)
        state.inflight_count -= 1
        if actual_count:
            state.completions.append((now_ns, actual_count))
            state.completed_count += actual_count
        if actual_count > 1:
            raise OutputOverrunError(request_id, user, actual_count)

    def available(self, user: str) -> int:
        """Return the number of slots that can be reserved immediately."""
        return self.statistics(user).available

    def ready_at_ns(self, user: str) -> int | None:
        """Return the earliest admission time if in-flight work stays pending.

        Return the current clock value when a slot is available. None means that
        time alone cannot free a slot; a safe completion must arrive first.
        """
        return self.statistics(user).ready_at_ns

    def statistics(self, user: str) -> OutputBudgetStatistics:
        """Snapshot current per-user accounting after expiring old completions."""
        now_ns = self._now_ns()
        limit = self._limit_for(user)
        state = self._users.get(user)
        if state is None:
            return OutputBudgetStatistics(limit, 0, 0, limit, now_ns)
        self._prune(state, now_ns)
        available = max(0, limit - state.inflight_count - state.completed_count)
        return OutputBudgetStatistics(
            limit,
            state.inflight_count,
            state.completed_count,
            available,
            self._ready_at_ns(state, limit, now_ns),
        )

    def cleanup_idle_users(self) -> int:
        """Remove only users with no in-flight slots or recent output events.

        Returns:
            The number of removed accounting states. Configured limits remain.

        """
        now_ns = self._now_ns()
        removed = 0
        for user, state in list(self._users.items()):
            self._prune(state, now_ns)
            if not state.inflight_count and not state.completions:
                del self._users[user]
                removed += 1
        return removed

    def _limit_for(self, user: str) -> int:
        return self._per_user_limits.get(user, self._default_limit)

    def _now_ns(self) -> int:
        now_ns = self._clock()
        if self._last_now_ns is not None and now_ns < self._last_now_ns:
            raise ValueError("clock must be nondecreasing")
        self._last_now_ns = now_ns
        return now_ns

    @staticmethod
    def _validate_limit(limit: int) -> None:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("output limits must be positive integers")

    @staticmethod
    def _prune(state: _UserBudget, now_ns: int) -> None:
        cutoff_ns = now_ns - _WINDOW_NS
        while state.completions and state.completions[0][0] <= cutoff_ns:
            _, count = state.completions.popleft()
            state.completed_count -= count

    @staticmethod
    def _ready_at_ns(state: _UserBudget, limit: int, now_ns: int) -> int | None:
        used = state.inflight_count + state.completed_count
        if used < limit:
            return now_ns
        if state.inflight_count >= limit:
            return None
        needed = used - limit + 1
        for completed_ns, count in state.completions:
            needed -= count
            if needed <= 0:
                return completed_ns + _WINDOW_NS
        raise AssertionError("completion counts do not match their events")

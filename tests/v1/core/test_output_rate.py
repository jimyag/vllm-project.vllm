# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.v1.core.sched.output_rate import (
    DuplicateReservationError,
    OutputOverrunError,
    RollingOutputBudget,
    UnknownReservationError,
)

pytestmark = pytest.mark.cpu_test
SECOND_NS = 1_000_000_000


class FakeClock:
    def __init__(self) -> None:
        self.now_ns = 0

    def __call__(self) -> int:
        return self.now_ns


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


def test_default_and_overrides_are_per_user_and_copied(clock: FakeClock) -> None:
    overrides = {"A": 100, "B": 500}
    budget = RollingOutputBudget(2, overrides, clock=clock)
    overrides["A"] = 1

    for user, limit in (("A", 100), ("B", 500)):
        for i in range(limit):
            request_id = f"{user}-{i}"
            assert budget.reserve(request_id, user)
            budget.finish(request_id, 1)
        assert not budget.reserve(f"{user}-blocked", user)
        assert budget.statistics(user).completed_count == limit
    assert budget.reserve("unknown-1", "unknown")
    assert budget.reserve("unknown-2", "unknown")
    assert not budget.reserve("unknown-3", "unknown")
    assert budget.available("another-unknown") == 2


def test_completed_and_inflight_streams_share_one_budget(clock: FakeClock) -> None:
    budget = RollingOutputBudget(2, clock=clock)
    assert budget.reserve("stream-1", "user")
    assert budget.reserve("stream-2", "user")
    budget.finish("stream-1", 1)

    stats = budget.statistics("user")
    assert (stats.inflight_count, stats.completed_count, stats.available) == (1, 1, 0)
    assert not budget.reserve("stream-3", "user")
    assert budget.ready_at_ns("user") == SECOND_NS
    clock.now_ns = SECOND_NS
    assert budget.reserve("stream-3", "user")
    assert budget.ready_at_ns("user") is None


def test_pending_cancellation_keeps_slot_until_safe_completion(
    clock: FakeClock,
) -> None:
    budget = RollingOutputBudget(1, clock=clock)
    assert budget.reserve("cancel-pending", "user")
    clock.now_ns = 60 * SECOND_NS

    assert budget.cleanup_idle_users() == 0
    assert budget.available("user") == 0
    assert budget.ready_at_ns("user") is None
    assert not budget.reserve("next", "user")
    budget.finish("cancel-pending", 0)  # Backend has now confirmed safe completion.
    assert budget.reserve("next", "user")


def test_delayed_completion_is_charged_from_finish_with_ns_boundary(
    clock: FakeClock,
) -> None:
    budget = RollingOutputBudget(1, clock=clock)
    assert budget.reserve("slow", "user")
    clock.now_ns = 10 * SECOND_NS
    assert not budget.reserve("too-early", "user")
    budget.finish("slow", 1)

    assert budget.ready_at_ns("user") == 11 * SECOND_NS
    clock.now_ns = 11 * SECOND_NS - 1
    assert not budget.reserve("one-ns-early", "user")
    clock.now_ns += 1
    assert budget.reserve("at-boundary", "user")


def test_window_rolls_by_completion_order_instead_of_fixed_seconds(
    clock: FakeClock,
) -> None:
    budget = RollingOutputBudget(2, clock=clock)
    assert budget.reserve("slow", "user")
    assert budget.reserve("fast", "user")
    clock.now_ns = 900_000_000
    budget.finish("fast", 1)
    clock.now_ns = 1_100_000_000
    budget.finish("slow", 1)
    assert not budget.reserve("next", "user")
    assert budget.ready_at_ns("user") == 1_900_000_000

    clock.now_ns = 1_900_000_000 - 1
    assert budget.available("user") == 0
    clock.now_ns += 1
    assert budget.available("user") == 1
    assert budget.reserve("next", "user")
    budget.finish("next", 1)
    clock.now_ns = 2_100_000_000
    assert budget.available("user") == 1


def test_zero_output_refunds_and_request_id_can_be_reused(clock: FakeClock) -> None:
    budget = RollingOutputBudget(1, clock=clock)
    for _ in range(3):
        assert budget.reserve("request", "user")
        budget.finish("request", 0)
        assert budget.available("user") == 1
    assert budget.reserve("request", "user")
    budget.finish("request", 1)
    assert not budget.reserve("request", "user")


@pytest.mark.parametrize("other_user", ["user", "another-user"])
def test_duplicate_inflight_id_cannot_reserve_or_transfer_slot(
    clock: FakeClock, other_user: str
) -> None:
    budget = RollingOutputBudget(1, clock=clock)
    assert budget.reserve("request", "user")
    before = budget.statistics("user")
    with pytest.raises(DuplicateReservationError, match="request"):
        budget.reserve("request", other_user)
    assert budget.statistics("user") == before
    assert budget.available("another-user") == 1
    budget.finish("request", 0)
    assert budget.reserve("request", other_user)


def test_unknown_or_repeated_finish_cannot_refund_real_output(clock: FakeClock) -> None:
    budget = RollingOutputBudget(1, clock=clock)
    with pytest.raises(UnknownReservationError, match="missing"):
        budget.finish("missing", 0)
    assert budget.reserve("request", "user")
    budget.finish("request", 1)
    with pytest.raises(UnknownReservationError, match="request"):
        budget.finish("request", 0)
    assert budget.available("user") == 0


@pytest.mark.parametrize("invalid_count", [-1, True, 1.5, "1", None])
def test_invalid_completion_keeps_reservation(
    clock: FakeClock, invalid_count: object
) -> None:
    budget = RollingOutputBudget(1, clock=clock)
    assert budget.reserve("request", "user")
    with pytest.raises(ValueError, match="nonnegative integer"):
        budget.finish("request", invalid_count)  # type: ignore[arg-type]
    assert budget.statistics("user").inflight_count == 1
    assert not budget.reserve("next", "user")
    budget.finish("request", 0)
    assert budget.reserve("next", "user")


def test_overrun_raises_after_charging_all_output_and_cannot_be_refunded(
    clock: FakeClock,
) -> None:
    budget = RollingOutputBudget(1, clock=clock)
    assert budget.reserve("request", "user")
    with pytest.raises(OutputOverrunError, match="all actual outputs") as error:
        budget.finish("request", 3)
    assert error.value.actual_count == 3
    stats = budget.statistics("user")
    assert (stats.inflight_count, stats.completed_count, stats.available) == (0, 3, 0)
    with pytest.raises(UnknownReservationError):
        budget.finish("request", 0)
    assert not budget.reserve("next", "user")
    assert budget.available("other-user") == 1
    clock.now_ns = SECOND_NS - 1
    assert budget.cleanup_idle_users() == 0
    assert not budget.reserve("next", "user")
    clock.now_ns += 1
    assert budget.reserve("next", "user")


def test_ready_time_waits_for_enough_weighted_events_after_overrun(
    clock: FakeClock,
) -> None:
    budget = RollingOutputBudget(2, clock=clock)
    assert budget.reserve("first", "user")
    assert budget.reserve("second", "user")
    budget.finish("first", 1)
    clock.now_ns = 100_000_000
    with pytest.raises(OutputOverrunError):
        budget.finish("second", 3)

    assert budget.ready_at_ns("user") == 1_100_000_000
    clock.now_ns = SECOND_NS
    assert budget.available("user") == 0
    assert budget.ready_at_ns("user") == 1_100_000_000
    clock.now_ns = 1_100_000_000
    assert budget.available("user") == 2
    assert budget.ready_at_ns("user") == clock.now_ns


def test_cleanup_retains_recent_events_and_pending_work(clock: FakeClock) -> None:
    budget = RollingOutputBudget(1, {"recent": 2}, clock=clock)
    assert budget.reserve("pending", "pending-user")
    assert budget.reserve("old", "old-user")
    budget.finish("old", 1)
    assert budget.reserve("empty", "empty-user")
    budget.finish("empty", 0)
    clock.now_ns = SECOND_NS - 1
    assert budget.reserve("recent-1", "recent")
    budget.finish("recent-1", 1)
    assert budget.reserve("recent-2", "recent")
    budget.finish("recent-2", 1)

    assert budget.cleanup_idle_users() == 1
    clock.now_ns = SECOND_NS
    assert budget.cleanup_idle_users() == 1
    assert budget.available("pending-user") == 0
    assert budget.available("recent") == 0
    assert budget.available("old-user") == 1
    clock.now_ns = 2 * SECOND_NS - 1
    assert budget.cleanup_idle_users() == 1
    assert budget.available("recent") == 2
    budget.finish("pending", 0)
    assert budget.cleanup_idle_users() == 1
    assert budget.cleanup_idle_users() == 0


@pytest.mark.parametrize("invalid_limit", [0, -1, True, 1.5, "1"])
@pytest.mark.parametrize("override", [False, True])
def test_limits_must_be_positive_integers(
    invalid_limit: object, override: bool
) -> None:
    with pytest.raises(ValueError, match="positive integers"):
        if override:
            RollingOutputBudget(1, {"user": invalid_limit})  # type: ignore[dict-item]
        else:
            RollingOutputBudget(invalid_limit)  # type: ignore[arg-type]


def test_backwards_clock_cannot_release_inflight_work(clock: FakeClock) -> None:
    clock.now_ns = 10
    budget = RollingOutputBudget(1, clock=clock)
    assert budget.reserve("request", "user")
    clock.now_ns = 9
    with pytest.raises(ValueError, match="nondecreasing"):
        budget.finish("request", 0)
    clock.now_ns = 10
    assert budget.available("user") == 0
    budget.finish("request", 0)
    assert budget.available("user") == 1

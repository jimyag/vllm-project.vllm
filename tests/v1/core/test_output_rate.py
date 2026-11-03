# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.v1.core.sched.output_rate import OutputRateLimiter

pytestmark = pytest.mark.cpu_test
SECOND_NS = 1_000_000_000
MS = 1_000_000


class FakeClock:
    def __init__(self) -> None:
        self.now_ns = 0

    def __call__(self) -> int:
        return self.now_ns


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


def _take_all(limiter: OutputRateLimiter, user: str) -> int:
    taken = 0
    while limiter.try_take(user):
        taken += 1
    return taken


def test_default_and_overrides_are_per_user_and_copied(clock: FakeClock) -> None:
    overrides = {"A": 10, "B": 50}
    limiter = OutputRateLimiter(1, overrides, burst_ns=SECOND_NS, clock=clock)
    overrides["A"] = 1000

    # A burst of one second admits rate + 1 outputs at once.
    assert _take_all(limiter, "A") == 11
    assert _take_all(limiter, "B") == 51
    assert _take_all(limiter, "unknown") == 2
    assert _take_all(limiter, "another-unknown") == 2


def test_outputs_are_spaced_by_rate_after_the_burst(clock: FakeClock) -> None:
    limiter = OutputRateLimiter(10, burst_ns=100 * MS, clock=clock)
    assert _take_all(limiter, "user") == 2
    assert limiter.ready_at_ns("user") == 100 * MS

    clock.now_ns = 100 * MS - 1
    assert not limiter.try_take("user")
    clock.now_ns += 1
    assert limiter.try_take("user")
    assert not limiter.try_take("user")
    assert limiter.ready_at_ns("user") == 200 * MS


@pytest.mark.parametrize("rate", [3, 7, 100, 333])
def test_every_window_is_bounded_and_no_credit_is_lost(
    clock: FakeClock, rate: int
) -> None:
    burst_ns = 100 * MS
    step_ns = 25 * MS
    limiter = OutputRateLimiter(rate, burst_ns=burst_ns, clock=clock)
    times: list[int] = []
    for step in range(10 * SECOND_NS // step_ns):
        clock.now_ns = step * step_ns
        times.extend([clock.now_ns] * _take_all(limiter, "user"))

    # GCRA bound: any window of W admits at most (W + burst) * rate + 1.
    bound = (SECOND_NS + burst_ns) * rate // SECOND_NS + 1
    for start in times:
        in_window = sum(start <= t < start + SECOND_NS for t in times)
        assert in_window <= bound
    # Steps shorter than the burst never let credit lapse.
    assert len(times) >= 10 * rate - 1


def test_refund_returns_an_undispatched_output(clock: FakeClock) -> None:
    limiter = OutputRateLimiter(1, burst_ns=0, clock=clock)
    assert limiter.try_take("user")
    assert not limiter.try_take("user")
    limiter.refund("user")
    assert limiter.try_take("user")
    assert limiter.ready_at_ns("user") == SECOND_NS


def test_refund_after_idle_gap_gives_no_extra_credit(clock: FakeClock) -> None:
    limiter = OutputRateLimiter(1, burst_ns=0, clock=clock)
    assert limiter.try_take("user")
    clock.now_ns = 10 * SECOND_NS
    assert limiter.try_take("user")
    limiter.refund("user")
    assert limiter.try_take("user")
    assert not limiter.try_take("user")


def test_cleanup_drops_only_idle_users_without_changing_behavior(
    clock: FakeClock,
) -> None:
    limiter = OutputRateLimiter(1, {"busy": 1}, burst_ns=0, clock=clock)
    assert limiter.try_take("idle")
    assert limiter.try_take("busy")
    clock.now_ns = SECOND_NS
    assert limiter.try_take("busy")

    assert limiter.cleanup_idle_users() == 1
    assert limiter.ready_at_ns("idle") == SECOND_NS
    assert limiter.ready_at_ns("busy") == 2 * SECOND_NS
    assert limiter.try_take("idle")
    assert not limiter.try_take("busy")


@pytest.mark.parametrize("invalid_limit", [0, -1, True, 1.5])
def test_limits_must_be_positive_integers(invalid_limit) -> None:
    with pytest.raises(ValueError, match="positive integers"):
        OutputRateLimiter(invalid_limit)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="positive integers"):
        OutputRateLimiter(1, {"user": invalid_limit})  # type: ignore[dict-item]

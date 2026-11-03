# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import queue
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm.v1.engine import EngineCoreRequestType
from vllm.v1.engine.core import EngineCoreProc

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize(
    "request_type",
    [
        EngineCoreRequestType.ADD,
        EngineCoreRequestType.ABORT,
        EngineCoreRequestType.WAKEUP,
    ],
)
def test_output_rate_wait_wakes_for_input(request_type):
    """An input arriving during the quota wait is handled in the same step."""
    waiting = threading.Event()

    class NotifyingQueue(queue.Queue):
        def get(self, *args, **kwargs):
            waiting.set()
            return super().get(*args, **kwargs)

    inputs = NotifyingQueue()
    message = (request_type, None)

    def send_input():
        if waiting.wait(timeout=1):
            inputs.put(message)

    sender = threading.Thread(target=send_input)
    core = SimpleNamespace(
        step_fn=lambda: ({}, False),
        output_queue=queue.Queue(),
        input_queue=inputs,
        scheduler=SimpleNamespace(
            has_requests=lambda: True,
            get_output_rate_wait_seconds=lambda: 0.05,
        ),
        post_step=Mock(),
        _handle_client_request=Mock(),
    )
    sender.start()
    try:
        assert not EngineCoreProc._process_engine_step(core)
    finally:
        sender.join(timeout=2)
    assert not sender.is_alive()
    core._handle_client_request.assert_called_once_with(*message)
    core.post_step.assert_called_once_with(False)


def test_output_rate_wait_timeout_retries_without_losing_outputs():
    outputs = queue.Queue()
    core = SimpleNamespace(
        step_fn=lambda: ({0: "housekeeping"}, False),
        output_queue=outputs,
        input_queue=queue.Queue(),
        scheduler=SimpleNamespace(
            has_requests=lambda: True,
            get_output_rate_wait_seconds=lambda: 0.001,
        ),
        post_step=Mock(),
        _handle_client_request=Mock(),
    )
    assert not EngineCoreProc._process_engine_step(core)
    assert outputs.get_nowait() == (0, "housekeeping")
    core._handle_client_request.assert_not_called()


def test_output_rate_productive_step_does_not_wait():
    core = SimpleNamespace(
        step_fn=lambda: ({}, True),
        output_queue=queue.Queue(),
        scheduler=Mock(),
        post_step=Mock(),
    )
    assert EngineCoreProc._process_engine_step(core)
    core.scheduler.get_output_rate_wait_seconds.assert_not_called()

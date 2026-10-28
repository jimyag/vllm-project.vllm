# Per-user output rate limiting

This experimental feature limits generated output tokens per user within one
vLLM engine. It does not count prompt tokens, truncate answers, or reject a
request because its output allowance is exhausted. A request without allowance
stays queued or pauses decoding while other users can make progress.

## Start a server

The implementation is based on vLLM v0.32.0, commit
`5df63a7203cdb9328d824a27b682e77c7b9928be`.

```bash
vllm serve <text-decoder-model> \
  --no-async-scheduling \
  --default-user-output-tokens-per-second 100 \
  --user-output-tokens-per-second '{"user-a":100,"user-b":500}'
```

Both limits must be positive integers. The default is required when overrides
are supplied. Leaving the default unset disables the feature and preserves the
existing scheduler behavior. Omitted async scheduling is set to false when the
feature is enabled; explicitly enabling async scheduling is an error.

A trusted gateway supplies the authenticated identity through `vllm_xargs`:

```json
{
  "model": "<served-model-name>",
  "messages": [{"role": "user", "content": "Explain continuous batching."}],
  "max_tokens": 256,
  "stream": true,
  "stream_options": {"include_usage": true},
  "vllm_xargs": {"output_rate_user": "user-a"}
}
```

The OpenAI-compatible `user` field is not the identity for this feature.
Requests with no `output_rate_user` share the `__anonymous__` budget. Different
nonempty identities receive separate default budgets unless an override exists.
The vLLM endpoint must not be directly accessible to untrusted clients: this
field is not authentication. The gateway must replace any caller-provided
`output_rate_user` with its verified identity. Keep all concurrent requests for
one identity on the same engine. Sending them to two engines gives two budgets.
This patch does not implement a distributed identity owner or a gateway.

## Accounting and waiting

The scheduler reserves one possible output before dispatching a step that can
sample a token. The final prefill step also needs a reservation because it can
produce the first output token. Earlier prefill chunks do not consume output
allowance. If allowance is exhausted and chunked prefill is enabled, prefill can
progress up to the position before that first output is sampled. When chunked
prefill is disabled, including for non-causal PrefixLM attention, the entire
prefill waits for allowance so its attention and KV semantics remain intact.

All requests for an identity share the same rolling one-second ledger:

```text
uncompleted output reservations + outputs completed in the last second <= limit
```

A reservation does not expire while model execution is in flight. Completed
outputs remain charged for a full second after the scheduler observes the model
result. Canceled requests that already sampled a token are still charged. EOS,
hidden reasoning, and tool-call tokens count even if a response parser does not
show them as ordinary text. Raw sampled token IDs are counted before stop
processing can trim them.

Requests with available output allowance are not required to use it all. The
limit is a maximum, not a throughput guarantee. A rolling-window ceiling permits
up to the configured number of tokens in a burst; it does not guarantee evenly
spaced tokens. HTTP buffering and network delivery may also group tokens.

Peers already admitted into RUNNING with the same identity and priority rotate
to avoid one stream always taking the next available output slot. KV admission
and existing scheduling priorities still apply. Requests waiting for an active
slot or for KV memory are not guaranteed equal progress with RUNNING requests.

Waiting inside `schedule()` would prevent the engine from processing new
requests and cancellation. Instead, after an empty quota-blocked step, the
engine waits on its input queue for at most 50 ms or until the next allowance
expires, whichever is sooner. New input wakes this wait immediately. Existing
worker cleanup is preserved, so an empty housekeeping step may still be sent to
the model executor.

## Supported configuration

The first implementation supports synchronous, text-only attention decoder
models with data parallel size 1 and pipeline parallel size 1. Tensor parallelism
within an engine can still be used. Unsupported combinations are rejected at
configuration time, including speculative decoding, custom schedulers,
multimodal/encoder-decoder/diffusion/hybrid/attention-free models, and
KV/EC/auxiliary connectors. Beam search is rejected before starting generation
when the feature is enabled. Its separate candidate expansion is outside this
sampled-token accounting contract; ordinary sampling remains supported.

Paused requests retain KV memory. Use normal admission limits and an external
queue to bound active requests. Existing capacity failures, invalid requests,
network failures, and hardware failures are not converted into successful
responses by this feature. In particular, `max_num_queued_reqs` can still reject
with 503; an external gateway is responsible for its own queueing policy.

The ledger is engine-local and in memory. Restarting an engine resets its
history. It does not provide a strict cross-restart or cross-engine ceiling.
Production routing must fence the old owner and preserve or conservatively
wait out outstanding work before assigning a user to another engine. Merely
changing a load-balancer route is insufficient.

## Standard and Fast pools

Run Standard and Fast on separate GPU groups if capacity isolation is required.
Use the same model and precision initially; tune active concurrency in each
pool using measurements. `gpu_memory_utilization` is not a GPU compute-share
control and does not provide performance isolation between processes.

A higher output allowance does not make an intrinsically slower model decode
faster. Verify single-stream speed and loaded inter-token latency before
promising a Fast throughput target. This patch implements output allowance;
it does not implement pool management, capacity reservations, or billing.

## Validation

CPU tests exercise the real scheduler and its KV manager with controlled model
runner results. No GPU kernels are involved in those tests.

```bash
.venv/bin/python -m pytest tests/v1/core/test_output_rate.py -q
.venv/bin/python -m pytest tests/v1/core/test_scheduler.py -k output_rate -q
.venv/bin/python -m pytest tests/test_config.py -k user_output_rate -q
```

Before a GPU deployment, also validate real streaming generation with multiple
concurrent requests for A and B, long prompts, prefix-cache hits, short answers,
EOS, client cancellation, and a known final `completion_tokens` total. Confirm
answers are complete, throttled requests resume, and unthrottled users progress.
Observe generation events in the engine; client receive times alone cannot
prove a strict generation-rate ceiling because network buffering can batch
otherwise separate outputs.

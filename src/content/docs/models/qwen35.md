---
title: Qwen3.5 Dense
description: "Running Qwen3.5 Dense on pegainfer: supported sizes, build, launch, serving performance, and hybrid-attention architecture notes."
tableOfContents:
  minHeadingLevel: 2
  maxHeadingLevel: 4
---

Qwen3.5 is a hybrid-attention model line: 3 of every 4 layers use linear
attention (gated delta rule), and only every 4th layer is full attention.
pegainfer serves the complete dense family — 0.8B, 2B, 4B, 9B, and 27B,
text-only — behind the `qwen35` cargo feature with CUDA Graph decode and
paged KV cache for the full-attention layers.

## Build

Qwen3.5 needs Python at **build time** because its linear-attention prefill
kernels are Triton AOT-generated.
There is no Python at runtime — the compiled kernels link into the same
single Rust binary.

```bash
# One-time: a Python environment with Triton for the AOT step
uv venv && uv pip install triton

# pegainfer picks up .venv/bin/python automatically, or point at one:
export PEGAINFER_TRITON_PYTHON=.venv/bin/python
```

## Launch

From the pegainfer workspace root:

### Qwen3.5-4B

```bash
huggingface-cli download Qwen/Qwen3.5-4B --local-dir models/Qwen3.5-4B

export CUDA_HOME=/usr/local/cuda
cargo run --release --features qwen35 -- --model-path models/Qwen3.5-4B
```

The server exposes an OpenAI-compatible `/v1/completions` endpoint:

```bash
curl -s http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "models/Qwen3.5-4B", "prompt": "The capital of France is", "max_tokens": 32}'
```

Streaming:

```bash
curl -N http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "models/Qwen3.5-4B", "prompt": "Write a haiku about Rust:", "max_tokens": 64, "stream": true}'
```

The `model` field must match the served model id — by default the
`--model-path` value, or whatever `--served-model-name` sets
(`curl http://localhost:8000/v1/models` shows it).

### Qwen3.5-0.8B

```bash
huggingface-cli download Qwen/Qwen3.5-0.8B --local-dir models/Qwen3.5-0.8B

cargo run --release --features qwen35 -- --model-path models/Qwen3.5-0.8B
```

### Qwen3.5-2B

```bash
huggingface-cli download Qwen/Qwen3.5-2B --local-dir models/Qwen3.5-2B

cargo run --release --features qwen35 -- --model-path models/Qwen3.5-2B
```

### Qwen3.5-9B

```bash
huggingface-cli download Qwen/Qwen3.5-9B --local-dir models/Qwen3.5-9B

cargo run --release --features qwen35 -- --model-path models/Qwen3.5-9B
```

### Qwen3.5-27B

```bash
huggingface-cli download Qwen/Qwen3.5-27B --local-dir models/Qwen3.5-27B

cargo run --release --features qwen35 -- --model-path models/Qwen3.5-27B
```

All five sizes are gated by the same HF bf16 logits golden tests
(short prompts plus 4097/8192-token long prompts per size).

## Performance

Two hardware suites are reported below: Qwen3.5-4B on an RTX 5070 Ti,
and GH200 family benchmarks for 0.8B, 2B, 9B, and 27B. Results from the
two suites are not directly comparable because the hardware and benchmark
runs differ.

### RTX 5070 Ti: Qwen3.5-4B

Measured on **1x RTX 5070 Ti 16GB**, driver 610.43.02, CUDA 13.3 build,
Qwen3.5-4B BF16 weights, TP1, CUDA Graph decode on, pegainfer main
`baaffd0`. `vllm-bench` client on localhost, random dataset, 1024-token
prompts, 128-token outputs, greedy, seed 42 — reproducible via
`tools/bench/run_serving_bench.sh` in the repo.

Poisson arrivals (`QPS n`) and fixed in-flight concurrency (`c=N`):

| load | req/s | out tok/s | TTFT p50 / p99 | TPOT p50 / p99 |
| ---: | ---: | ---: | ---: | ---: |
| c=1 | 0.60 | 77 | 106 / 113 ms | 12.2 / 12.2 ms |
| QPS 1 | 0.97 | 125 | 124 / 236 ms | 15.6 / 20.2 ms |
| QPS 2 | 1.94 | 249 | 129 / 377 ms | 20.1 / 29.6 ms |
| c=4 | 1.80 | 230 | 235 / 346 ms | 15.7 / 16.5 ms |

Single-stream decode (`c=1`) runs at 12.2 ms/token — about 82 tok/s per
request. All rows completed every request at the full 128-token output.

### GH200: 0.8B, 2B, 9B, and 27B

All four sizes were measured on **1x GH200 120GB** (aarch64, sm_90), BF16,
TP1, with random 1024-token prompts and 128-token greedy outputs. The 0.8B
and 2B runs used pegainfer main `aea46ed` and `vllm bench serve` 0.23.0
with range ratio 0 and no seed. The 9B and 27B runs used `ffb959c4` and the
benchmark client's default seed.

#### Qwen3.5-0.8B

The model loads in 0.15 s and the server is ready in 4.2 s. The load-time budget is 77.5 GB
of paged KV, a 1.5 GB prefill-scratch reserve, and a 2.4 GB recurrent-state
reserve.

| load | req/s | out tok/s | TTFT p50 / p99 | TPOT p50 / p99 |
| ---: | ---: | ---: | ---: | ---: |
| c=1 | 2.96 | 379 | 22.5 / 29.7 ms | 2.48 / 2.54 ms |
| c=4 | 7.94 | 1016 | 24.6 / 922 ms | 3.08 / 3.18 ms |
| c=8 | 12.50 | 1600 | 28.6 / 1081 ms | 3.61 / 3.86 ms |
| QPS 1 | 0.98 | 126 | 19.9 / 23.0 ms | 2.48 / 2.95 ms |
| QPS 2 | 1.96 | 250 | 19.9 / 26.7 ms | 2.55 / 2.98 ms |
| QPS 4 | 3.91 | 500 | 22.0 / 285 ms | 2.74 / 3.21 ms |
| QPS 8 | 7.82 | 1001 | 23.3 / 634 ms | 3.15 / 4.27 ms |
| QPS 10 | 9.77 | 1251 | 23.8 / 945 ms | 3.40 / 4.38 ms |
| QPS 12 | 11.40 | 1459 | 23.4 / 814 ms | 3.56 / 4.52 ms |
| QPS 16 | 15.38 | 1968 | 24.1 / 1002 ms | 3.88 / 5.02 ms |

QPS 16 is not saturated. A 4097-token long-context run completes 4/4 with
TTFT p50 63.2 ms. Under c=120, 120/120 requests complete with TTFT p50
4.76 s, TPOT p50 18.6 ms, and no server errors.

The short- and long-context HF logits golden gates pass all five replay
surfaces (mean error 0.027–0.030, p99 at most 0.115, max 0.276). Greedy
output is token-identical to HF on 4/6 24-token prompts. Tool calling
returns a valid `get_weather` call.

#### Qwen3.5-2B

Measured with the same GH200 and benchmark setup as 0.8B. The model loads
in 0.25 s and the server is ready in 4.7 s. The load-time budget is 75.3 GB
of paged KV, a 1.9 GB prefill-scratch reserve, and a 2.4 GB recurrent-state
reserve.

| load | req/s | out tok/s | TTFT p50 / p99 | TPOT p50 / p99 |
| ---: | ---: | ---: | ---: | ---: |
| c=1 | 2.35 | 301 | 24.9 / 30.2 ms | 3.16 / 3.20 ms |
| c=4 | 6.84 | 875 | 27.7 / 426 ms | 4.04 / 4.08 ms |
| c=8 | 10.30 | 1318 | 30.4 / 1109 ms | 4.84 / 5.13 ms |
| QPS 1 | 0.98 | 125 | 22.3 / 25.8 ms | 3.17 / 3.85 ms |
| QPS 2 | 1.94 | 249 | 26.1 / 36.8 ms | 3.34 / 3.89 ms |
| QPS 4 | 3.87 | 496 | 26.8 / 418 ms | 3.70 / 4.48 ms |
| QPS 8 | 7.76 | 994 | 28.3 / 644 ms | 4.43 / 5.94 ms |
| QPS 10 | 9.72 | 1245 | 28.9 / 649 ms | 4.74 / 6.03 ms |
| QPS 12 | 11.10 | 1421 | 29.5 / 831 ms | 5.02 / 6.99 ms |
| QPS 16 | 15.35 | 1965 | 30.4 / 973 ms | 5.81 / 7.87 ms |

QPS 16 is not saturated. A 4097-token long-context run completes 4/4 with
TTFT p50 80.6 ms. Under c=120, 120/120 requests complete with TTFT p50
6.0 s, TPOT p50 23.8 ms, no server errors, and no unknown token IDs.

The short- and long-context HF logits golden gates pass all five replay
surfaces (mean error 0.023–0.029, p99 at most 0.110, max 0.131). Greedy
output is token-identical to HF on 5/6 24-token prompts. Tool calling
returns a valid `get_weather` call. Both small sizes exercise the previously
untested GDN expansion-factor-1 path.

#### Qwen3.5-9B

Load to HTTP-ready is 5.6 s warm.

| load | req/s | out tok/s | TTFT p50 / p99 | TPOT p50 / p99 |
| ---: | ---: | ---: | ---: | ---: |
| c=1 | 1.02 | 131 | 51 / 55 ms | 7.3 / 7.3 ms |
| c=4 | 2.75 | 352 | 64 / 794 ms | 10.4 / 10.4 ms |
| c=8 | 4.03 | 516 | 110 / 1135 ms | 14.0 / 14.4 ms |
| QPS 1 | 0.95 | 122 | 59 / 91 ms | 7.9 / 10.1 ms |
| QPS 2 | 1.86 | 238 | 62 / 137 ms | 9.8 / 12.2 ms |
| QPS 4 | 3.58 | 459 | 70 / 174 ms | 15.5 / 22.2 ms |
| QPS 8 | 5.69 | 729 | 366 / 2546 ms | 60.3 / 70.9 ms |
| QPS 10 | 6.33 | 806 | 1.3 / 5.4 s | 66.7 / 70.3 ms |
| QPS 12 | 6.48 | 830 | 4.5 / 8.0 s | 68.9 / 70.0 ms |
| QPS 16 | 6.70 | 857 | 6.4 / 16.0 s | 68.7 / 70.0 ms |

The single GPU saturates around 6.7 req/s and ~857 output tok/s at this shape. Long-context at in=4097 / out=32, c=1 holds TTFT p50 at 194 ms; a c=120 overload with 4096-token prompts completes 120/120 with no OOM — TTFT there is pure queueing. A `get_weather` tool-call round-trip through `/v1/chat/completions` returns well-formed `tool_calls`.

Greedy output matches HF `transformers` (bf16, same GPU) token-for-token on 5 of 6 test prompts over the first 20 tokens; the sixth flips at around token 13. The per-size HF logits golden gate passes (mean logit delta 0.022–0.024, p99 ≤ 0.090).

#### Qwen3.5-27B

Load to HTTP-ready is 5.6 s warm. The load-time budget is 17.2 GB of paged
KV, a 5.5 GB prefill-scratch reserve, and an 18.8 GB recurrent-state reserve
— two ~147 MB linear-attention states budgeted per decode slot across the
full 64-slot capacity.

| load | req/s | out tok/s | TTFT p50 / p99 | TPOT p50 / p99 |
| ---: | ---: | ---: | ---: | ---: |
| c=1 | 0.37 | 47 | 159 / 173 ms | 20.3 / 20.3 ms |
| c=4 | 1.02 | 131 | 211 / 1309 ms | 28.3 / 28.5 ms |
| c=8 | 1.55 | 198 | 339 / 1495 ms | 37.5 / 39.1 ms |
| c=32 | 2.35 | 301 | 0.7 / 6.5 s | 96.3 / 98.7 ms |
| c=48 | 2.48 | 318 | 0.9 / 10.1 s | 134.5 / 140.7 ms |
| QPS 1 | 0.87 | 112 | 219 / 511 ms | 29.2 / 32.9 ms |
| QPS 2 | 1.57 | 201 | 225 / 709 ms | 43.2 / 50.1 ms |
| QPS 4 | 2.33 | 299 | 383 / 1234 ms | 107.5 / 124.3 ms |
| QPS 8 | 2.62 | 336 | 4.7 / 22.4 s | 164.5 / 180.0 ms |
| QPS 10 | 2.58 | 331 | 17.0 / 36.1 s | 175.1 / 185.0 ms |
| QPS 12 | 2.64 | 338 | 20.9 / 42.8 s | 177.5 / 185.0 ms |
| QPS 16 | 2.67 | 342 | 27.5 / 66.2 s | 179.8 / 184.6 ms |

The single GPU saturates around 2.7 req/s and ~340 output tok/s at this shape; past QPS 8 throughput is flat and TTFT grows with queueing. Long-context at in=4097 / out=32, c=1 holds TTFT p50 at 587 ms; a c=120 overload with 4096-token prompts completes 120/120 with zero server-side errors. A `get_weather` tool-call round-trip returns well-formed `tool_calls`.

Greedy output matches HF `transformers` (bf16, same GPU) token-for-token on 4 of 6 test prompts over the first 20 tokens, with flips at near-tie logit positions. The per-size HF logits golden gate passes (mean logit delta 0.020–0.022, max 0.206).

## Notes

- Only the full-attention layers (1 in 4) keep a paged KV cache; the linear-attention layers carry a fixed-size per-request recurrent state (~49 MB at 9B, ~147 MB at 27B), so KV memory grows with context length at 1/4 the rate of a full-attention stack.
- The recurrent state is reserved at load for the full decode-batch capacity, ahead of KV-pool sizing — at 27B, 18.8 GB for the default 64 decode slots (two ~147 MB states per slot). `--max-batch` (1..=64, default 64) lowers that capacity and hands most of the freed reserve back to KV-pool sizing on tighter-VRAM GPUs.
- Token selection is bounded to the tokenizer-decodable vocab (248077 ids; the checkpoint pads `lm_head` to 248320), so sampling never lands on an id the tokenizer cannot decode.
- CUDA Graph decode is always on for Qwen3.5 — the batched decode path is built around graph replay, and the server rejects `--cuda-graph=false`. Greedy and sampled decoding are supported; prefix caching is not yet wired up for the hybrid KV/recurrent state.

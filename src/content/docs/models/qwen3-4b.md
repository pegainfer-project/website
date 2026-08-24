---
title: Qwen3 Dense
description: "Running Qwen3 Dense on pegainfer: supported sizes, launch, serving performance, speculative decoding, and architecture notes."
tableOfContents:
  minHeadingLevel: 2
  maxHeadingLevel: 4
---

The complete Qwen3 dense family — 0.6B, 1.7B, 4B, 8B, 14B, and 32B — is
the default pegainfer model line: pure Rust + CUDA, no Python at build time or runtime,
full-attention GQA, paged KV cache, prefix caching, CUDA Graph decode,
optional pegaflow KV offload, and DSpark speculative decoding.

## Launch

From the pegainfer workspace root:

### Qwen3-4B

```bash
huggingface-cli download Qwen/Qwen3-4B --local-dir models/Qwen3-4B

export CUDA_HOME=/usr/local/cuda
cargo run --release
```

The default model path is `models/Qwen3-4B`, and `pegainfer-server` is the
workspace default member. To pass an explicit model path or port:

```bash
cargo run --release -p pegainfer-server -- \
  --model-path models/Qwen3-4B \
  --port 8000
```

The server exposes an OpenAI-compatible `/v1/completions` endpoint:

```bash
curl -s http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "models/Qwen3-4B", "prompt": "The capital of France is", "max_tokens": 32}'
```

Streaming:

```bash
curl -N http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "models/Qwen3-4B", "prompt": "Write a haiku about Rust:", "max_tokens": 64, "stream": true}'
```

Useful Qwen3 flags:

```bash
# Disable CUDA Graph for debugging
cargo run --release -- --cuda-graph=false

# Pure host-tier KV offload benchmark mode
cargo run --release -- \
  --kv-offload \
  --kv-offload-host-gib 16 \
  --no-prefix-cache

# DSpark speculative decoding (greedy, single-GPU)
cargo run --release -- \
  --model-path models/Qwen3-4B \
  --dflash-draft-model-path models/dspark_qwen3_4b_block7
```

### Qwen3-0.6B

```bash
huggingface-cli download Qwen/Qwen3-0.6B --local-dir models/Qwen3-0.6B

cargo run --release -- --model-path models/Qwen3-0.6B
```

### Qwen3-1.7B

```bash
huggingface-cli download Qwen/Qwen3-1.7B --local-dir models/Qwen3-1.7B

cargo run --release -- --model-path models/Qwen3-1.7B
```

### Qwen3-8B

Qwen3-8B uses the same architecture (4096 hidden, 12288 intermediate, 36
layers) and runs on the same single GPU — just point `--model-path` at the
8B weights. No feature flags or build changes needed.

```bash
cargo run --release -- --model-path models/Qwen3-8B
```

### Qwen3-14B

Qwen3-14B's GQA group — 40 query heads over 8 KV heads — has no compiled decode kernel, so decode reroutes through the batched eager prefill path (logged as a `WARN` at load). Serving works normally; the trade-off is eager per-step decode, with batched throughput that scales well.

```bash
huggingface-cli download Qwen/Qwen3-14B --local-dir models/Qwen3-14B

cargo run --release -- --model-path models/Qwen3-14B
```

### Qwen3-32B

Qwen3-32B's BF16 weights (~63 GB) need a single large-VRAM GPU
(GH200/H200 class).

```bash
huggingface-cli download Qwen/Qwen3-32B --local-dir models/Qwen3-32B

cargo run --release -- --model-path models/Qwen3-32B
```

Tool calling goes through `/v1/chat/completions` with a `tools` array; a
`get_weather` round-trip returns:

```json
{"choices":[{"message":{"role":"assistant","tool_calls":[{"function":{"name":"get_weather",
  "arguments":"{\"city\": \"Paris\"}"}}]},"finish_reason":"tool_calls"}]}
```

## Performance

Two serving-benchmark suites are reported below — an engine comparison against vLLM on a consumer RTX 5090, and GH200 family benchmarks covering all six dense sizes — followed by KV offload and speculative-decoding results.

### RTX 5090: pegainfer vs vLLM

Measured on **1x RTX 5090 32GB**, driver 590.48.01, CUDA 13.1 build,
Qwen3-4B BF16 weights, TP1. pegainfer main `70888b2`, vLLM 0.24.0, same
`vllm bench serve` client, same host, same GPU, prefix cache on, seed 42,
input 1024 / output 128 for the QPS sweep.

These tables were measured with the bench harness of the time: one seed for every sweep point against one long-lived server with the prefix cache on, so later points could replay prompts earlier points had already cached. Both engines saw the identical prompt stream, so the side-by-side is like-for-like; the absolute numbers are order-dependent, and the bench script has since moved to per-point seeds, so re-runs are not directly comparable to these tables.

#### Footprint

| Metric | pegainfer | vLLM 0.24.0 |
| --- | ---: | ---: |
| RSS before stress, loaded and idle | **771 MB** | 3814 MB |
| RSS after stress | **1064 MB** | 3863 MB |
| Startup to HTTP ready, cold | **2.99 s** | 70.0 s |
| Startup, warm compile cache | **~3.0 s** | 32.7 s |
| GPU memory, default utilization | 28832 MiB | 30290 MiB |

pegainfer is a single process; vLLM RSS is summed over its process tree.
The pegainfer RSS peak during load is transient while reading safetensors
through `mmap`; steady-state settles at 771 MB after load.

#### Qwen3-4B Serving Load

Poisson arrivals, 1024-token prompts, 128-token outputs, greedy
(`--temperature 0`):

| QPS | pegainfer out tok/s | vLLM out tok/s | pegainfer TTFT p50 | vLLM TTFT p50 | pegainfer TPOT p50 | vLLM TPOT p50 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 126.3 | 126.2 | 45.2 ms | 54.9 ms | 6.53 ms | 6.71 ms |
| 2 | 252.3 | 252.2 | 30.3 ms | 38.4 ms | 6.93 ms | 7.08 ms |
| 4 | 504.1 | 503.3 | 48.8 ms | 38.7 ms | 8.30 ms | 7.95 ms |
| 8 | 1007.8 | 1006.9 | 51.1 ms | 66.9 ms | 11.39 ms | 11.97 ms |
| 10 | 1258.3 | 1256.3 | 53.4 ms | 76.3 ms | 13.55 ms | 14.11 ms |
| 12 | 1507.7 | 1506.2 | 60.0 ms | 106.0 ms | 16.75 ms | 18.36 ms |
| 16 | **1979.9** | 1687.9 | **203.8 ms** | 3832.3 ms | **46.92 ms** | 79.42 ms |

Low load (QPS 1–4) is comparable. At QPS 8–12 pegainfer leads on both TTFT
and TPOT. At QPS 16 both systems are overloaded, but pegainfer edges ahead
on throughput (1980 vs 1688 output tok/s) and stays 19× lower on TTFT.

#### Qwen3-8B Serving Load

Same harness, Qwen3-8B BF16, single RTX 5090 (32 GB). The 8B model is 2×
the weights of 4B; throughput scales accordingly until the GPU saturates
around QPS 8:

| QPS | pegainfer out tok/s | vLLM out tok/s | pegainfer TTFT p50 | vLLM TTFT p50 | pegainfer TPOT p50 | vLLM TPOT p50 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 125.1 | 125.0 | 82.2 ms | 97.4 ms | 11.55 ms | 11.63 ms |
| 2 | 249.9 | 250.0 | 54.1 ms | 61.5 ms | 11.46 ms | 11.57 ms |
| 4 | 498.6 | 498.5 | 88.1 ms | 103.6 ms | 16.08 ms | 16.24 ms |
| 8 | 991.9 | 990.4 | 148.0 ms | 235.1 ms | 30.97 ms | 35.56 ms |

#### Warm Prefix-Cache TTFT

For multi-turn chat and agent workloads, most of the prompt often lands as
a warm prefix-cache hit. In this sweep, the same prompt group is sent cold
once to populate GPU KV cache, then sent warm:

| Input length | pegainfer cold | pegainfer warm p50 | pegainfer warm p99 | vLLM warm p50 | vLLM warm p99 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 16.2 ms | 8.5 ms | 8.8 ms | 14.5 ms | 19.1 ms |
| 512 | 24.6 ms | 8.6 ms | 8.8 ms | 16.0 ms | 16.4 ms |
| 1024 | 44.0 ms | 9.2 ms | 9.5 ms | 18.4 ms | 19.0 ms |
| 2048 | 92.0 ms | 10.4 ms | 10.8 ms | 23.7 ms | 24.4 ms |
| 4096 | 211.5 ms | 12.7 ms | 13.4 ms | 34.1 ms | 36.2 ms |
| 8192 | 460.0 ms | 21.6 ms | 22.8 ms | 58.6 ms | 59.9 ms |
| 16384 | 1143.9 ms | **26.3 ms** | 27.9 ms | 95.6 ms | 98.2 ms |

pegainfer wins warm TTFT at every measured length; the 16k warm-cache path
is 3.6× faster than vLLM p50.

### GH200 Family Benchmarks

All six dense sizes were measured on **1x GH200 120GB** (aarch64, sm_90),
BF16, TP1, with random 1024-token prompts and 128-token greedy outputs.
The 0.6B and 1.7B runs used pegainfer main `aea46ed` and `vllm bench serve`
0.23.0 with range ratio 0 and no seed. The 4B through 32B runs used main
`c116077b` and seed 42; each benchmark point started a fresh server so it
could not reuse prefixes cached by an earlier point. Their low-load p99
therefore includes first-use process costs. `c=N` holds N requests in
flight; `QPS n` uses Poisson arrivals.

The `c=8` column is the one measured at the same load for all six sizes.
The 0.6B and 1.7B ladders stop at QPS 16; 4B through 32B run to QPS 32.

| Model | Decode path | c=8 out tok/s | Saturation | Queueing knee |
| --- | --- | ---: | ---: | ---: |
| 0.6B | CUDA Graph | 2805 | not reached | not reached |
| 1.7B | CUDA Graph | 2094 | not reached | not reached |
| 4B | CUDA Graph | 1204 | not reached, ≥3.7k tok/s at QPS 32 | QPS 24→32 |
| 8B | CUDA Graph | 836 | ~2.6k tok/s | QPS 20→24 |
| 14B | batched eager reroute | 504 | ~1.5k tok/s | QPS 12→16 |
| 32B | CUDA Graph | 253 | ~660–675 tok/s | QPS 4→6 |

#### Qwen3-0.6B

The model loads in 0.12 s and the server is ready in 3.4 s. The profiled KV
budget is 84885 MiB (48506 blocks).

| load | req/s | out tok/s | TTFT p50 / p99 | TPOT p50 / p99 |
| ---: | ---: | ---: | ---: | ---: |
| c=1 | 4.10 | 524 | 12.0 / 22.8 ms | 1.81 / 1.84 ms |
| c=4 | 11.58 | 1483 | 14.1 / 973 ms | 2.09 / 2.17 ms |
| c=8 | 21.92 | 2805 | 12.7 / 725 ms | 2.31 / 2.42 ms |
| QPS 1 | 0.99 | 126 | 9.5 / 19.0 ms | 1.81 / 1.99 ms |
| QPS 2 | 1.97 | 252 | 9.1 / 16.7 ms | 1.81 / 1.99 ms |
| QPS 4 | 3.93 | 503 | 9.5 / 105 ms | 1.89 / 2.02 ms |
| QPS 8 | 7.87 | 1007 | 11.1 / 344 ms | 1.98 / 2.53 ms |
| QPS 10 | 9.84 | 1260 | 10.4 / 492 ms | 1.99 / 2.26 ms |
| QPS 12 | 11.35 | 1453 | 10.1 / 835 ms | 2.03 / 2.27 ms |
| QPS 16 | 14.94 | 1913 | 10.1 / 920 ms | 2.09 / 2.49 ms |

QPS 16 is not saturated; c=8 reaches 2805 output tok/s. A 4097-token
long-context run completes 4/4 with TTFT p50 36.7 ms. Under c=120 with
4096-token prompts, 120/120 requests complete with TTFT p50 2.86 s, TPOT
p50 15.0 ms, and no server errors.

The HF logits golden gate passes all six execution paths (sequential bs=1
eager: mean 0.0387, p99 0.136, max 0.498). Greedy output is token-identical
to HF on 2/6 24-token prompts; the other continuations remain coherent.
Tool calling returns a valid `get_weather` call.

#### Qwen3-1.7B

The model loads in 0.25 s and the server is ready in 3.9 s. The profiled KV
budget is 82714 MiB (47265 blocks).

| load | req/s | out tok/s | TTFT p50 / p99 | TPOT p50 / p99 |
| ---: | ---: | ---: | ---: | ---: |
| c=1 | 3.01 | 385 | 16.0 / 21.9 ms | 2.49 / 2.51 ms |
| c=4 | 8.14 | 1042 | 18.5 / 965 ms | 2.77 / 2.86 ms |
| c=8 | 16.36 | 2094 | 19.0 / 979 ms | 3.06 / 3.29 ms |
| QPS 1 | 0.98 | 126 | 9.4 / 15.4 ms | 2.48 / 2.66 ms |
| QPS 2 | 1.96 | 250 | 10.0 / 17.3 ms | 2.52 / 2.67 ms |
| QPS 4 | 3.91 | 501 | 10.2 / 14.7 ms | 2.58 / 2.69 ms |
| QPS 8 | 7.82 | 1001 | 13.1 / 25.6 ms | 2.70 / 3.41 ms |
| QPS 10 | 9.78 | 1252 | 11.4 / 888 ms | 2.70 / 3.27 ms |
| QPS 12 | 11.20 | 1434 | 11.5 / 974 ms | 2.73 / 3.14 ms |
| QPS 16 | 15.02 | 1923 | 11.5 / 760 ms | 2.89 / 3.61 ms |

QPS 16 is not saturated. A 4097-token long-context run completes 4/4 with
TTFT p50 46.3 ms. Under c=120, 120/120 requests complete with TTFT p50
3.48 s, TPOT p50 19.7 ms, no server errors, and no unknown token IDs.

The HF logits golden gate passes all six execution paths (sequential bs=1
eager: mean 0.0332, p99 0.123, max 0.250). Greedy output is token-identical
to HF on 5/6 24-token prompts. Tool calling returns a valid `get_weather`
call.

#### Qwen3-4B

The profiled KV budget is 34783 blocks.

| load | req/s | out tok/s | TTFT p50 / p99 | TPOT p50 / p99 |
| ---: | ---: | ---: | ---: | ---: |
| c=1 | 1.74 | 223 | 26 / 35 ms | 4.3 / 4.3 ms |
| c=4 | 5.46 | 699 | 28 / 891 ms | 4.9 / 5.1 ms |
| c=8 | 9.41 | 1204 | 29 / 1120 ms | 5.6 / 5.7 ms |
| c=16 | 14.19 | 1817 | 64 / 1829 ms | 7.1 / 7.4 ms |
| c=32 | 20.52 | 2626 | 48 / 2130 ms | 10.6 / 11.1 ms |
| c=64 | 26.98 | 3454 | 53 / 2054 ms | 17.4 / 17.6 ms |
| QPS 1 | 0.97 | 124 | 29 / 40 ms | 4.5 / 5.0 ms |
| QPS 2 | 1.92 | 246 | 27 / 37 ms | 4.6 / 5.0 ms |
| QPS 4 | 3.84 | 492 | 30 / 99 ms | 4.9 / 5.5 ms |
| QPS 8 | 7.69 | 985 | 30 / 412 ms | 5.6 / 6.6 ms |
| QPS 10 | 9.43 | 1207 | 30 / 425 ms | 5.8 / 7.1 ms |
| QPS 12 | 11.51 | 1473 | 30 / 992 ms | 6.2 / 8.5 ms |
| QPS 16 | 15.01 | 1921 | 32 / 1103 ms | 7.2 / 8.9 ms |
| QPS 20 | 18.95 | 2425 | 36 / 1087 ms | 9.4 / 12.5 ms |
| QPS 24 | 22.43 | 2871 | 41 / 1059 ms | 12.5 / 14.5 ms |
| QPS 32 | 28.96 | 3707 | 239 / 2064 ms | 30.3 / 31.2 ms |

QPS 32, the last measured point, still adds throughput over QPS 24 (2871 → 3707 out tok/s) while TTFT p50 jumps from 41 to 239 ms — the ladder ends at ≥3.7k tok/s without reaching a plateau. The interactive band ends around QPS 24, where TPOT p50 crosses from 12.5 to 30 ms.

#### Qwen3-8B

The profiled KV budget is 31057 blocks.

| load | req/s | out tok/s | TTFT p50 / p99 | TPOT p50 / p99 |
| ---: | ---: | ---: | ---: | ---: |
| c=1 | 1.17 | 150 | 34 / 43 ms | 6.4 / 6.5 ms |
| c=4 | 3.70 | 473 | 40 / 871 ms | 7.4 / 7.4 ms |
| c=8 | 6.53 | 836 | 62 / 1417 ms | 8.3 / 8.5 ms |
| c=16 | 10.50 | 1344 | 72 / 1248 ms | 10.6 / 10.9 ms |
| c=32 | 14.42 | 1846 | 76 / 2288 ms | 15.5 / 15.9 ms |
| c=64 | 18.55 | 2375 | 80 / 2373 ms | 25.3 / 29.0 ms |
| QPS 1 | 0.96 | 123 | 39 / 43 ms | 6.7 / 7.3 ms |
| QPS 2 | 1.89 | 242 | 40 / 58 ms | 7.0 / 7.6 ms |
| QPS 4 | 3.77 | 483 | 42 / 69 ms | 7.7 / 9.1 ms |
| QPS 8 | 7.55 | 967 | 44 / 327 ms | 9.2 / 11.7 ms |
| QPS 10 | 9.39 | 1201 | 46 / 859 ms | 10.1 / 12.6 ms |
| QPS 12 | 11.16 | 1429 | 48 / 885 ms | 11.3 / 13.8 ms |
| QPS 16 | 14.87 | 1903 | 60 / 861 ms | 16.9 / 19.3 ms |
| QPS 20 | 17.72 | 2268 | 119 / 1122 ms | 31.5 / 38.7 ms |
| QPS 24 | 19.99 | 2559 | 0.6 / 2.0 s | 44.8 / 47.0 ms |
| QPS 32 | 20.59 | 2636 | 3.4 / 7.0 s | 45.5 / 47.6 ms |

Throughput saturates around 20 req/s and ~2.6k out tok/s — QPS 24–32 hold 2559–2636 tok/s. The interactive band ends around QPS 16, where TPOT p50 crosses from 17 to 32 ms.

#### Qwen3-14B

Decode runs on the batched eager reroute (no CUDA Graph at this size). Load to HTTP-ready is 8.7 s warm; the profiled KV budget is 57.5 GB (23020 blocks).

| load | req/s | out tok/s | TTFT p50 / p99 | TPOT p50 / p99 |
| ---: | ---: | ---: | ---: | ---: |
| c=1 | 0.66 | 84 | 57 / 68 ms | 11.5 / 11.5 ms |
| c=4 | 2.24 | 287 | 71 / 982 ms | 12.7 / 12.7 ms |
| c=8 | 3.94 | 504 | 130 / 848 ms | 14.2 / 14.7 ms |
| c=16 | 6.06 | 776 | 133 / 1771 ms | 17.8 / 24.0 ms |
| c=32 | 9.24 | 1182 | 131 / 2552 ms | 25.2 / 25.8 ms |
| c=64 | 11.51 | 1473 | 136 / 4129 ms | 41.6 / 42.0 ms |
| QPS 1 | 0.93 | 119 | 70 / 104 ms | 12.2 / 13.2 ms |
| QPS 2 | 1.81 | 232 | 71 / 115 ms | 12.8 / 14.3 ms |
| QPS 4 | 3.61 | 463 | 75 / 165 ms | 14.6 / 18.7 ms |
| QPS 6 | 5.43 | 695 | 81 / 329 ms | 17.3 / 21.2 ms |
| QPS 8 | 7.25 | 928 | 87 / 254 ms | 21.2 / 27.0 ms |
| QPS 10 | 8.90 | 1140 | 112 / 1157 ms | 27.9 / 32.4 ms |
| QPS 12 | 10.51 | 1345 | 165 / 980 ms | 41.9 / 48.9 ms |
| QPS 16 | 11.52 | 1475 | 1.4 / 3.3 s | 71.6 / 79.4 ms |
| QPS 20 | 11.88 | 1520 | 3.8 / 7.7 s | 74.4 / 78.8 ms |
| QPS 24 | 11.87 | 1519 | 6.0 / 12.8 s | 76.2 / 82.2 ms |
| QPS 32 | 12.13 | 1553 | 11.1 / 22.3 s | 78.9 / 79.2 ms |

The eager batched-decode path saturates around 12 req/s and ~1.5k out tok/s — QPS 16–32 hold 1475–1553 tok/s while TTFT grows with queueing, and c=64 lands on the same plateau. The interactive band ends around QPS 10–12, where TPOT p50 crosses from 28 to 42 ms.

From the earlier `ffb959c4` sweep on the same GPU class: long-context at in=4097 / out=32, c=1 holds TTFT p50 at 224 ms; a c=120 overload with 4096-token prompts (~507k aggregate demanded tokens against the 23020-block pool) completes 120/120 with zero server-side errors; a `get_weather` tool-call round-trip returns well-formed `tool_calls`. Greedy output matches HF `transformers` (bf16, same GPU class) token-for-token on 4 of 6 test prompts over the first 20 tokens. One conspicuous completion, web-forum mimicry on a malformed arithmetic prompt, is token-for-token identical in HF. The per-size HF logits golden gate passes at 14B.

#### Qwen3-32B

Load to HTTP-ready is 46 s cold; the profiled KV budget is 21.4 GB (5360 blocks) next to the 63 GB of weights.

| load | req/s | out tok/s | TTFT p50 / p99 | TPOT p50 / p99 |
| ---: | ---: | ---: | ---: | ---: |
| c=1 | 0.35 | 45 | 134 / 144 ms | 21.4 / 21.4 ms |
| c=4 | 1.16 | 149 | 163 / 1113 ms | 25.2 / 25.3 ms |
| c=8 | 1.98 | 253 | 289 / 1783 ms | 28.7 / 29.7 ms |
| c=16 | 3.05 | 391 | 291 / 2555 ms | 38.2 / 39.2 ms |
| c=32 | 4.20 | 538 | 292 / 4389 ms | 56.8 / 57.8 ms |
| c=64 | 5.18 | 663 | 303 / 8855 ms | 92.9 / 93.8 ms |
| QPS 1 | 0.87 | 111 | 156 / 276 ms | 25.2 / 29.2 ms |
| QPS 2 | 1.65 | 211 | 160 / 329 ms | 29.5 / 37.9 ms |
| QPS 4 | 3.24 | 414 | 261 / 561 ms | 51.6 / 66.5 ms |
| QPS 6 | 4.37 | 560 | 0.7 / 2.0 s | 79.7 / 104.2 ms |
| QPS 8 | 4.81 | 616 | 1.9 / 6.1 s | 94.0 / 105.6 ms |
| QPS 10 | 4.78 | 612 | 5.7 / 11.3 s | 103.8 / 108.8 ms |
| QPS 12 | 5.00 | 640 | 8.5 / 16.9 s | 104.7 / 106.2 ms |
| QPS 16 | 5.10 | 652 | 12.9 / 28.0 s | 105.7 / 106.0 ms |
| QPS 20 | 5.13 | 656 | 20.1 / 39.4 s | 106.2 / 106.5 ms |
| QPS 24 | 5.26 | 673 | 24.9 / 49.3 s | 106.0 / 106.4 ms |
| QPS 32 | 5.27 | 675 | 36.3 / 71.5 s | 105.8 / 106.6 ms |

The single GPU saturates around 5.2 req/s and ~660–675 out tok/s — QPS 12–32 hold 640–675 tok/s while TTFT is pure queueing, and c=64, past the ≤32-batch SplitKv decode-attention boundary, lands on the same plateau. The interactive band ends around QPS 4–6 (TPOT p50 52 → 80 ms, TTFT p50 261 → 709 ms).

From the earlier `5959f05` run: greedy output matches HF `transformers` (bf16, same GPU) token-for-token on 4 of 5 test prompts over the first 20 tokens. The fifth diverges at the second generated token, where HF's own top-4 logits sit within a 0.375 spread and pegainfer emits HF's second-ranked token, 0.25 below the top.

### KV Offload

With `--kv-offload`, sealed Qwen3 KV blocks can be restored from the
pegaflow host tier instead of recomputing full prefill. Measured on the
same RTX 5090 setup as the engine comparison above. The pure-L2 mode
below disables cross-request HBM prefix reuse, so every prefix hit is
restored from host DRAM:

```bash
cargo run --release -- \
  --kv-offload \
  --kv-offload-host-gib 16 \
  --no-prefix-cache
```

| Input length | Cold full prefill | L2 warm p50, host restore | Speedup |
| ---: | ---: | ---: | ---: |
| 256 | 25.4 ms | 9.8 ms | 2.6x |
| 512 | 25.6 ms | 11.6 ms | 2.2x |
| 1024 | 45.3 ms | 15.4 ms | 2.9x |
| 2048 | 92.5 ms | 22.9 ms | 4.0x |
| 4096 | 211.1 ms | 37.5 ms | 5.6x |
| 8192 | 461.3 ms | 71.4 ms | 6.5x |
| 16384 | 1140.5 ms | **125.5 ms** | 9.1x |

At 16k, the tiering picture is: HBM hit about 26 ms, host-tier restore
about 126 ms, cold prefill about 1.14 s.

### DSpark Speculative Decoding

[DSpark](https://huggingface.co/deepseek-ai/dspark_qwen3_4b_block7)
(DeepSeek-AI, Jun 2026) adds a semi-autoregressive Markov head to a DFlash
parallel drafter, raising accepted draft length by conditioning each block
position on the previously sampled token. pegainfer supports it behind
`--dflash-draft-model-path` — the drafter checkpoint goes in, the target
model serves as-is, and greedy verify keeps output lossless.

```bash
# Download the released DSpark block7 drafter
huggingface-cli download deepseek-ai/dspark_qwen3_4b_block7 \
  --local-dir models/dspark_qwen3_4b_block7

# Launch with speculative decoding (greedy, single-GPU)
cargo run --release -- \
  --model-path models/Qwen3-4B \
  --dflash-draft-model-path models/dspark_qwen3_4b_block7
```

Single-stream TPOT drops from 5.8 ms to 3.0 ms — roughly 2× decode
speedup from amortizing target forwards over accepted drafts. Concurrency
sweep on the same RTX 5090 setup as the engine comparison above, greedy,
sharegpt + SPEED-Bench (coding) datasets:

**ShareGPT:**

| Concurrency | baseline tok/s | DSpark tok/s | baseline TPOT p50 | DSpark TPOT p50 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 170 | **381** | 5.83 ms | 2.96 ms |
| 4 | 576 | **1288** | 6.72 ms | 3.59 ms |

**SPEED-Bench (coding):**

| Concurrency | baseline tok/s | DSpark tok/s | baseline TPOT p50 | DSpark TPOT p50 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 164 | **314** | 5.87 ms | 3.07 ms |
| 4 | 574 | **988** | 6.73 ms | 3.77 ms |

DSpark gains 2.2× throughput on ShareGPT and 1.7–1.9× on coding, roughly halving TPOT on both.

DFlash (the non-Markov predecessor,
[`dflash_qwen3_4b_block7`](https://huggingface.co/deepseek-ai/dflash_qwen3_4b_block7))
is also supported via the same flag with a DFlash-format drafter checkpoint.
DSpark is the recommended drafter for Qwen3-4B.

## Architecture Notes

- Full attention with grouped-query attention: 32 query heads, 8 KV heads, head dim 128, 36 layers. Qwen3-14B widens to 40 query heads over 40 layers (GQA group 5 — batched eager decode reroute); Qwen3-32B scales to 64 query heads and 64 layers (GQA group 8).
- Paged KV cache uses full-lifetime admission, so requests that cannot fit
  are rejected instead of hanging under memory pressure.
- Prefix cache is on by default; `--no-prefix-cache` disables GPU prefix
  matching, or becomes pure-L2 host restore mode when combined with
  `--kv-offload`.
- CUDA Graph decode uses pre-allocated buffers and can be disabled with
  `--cuda-graph=false` for debugging.
- DSpark/DFlash speculative decoding is single-GPU, greedy-only, and forces
  prefix caching off (the drafter needs clean target hidden states).

---
title: Gemma 4
description: "Running Gemma 4 on pegainfer: the 12B dense line and the 26B-A4B routed line, build with the gemma4 feature, launch, the memory envelope a slot count buys, long-context profiles, and heterogeneous attention notes."
tableOfContents:
  minHeadingLevel: 2
  maxHeadingLevel: 3
---

Gemma 4 uses two kinds of attention in one model. Five of every six layers use sliding-window local attention at head dim 256. Every sixth layer, and the last one, uses global attention at head dim 512 with a single KV head. pegainfer serves the Gemma 4 text stack behind the `gemma4` cargo feature. Each kind gets its own paged KV pool, and decode runs as bucketed CUDA graphs with per-iteration scheduling.

Two checkpoints are served today. 12B is the dense line. 26B-A4B is the routed one: 128 routed experts with top-8 routing beside a dense branch on every MoE layer, served from the published NVFP4 checkpoint with the expert weights kept packed on the device. Everything below applies to both lines unless a section says otherwise.

## Build

```bash
cargo build --release --features gemma4 -p pegainfer-server
```

No Python at build time or runtime.

## Launch

```bash
huggingface-cli download google/gemma-4-12B-it --local-dir models/gemma-4-12B-it

target/release/pegainfer \
  --model-path models/gemma-4-12B-it \
  --served-model-name gemma-4-12b-it \
  --port 8000
```

```bash
curl -s localhost:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"gemma-4-12b-it","prompt":"<bos>The capital of France is",
       "max_tokens":16,"temperature":0}'
```

`/v1/completions` takes the prompt verbatim. BOS comes from the chat template, so a raw completions prompt has to carry its own `<bos>`. Without it the model reads a different first position. `/v1/chat/completions` renders the template and needs no such prefix.

Gemma 4's published generation defaults are sampled, not greedy (`temperature 1.0`, `top_k 64`, `top_p 0.95`), so a greedy comparison has to pass `"temperature": 0` explicitly.

The routed line launches the same way, pointed at the NVFP4 checkpoint:

```bash
target/release/pegainfer \
  --model-path models/Gemma-4-26B-A4B-NVFP4 \
  --port 8000
```

The loader validates the routed layer's manifest before the first upload and fails closed on a missing, extra, mis-shaped or mis-typed tensor. The experts are repacked once at load for the Marlin NVFP4 GEMM and read in their stored format from then on; the dense projections stay BF16.

## Memory footprint

The KV pools are allocated for every decode slot at startup, so the footprint is fixed before the first request arrives. At the defaults (16 slots, an 8192-token ceiling) 12B BF16 sits at 32.3 GiB while idle. That is 22.18 GiB of weights, 9.27 GiB of pools, and the rest CUDA context, RoPE tables, step buffers and the captured decode graphs. A 32 GiB card cannot start that configuration.

One request needs about 2.6 GiB of pool, not 9.27. The slot count is what moves the floor: `PEGAINFER_DECODE_SLOTS` takes 1 to 16 and trades concurrency for headroom.

These figures were measured on a 48 GiB card, which is the hardware used throughout this page, not a requirement.

| Configuration | Idle resident |
| --- | ---: |
| Default (8192 ceiling x 16 slots) | 32.3 GiB |
| `--cuda-graph=false` | 32.2 GiB |
| 64K ceiling x 16 slots | 46.0 GiB |
| 128K ceiling x 8 slots | 43.5 GiB |
| 262K ceiling x 4 slots | 42.6 GiB |

26B-A4B on the long-context profile below held 29,917 MiB at its peak through the measurement further down, weights and pools included. That is a peak under load rather than the idle resident figure this table reports.

## Scheduling

One engine loop admits whatever the pools can hold, up to the slot count. Then every active request advances one token in a single batched decode step, sharing one pass over the weights.

A new prompt does not freeze the batch. Its rows are computed in the same step as the live streams' next token. Under a flood of one-token requests a stream's gap between tokens stays around 30 ms, whether 16, 48 or 96 requests are queued. Up to four prompts that arrive together are handled in one step, which keeps a burst from admitting one request at a time.

A request past the slot count waits at the head of the queue. It is only refused when nothing else is running, because then no other request's pages can free up.

The two kinds of attention get separate budgets, because they store different shapes and free their pages on different rules. Local layers free the front of the window past 1024 tokens. Global layers never free anything, so their pool is the slot count times the ceiling.

| Knob | Default | What it binds |
| --- | --- | --- |
| `PEGAINFER_DECODE_SLOTS` | 16 | requests beyond this queue |
| `PEGAINFER_MAX_CONTEXT` | 8192 | prompt + `max_tokens`, checked at admission |
| Page size | 16 tokens | both families |
| Sliding window | 1024 tokens | local family only |

## Opt-in profiles

Four profiles. Leave them alone and you get an 8192-token ceiling, 16 decode slots, no prefix cache, no async lane and whole-prompt prefill. The cache, the lane and the chunked walk allocate nothing until you set them.

### Conversation prefix cache

`PEGAINFER_PREFIX_CACHE=K` keeps up to K finished prompt states. The next turn of a conversation then starts where its history ended. On a hit the engine copies those pages back and prefills only the new part of the prompt, and `Scheduled` reports how much was reused as `cached_tokens`.

Two things never hit. Prompts longer than half the serving ceiling are not saved, and a prompt that reaches back past the freed sliding window cannot be rebuilt.

The cache allocates its pages at startup. At `K=16` the idle footprint is 38.3 GiB, against 32.3 GiB with the cache off.

### Async prefill lane

`PEGAINFER_ASYNC_PREFILL=green:NN` runs a new prompt's prefill on its own CUDA stream while decode keeps going. A Green Context caps that stream at roughly NN% of the SMs. The cap is the point: without it, prefill takes the whole GPU and decode stalls.

Under a flood of sixteen ~1900-token prompts at `green:35`, a live stream's worst gap between tokens drops from 387–452 ms to 75–76 ms, and its p99 drops from 385–432 ms to 39–40 ms. The flood pays for that. Its own TTFT p50 rises from 3.3–3.7 s to 9.8–10.3 s.

Use it when decode latency matters more than prefill latency. At light load it only costs TTFT, and an idle lane allocates nothing.

### Chunked walk

`PEGAINFER_MIX_CHUNK_TOKENS=N` limits how many prompt rows one step computes. Live streams then advance once per segment instead of waiting for a whole prompt. Pages are reserved one round at a time, so a prompt only holds its window plus the segment being written, not its whole length.

What you trade is segment size, and it can go either way. Under a flood of sixteen ~3900-token prompts, `N=2048` cut a live stream's p99 gap from 855–975 ms to 501–526 ms. At ~1900-token prompts one round can cover two prompts, and the same setting raised that p99 from 432–468 ms to 519–537 ms. Set it when prompts are long enough to span several segments.

### Raised ceiling

`PEGAINFER_MAX_CONTEXT=N` sets the serving ceiling anywhere from 1024 to the checkpoint's 262144, so it lowers the ceiling as well as raises it. Above the 8192 default you must also set the chunked walk, because one whole scan would hold the entire context in sliding pages. The async prefill lane is refused above the default for the same reason.

At the full ceiling a 49K-token prompt returns its first token in about 13 s, and a 204K-token prompt in about 125 s. Prefill costs roughly 0.21 ms per token, plus a quadratic global-attention term that catches up with the linear one around 200K. Decode slows down much less: the median gap between tokens goes from 28.9 ms at a 13K history to 34.9 ms at 200K. Short prompts are unaffected, still ~148 tok/s at c=8 with a 64K ceiling and 16 slots.

Smaller segments are nearly free here. With two ~49K prompts arriving together, a live stream's p99 gap falls from 2613–2673 ms at `N=8192` to 381–403 ms at `N=1024`, and admission latency does not move. At `N=512` the per-step cost shows up: the stall drops to 214–224 ms, but admission and wall time cost about 13% more. `N=1024` is the recommended long-context profile; `512` is the tail-protective option.

The long-context profile both measurements below use is:

```bash
PEGAINFER_MAX_CONTEXT=262144 PEGAINFER_MIX_CHUNK_TOKENS=2496 PEGAINFER_DECODE_SLOTS=2
```

## Performance

Both lines were measured against vLLM 0.23.1rc1 on the same card. vLLM's own `vllm bench serve` client drove both engines, so the metric definitions, the percentile maths and the request pacing are vLLM's code, used the same way on both sides.

Single GPU (sm_89, x86_64), 49,140 MiB, driver 570.211.01, CUDA 12.9.

The protocol is the same for both lines. One request in flight, four prompt lengths, four rounds interleaved so each engine leads twice, and the prompt-length order rotating each round so no length always runs in the same thermal position. Each engine started fresh per phase, and startup time enters no metric. The seed is the round number, so both engines within a round see identical prompts and different rounds use different ones. One discarded warm-up per length per boot, then three kept requests — 12 per engine per length, 96 in all. A cell counted only if every generation was exactly 256 tokens, and all 32 passed on both lines. Prefix caching is off on both sides, and random prompts share no prefix by construction.

vLLM ran at its defaults with `--max-model-len 262144` and `--gpu-memory-utilization 0.92`, which restate what it would have chosen anyway, plus `--no-enable-prefix-caching` and `--language-model-only` for symmetry. No `VLLM_*` environment variable was set. It picked a 2496-token chunked prefill width for itself, and ran with `torch.compile`, CUDA graphs and FlashInfer sampling. pegainfer ran the long-context profile above, with the prefill width set to vLLM's choice.

### 12B against vLLM

Checkpoint `google/gemma-4-12B-it` at revision `707f0a3b8a3c`. BF16 weights and KV on both engines. pegainfer `e38dfdd8`.

![Time to first token and time per output token versus prompt length, PegaInfer against vLLM, at 10.6K / 40K / 81.7K / 163K prompt tokens](/models/gemma4/perf.svg)

*One request in flight, four prompt lengths. Medians of four interleaved rounds, 12 kept requests per point.*

The two prefill widths are close but not equal. pegainfer was given vLLM's 2496 rather than a width of its own choosing, but its steps round down to whole 128-row tiles, so the effective width was 2432. vLLM will not go below 2496 for this model, so the two cannot be set to the same number. The smaller width means more steps on the pegainfer side.

The figure plots medians. The table below adds end-to-end latency, and the paired deltas computed inside each round, where both engines saw the same prompts. Negative means pegainfer is faster.

| prompt tokens | E2EL pegainfer | E2EL vLLM | E2EL Δ | paired Δ TTFT | paired Δ TPOT | paired Δ E2EL |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10,602 | 10.34 s | 10.84 s | −4.6% | −0.5 … −4.1% | −4.1 … −7.3% | −3.7 … −5.8% |
| 40,002 | 20.81 s | 23.36 s | −10.9% | −11.3 … −12.9% | −8.7 … −11.1% | −10.5 … −12.3% |
| 81,653 | 41.35 s | 48.92 s | −15.5% | −16.4 … −18.2% | −7.6 … −9.6% | −14.8 … −16.5% |
| 163,336 | 96.45 s | 125.00 s | −22.8% | −23.4 … −24.8% | −7.8 … −8.9% | −22.1 … −23.5% |

Round-to-round spread is small. The widest is 4.0%, on vLLM at the shortest prompt. At the longest prompt it is 1.7% for pegainfer and 0.2% for vLLM.

The shortest prompt is the one to be careful with. At 10,602 tokens the two engines' own ranges overlap, and the paired deltas, while all four negative, run from −0.5% to −4.1%. Read that point as roughly even, not a win.

Peak GPU memory, sampled every 100 ms, was 34,354 MiB against vLLM's 47,308 MiB. vLLM's number is what its 0.92 utilization setting reserves up front, not what it actually used.

### 26B-A4B against vLLM

The ModelOpt NVFP4 export of `google/gemma-4-26B-A4B-it`. pegainfer `e7a41975`.

This run adds one flag to the four above: `--kv-cache-dtype bfloat16`. The checkpoint's quantization config declares an FP8 KV cache, and vLLM honours it by storing the cache as fp8_e4m3 by default. pegainfer stores BF16, so the like-for-like comparison sets vLLM to BF16 too, and the second table below reports vLLM at its own default, because that default costs it real time at depth.

![Time to first token and time per output token versus prompt length, pegainfer against vLLM at like-for-like KV precision, at 10.6K / 40K / 81.7K / 163K prompt tokens](/models/gemma4/perf-26b.svg)

*One request in flight, four prompt lengths, the KV cache at BF16 on both engines. Medians of four interleaved rounds, 12 kept requests per point.*

| prompt tokens | E2EL pegainfer | E2EL vLLM | E2EL Δ | paired Δ TTFT | paired Δ TPOT | paired Δ E2EL |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10,602 | 3.20 s | 3.30 s | −2.9% | −4.7 … −8.8% | −0.0 … −0.8% | −1.5 … −3.4% |
| 40,002 | 7.27 s | 8.44 s | −13.8% | −16.4 … −18.4% | −4.7 … −5.8% | −12.8 … −14.5% |
| 81,653 | 16.28 s | 21.45 s | −24.1% | −26.2 … −27.5% | −6.1 … −7.0% | −23.6 … −24.8% |
| 163,336 | 43.91 s | 72.98 s | −39.8% | −41.3 … −41.4% | −7.0 … −7.7% | −39.7 … −39.8% |

Round-to-round spread is at most 3.4%, on vLLM at the shortest prompt; pegainfer stays within 1.9% everywhere and 0.7% at the longest prompt.

**vLLM at its own default.** The same protocol with vLLM storing its KV cache as fp8_e4m3, the four shared flags only, against pegainfer at BF16 KV, measured the same morning on the commit one frontend change earlier (`ea02a9f7`; the change since removes a per-request cost of about 40 ms on pegainfer's side, visible only at the shortest prompt):

| prompt tokens | E2EL pegainfer | E2EL vLLM (fp8 KV) | E2EL Δ | paired Δ TTFT | paired Δ TPOT |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10,602 | 3.23 s | 3.12 s | +3.4% | +5.6 … +12.7% | +2.3 … +2.7% |
| 40,002 | 7.31 s | 7.28 s | +0.4% | −0.8 … +2.7% | +1.1 … +1.5% |
| 81,653 | 16.24 s | 16.51 s | −1.6% | −3.2 … −2.0% | +3.5 … +4.3% |
| 163,336 | 43.93 s | 45.91 s | −4.3% | −5.7 … −4.9% | +8.0 … +10.3% |

pegainfer's numbers are the same in both tables; the difference between them is what the FP8 cache buys vLLM's Triton attention at depth, where its 163K time to first token drops from 69.6 s to 43.0 s. At equal precision the gap grows with prompt length. At vLLM's default the two engines are close, with vLLM ahead at 10.6K and pegainfer ahead from 81.7K on.

Peak GPU memory, sampled every 100 ms, was 29,917 MiB against vLLM's 45,641 MiB. vLLM's number is what its 0.92 utilization setting reserves up front, not what it actually used.

### Caveats

Part of the gap at long prompts comes from the kernels, not from the engines as a whole. vLLM serves both checkpoints through its Triton attention backend, which its configuration layer selects because FlashAttention does not cover Gemma 4's two head dimensions in this version. pegainfer has kernels written for those two dimensions. On the routed line vLLM also decompresses the experts through its Marlin NvFp4 path, because the card has no native FP4; pegainfer's expert GEMM is a Marlin path too. So the deeper points compare what each engine ships for these models today, not two implementations of the same kernel, and we did not measure how much of the gap that accounts for.

The two engines also get here differently. vLLM covers 262,144 positions out of the box, while pegainfer refuses a prompt this long until someone sets the profile above. That extra setup is on pegainfer's side. Within that profile, the one setting we could have picked in our own favour is the prefill chunk width, and we used vLLM's.

These numbers say nothing about throughput under concurrency, multi-turn workloads, accuracy, or prompt lengths other than these four. Both engines were configured for the full 262,144-token context, but the longest prompt actually run is 163,336. One machine, one operator.

## Reproducibility under concurrency

A request decoded in a batch does not get the same logprobs as the same request decoded alone. What changes it is the sequence of padded batch widths its decode steps run at. That sequence depends on when the other requests in the batch start and finish.

| Contrast | Trajectory | Row's tokens | max abs delta logprob |
| --- | --- | ---: | ---: |
| One batch repeated three times | same | identical | 0.000000 |
| Companions replaced, same lengths, different content | same | identical | 0.000000 |
| Companions replaced, lengths from 2 to 1601 tokens | same | identical | 0.000000 |
| Seven short companions against seven long ones | changed | differ | 0.595163 |
| Alone against in a batch of eight | changed | differ | 0.623022 |

Keep that sequence the same and change what the other requests contain, or how long their prompts are, and the row comes out bit-identical. No other request leaks into it.

So greedy output repeats for a given workload on an otherwise idle GPU, but not across different workloads. Send the same requests the same way and you get the same tokens. Send them alongside different traffic and the batch widths change, which can flip a near-tie. Another process on the same GPU has the same effect.

Decode steps run at power-of-two batch sizes and replay as CUDA graphs captured at startup, one per size. `--cuda-graph=false` runs the same steps eagerly. Padding applies either way, so both modes do the same arithmetic.

## Architecture notes

| Property | 12B |
| --- | --- |
| Layers | 48 (8 global, 40 local) |
| Local : global pattern | every sixth layer is global, and so is the last |
| Attention heads | 16 |
| Local KV heads / head dim | 8 / 256 |
| Global KV heads / head dim | 1 / 512 |
| Local RoPE | full 256, theta 10k |
| Global RoPE | proportional 128/512, theta 1M |
| Sliding window | 1024 |
| Vocabulary | 262,144 |
| Max context | 262,144 |
| Final logit softcap | 30 |

- **Global layers have no `v_proj` tensor.** `attention_k_eq_v` is built into the checkpoint rather than decided at runtime: V is taken from the K projection's output. The two split straight after that. K goes through `k_norm` and then RoPE, V through a scale-free norm and no RoPE. Both are still written out, so the saving on global KV is in how the cache is stored, not a pointer alias.
- **The proportional RoPE is not ordinary partial RoPE.** Only 128 of the global head's 512 dimensions rotate, but the frequency denominator stays the full head dimension.
- **The text graph also needs** scaled token embeddings, RMSNorm before and after both attention and the feed-forward block, a scale-free V norm, tied embeddings, GELU-tanh, a residual layer scalar, and final logit softcapping.
- **The routed line adds a parallel expert branch**, not a shared-expert variant: every MoE layer computes its dense branch and its 128 routed experts at top-8, then combines the two. The attention and KV stack is the one above.
- 12B ships one `model.safetensors` with no index file. Its vision and audio tensors sit under `model.vision_embedder.*`, `model.embed_vision.*` and `model.embed_audio.*`. Serving is text-only: those tokens are rejected before embedding and suppressed before sampling.
- `eos_token_id` appears three times in the checkpoint's config files, with three different values. `generation_config.json` is the one the engine follows.

## Limits

- **Single GPU.** No tensor parallelism for this line yet.
- **No prefix sharing between requests.** Two live requests with the same prefix each pay for it. The conversation cache above works across turns of one conversation, not across concurrent requests.
- **Prompts prefill whole by default**, unless the chunked walk limits them.
- **KV capacity is not reported to the frontend**, so its capacity metrics stay empty for this model.
- **26B-A4B serves the NVFP4 checkpoint only.** BF16 experts are not served.
- **Text only.** Multimodal inputs are not supported, and the checkpoints' vision tower is skipped at load.
- 31B is on the roadmap, not served today.

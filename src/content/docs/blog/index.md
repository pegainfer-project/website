---
title: Blogs
description: Release notes, benchmarks, runtime design notes, and model bring-up stories from pegainfer.
---

## Posts

| Date | Post | Summary |
| --- | --- | --- |
| 2026-07-28 | [The Fastest H2D Path Wasn’t the Fastest Weight Loader](/blog/weight-loading/) | Safetensors index addressing, TP/FP8/EP sharding, two-slot pinned staging, and a startup-time comparison with vLLM. |
| 2026-07-17 | [Speculative Decoding](/blog/speculative-decoding/) | From output entropy and verification correctness to EAGLE draft models and dynamic verify length. |
| 2026-07-10 | [See Qwen3 Decode as a CUDA Graph](/blog/cuda-graph-export/) | One flag exports a detailed DOT for LLMs and a folded high-resolution PNG for people. |
| 2026-06-20 | [Co-locating Prefill and Decode on One GPU: Green Contexts for Higher Throughput](/blog/green-ctx/) | CUDA Green Contexts, SM partitioning, prefill/decode overlap, and PegaInfer benchmark results. |
| 2026-06-13 | [PegaInfer 0.1.0: Writing a Production-Grade Inference Engine in Rust](/blog/pegainfer-010/) | Rust runtime story, RTX 5090 serving benchmarks, prefix-cache TTFT, and pegaflow KV offload. |

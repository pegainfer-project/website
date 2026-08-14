---
title: GLM-5.2
description: "Running GLM-5.2-FP8 on pegainfer: build with the glm52 feature, EP decode topologies, native MTP speculative decoding, and P/D disaggregation across nodes."
---

GLM-5.2 is zai-org's large MoE model: 78 transformer layers, 256 routed
experts with 8 active per token plus one shared expert, DeepSeek-style
sparse attention (a lightweight indexer selects the top 2048 context
tokens per query), and one checkpoint-native MTP layer for speculative
decoding. The released checkpoint is FP8 (about 704 GB). pegainfer
serves it behind the `glm52` cargo feature.

GLM-5.2 support is Blackwell-only (compute capability ≥ 10.0); launch
fails closed on older GPUs. Everything on this page was run on GB300
nodes with 4 GPUs and an RDMA NIC each.

## Deployment shape

The recommended deployment is [P/D disaggregated](#pd-disaggregation):
prefill and decode run on separate nodes, connected by a KV handoff.
`--moe-topo` selects each side's sharding:

- **Prefill** — `--moe-topo tp4 --glm52-prefill-only`: 4-GPU tensor
  parallel, computes prompt KV and the initial tokens, then hands off.
- **Decode** — `--moe-topo epN` (expert parallelism): the 256 experts
  shard across N ranks, from `ep4` (one 4-GPU node) up to `ep64`
  within one NVLink domain.

An EP instance also serves complete requests on its own, with prefill
co-located — the simplest way to bring the model up, at a decode
throughput cost under load (see [Performance](#performance)).

## Build

The `glm52` feature builds the vendored DeepGEMM, FlashMLA, and DeepEP
kernels, so clone with submodules. DeepEP needs an NCCL ≥ 2.30.4
install at build time; the `nvidia-nccl-cu13` wheel works directly:

```bash
git clone --recurse-submodules https://github.com/pegainfer-project/pegainfer
cd pegainfer

pip download 'nvidia-nccl-cu13>=2.30.4' --no-deps -d /tmp/nccl
unzip /tmp/nccl/*.whl 'nvidia/nccl/*' -d /tmp/nccl
export PEGAINFER_NCCL_ROOT=/tmp/nccl/nvidia/nccl

export CUDA_HOME=/usr/local/cuda
cargo build --release --features glm52
```

cudarc loads `libnccl.so.2` at runtime, and build and runtime must
resolve to the same install — keep `$PEGAINFER_NCCL_ROOT/lib` on
`LD_LIBRARY_PATH` when launching. A mismatched system NCCL on the
library path can be silently picked up and fail the first all-reduce.

## Single-node serving

The four-GPU EP4 launch, with native MTP as the drafter:

```bash
export LD_LIBRARY_PATH=$PEGAINFER_NCCL_ROOT/lib:$LD_LIBRARY_PATH
export GLM52_DECODE_SLOTS=32 GLM52_MTP_DRAFTS=2

target/release/pegainfer-server \
  --model-path models/GLM-5.2-FP8 \
  --served-model-name glm-5.2-fp8 \
  --moe-topo ep4 \
  --glm52-native-mtp \
  --glm52-weight-staging \
  --max-model-len 131072
```

Pass `--max-model-len` explicitly; flag details are under
[Notes](#notes). The server exposes the OpenAI API:

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "glm-5.2-fp8", "messages": [{"role": "user", "content": "Why is the sky blue?"}], "max_tokens": 128}'
```

An 8-GPU node runs the same command with `--moe-topo ep8`. Wider EP
(16–64 ranks within one NVLink domain) runs one process per node, each
hosting its local ranks:

```bash
# node 0 of 4 (EP16) — binds the one-time bootstrap rendezvous
target/release/pegainfer-server ... --moe-topo ep16 \
  --glm52-ranks 0..4 --glm52-rendezvous <node0>:19200
# node 1
target/release/pegainfer-server ... --moe-topo ep16 \
  --glm52-ranks 4..8 --glm52-rendezvous <node0>:19200
```

## Native MTP

`--glm52-native-mtp` uses the checkpoint's own MTP layer as the
speculative drafter, for greedy and sampled requests. The engine
proposes up to `GLM52_MTP_DRAFTS` tokens per step (1–5, default 5) and
verifies them in the target's decode pass. On matched single-request
greedy decoding, accepted length is within 2% of the official vLLM
implementation.

Two knobs trade single-request latency against batch throughput, under
a shared 96-row step budget (`slots × (1 + drafts) ≤ 96`):

| Env | Default | Meaning |
| --- | --- | --- |
| `GLM52_DECODE_SLOTS` | 8 | Concurrent requests per rank |
| `GLM52_MTP_DRAFTS` | 5 | Draft span per proposal |

The default `8 × (1+5)` is the latency profile; throughput deployments
run `GLM52_DECODE_SLOTS=32 GLM52_MTP_DRAFTS=2`, and that is the
profile every command and measurement on this page uses. The two sides
of a P/D pair may disagree on the draft span — the decode side
truncates a longer proposal to its own setting.

## P/D disaggregation

Prefill and decode compete for the same GPUs, so under mixed load a
co-located node loses a large share of its decode throughput to
prompt processing (the measurements below put it at 1.6× at c=32).
The disaggregated deployment runs prefill on a dedicated TP4 node and
decode on an EP fleet, connected by three pieces:

- a **pegaflow metaserver** (from
  [novitalabs/pegaflow](https://github.com/novitalabs/pegaflow)) — the
  KV block catalog; the actual pages move over RDMA between engines;
- one or more **prefill instances** — `--moe-topo tp4
  --glm52-prefill-only`, native MTP on;
- one or more **decode instances** — any EP topology, native MTP on.

A prefill instance answers with the prompt's first sampled token plus a
5-token MTP proposal; the decode instance restores the transferred KV
pages, verifies that proposal as its first step, and continues
decoding. The handoff carries token ids, never re-tokenized text.

### Launch

Metaserver, on any host both engines can reach:

```bash
pegaflow-metaserver --addr 0.0.0.0:20056 --http-addr 0.0.0.0:20057
```

Prefill node (4 GPUs, TP4):

```bash
export NCCL_MNNVL_ENABLE=0   # GB300 bare metal: MNNVL breaks the TP4 all-reduce
target/release/pegainfer-server \
  --model-path models/GLM-5.2-FP8 \
  --served-model-name glm-5.2-fp8 --port 8000 \
  --tp-size 4 --moe-topo tp4 --glm52-prefill-only \
  --glm52-native-mtp --glm52-weight-staging \
  --max-model-len 131072 \
  --kv-offload --kv-offload-host-gib 640 \
  --kv-p2p-metaserver-addr http://<metaserver-ip>:20056 \
  --kv-p2p-advertise-addr <this-node-ip>:20103 \
  --kv-p2p-nics <rdma-nic>
```

Decode node (4 GPUs, EP4):

```bash
target/release/pegainfer-server \
  --model-path models/GLM-5.2-FP8 \
  --served-model-name glm-5.2-fp8 --port 8000 \
  --moe-topo ep4 \
  --glm52-native-mtp --glm52-weight-staging \
  --max-model-len 131072 \
  --kv-offload --kv-offload-host-gib 600 \
  --kv-p2p-metaserver-addr http://<metaserver-ip>:20056 \
  --kv-p2p-advertise-addr <this-node-ip>:20114 \
  --kv-p2p-nics <rdma-nic>
```

`--kv-p2p-advertise-addr` must be a literal `ip:port` reachable from
the peer. Pick ports below your ephemeral port range
(`net.ipv4.ip_local_port_range`, typically 32768+) — otherwise an
unrelated outbound connection can randomly occupy the port before the
engine binds it.

### Router

Requests enter through a router that makes two hops per request:

1. Send the request to a prefill instance with `max_tokens: 1` and
   `"kv_transfer_params": {"do_remote_decode": true, ...}`. The
   response carries a `kv_transfer_params` handoff envelope.
2. Send the original request, plus that envelope verbatim, to a decode
   instance, and stream its response to the client.

This is the same two-hop contract as the
[vLLM NIXL connector](https://docs.vllm.ai/en/stable/features/nixl_connector_usage/),
so NIXL-aware routers work unchanged. A minimal single-file proxy
(Python, stdlib only) is enough to drive it:

```bash
python3 pd_proxy.py --port 10001 \
  --prefill http://<prefill-node>:8000 \
  --decode http://<decode-node>:8000
```

<details>
<summary><code>pd_proxy.py</code></summary>

```python
#!/usr/bin/env python3
"""Minimal P/D disaggregation proxy for pegainfer GLM5.2 (stdlib only).

Two-hop flow per request, following the vLLM NIXL connector convention:

1. Send the request to a prefill server with max_tokens=1 and
   kv_transfer_params.do_remote_decode=true. The prefill server computes
   the prompt KV, saves it, and returns a kv_transfer_params handoff
   envelope (for GLM5.2 native MTP: the anchor token id plus draft
   token ids).
2. Send the original request, plus that envelope verbatim, to a decode
   server. The decode server restores the KV, verifies the proposal,
   and streams tokens back.

The envelope carries token *ids*; the proxy never rebuilds the P/D
boundary from text. Do not edit the prompt between the two hops — the
decode side asserts it tokenizes to the same committed length.

Usage:
  python3 pd_proxy.py --port 10001 \
    --prefill http://<prefill-node>:8000 --decode http://<decode-node>:8000
"""

import argparse
import itertools
import json
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PREFILL_KV_TRANSFER_PARAMS = {
    "do_remote_decode": True,
    "do_remote_prefill": False,
    "remote_engine_id": None,
    "remote_block_ids": None,
    "remote_host": None,
    "remote_port": None,
}

GENERATE_PATHS = ("/v1/completions", "/v1/chat/completions")


def post(url: str, body: dict, timeout: float):
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return urllib.request.urlopen(request, timeout=timeout)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    prefill_iter = None  # set by main()
    decode_iter = None
    timeout_s = 1800.0

    def log_message(self, fmt, *args):  # quiet per-request lines
        pass

    def _reply(self, status: int, payload: bytes, content_type="application/json"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _reply_error(self, status: int, message: str):
        self._reply(status, json.dumps({"error": {"message": message}}).encode())

    def do_GET(self):
        if self.path in ("/health", "/v1/models"):
            try:
                with urllib.request.urlopen(
                    next(self.decode_iter) + self.path, timeout=10
                ) as upstream:
                    self._reply(upstream.status, upstream.read())
            except Exception as exc:  # noqa: BLE001
                self._reply_error(502, f"decode upstream: {exc}")
        else:
            self._reply_error(404, f"no route for GET {self.path}")

    def do_POST(self):
        if self.path not in GENERATE_PATHS:
            self._reply_error(404, f"no route for POST {self.path}")
            return
        try:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        except (TypeError, ValueError) as exc:
            self._reply_error(400, f"bad request body: {exc}")
            return

        # Hop 1: prefill-only request. max_tokens=1 stops the prefill
        # server after the prompt forward + one sampled token.
        prefill_request = dict(body)
        prefill_request["stream"] = False
        prefill_request.pop("stream_options", None)
        prefill_request["max_tokens"] = 1
        if "max_completion_tokens" in prefill_request:
            prefill_request["max_completion_tokens"] = 1
        if prefill_request.get("min_tokens"):
            prefill_request["min_tokens"] = 1
        prefill_request["kv_transfer_params"] = PREFILL_KV_TRANSFER_PARAMS
        try:
            with post(
                next(self.prefill_iter) + self.path, prefill_request, self.timeout_s
            ) as response:
                prefill_response = json.load(response)
        except urllib.error.HTTPError as exc:
            self._reply(exc.code, exc.read())
            return
        except Exception as exc:  # noqa: BLE001
            self._reply_error(502, f"prefill upstream: {exc}")
            return

        # Hop 2: the original request plus the handoff envelope.
        decode_request = dict(body)
        handoff = prefill_response.get("kv_transfer_params")
        if handoff is not None:
            decode_request["kv_transfer_params"] = handoff
        try:
            upstream = post(
                next(self.decode_iter) + self.path, decode_request, self.timeout_s
            )
        except urllib.error.HTTPError as exc:
            self._reply(exc.code, exc.read())
            return
        except Exception as exc:  # noqa: BLE001
            self._reply_error(502, f"decode upstream: {exc}")
            return

        with upstream:
            if not body.get("stream"):
                self._reply(upstream.status, upstream.read())
                return
            # Pass SSE chunks through unbuffered.
            self.send_response(upstream.status)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            while True:
                chunk = upstream.read1(65536)
                if not chunk:
                    break
                self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
            self.wfile.write(b"0\r\n\r\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=10001)
    parser.add_argument("--prefill", action="append", required=True,
                        help="prefill base URL, repeatable (round-robin)")
    parser.add_argument("--decode", action="append", required=True,
                        help="decode base URL, repeatable (round-robin)")
    parser.add_argument("--timeout-secs", type=float, default=1800)
    args = parser.parse_args()

    Handler.prefill_iter = itertools.cycle(u.rstrip("/") for u in args.prefill)
    Handler.decode_iter = itertools.cycle(u.rstrip("/") for u in args.decode)
    Handler.timeout_s = args.timeout_secs
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"pd_proxy on {args.host}:{args.port} "
          f"prefill={args.prefill} decode={args.decode}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
```

</details>

Then talk to the proxy like any OpenAI endpoint:

```bash
curl -s http://localhost:10001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "glm-5.2-fp8", "messages": [{"role": "user", "content": "Why is the sky blue?"}], "max_tokens": 128, "stream": true}'
```

### Sizing the host KV pool

`--kv-offload-host-gib` sizes the pinned host pool that holds sealed KV
pages on each side. Transfers fetch in 256 MiB chunks, and a decode
instance that cannot allocate a chunk within the 15-second handoff
deadline rejects the request — an undersized pool (the 8 GiB default)
produces rejects under multi-turn load long before memory pressure is
visible anywhere else. Size it to a large fraction of free host DRAM;
the pool is allocated up front, so startup RSS reflects it.

## Performance

Both deployments measured with the same `vllm-bench` client and
workload: random dataset, 1024-token prompts (range ratio 0.8),
128-token outputs, greedy, ignore-eos, native MTP,
`GLM52_DECODE_SLOTS=32 GLM52_MTP_DRAFTS=2`.

Single node, EP4 on 4×GB300 (prefill and decode co-located):

| load | req/s | out tok/s | TTFT p50 / p99 | TPOT p50 / p99 |
| ---: | ---: | ---: | ---: | ---: |
| c=1 | 0.53 | 63 | 576 / 581 ms | 11.4 / 12.0 ms |
| c=8 | 2.69 | 305 | 730 / 1,306 ms | 18.0 / 28.7 ms |
| c=32 | 5.48 | 628 | 3,358 / 4,834 ms | 25.7 / 37.1 ms |

Two nodes, P/D disaggregated: one TP4 prefill node plus one EP4 decode
node (4×GB300 each), requests entering through `pd_proxy.py`:

| load | req/s | out tok/s | TTFT p50 / p99 | TPOT p50 / p99 |
| ---: | ---: | ---: | ---: | ---: |
| c=1 | 0.69 | 82 | 208 / 215 ms | 10.2 / 12.4 ms |
| c=8 | 3.81 | 433 | 295 / 669 ms | 14.1 / 21.2 ms |
| c=32 | 8.88 | 1,018 | 729 / 2,436 ms | 23.0 / 35.2 ms |
| c=64 | 13.97 | 1,602 | 1,766 / 3,452 ms | 22.4 / 35.8 ms |

All rows completed every request. At c=32 the disaggregated pair
serves 1.6× the output throughput of the single node at 4.6× lower
median TTFT — co-located prefill steals decode steps, and moving it
to a dedicated node gives both back. Decode latency is flat from c=32
to c=64 while TTFT grows: the single prefill instance is the
bottleneck at c=64, and adding prefill instances behind the router
moves that envelope.

MTP acceptance depends on how predictable the continuation is: these
are greedy numbers, and sampled decoding at high temperature accepts
fewer drafts (the same c=1 run at temperature 1 measured about 28 ms
TPOT).

## Notes

- `--max-model-len`: pass it explicitly. When omitted, GLM5.2 derives
  the cap from post-weight-load VRAM, and on the current build that
  path can over-commit and fail EP4 startup with CUDA out-of-memory.
- `EP_DISABLE_GIN=1` on machines without an RDMA NIC — the NCCL GIN
  probe fails at startup otherwise.
- `--glm52-weight-staging` stages checkpoint bytes through pinned
  buffers and speeds up warm restarts; leave it off for a cold start
  from a network filesystem.
- Prefix caching is on by default and works across P/D turns: repeat
  turns of a conversation reuse prompt KV instead of re-prefilling the
  history.
- Sampling supports `temperature`, `top_p`, `top_k`, `min_p`, and an
  engine-level seed. `logprobs`, `n > 1`, and the presence, frequency,
  and repetition penalties are not part of the GLM5.2 contract.
- A community DSpark drafter is also supported on EP decode
  (`--dflash-draft-model-path`), but it is mutually exclusive with
  prefix caching, KV offload, and P/D; native MTP is the deployment
  default.
- `--dump-graph-png` exports the live decode CUDA graph of rank 0, as
  on the other model lines.

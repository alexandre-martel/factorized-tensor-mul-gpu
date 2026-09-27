"""Optional CUDA-graph comparison (separate from the required, uncaptured results).

All three methods get the same treatment: warm up, capture one call into a CUDA graph
with a static input buffer, then time "copy the new input into the buffer + replay" with
the harness's CUDA-event method. Capture time and the graph's extra memory are reported,
and the replayed output is checked against an FP32 oracle.

    python benchmarks/graphs.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from factorized_inference import (  # noqa: E402
    TRSpec, dense_forward, make_cores, materialize_dense_weight, prepare_optimized, tr_forward_reference,
)
from tune import CASES, stream_ms  # noqa: E402


def graph_row(fn, x, expected):
    static_x = x.clone()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):  # warm-up (compilation, launcher caches) off the capture
        for _ in range(3):
            fn(static_x)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph, before, start = torch.cuda.CUDAGraph(), torch.cuda.memory_allocated(), time.perf_counter()
    with torch.cuda.graph(graph):
        static_y = fn(static_x)
    torch.cuda.synchronize()
    capture_ms, graph_mib = (time.perf_counter() - start) * 1e3, (torch.cuda.memory_allocated() - before) / 2**20

    def replay():
        static_x.copy_(x)  # a new input has to be copied into the captured buffer
        graph.replay()

    ms = stream_ms(replay)
    ok = torch.allclose(static_y.float(), expected, rtol=2e-2, atol=2e-2)
    return ms, capture_ms, graph_mib, ok


def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    print(f"{torch.cuda.get_device_name()}: CUDA-graph replay (input copy included)")
    for rank, tokens in CASES:
        spec = TRSpec(rank=rank)
        cores = make_cores(spec, device="cuda", dtype=torch.float16)
        x = torch.randn(tokens, spec.in_features, device="cuda", dtype=torch.float16,
                        generator=torch.Generator(device="cuda").manual_seed(tokens))
        expected = dense_forward(x.float(), materialize_dense_weight([c.float() for c in cores], spec))
        weight = materialize_dense_weight(cores, spec)
        methods = {"dense": lambda v: dense_forward(v, weight),
                   "factorized_reference": lambda v: tr_forward_reference(v, cores, spec),
                   "factorized_optimized": prepare_optimized(cores, spec)}
        for name, fn in methods.items():
            ms, capture_ms, graph_mib, ok = graph_row(fn, x, expected)
            print(f"R={rank:2d} T={tokens:2d} {name:21s} replay {ms * 1e3:7.1f} us   capture {capture_ms:6.1f} ms"
                  f"   graph memory {graph_mib:6.2f} MiB   correct {ok}", flush=True)


if __name__ == "__main__":
    main()

"""Host cost of one call at R=8 and 1 token, for each launch path (GPU).

dispatch: wall time per call without synchronisation.
stream: CUDA-event stream latency, median of 5 blocks of 20 calls (the harness metric).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import triton

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from factorized_inference import (  # noqa: E402
    TRSpec, dense_forward, make_cores, materialize_dense_weight, prepare_optimized,
)
from factorized_inference.fused_kernel import FusedTR, _tr_fused_kernel, default_config  # noqa: E402
from tune import dispatch_us, stream_ms  # noqa: E402

spec, tokens = TRSpec(rank=8), 1
cores = make_cores(spec, device="cuda", dtype=torch.float16)
x = torch.randn(tokens, spec.in_features, device="cuda", dtype=torch.float16)
fused = FusedTR(cores, spec)
fused(x)  # compiles and caches the launch arguments
_, grid, _, args, _ = fused._launchers[tokens]
warps, stages = default_config(spec, tokens)[4:]
kernel = _tr_fused_kernel[grid](x, *args, num_warps=warps, num_stages=stages)  # compiled kernel
stream = triton.runtime.driver.active.get_current_stream
weight = materialize_dense_weight(cores, spec)
run = prepare_optimized(cores, spec)

paths = {
    "Triton JIT launch": lambda: _tr_fused_kernel[grid](x, *args, num_warps=warps, num_stages=stages),
    "compiled launcher + new output": lambda: (
        torch.empty(tokens, spec.out_features, device=x.device, dtype=x.dtype),
        kernel.run(*grid, stream(x.device.index), kernel.function, kernel.packed_metadata, None, None, None, x, *args)),
    "C launch + reused output (FusedTR)": lambda: fused(x),
    "prepare_optimized callable": lambda: run(x),
    "dense F.linear": lambda: dense_forward(x, weight),
}
print(f"{torch.cuda.get_device_name()}, config {default_config(spec, tokens)}")
for name, fn in paths.items():
    fn()
    print(f"{name:36s} dispatch {dispatch_us(fn):6.1f} us   stream {stream_ms(fn) * 1e3:6.1f} us", flush=True)
for name, fn in {"torch.empty(1, 2880) alone": lambda: torch.empty(tokens, spec.out_features, device="cuda",
                                                                     dtype=torch.float16),
                 "current stream query alone": lambda: stream(0)}.items():
    print(f"{name:36s} dispatch {dispatch_us(fn):6.1f} us", flush=True)

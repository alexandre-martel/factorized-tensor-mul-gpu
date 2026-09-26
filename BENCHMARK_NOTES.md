# What the supplied benchmark measures

Reference for interpreting the supplied measurements.

## Isolation and lifecycle

Each method/token-count case runs in a new process with identical deterministic inputs and factors. The dense worker discards its factor tensors after materializing the weight. Factorized workers do not load the dense baseline during measurement. The optimized worker retains the original cores as well as any state created by `prepare_optimized`, so the cost of keeping the source factors is included.

Explicit preparation and the first forward call are timed separately. Lazy compilation in the first call is included in `first_call_ms`; initialization spanning subsequent warm-up calls must be disclosed separately. Twenty warm-up calls precede steady measurements. Preparation is repeated per isolated case, not per measured iteration. A real model would normally prepare each layer once and may amortize preparation differently.

The correctness oracle is constructed only after performance measurement. Profiling follows correctness and also does not contribute to reported latency or memory.

## Timing

- `host_synchronized_*`: wall-clock latency around one call, including Python dispatch and the trailing device synchronization. There are 100 raw samples by default.
- `cuda_event_stream_*`: five CUDA-event measurements, each bracketing 20 calls by default, divided by the call count. These amortize event overhead but can still include gaps while the host submits work. They are not sums of kernel durations.
- `preparation_ms`: explicit factor preparation, or dense baseline materialization.
- `first_call_ms`: first forward invocation, including lazy work.

Required results use an uncaptured execution path. The harness does not capture CUDA graphs; do not capture inside your submitted callable for these runs. If using `torch.compile`, disable any automatic CUDA-graph mode and record your compilation options. Optional graph measurements must be separate, use equivalent capture/replay treatment for all methods, and include necessary input copies plus graph storage/capture costs.

CUDA-event stream latency is the primary comparison metric. It is not kernel-only time. Repeated use of one weight/input set also creates a warm-cache microbenchmark; do not assume it represents traffic across a complete model.

## Memory

CUDA values use PyTorch's allocation counters. They exclude some library/context allocations and allocations made directly by custom code outside the PyTorch allocator. Disclose and measure those separately if your implementation uses them. CPU GPU-memory fields are unavailable, not zero.

- `representation_logical_bytes`: dense weight bytes for dense, original factor bytes for factorized methods.
- `input_logical_bytes`: input tensor bytes.
- `resident_after_warmup_allocated_bytes`: all live PyTorch allocations before timing, including persistent prepared state/workspace. This is not subtracted away from the reported total.
- `extra_resident_allocated_bytes`: resident allocation beyond the context baseline, original representation and input. This is an accounting aid, not an exact inventory of caches; allocator rounding can affect it.
- `preparation_and_warmup_peak_allocated_bytes`: startup peak, including input/weight creation, packing, lazy initialization and warm-up. Dense startup includes reconstruction in this synthetic harness; a real dense checkpoint might load differently.
- `steady_peak_allocated_bytes`: total peak live tensor allocation during synchronized host timing in the isolated worker.
- `incremental_workspace_and_output_bytes`: that peak minus the pre-timing resident value. This is temporary overhead, not total serving memory.
- `steady_peak_reserved_bytes`: allocator-reserved peak; reserved blocks from preparation may remain in the pool.

Memory counters are collected on the host-timed pass before event timing. If your method changes its state later or uses a different event/graph execution path, measure and explain that separately. The one-operator footprint does not include model attention, KV cache, scheduler, communication or the rest of a serving system.

## Fair comparisons

All required methods use the same generated FP16 factors and FP16 inputs; the dense baseline stores the FP16 reconstruction. TF32 and reduced-precision FP16 matmul reductions are disabled in the harness; record any deliberate precision changes. Higher precision is used only for correctness oracles after measurement.

The factorized reference has a fixed pairwise contraction order. Optional `opt_einsum` packages cannot silently change it. A different contraction order is allowed alongside the required custom-kernel implementation.

For a complete system, distinguish operator numerical error, model quality, and serving throughput/latency. Compare against appropriate optimized baselines under matched hardware, workload and quality conditions.

## CPU smoke check

This checks functionality, not GPU performance.

```bash
python benchmarks/benchmark.py --device cpu --dtype float32 \
  --input-modes 4,4,4 --output-modes 4,4,4 --rank 2 \
  --tokens 1,4 --warmup 2 --iterations 5 --output results/cpu_smoke.json
```

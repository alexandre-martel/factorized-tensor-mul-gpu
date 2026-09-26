# Factorized inference kernel exercise

Profile a factorized linear operator, implement a custom GPU kernel, and explain its performance. This is a synthetic exercise using randomly generated weights.

## Your task

1. Profile the reference on an NVIDIA GPU and identify the main bottleneck.
2. Write or substantially adapt a **custom GPU kernel** for a contraction or a meaningful fused part of the computation. Use Triton, CUDA C++, CuTe DSL or an equivalent tool, and run your kernel in all five required GPU cases.
3. Check correctness and compare your implementation with the factorized reference and dense baseline.
4. Explain the results, memory costs and remaining limitations.

A `torch.compile` or CUDA-graph wrapper alone does not meet the task. You may use those tools alongside your kernel, subject to the measurement rules below. A correct implementation with a well-supported negative result is valid; beating dense is an objective, not a passing requirement.

Take as much time as you need. One implemented kernel idea and a concise report are sufficient.

## The operation

The reference applies a three-core tensor-ring linear operator without reconstructing its full weight matrix. The default workload has **1,920 input features and 2,880 output features**, with input modes `(8,12,20)` and output modes `(12,10,24)`.

Start in `src/factorized_inference/submission.py`. The harness calls `prepare_optimized(cores, spec)`, then repeatedly calls the returned function with `x`. A reference CPU fallback is fine.

Keep the factorized weights and do not construct the complete dense matrix, even temporarily. Packing, partial contractions and bounded tile reconstruction are allowed; account for their preparation and memory costs. The exact mathematics and interface are in [IMPLEMENTATION.md](IMPLEMENTATION.md).

## Setup

We will provide funded NVIDIA GPU access and confirm the GPU model and connection details separately. Use Python 3.11 or newer; package dependencies are pinned.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
pytest -q
```

On the supplied GPU, the CUDA tests should run without skips. A short CPU smoke-check command is in [BENCHMARK_NOTES.md](BENCHMARK_NOTES.md).

## Measure

Run both commands on the same GPU:

```bash
python benchmarks/benchmark.py --device cuda --dtype float16 \
  --rank 8 --tokens 1,8,32 --output results/rank8.json --profile-dir traces/rank8
python benchmarks/benchmark.py --device cuda --dtype float16 \
  --rank 16 --tokens 1,32 --output results/rank16.json
python -m pip freeze > results/environment.txt
```

Use CUDA-event stream latency for headline comparisons, and include host latency and memory for every case. Required results use **no CUDA graph capture**; optional graph comparisons belong in a separate section and must treat all methods consistently. Explain the token-1 profiler evidence and show where your custom kernel runs.

The harness records raw samples, correctness errors, preparation costs and memory. See [BENCHMARK_NOTES.md](BENCHMARK_NOTES.md) for definitions and timing boundaries.

## Submit and discuss

Reply to the invitation with your repository or ZIP, completed `REPORT_TEMPLATE.md`, raw results, environment details and profiler artifacts. Include commands to reproduce your work. AI tools are welcome; disclose their use and be ready to explain your implementation.

We will hold a **45-minute discussion** with the researchers about your kernel, the trade-offs, and what you would measure before claiming an improvement in a complete inference system.

# Performance report

Keep answers concise and link to supporting artifacts.

## 1. Findings

What did you implement, what happened, and what remains uncertain?

## 2. Kernel and evidence

- Kernel source files/entry point, toolchain, and any upstream code adapted:
- Main reference bottleneck and profiler evidence:
- Your kernel's calculation, layout and memory movement; where it appears in the trace:
- Preparation, persistent state, temporary buffers and remaining bottlenecks:

## 3. Correctness and results

Summarize tests, tolerances, observed errors and any harness changes. Include all five required cases and all three methods below, or attach an equivalent generated table and the raw JSON.

| Rank | Tokens | Method | Host median ms | CUDA stream median ms | Resident MiB | Steady allocated peak MiB | Incremental workspace/output MiB |
|---|---|---|---|---|---|---|---|
| | | | | | | | | |

Compare against reference and dense using CUDA-event stream latency. Include preparation/first-call costs and the memory breakdown from `BENCHMARK_NOTES.md`. Explain unfavorable cases and changes across ranks/token counts. Required results are uncaptured; label any optional CUDA-graph comparisons separately.

## 4. System implications

What would change with larger ranks or batches? What integration work and measurements would establish whether this helps a complete inference system? What would make you change approach?

## 5. Reproduction and disclosure

- Commands, source revision/archive identifier, environment and dependencies:
- Raw results and profiler artifacts:
- GPU hours used and whether compute was stopped:
- AI tools used and what you independently verified:

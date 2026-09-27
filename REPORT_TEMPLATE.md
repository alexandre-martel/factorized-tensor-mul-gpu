# Performance report

All numbers come from one NVIDIA A100-SXM4-80GB (Verda, driver 580.178.04, torch 2.14.0+cu130, Triton 3.8.0). Latency is the harness CUDA-event stream median unless stated otherwise. Raw files are in `results/` and `traces/`.

## 1. Findings

I wrote one Triton kernel that computes the whole three-core contraction in a single launch. It runs in all five required GPU cases and all 22 tests pass on the A100 with no skips.

Stream latency, median of three full benchmark runs (`results/rank*.json` and `results/repeats/`):

| Rank | Tokens | Dense | Reference | Fused kernel | Fused vs dense | Fused vs reference |
|---|---|---|---|---|---|---|
| 8 | 1 | 15.7 us | 272.1 us | 10.5 us | 1.50x faster | 26x faster |
| 8 | 8 | 22.7 us | 284.9 us | 17.1 us | 1.33x faster | 17x faster |
| 8 | 32 | 15.4 us | 273.4 us | 40.8 us | 2.6x slower | 6.7x faster |
| 16 | 1 | 16.7 us | 293.1 us | 34.6 us | 2.1x slower | 8.5x faster |
| 16 | 32 | 15.9 us | 409.9 us | 355.2 us | 22x slower | 1.15x faster |

The fused path allocates at most 1.34 MiB at steady state. Dense needs 18.7 to 19.0 MiB and the reference up to 138.6 MiB.

What happened:
- The reference is limited by host dispatch at small token counts. It launches 9 kernels per call and the GPU is busy for only 36 of 270 us.
- Fusion removes the extra launches and the intermediate tensors. The largest gains after that came from the host side. Triton's normal Python launch costs 17 to 22 us per call, more than the kernel itself.
- At 32 tokens and at rank 16 the fused kernel is compute bound and dense is faster. Rank 16 with 32 tokens cannot beat dense on this GPU. Its useful work is 6.3 GFLOP, which takes 20 us even at the 312 TFLOPS FP16 peak, while dense takes 16 us.

How the result was reached (harness stream latency, one run per session, R=8):

| Step | Change | 1 token | 8 tokens |
|---|---|---|---|
| 1 | Fused kernel, Triton JIT launch | 37.5 us | 37.2 us |
| 2 | Reuse the compiled launcher | 18.5 us | 24.7 us |
| 3 | Split-K with a fix-up pass instead of a zeroed output | 15.2 us | 20.6 us |
| 4 | Call the C launcher directly, reuse the output buffer | 9.4 us | 22.2 us |
| 5 | Wider tuning (pipeline stages, warps, larger tiles) | 8.8 us | 20.9 us |
| Final | Same code, three runs | 10.5 us | 17.1 us |

Each session used a new instance, so dense also moved between sessions (13.9 to 16.0 us at 1 token, 17.9 to 23.1 us at 8 tokens). Per-session files are in `results/history/`.

What remains uncertain:
- This is a warm-cache microbenchmark on one GPU. Run-to-run variation is small for most cases but reaches 20% for some (fused R=8 at 32 tokens: 40.7 to 49.7 us).
- The tuned configs in `tuned_configs.json` are specific to this A100.
- The fast launch path uses Triton 3.8 launcher internals. It falls back to Triton's own launcher when those do not apply, but it may need changes for other Triton versions.

## 2. Kernel and evidence

- Kernel source files/entry point, toolchain, and any upstream code adapted:
  - `src/factorized_inference/fused_kernel.py`: the kernel `_tr_fused_kernel` and the launcher class `FusedTR`.
  - `src/factorized_inference/submission.py`: `prepare_optimized` uses `FusedTR` for CUDA FP16/BF16 inputs and the reference otherwise.
  - Toolchain: Triton 3.8.0 as installed with torch 2.14.0 (CUDA 13.0). No upstream kernel code was adapted.
  - Support files: `benchmarks/tune.py` (config sweep), `benchmarks/host_overhead.py` (launch cost breakdown), `benchmarks/graphs.py` (optional CUDA graph comparison), `tests/test_fused_kernel.py` (CPU interpreter tests), `scripts/gpu_session.sh` (full GPU run).

- Main reference bottleneck and profiler evidence:
  - Token-1 trace, rank 8 (`traces/rank8/factorized_reference_rank8_tokens1.json`): 9 kernels per call. They are 5 elementwise copies from the einsum permutes (3.9 us each), 3 CUTLASS tensor-core GEMMs (3.9 to 4.6 us) and one cuBLASLt split-K reduction (3.1 us).
  - GPU time is 35.8 us per call while the stream latency is 270 us. About 87% of the time is host work: einsum planning, allocations and launches.
  - The intermediates go through global memory. The reference allocates 1.0 MiB per call at 1 token and 130 MiB at rank 16 with 32 tokens.
  - Dense runs one cuBLAS tensor-core GEMM (`ampere_fp16_s16816gemm...`) of 13.9 us. Its 10.5 MiB weight stays in the 40 MB L2 in this benchmark.

- Your kernel's calculation, layout and memory movement; where it appears in the trace:
  - Order: x is contracted with B over j, then with A over i and b, then with C over k, a and c. The R cubed step sums over i (8 values) instead of j (12 values). Useful work is 29.5 MFLOP per token at rank 8 and 197 at rank 16, against 39.8 and 277 for the reference order.
  - Work split: one program handles BT tokens, one q, PC values of p and one of KS slices of k. It loops over its k values in chunks of KC. Each chunk runs three `tl.dot` calls with FP16 inputs and FP32 accumulation:
    - dot 1: B as [(b,c), j] times x as [j, (i,t,k)] gives U as [(b,c), (i,t,k)].
    - dot 2: A as [(p,a), (i,b)] times U as [(i,b), (t,k,c)] gives V as [(p,a), (t,k,c)].
    - dot 3: V as [(t,p), (k,a,c)] times C as [(k,a,c), r] is added to the output tile [(t,p), r].
  - Between dots, U and V are reshaped and permuted in registers and shared memory. No intermediate is written to global memory.
  - Padding: j goes from 12 to 16, r from 24 to 32 and p from 12 to 16 when PC is 16. Padding exists only in registers and is masked. The cores are read in place, with no packed copy.
  - Split-K: with KS > 1, each slice stores its FP32 partial tile. A per-tile atomic counter finds the last slice, which adds the partials in a fixed order, writes y and resets the counter. The result is deterministic and no zeroed output or extra launch is needed.
  - Launch: the first call for a token count uses Triton's JIT launch. Later calls call Triton's generated C launch function directly with prebuilt arguments and write into an output buffer reused per token count.
  - In the trace (`traces/rank8/factorized_optimized_rank8_tokens1.json`), `_tr_fused_kernel` is the only kernel per call and takes 9.05 us, against 13.9 us for the dense GEMM.

  Tuned configs (`src/factorized_inference/tuned_configs.json`, from the sweep in `results/tune.json`):

  | Rank | Tokens | BT, PC, KC, KS, warps, stages | Programs |
  |---|---|---|---|
  | 8 | 1 | 1, 16, 2, 10, 4, 3 | 100 |
  | 8 | 8 | 4, 8, 1, 5, 4, 3 | 200 |
  | 8 | 32 | 2, 16, 2, 2, 4, 3 | 320 |
  | 16 | 1 | 1, 16, 2, 10, 4, 1 | 100 |
  | 16 | 32 | 4, 4, 1, 2, 4, 4 | 480 |

  The sweep checks every config against an FP32 dense oracle before and after timing. All 612 configs measured in the last sweep were correct.

- Preparation, persistent state, temporary buffers and remaining bottlenecks:
  - Preparation takes 0.07 to 0.08 ms. It only validates shapes, since nothing is packed.
  - The first call takes 0.43 to 0.44 s. It includes loading the compiled kernel from Triton's disk cache and building the launcher. The kernels had been compiled earlier in the same session by the test run, so a cold compile is not included.
  - Persistent state per token count: the output buffer (5.6 KiB at 1 token, 180 KiB at 32 tokens), the FP32 split-K workspace (KS x tokens x 2880 x 4 bytes) and int32 tile counters (under 1 KiB). The harness reports 119 KiB of extra resident memory at 1 token, 496 KiB at 8 tokens and 901 KiB at 32 tokens. The incremental workspace during timing is 0.
  - The dense weight is never built, even partly.
  - At 1 token the kernel is latency bound: 9 us for about 40 MFLOP.
  - At 8 to 32 tokens and at rank 16 it is compute bound with low efficiency. At rank 8 with 32 tokens it executes about 1.4 GFLOP (0.94 useful) in 41 us, about 10% of peak. The causes are small tensor-core tiles, two on-chip transposes per chunk, padding (+33% on j and r) and dot 1 recomputed for each block of p.
  - Host dispatch is 6.7 to 8 us per call (`results/host_overhead.txt`):

  | Launch path, rank 8, 1 token | Host dispatch | Stream latency |
  |---|---|---|
  | Triton JIT launch | 22.0 us | 23.4 us |
  | Compiled launcher and a new output tensor | 13.3 us | 14.0 us |
  | C launch and reused output (`FusedTR`) | 6.7 us | 10.4 us |
  | `prepare_optimized` callable | 8.0 us | 10.5 us |
  | Dense `F.linear` | 16.2 us | 17.2 us |

## 3. Correctness and results

- Tests: `pytest -q` on the A100 gives 22 passed and 0 skipped (`results/pytest.txt`). This includes the five required CUDA cases and the CUDA independence test, whose second call to `run_a` goes through the direct launcher. `tests/test_fused_kernel.py` also runs the kernel through Triton's CPU interpreter on 16 cases, including split-K configs and small shapes, against an FP64 oracle. With `TRITON_INTERPRET=1`, the provided CPU tests pass through the kernel at the FP32 tolerance of 1e-4.
- Tolerance: FP16 with atol = rtol = 0.02 against the FP32 dense oracle of the harness. The fused path has a max absolute error of 1.3e-3 to 2.4e-3, a mean absolute error of 2.8e-4 and a relative L2 error of 3.6e-4, the same level as dense and the reference.
- Harness changes: none. `benchmarks/benchmark.py` and the provided tests are unchanged. One command change: the README uses `--device cuda`, but with the pinned torch 2.14 the harness call `torch.cuda.set_device(torch.device("cuda"))` raises "Expected a torch.device with a specified index". I ran the same commands with `--device cuda:0`, which is the same GPU on this single-GPU machine.

Canonical run (`results/rank8.json`, `results/rank16.json`):

| Rank | Tokens | Method | Host median ms | CUDA stream median ms | Resident MiB | Steady allocated peak MiB | Incremental workspace/output MiB |
|---|---|---|---|---|---|---|---|
| 8 | 1 | dense | 0.0332 | 0.0160 | 18.68 | 18.68 | 0.006 |
| 8 | 1 | factorized_reference | 0.2894 | 0.2696 | 8.21 | 9.22 | 1.011 |
| 8 | 1 | factorized_optimized | 0.0229 | 0.0105 | 0.21 | 0.21 | 0.000 |
| 8 | 8 | dense | 0.0360 | 0.0221 | 18.70 | 18.75 | 0.044 |
| 8 | 8 | factorized_reference | 0.2845 | 0.2805 | 8.24 | 16.22 | 7.983 |
| 8 | 8 | factorized_optimized | 0.0289 | 0.0171 | 0.60 | 0.60 | 0.000 |
| 8 | 32 | dense | 0.0307 | 0.0154 | 18.79 | 18.96 | 0.176 |
| 8 | 32 | factorized_reference | 0.2882 | 0.2734 | 8.33 | 41.59 | 33.265 |
| 8 | 32 | factorized_optimized | 0.0533 | 0.0407 | 1.08 | 1.08 | 0.000 |
| 16 | 1 | dense | 0.0348 | 0.0163 | 18.68 | 18.68 | 0.006 |
| 16 | 1 | factorized_reference | 0.3114 | 0.2895 | 8.47 | 12.51 | 4.043 |
| 16 | 1 | factorized_optimized | 0.0467 | 0.0346 | 0.46 | 0.46 | 0.000 |
| 16 | 32 | dense | 0.0341 | 0.0164 | 18.79 | 18.96 | 0.176 |
| 16 | 32 | factorized_reference | 0.4924 | 0.4099 | 8.58 | 138.64 | 130.059 |
| 16 | 32 | factorized_optimized | 0.4002 | 0.3848 | 1.34 | 1.34 | 0.000 |

Variation over three full runs of both commands (stream latency, min / median / max, in us):

| Rank | Tokens | Dense | Reference | Fused |
|---|---|---|---|---|
| 8 | 1 | 15.4 / 15.7 / 16.0 | 269.6 / 272.1 / 277.9 | 8.8 / 10.5 / 10.7 |
| 8 | 8 | 22.1 / 22.7 / 23.1 | 280.5 / 284.9 / 289.6 | 17.0 / 17.1 / 17.2 |
| 8 | 32 | 15.2 / 15.4 / 16.2 | 273.2 / 273.4 / 279.6 | 40.7 / 40.8 / 49.7 |
| 16 | 1 | 16.3 / 16.7 / 16.7 | 289.5 / 293.1 / 297.1 | 33.8 / 34.6 / 41.5 |
| 16 | 32 | 15.6 / 15.9 / 16.4 | 383.0 / 409.9 / 441.9 | 327.9 / 355.2 / 384.8 |

Preparation and first call:
- Dense spends 137 to 146 ms building the weight. Its first call takes 0.1 to 0.2 ms, and 9.3 ms at 8 tokens.
- The reference has no preparation. Its first call takes 130 to 145 ms.
- The fused path takes 0.07 to 0.08 ms to prepare and 0.43 to 0.44 s for its first call.

Memory breakdown (harness counters):
- Representation: dense stores 10.55 MiB. The factorized methods store 87 KiB of cores at rank 8 and 348 KiB at rank 16.
- Dense and the reference keep 8.1 MiB of extra resident memory, which is the cuBLAS workspace.
- The fused path keeps 119 to 901 KiB: the output buffer and the split-K scratch.
- Startup peak: dense 30.7 to 35.3 MiB (weight reconstruction), reference up to 138.6 MiB, fused at most 1.34 MiB.
- Reserved peak: dense 46 MiB, reference 22 to 152 MiB, fused 2 MiB.

Unfavorable cases and trends:
- Dense cost is almost flat in the token count. It reads the same 10.5 MiB weight, which stays in L2 here.
- The fused cost grows with tokens and with R cubed. At rank 8 it wins at 1 and 8 tokens and loses at 32, so the crossover is between 8 and 32 tokens.
- Rank 16 costs 6.7 times more useful work per token than rank 8. Dense wins at both token counts, and at 32 tokens the fused kernel is only 1.15x faster than the reference.
- At 8 tokens, part of the win comes from dense's own host cost. Dense `F.linear` needs 16 us of host dispatch per call at 1 token, and under CUDA graphs dense drops from 22.7 to 14.5 us at 8 tokens.

Optional CUDA graph comparison (separate from the required results, `results/graphs.txt`). Each method is captured once with a static input buffer. The timed step copies the input into that buffer and replays the graph, for all three methods.

| Rank | Tokens | Dense replay | Reference replay | Fused replay |
|---|---|---|---|---|
| 8 | 1 | 17.5 us | 37.3 us | 13.2 us |
| 8 | 8 | 14.5 us | 64.2 us | 24.2 us |
| 8 | 32 | 17.9 us | 141.7 us | 51.4 us |
| 16 | 1 | 17.4 us | 50.7 us | 43.7 us |
| 16 | 32 | 17.9 us | 424.8 us | 331.0 us |

- Capture takes 0.6 to 2.0 ms for each method.
- The graphs add 8.1 to 8.3 MiB for dense and the reference (cuBLAS workspace) and nothing for the fused path, whose buffers already exist.
- All replayed outputs are correct.
- With graphs the host cost disappears. The fused kernel keeps its lead at 1 token (13.2 against 17.5 us) and stays 2.8x faster than the reference there. At 8 tokens dense is faster once its host cost is removed.

## 4. System implications

- Larger ranks: the middle contraction costs 19,200 x R cubed multiply-adds per token. Rank 16 is already slower than dense at 1 token, so higher ranks only make sense where the weights are memory bound.
- Larger batches: dense stays close to flat until it becomes compute bound, around 150 tokens by a simple roofline estimate for this GPU. The factorized cost grows linearly, so it only fits small decode batches.
- Cold weights: in a real model the weights do not stay in L2. Many layers of 10.5 MiB exceed the 40 MB L2, so dense would read each weight from HBM (at least 5 us per layer at 2 TB/s). The cores of one layer (87 KiB) are small enough to stay on chip. This benchmark therefore favours dense.
- What to measure before claiming a gain in a complete system:
  - A chain of layers with distinct weights, to get cold-cache behaviour.
  - A full decode step under CUDA graphs, since serving engines use them.
  - Tokens per second and latency at realistic batch sizes.
  - Comparisons against strong dense baselines such as weight-only quantised GEMV kernels.
  - The quality of a model actually compressed to this rank. These weights are synthetic.
- Integration work: the fused path already works under CUDA graph capture. It needs stream-ordered calls, because the output buffer and the counters are reused. The config table has to be tuned per GPU. Several projections (for example Q, K and V) could share one launch. There is no backward pass.
- I would change approach if a cold-cache full-model test still showed dense winning at decode batch sizes, or if acceptable quality needed rank 16 or more. Options then are merging two cores offline (more memory, less compute), reconstructing bounded weight tiles inside a GEMV kernel, or rewriting the kernel in CUDA or CuTe to remove the on-chip transposes.

## 5. Reproduction and disclosure

- Commands, source revision/archive identifier, environment and dependencies:
  - Local CPU checks: `pip install -e '.[test]'`, then `pytest -q`. With Triton installed, `TRITON_INTERPRET=1 pytest -q` runs the provided tests through the kernel.
  - GPU: `bash scripts/gpu_session.sh` from the repo root. It creates the environment, checks the kernel, runs `pytest`, the two README benchmark commands with `--device cuda:0`, two repeats of them, `host_overhead.py`, `graphs.py` and `pip freeze`. `TUNE=1 bash scripts/gpu_session.sh` redoes the config sweep first.
  - Source revision: git commit `7b97560` was used for the final measurements.
  - Environment: A100-SXM4-80GB, driver 580.178.04, Ubuntu 24.04 (Linux 6.8), Python 3.12.14, torch 2.14.0+cu130, triton 3.8.0, numpy 2.3.5 (`results/environment.txt`, `results/nvidia-smi.txt`).
- Raw results and profiler artifacts:
  - `results/rank8.json` and `results/rank16.json`: canonical run.
  - `results/repeats/`: the two repeats.
  - `results/pytest.txt`, `results/host_overhead.txt`, `results/graphs.txt`.
  - `results/tune.json` and `results/tune.txt`: the sweep behind the committed configs.
  - `results/history/`: the five development sessions.
  - `traces/rank8/`: token-1 traces for the three methods.
- GPU hours used and whether compute was stopped: about 1.0 GPU hour on one A100-SXM4-80GB, over six sessions of 6 to 21 minutes, for about 1.6 USD. Every instance was deleted at the end of its session.
- AI tools used and what you independently verified: Claude (Anthropic) was used to help write the kernel, the scripts and this report. Every number here comes from the result files listed above. Correctness was checked with the provided tests on the GPU, with the interpreter tests on CPU and by the harness's own oracle check in every run.

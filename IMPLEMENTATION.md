# Operator and interface reference

## Mathematics

The input is `[tokens, in_features]`. Each feature dimension is a product of three modes. Core shapes are `[rank, output_mode, input_mode, rank]`.

```text
W[(p,q,r), (i,j,k)] = sum over a,b,c of A[a,p,i,b] B[b,q,j,c] C[c,r,k,a]
y[t,p,q,r] = sum over i,j,k of x[t,i,j,k] W[(p,q,r), (i,j,k)]
```

The repeated ring indices close the cycle. The reference uses a fixed sequence of pairwise contractions on the factors and input. The dense baseline constructs the same synthetic weight once. Agreement between them establishes numerical correctness of this operator, not the quality of a compressed pretrained model.

## Implementation

`prepare_optimized(cores, spec)` returns a callable accepting `x`. The starter delegates to `tr_forward_optimized`; replace either entry point as needed. It is a correctness placeholder, not a completed custom-kernel submission.

Your GPU kernel must perform a substantive part of the factorized calculation in every required GPU case. A wrapper, no-op, trivial copy or compiler-generated kernel without your own kernel implementation is insufficient. You may adapt an existing kernel; identify its source, license and your changes. Explain the kernel's arithmetic, layout and memory movement, and identify it in the profiler trace.

## Interface requirements

- Inputs and cores remain unmodified. A prepared callable must handle changing inputs/token counts; different weight sets must remain independent.
- Support the contiguous 2D inputs and three-mode shapes in the supplied tests. Preserve output shape, device and dtype. Required GPU execution uses FP16; small CPU checks use FP32. Backward/training and other input layouts are outside scope.
- Outputs must be usable on the caller's stream. Document reusable output storage; callers consume it before the next invocation.
- Never construct or retain a complete dense weight table, including equivalent layouts or precisions. Baseline and correctness-oracle code are exempt. Report packed factors, partial contractions, tiles and workspace. Storage expansion is a trade-off to measure, not a compression saving.
- Include necessary per-input work in timing. Preserve comparison workloads and tolerances; disclose harness changes. Use only code and data you are authorized to share.

## Correctness

The benchmark compares to an FP32 dense oracle built from the same FP16 factors. Small tests also use FP64. Acceptance is:

```text
abs(actual - expected) <= atol + rtol * abs(expected)
FP16: atol = rtol = 0.02
FP32: atol = rtol = 0.0001
```

Report maximum/mean absolute error and relative L2 error. These tolerances are not a universal percentage-error guarantee. The tests check numerics and interface behavior; reviewers separately verify custom-kernel use and implementation ownership.

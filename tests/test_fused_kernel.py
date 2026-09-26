"""Checks the fused Triton kernel without a GPU, through Triton's CPU interpreter."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

pytest.importorskip("triton")

SCRIPT = r"""
import torch
from factorized_inference import TRSpec, make_cores, materialize_dense_weight, prepare_optimized
from factorized_inference.fused_kernel import default_config, tr_forward_fused
R8, R16, TINY = TRSpec(rank=8), TRSpec(rank=16), TRSpec((2, 3, 2), (2, 4, 3), 3)
cases = [(R8, t, None) for t in (1, 3, 4, 8, 32)] + [(R16, t, None) for t in (1, 32)]
cases += [(TINY, t, None) for t in (1, 5)] + [(TRSpec((4, 4, 4), (4, 4, 4), 2), 6, None)]
# Explicit (BT, PC, KC, KS, warps): split-K, incl. uneven splits and a split past the k range.
cases += [(R8, 1, (1, 16, 2, 5, 4)), (R8, 1, (1, 16, 2, 10, 4)), (R16, 1, (1, 16, 2, 3, 8)),
          (R16, 32, (4, 4, 2, 2, 8)), (TINY, 1, (1, 16, 4, 2, 4))]
for spec, tokens, config in cases:
    cores = make_cores(spec, dtype=torch.float16, seed=tokens)
    x = torch.randn(tokens, spec.in_features, generator=torch.Generator().manual_seed(7)).half()
    expected = x.double() @ materialize_dense_weight([c.double() for c in cores], spec).T
    actual = tr_forward_fused(x, cores, spec, config) if config else prepare_optimized(cores, spec)(x)
    assert actual.dtype == torch.float16 and actual.shape == expected.shape
    torch.testing.assert_close(actual.double(), expected, rtol=2e-2, atol=2e-2)
    print(spec.rank, spec.input_modes, tokens, config or default_config(spec, tokens),
          f"max_abs_err={(actual.double() - expected).abs().max().item():.2e}")
"""


def test_fused_kernel_in_triton_interpreter():
    env = dict(os.environ, TRITON_INTERPRET="1")
    result = subprocess.run([sys.executable, "-c", SCRIPT], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    print(result.stdout)

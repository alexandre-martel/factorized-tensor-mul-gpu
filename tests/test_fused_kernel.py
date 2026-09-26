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
from factorized_inference.fused_kernel import _config
cases = [(TRSpec(rank=8), t) for t in (1, 3, 4, 8, 32)] + [(TRSpec(rank=16), t) for t in (1, 32)]
cases += [(TRSpec((2, 3, 2), (2, 4, 3), 3), t) for t in (1, 5)] + [(TRSpec((4, 4, 4), (4, 4, 4), 2), 6)]
for spec, tokens in cases:
    cores = make_cores(spec, dtype=torch.float16, seed=tokens)
    x = torch.randn(tokens, spec.in_features, generator=torch.Generator().manual_seed(7)).half()
    expected = x.double() @ materialize_dense_weight([c.double() for c in cores], spec).T
    actual = prepare_optimized(cores, spec)(x)
    assert actual.dtype == torch.float16 and actual.shape == expected.shape
    torch.testing.assert_close(actual.double(), expected, rtol=2e-2, atol=2e-2)
    print(spec.rank, spec.input_modes, tokens, _config(spec, tokens < 4)[:3],
          f"max_abs_err={(actual.double() - expected).abs().max().item():.2e}")
"""


def test_fused_kernel_in_triton_interpreter():
    env = dict(os.environ, TRITON_INTERPRET="1")
    result = subprocess.run([sys.executable, "-c", SCRIPT], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    print(result.stdout)

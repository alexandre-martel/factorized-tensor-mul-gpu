from __future__ import annotations

import os
from typing import Callable, Sequence

import torch

from .reference import TRSpec, _validate, tr_forward_reference

try:
    from .fused_kernel import tr_forward_fused
except ImportError:  # Triton unavailable (e.g. CPU-only install): reference fallback only.
    tr_forward_fused = None

# TRITON_INTERPRET=1 runs the Triton kernel on CPU tensors (used by the CPU test suite).
_INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"


def _use_kernel(x: torch.Tensor) -> bool:
    if tr_forward_fused is None:
        return False
    return _INTERPRET or (x.is_cuda and x.dtype in (torch.float16, torch.bfloat16))


def tr_forward_optimized(
    x: torch.Tensor,
    cores: Sequence[torch.Tensor],
    spec: TRSpec,
) -> torch.Tensor:
    """Fused single-kernel Triton path on CUDA FP16/BF16; reference fallback otherwise."""

    if _use_kernel(x):
        _validate(x, cores, spec)
        return tr_forward_fused(x.contiguous(), [c.contiguous() for c in cores], spec)
    return tr_forward_reference(x, cores, spec)


def prepare_optimized(
    cores: Sequence[torch.Tensor],
    spec: TRSpec,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Validate once and bind the cores; the kernel reads them in place (no extra state).

    Each call returns a newly allocated output; nothing is cached between calls, so
    prepared callables for different weights are independent.
    """

    cores = tuple(c.contiguous() for c in cores)  # no copy for contiguous cores
    if not _use_kernel(cores[0]):
        return lambda x: tr_forward_reference(x, cores, spec)
    _validate(torch.empty(0, spec.in_features, dtype=cores[0].dtype), cores, spec)

    def run(x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != spec.in_features:
            raise ValueError(f"Expected x with shape [tokens, {spec.in_features}], got {tuple(x.shape)}")
        return tr_forward_fused(x.contiguous(), cores, spec)

    return run

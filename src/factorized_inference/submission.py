from __future__ import annotations

import os
from typing import Callable, Sequence

import torch

from .reference import TRSpec, _validate, tr_forward_reference

try:
    from .fused_kernel import FusedTR, tr_forward_fused
except ImportError:  # Triton unavailable (e.g. CPU-only install): reference fallback only.
    FusedTR = tr_forward_fused = None

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
    """Validate once and bind the cores; the kernel reads them in place (no weight copy).

    Reusable output storage: on CUDA, the returned tensor is a buffer reused by the next
    call with the same token count, so consume it (or copy it) before calling again.
    Retained state, owned by this callable (prepared callables for different weights are
    independent): per token count, the compiled kernel's launch arguments, the output
    buffer and the split-K scratch.
    """

    cores = tuple(c.contiguous() for c in cores)  # no copy for contiguous cores
    if not _use_kernel(cores[0]):
        return lambda x: tr_forward_reference(x, cores, spec)
    _validate(torch.empty(0, spec.in_features, dtype=cores[0].dtype), cores, spec)
    fused = FusedTR(cores, spec)

    def run(x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != spec.in_features:
            raise ValueError(f"Expected x with shape [tokens, {spec.in_features}], got {tuple(x.shape)}")
        return fused(x.contiguous())

    return run

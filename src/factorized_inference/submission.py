from __future__ import annotations

import os
from typing import Callable, Sequence

import torch

from .reference import TRSpec, _validate, tr_forward_reference

try:
    from .fused_kernel import FusedTR, tr_forward_fused
except ImportError:  # no Triton: reference only
    FusedTR = tr_forward_fused = None

# TRITON_INTERPRET=1 runs the kernel on CPU tensors (used in tests).
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
    """Fused kernel for CUDA FP16/BF16 inputs, reference otherwise."""

    if _use_kernel(x):
        _validate(x, cores, spec)
        return tr_forward_fused(x.contiguous(), [c.contiguous() for c in cores], spec)
    return tr_forward_reference(x, cores, spec)


def prepare_optimized(
    cores: Sequence[torch.Tensor],
    spec: TRSpec,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Returns a callable that runs the fused kernel on CUDA FP16/BF16 inputs.

    The cores are read in place. The returned tensor is reused by the next call with the
    same token count, so it must be consumed first. The callable keeps, per token count,
    the launch arguments, the output buffer and the split-K scratch.
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

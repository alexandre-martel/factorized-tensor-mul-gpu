from __future__ import annotations

from typing import Callable, Sequence

import torch

from .reference import TRSpec, tr_forward_reference


def tr_forward_optimized(
    x: torch.Tensor,
    cores: Sequence[torch.Tensor],
    spec: TRSpec,
) -> torch.Tensor:
    """Candidate implementation.

    Replace this placeholder with factorized execution that uses your custom
    GPU kernel on the required CUDA workloads. A reference CPU fallback is fine.
    Compilation/graph wrapping alone does not meet the implementation task.
    Do not reconstruct the complete dense weight, even temporarily.
    """

    return tr_forward_reference(x, cores, spec)


def prepare_optimized(
    cores: Sequence[torch.Tensor],
    spec: TRSpec,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Optional one-time preparation; return a callable taking x.

    The harness calls this once per isolated method/shape case. You may prepack
    factors, compile your kernel, or allocate reusable workspace here. Report that work
    and all retained storage. Never construct the complete dense weight, even
    temporarily. A prepared callable must handle changing inputs/token counts;
    separate prepared callables for different weights must remain independent.

    You can leave this wrapper unchanged and edit tr_forward_optimized, or
    implement your optimization here. The harness and tests use this entry point.
    """
    return lambda x: tr_forward_optimized(x, cores, spec)

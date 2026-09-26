from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from operator import mul
from typing import Sequence

import torch


def _product(values: Sequence[int]) -> int:
    return reduce(mul, values, 1)


@dataclass(frozen=True)
class TRSpec:
    """Shape of a three-core tensor-ring linear layer.

    A dense [out_features, in_features] weight is represented by three cores.
    Core k has shape [rank, output_mode[k], input_mode[k], rank].
    """

    input_modes: tuple[int, int, int] = (8, 12, 20)
    output_modes: tuple[int, int, int] = (12, 10, 24)
    rank: int = 8

    def __post_init__(self) -> None:
        if len(self.input_modes) != 3 or len(self.output_modes) != 3:
            raise ValueError("Exactly three input and output modes are required")
        if any(v <= 0 for v in (*self.input_modes, *self.output_modes, self.rank)):
            raise ValueError("All modes and the rank must be positive")

    @property
    def in_features(self) -> int:
        return _product(self.input_modes)

    @property
    def out_features(self) -> int:
        return _product(self.output_modes)

    @property
    def factor_parameters(self) -> int:
        return sum(
            self.rank * o * i * self.rank
            for o, i in zip(self.output_modes, self.input_modes)
        )

    @property
    def dense_parameters(self) -> int:
        return self.in_features * self.out_features


def make_cores(
    spec: TRSpec,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create deterministic teaching weights with bounded magnitude."""

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    cores = []
    for output_mode, input_mode in zip(spec.output_modes, spec.input_modes):
        core = torch.randn(
            spec.rank,
            output_mode,
            input_mode,
            spec.rank,
            generator=generator,
            device=device,
            dtype=dtype,
        )
        core = core * (1.0 / (spec.rank * input_mode) ** 0.5)
        cores.append(core)
    return tuple(cores)  # type: ignore[return-value]


def _validate(
    x: torch.Tensor,
    cores: Sequence[torch.Tensor],
    spec: TRSpec,
) -> None:
    if x.ndim != 2 or x.shape[1] != spec.in_features:
        raise ValueError(
            f"Expected x with shape [tokens, {spec.in_features}], got {tuple(x.shape)}"
        )
    if len(cores) != 3:
        raise ValueError(f"Expected three cores, got {len(cores)}")
    for index, (core, output_mode, input_mode) in enumerate(
        zip(cores, spec.output_modes, spec.input_modes)
    ):
        expected = (spec.rank, output_mode, input_mode, spec.rank)
        if tuple(core.shape) != expected:
            raise ValueError(f"Core {index} should have shape {expected}, got {tuple(core.shape)}")


def tr_forward_reference(
    x: torch.Tensor,
    cores: Sequence[torch.Tensor],
    spec: TRSpec,
) -> torch.Tensor:
    """Correct but deliberately unoptimized direct factorized execution.

    Indices:
      x:             t i j k
      first core:    a p i b
      second core:   b q j c
      third core:    c r k a
      output:        t p q r

    The repeated a/b/c indices close the tensor ring. The implementation never
    constructs the full dense weight.
    """

    _validate(x, cores, spec)
    first, second, third = cores
    x_modes = x.reshape(x.shape[0], *spec.input_modes)
    # Explicit pairwise contractions keep the baseline independent of optional
    # opt_einsum installation and its path-search strategy.
    intermediate = torch.einsum("tijk,apib->tjkapb", x_modes, first)
    intermediate = torch.einsum("tjkapb,bqjc->tkapqc", intermediate, second)
    output = torch.einsum("tkapqc,crka->tpqr", intermediate, third)
    return output.reshape(x.shape[0], spec.out_features)


def materialize_dense_weight(
    cores: Sequence[torch.Tensor],
    spec: TRSpec,
) -> torch.Tensor:
    """Build the equivalent dense weight for validation and baseline timing only.

    The returned tensor has shape [out_features, in_features]. It must not be
    cached by the submitted factorized implementation.
    """

    if len(cores) != 3:
        raise ValueError(f"Expected three cores, got {len(cores)}")
    first, second, third = cores
    partial = torch.einsum("apib,bqjc->apiqjc", first, second)
    weight = torch.einsum("apiqjc,crka->pqrijk", partial, third)
    return weight.reshape(spec.out_features, spec.in_features)


def dense_forward(x: torch.Tensor, dense_weight: torch.Tensor) -> torch.Tensor:
    """Dense baseline using a pre-materialized [out_features, in_features] weight."""

    return torch.nn.functional.linear(x, dense_weight)

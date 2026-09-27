"""Single-launch fused tensor-ring forward pass (Triton).

    y[t,p,q,r] = sum_{i,j,k,a,b,c} x[t,i,j,k] A[a,p,i,b] B[b,q,j,c] C[c,r,k,a]

Contraction order x.B -> A -> C: the R^3 step sums over the smallest input mode (i).
A program owns BT tokens, one output index q, PC output indices p and 1/KS of the k range
(split-K: KS > 1 adds partial outputs atomically into a zeroed y). It loops over its k
range in chunks of KC and keeps every intermediate on chip (registers / shared memory):

    U[(b,c),(i,t,k)] = sum_j     B[b,q,j,c] x[t,i,j,k]        dot 1, then on-chip transpose
    V[(p,a),(t,k,c)] = sum_{i,b} A[a,p,i,b] U[(i,b),(t,k,c)]  dot 2, then on-chip transpose
    y[(t,p),r]      += sum_{k,a,c} V[(t,p),(k,a,c)] C[c,r,k,a]  dot 3

Non-power-of-two sizes are padded in registers and masked, never in memory: the cores
are read in place (no packed copy, no workspace, no dense weight).
"""

from __future__ import annotations

import functools
import json
from pathlib import Path

import torch
import triton
import triton.language as tl

# Fastest config per case on the benchmark GPU, written by benchmarks/tune.py.
TUNED_PATH = Path(__file__).with_name("tuned_configs.json")


@triton.jit(do_not_specialize=["T"])  # one binary for every token count
def _tr_fused_kernel(x_ptr, a_ptr, b_ptr, c_ptr, y_ptr, T,
                     I0: tl.constexpr, I1: tl.constexpr, I2: tl.constexpr,
                     O0: tl.constexpr, O1: tl.constexpr, O2: tl.constexpr, R: tl.constexpr,
                     IP: tl.constexpr, JP: tl.constexpr, OP: tl.constexpr, RP: tl.constexpr,
                     BT: tl.constexpr, PC: tl.constexpr, KC: tl.constexpr,
                     KS: tl.constexpr, CPS: tl.constexpr):
    t0 = tl.program_id(0) * BT
    q = tl.program_id(1)
    p0 = tl.program_id(2) // KS * PC
    k_lo = tl.program_id(2) % KS * (CPS * KC)

    # Loop-invariant operands: B[b,q,j,c] as [(b,c), j] and A[a,p,i,b] as [(p,a), (i,b)].
    bc = tl.arange(0, RP * RP)[:, None]
    b, c = bc // RP, bc % RP
    j = tl.arange(0, JP)[None, :]
    b_tile = tl.load(b_ptr + ((b * O1 + q) * I1 + j) * R + c,
                     mask=(b < R) & (c < R) & (j < I1), other=0.0)
    pa = tl.arange(0, PC * RP)[:, None]
    p, a = p0 + pa // RP, pa % RP
    ib = tl.arange(0, IP * RP)[None, :]
    i, b = ib // RP, ib % RP
    a_tile = tl.load(a_ptr + ((a * O0 + p) * I0 + i) * R + b,
                     mask=(p < O0) & (a < R) & (i < I0) & (b < R), other=0.0)

    # Per-chunk gathers: x[t,i,j,k] as [j, (i,t,k)] and C[c,r,k,a] as [(k,a,c), r].
    itk = tl.arange(0, IP * BT * KC)[None, :]
    xi, xt, xk = itk // (BT * KC), t0 + itk // KC % BT, itk % KC
    jx = tl.arange(0, JP)[:, None]
    x_ptrs = x_ptr + xt * (I0 * I1 * I2) + xi * (I1 * I2) + jx * I2 + xk
    x_mask = (xt < T) & (xi < I0) & (jx < I1)

    kac = tl.arange(0, KC * RP * RP)[:, None]
    ck, ca, cc = kac // (RP * RP), kac // RP % RP, kac % RP
    r = tl.arange(0, OP)[None, :]
    c_ptrs = c_ptr + ((cc * O2 + r) * I2 + ck) * R + ca
    c_mask = (ca < R) & (cc < R) & (r < O2)

    acc = tl.zeros((BT * PC, OP), dtype=tl.float32)
    
    for k0 in range(k_lo, k_lo + CPS * KC, KC):  # chunks past I2 are fully masked
        xs = tl.load(x_ptrs + k0, mask=x_mask & (k0 + xk < I2), other=0.0)
        u = tl.dot(b_tile, xs)
        u = tl.permute(tl.reshape(u, (RP, RP, IP, BT, KC)), (2, 0, 3, 4, 1))
        v = tl.dot(a_tile, tl.reshape(u, (IP * RP, BT * KC * RP)).to(a_tile.dtype))
        v = tl.permute(tl.reshape(v, (PC, RP, BT, KC, RP)), (2, 0, 3, 1, 4))
        c_tile = tl.load(c_ptrs + k0 * R, mask=c_mask & (k0 + ck < I2), other=0.0)
        acc = tl.dot(tl.reshape(v, (BT * PC, KC * RP * RP)).to(a_tile.dtype), c_tile, acc)

    tp = tl.arange(0, BT * PC)[:, None]
    yt, yp = t0 + tp // PC, p0 + tp % PC
    y_ptrs = y_ptr + yt * (O0 * O1 * O2) + (yp * O1 + q) * O2 + r
    y_mask = (yt < T) & (yp < O0) & (r < O2)
    if KS == 1:
        tl.store(y_ptrs, acc.to(y_ptr.dtype.element_ty), mask=y_mask)
    else:
        tl.atomic_add(y_ptrs, acc.to(y_ptr.dtype.element_ty), mask=y_mask, sem="relaxed")


@functools.lru_cache(maxsize=None)
def _pads(spec):
    """Register-tile sizes: every tl.dot dimension must be a power of two >= 16."""
    p2 = triton.next_power_of_2
    rp = max(4, p2(spec.rank))
    return (max(p2(spec.input_modes[0]), 16 // rp), max(16, p2(spec.input_modes[1])),
            max(16, p2(spec.output_modes[2])), rp)


def config_key(spec, tokens):
    return f"{spec.input_modes}|{spec.output_modes}|{spec.rank}|{tokens}"


def valid_config(spec, config):
    ip, _, _, rp = _pads(spec)
    bt, pc, kc, _, _ = config
    return min(bt * pc, pc * rp, bt * kc * rp, ip * bt * kc) >= 16


@functools.lru_cache(maxsize=None)
def default_config(spec, tokens):
    """(BT, PC, KC, KS, num_warps): tuned table if it has this case, else a safe heuristic."""
    tuned = json.loads(TUNED_PATH.read_text()) if TUNED_PATH.exists() else {}
    if config_key(spec, tokens) in tuned:
        return tuple(tuned[config_key(spec, tokens)])
    ip, _, _, rp = _pads(spec)
    # Dot-3 rows are (t, p): BT * PC = 16. For R > 8, (4, 4, KC=1) keeps shared memory at
    # 48 KB on sm_80+ (larger tiles spill or exceed the 99 KB limit of sm_86/sm_89).
    bt, pc = (1, 16) if tokens < 4 and rp <= 8 else (4, 4)
    kc = max(2 if rp <= 8 else 1, 16 // (bt * rp), 16 // (ip * bt))
    return bt, pc, kc, 1, 4 if rp <= 8 else 8


class FusedTR:
    """Prepared fused forward for one set of cores (contiguous, same dtype/device as x).

    The first call for a token count goes through Triton's JIT launch (compile or load
    from cache). Later calls reuse that compiled kernel's launcher directly, skipping the
    JIT dispatch (argument binding, specialisation key, cache lookup), which costs tens of
    microseconds of host time per call and dominates the latency at small token counts.
    """

    def __init__(self, cores, spec, config=None):
        self.cores, self.spec, self.config = tuple(cores), spec, config
        self._launchers = {}  # tokens -> (launcher, function, metadata, grid, zero_init, constexprs)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        tokens = x.shape[0]
        cached = self._launchers.get(tokens)
        if cached is None or x.data_ptr() % 16:  # the binary assumes 16-byte aligned pointers
            return self._jit_launch(x)
        launch, function, metadata, grid, zero_init, consts = cached
        y = (torch.zeros if zero_init else torch.empty)(tokens, self.spec.out_features, device=x.device, dtype=x.dtype)
        stream = triton.runtime.driver.active.get_current_stream(x.device.index)
        launch(*grid, stream, function, metadata, None, None, None, x, *self.cores, y, tokens, *consts)
        return y

    def _jit_launch(self, x):
        spec, tokens = self.spec, x.shape[0]
        bt, pc, kc, ks, warps = self.config or default_config(spec, tokens)
        cps = triton.cdiv(triton.cdiv(spec.input_modes[2], kc), ks)  # k chunks per split
        consts = (*spec.input_modes, *spec.output_modes, spec.rank, *_pads(spec), bt, pc, kc, ks, cps)
        grid = (triton.cdiv(tokens, bt), spec.output_modes[1], triton.cdiv(spec.output_modes[0], pc) * ks)
        y = (torch.zeros if ks > 1 else torch.empty)(tokens, spec.out_features, device=x.device, dtype=x.dtype)
        kernel = _tr_fused_kernel[grid](x, *self.cores, y, tokens, *consts, num_warps=warps)
        if x.is_cuda and x.data_ptr() % 16 == 0:  # the CPU interpreter has no compiled launcher
            self._launchers[tokens] = (kernel.run, kernel.function, kernel.packed_metadata, grid, ks > 1, consts)
        return y


def tr_forward_fused(x: torch.Tensor, cores, spec, config=None) -> torch.Tensor:
    """One-off launch through Triton's JIT path (no launcher reuse)."""
    return FusedTR(cores, spec, config)(x)

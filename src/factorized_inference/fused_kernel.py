"""Single-launch fused tensor-ring forward pass (Triton).

    y[t,p,q,r] = sum_{i,j,k,a,b,c} x[t,i,j,k] A[a,p,i,b] B[b,q,j,c] C[c,r,k,a]

Contraction order x.B -> A -> C: the R^3 step sums over the smallest input mode (i).
A program owns BT tokens, one output index q and PC output indices p. It loops over k
in chunks of KC and keeps every intermediate on chip (registers / shared memory):

    U[(b,c),(i,t,k)] = sum_j     B[b,q,j,c] x[t,i,j,k]        dot 1, then on-chip transpose
    V[(p,a),(t,k,c)] = sum_{i,b} A[a,p,i,b] U[(i,b),(t,k,c)]  dot 2, then on-chip transpose
    y[(t,p),r]      += sum_{k,a,c} V[(t,p),(k,a,c)] C[c,r,k,a]  dot 3

Non-power-of-two sizes are padded in registers and masked, never in memory: the cores
are read in place (no packed copy, no workspace, no dense weight).
"""

from __future__ import annotations

import functools

import torch
import triton
import triton.language as tl


@triton.jit
def _tr_fused_kernel(x_ptr, a_ptr, b_ptr, c_ptr, y_ptr, T,
                     I0: tl.constexpr, I1: tl.constexpr, I2: tl.constexpr,
                     O0: tl.constexpr, O1: tl.constexpr, O2: tl.constexpr, R: tl.constexpr,
                     IP: tl.constexpr, JP: tl.constexpr, OP: tl.constexpr, RP: tl.constexpr,
                     BT: tl.constexpr, PC: tl.constexpr, KC: tl.constexpr):
    t0 = tl.program_id(0) * BT
    q = tl.program_id(1)
    p0 = tl.program_id(2) * PC

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
    for k0 in range(0, I2, KC):
        xs = tl.load(x_ptrs + k0, mask=x_mask & (k0 + xk < I2), other=0.0)
        u = tl.dot(b_tile, xs)
        u = tl.permute(tl.reshape(u, (RP, RP, IP, BT, KC)), (2, 0, 3, 4, 1))
        v = tl.dot(a_tile, tl.reshape(u, (IP * RP, BT * KC * RP)).to(a_tile.dtype))
        v = tl.permute(tl.reshape(v, (PC, RP, BT, KC, RP)), (2, 0, 3, 1, 4))
        c_tile = tl.load(c_ptrs + k0 * R, mask=c_mask & (k0 + ck < I2), other=0.0)
        acc = tl.dot(tl.reshape(v, (BT * PC, KC * RP * RP)).to(a_tile.dtype), c_tile, acc)

    tp = tl.arange(0, BT * PC)[:, None]
    yt, yp = t0 + tp // PC, p0 + tp % PC
    tl.store(y_ptr + yt * (O0 * O1 * O2) + (yp * O1 + q) * O2 + r,
             acc.to(y_ptr.dtype.element_ty), mask=(yt < T) & (yp < O0) & (r < O2))


@functools.lru_cache(maxsize=None)
def _config(spec, small):
    """Tile sizes; every tl.dot dimension must be a power of two >= 16."""
    p2 = triton.next_power_of_2
    rp = max(4, p2(spec.rank))
    ip, jp, op = max(p2(spec.input_modes[0]), 16 // rp), max(16, p2(spec.input_modes[1])), max(16, p2(spec.output_modes[2]))
    bt, pc = (1, 16) if small else (4, 4)  # dot-3 rows are (t, p): BT * PC = 16
    kc = max(2, 16 // (bt * rp), 16 // (ip * bt))
    return bt, pc, kc, ip, jp, op, rp, 4 if rp <= 8 else 8


def tr_forward_fused(x: torch.Tensor, cores, spec) -> torch.Tensor:
    """Launch the fused kernel. x and cores must be contiguous and share a dtype/device."""
    tokens = x.shape[0]
    bt, pc, kc, ip, jp, op, rp, warps = _config(spec, tokens < 4)
    y = torch.empty(tokens, spec.out_features, device=x.device, dtype=x.dtype)
    grid = (triton.cdiv(tokens, bt), spec.output_modes[1], triton.cdiv(spec.output_modes[0], pc))
    _tr_fused_kernel[grid](x, *cores, y, tokens, *spec.input_modes, *spec.output_modes, spec.rank,
                           ip, jp, op, rp, bt, pc, kc, num_warps=warps)
    return y

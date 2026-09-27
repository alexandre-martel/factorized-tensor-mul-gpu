"""Tensor-ring forward pass in one Triton kernel.

y[t,p,q,r] = sum over i,j,k,a,b,c of x[t,i,j,k] A[a,p,i,b] B[b,q,j,c] C[c,r,k,a]

The order is x.B, then A, then C. A program handles BT tokens, one q, PC values of p and
one of KS slices of k. For each chunk of KC values of k it runs three dots:
    U[(b,c),(i,t,k)] = B[(b,c),j] @ x[j,(i,t,k)]
    V[(p,a),(t,k,c)] = A[(p,a),(i,b)] @ U[(i,b),(t,k,c)]
    y[(t,p),r]      += V[(t,p),(k,a,c)] @ C[(k,a,c),r]
Intermediates stay on chip. Padding exists only in registers and is masked. With KS > 1
each slice writes an FP32 partial and the last slice of a tile adds them in a fixed order.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path

import torch
import triton
import triton.language as tl

# Best config per case on the benchmark GPU, written by benchmarks/tune.py.
TUNED_PATH = Path(__file__).with_name("tuned_configs.json")


@triton.jit(do_not_specialize=["T"])  # same binary for every token count
def _tr_fused_kernel(x_ptr, a_ptr, b_ptr, c_ptr, y_ptr, ws_ptr, cnt_ptr, T,
                     I0: tl.constexpr, I1: tl.constexpr, I2: tl.constexpr,
                     O0: tl.constexpr, O1: tl.constexpr, O2: tl.constexpr, R: tl.constexpr,
                     IP: tl.constexpr, JP: tl.constexpr, OP: tl.constexpr, RP: tl.constexpr,
                     BT: tl.constexpr, PC: tl.constexpr, KC: tl.constexpr,
                     KS: tl.constexpr, CPS: tl.constexpr):
    t0 = tl.program_id(0) * BT
    q = tl.program_id(1)
    p0 = tl.program_id(2) // KS * PC
    ks = tl.program_id(2) % KS
    k_lo = ks * (CPS * KC)

    # B[b,q,j,c] as [(b,c), j] and A[a,p,i,b] as [(p,a), (i,b)], loaded once.
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

    # x[t,i,j,k] as [j, (i,t,k)] and C[c,r,k,a] as [(k,a,c), r], loaded per chunk.
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
    for k0 in range(k_lo, k_lo + CPS * KC, KC):  # chunks past I2 are masked
        xs = tl.load(x_ptrs + k0, mask=x_mask & (k0 + xk < I2), other=0.0)
        u = tl.dot(b_tile, xs)
        u = tl.permute(tl.reshape(u, (RP, RP, IP, BT, KC)), (2, 0, 3, 4, 1))
        v = tl.dot(a_tile, tl.reshape(u, (IP * RP, BT * KC * RP)).to(a_tile.dtype))
        v = tl.permute(tl.reshape(v, (PC, RP, BT, KC, RP)), (2, 0, 3, 1, 4))
        c_tile = tl.load(c_ptrs + k0 * R, mask=c_mask & (k0 + ck < I2), other=0.0)
        acc = tl.dot(tl.reshape(v, (BT * PC, KC * RP * RP)).to(a_tile.dtype), c_tile, acc)

    tp = tl.arange(0, BT * PC)[:, None]
    yt, yp = t0 + tp // PC, p0 + tp % PC
    y_off = yt * (O0 * O1 * O2) + (yp * O1 + q) * O2 + r
    y_mask = (yt < T) & (yp < O0) & (r < O2)
    if KS == 1:
        tl.store(y_ptr + y_off, acc.to(y_ptr.dtype.element_ty), mask=y_mask)
    else:
        split = T * (O0 * O1 * O2)  # workspace stride between slices
        tl.store(ws_ptr + ks * split + y_off, acc, mask=y_mask)
        tl.debug_barrier()  # partial stores happen before the counter update
        tile = (tl.program_id(0) * O1 + q) * ((O0 + PC - 1) // PC) + tl.program_id(2) // KS
        if tl.atomic_add(cnt_ptr + tile, 1, sem="acq_rel") == KS - 1:  # last slice of the tile
            total = tl.zeros((BT * PC, OP), dtype=tl.float32)
            for s in tl.static_range(KS):  # fixed order, so the result is deterministic
                total += tl.load(ws_ptr + s * split + y_off, mask=y_mask, other=0.0, cache_modifier=".cg")
            tl.store(y_ptr + y_off, total.to(y_ptr.dtype.element_ty), mask=y_mask)
            tl.atomic_xchg(cnt_ptr + tile, 0)  # reset for the next call


@functools.lru_cache(maxsize=None)
def _pads(spec):
    """Padded sizes. Every tl.dot dimension must be a power of two and at least 16."""
    p2 = triton.next_power_of_2
    rp = max(4, p2(spec.rank))
    return (max(p2(spec.input_modes[0]), 16 // rp), max(16, p2(spec.input_modes[1])),
            max(16, p2(spec.output_modes[2])), rp)


def config_key(spec, tokens):
    return f"{spec.input_modes}|{spec.output_modes}|{spec.rank}|{tokens}"


def valid_config(spec, config):
    ip, _, _, rp = _pads(spec)
    bt, pc, kc = config[:3]
    return min(bt * pc, pc * rp, bt * kc * rp, ip * bt * kc) >= 16


@functools.lru_cache(maxsize=None)
def default_config(spec, tokens):
    """(BT, PC, KC, KS, num_warps, num_stages) from the tuned table, else a heuristic."""
    tuned = json.loads(TUNED_PATH.read_text()) if TUNED_PATH.exists() else {}
    if config_key(spec, tokens) in tuned:
        return tuple(tuned[config_key(spec, tokens)])
    ip, _, _, rp = _pads(spec)
    # BT * PC = 16 rows for dot 3. For R > 8, (4, 4, KC=1) needs 48 KB of shared memory.
    bt, pc = (1, 16) if tokens < 4 and rp <= 8 else (4, 4)
    kc = max(2 if rp <= 8 else 1, 16 // (bt * rp), 16 // (ip * bt))
    return bt, pc, kc, 1, 4 if rp <= 8 else 8, 3


class FusedTR:
    """Prepared forward for one set of contiguous cores.

    The first call for a token count uses Triton's JIT launch. Later calls use the compiled
    kernel's C launcher with prebuilt arguments. The output buffer and the split-K scratch
    are allocated once per token count and reused, so an output must be consumed before the
    next call, and calls must be stream-ordered.
    """

    def __init__(self, cores, spec, config=None):
        self.cores, self.spec, self.config = tuple(cores), spec, config
        self._launchers = {}  # tokens -> (launch, grid, launch args, kernel args after x, y)
        self._scratch = {}  # tokens -> (output, workspace, counters)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        cached = self._launchers.get(x.shape[0])
        if cached is None or x.data_ptr() % 16:  # the binary assumes 16-byte alignment
            return self._jit_launch(x)
        launch, grid, head, args, y = cached
        launch(*grid, self._get_stream(x.device.index), *head, (x, *args))
        return y

    def _jit_launch(self, x):
        spec, tokens = self.spec, x.shape[0]
        bt, pc, kc, ks, warps, stages = self.config or default_config(spec, tokens)
        cps = triton.cdiv(triton.cdiv(spec.input_modes[2], kc), ks)  # k chunks per split
        consts = (*spec.input_modes, *spec.output_modes, spec.rank, *_pads(spec), bt, pc, kc, ks, cps)
        grid = (triton.cdiv(tokens, bt), spec.output_modes[1], triton.cdiv(spec.output_modes[0], pc) * ks)
        if tokens not in self._scratch:  # counters start at 0 and the kernel resets them
            size = ks * tokens * spec.out_features if ks > 1 else 1
            self._scratch[tokens] = (torch.empty(tokens, spec.out_features, device=x.device, dtype=x.dtype),
                                     torch.empty(size, device=x.device, dtype=torch.float32),
                                     torch.zeros(grid[0] * grid[1] * grid[2] // ks, device=x.device, dtype=torch.int32))
        y, *scratch = self._scratch[tokens]
        args = (*self.cores, y, *scratch, tokens, *consts)
        kernel = _tr_fused_kernel[grid](x, *args, num_warps=warps, num_stages=stages)
        if x.is_cuda and x.data_ptr() % 16 == 0:  # no compiled launcher in the CPU interpreter
            cl = kernel.run  # Python launcher, cl.launch is its C function
            if cl.global_scratch_size or cl.profile_scratch_size or cl.gsan_enabled:
                launch = lambda *a: cl(*a[:-1], *a[-1])  # noqa: E731
                head = (kernel.function, kernel.packed_metadata, None, None, None)
            else:
                launch = cl.launch
                head = (kernel.function, cl.launch_cooperative_grid, cl.launch_pdl, kernel.packed_metadata,
                        None, None, None, None, None, cl.arg_annotations, cl.kernel_signature)
            self._get_stream = triton.runtime.driver.active.get_current_stream
            self._launchers[tokens] = (launch, grid, head, args, y)
        return y


def tr_forward_fused(x: torch.Tensor, cores, spec, config=None) -> torch.Tensor:
    """Single call through Triton's JIT launch."""
    return FusedTR(cores, spec, config)(x)

"""Single-launch fused tensor-ring forward pass (Triton).

    y[t,p,q,r] = sum_{i,j,k,a,b,c} x[t,i,j,k] A[a,p,i,b] B[b,q,j,c] C[c,r,k,a]

Contraction order x.B -> A -> C: the R^3 step sums over the smallest input mode (i).
A program owns BT tokens, one output index q, PC output indices p and 1/KS of the k range
(split-K). It loops over its k range in chunks of KC and keeps every intermediate on chip
(registers / shared memory):

    U[(b,c),(i,t,k)] = sum_j     B[b,q,j,c] x[t,i,j,k]        dot 1, then on-chip transpose
    V[(p,a),(t,k,c)] = sum_{i,b} A[a,p,i,b] U[(i,b),(t,k,c)]  dot 2, then on-chip transpose
    y[(t,p),r]      += sum_{k,a,c} V[(t,p),(k,a,c)] C[c,r,k,a]  dot 3

Non-power-of-two sizes are padded in registers and masked, never in memory: the cores
are read in place (no packed copy, no dense weight). With KS > 1, each split stores its
FP32 partial in a workspace; the last split of a tile to finish (per-tile counter) sums
the KS partials in a fixed order, writes y and resets the counter: no zero-initialised
output, no extra launch, deterministic results.
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
    y_off = yt * (O0 * O1 * O2) + (yp * O1 + q) * O2 + r
    y_mask = (yt < T) & (yp < O0) & (r < O2)
    if KS == 1:
        tl.store(y_ptr + y_off, acc.to(y_ptr.dtype.element_ty), mask=y_mask)
    else:
        split = T * (O0 * O1 * O2)  # workspace stride between splits
        tl.store(ws_ptr + ks * split + y_off, acc, mask=y_mask)
        tl.debug_barrier()  # all threads' partial stores precede the release below
        tile = (tl.program_id(0) * O1 + q) * ((O0 + PC - 1) // PC) + tl.program_id(2) // KS
        if tl.atomic_add(cnt_ptr + tile, 1, sem="acq_rel") == KS - 1:  # last split of this tile
            total = tl.zeros((BT * PC, OP), dtype=tl.float32)
            for s in tl.static_range(KS):  # fixed order: deterministic sum
                total += tl.load(ws_ptr + s * split + y_off, mask=y_mask, other=0.0, cache_modifier=".cg")
            tl.store(y_ptr + y_off, total.to(y_ptr.dtype.element_ty), mask=y_mask)
            tl.atomic_xchg(cnt_ptr + tile, 0)  # ready for the next call


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
    from cache). Later calls skip Triton's Python dispatch (argument binding,
    specialisation key, cache lookup, launcher wrapper), which costs tens of microseconds
    of host time and dominates the latency at small token counts: they call the compiled
    kernel's C launch function with pre-built arguments.

    Reused storage, allocated once per token count: the output buffer (callers consume it
    before the next call, as the task allows) and the split-K scratch (FP32 partials +
    self-resetting tile counters). Calls on one callable must therefore be stream-ordered.
    """

    def __init__(self, cores, spec, config=None):
        self.cores, self.spec, self.config = tuple(cores), spec, config
        self._launchers = {}  # tokens -> (launch fn, grid, leading launch args, kernel args after x, y)
        self._scratch = {}  # tokens -> (output, workspace, counters)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        cached = self._launchers.get(x.shape[0])
        if cached is None or x.data_ptr() % 16:  # the binary assumes 16-byte aligned pointers
            return self._jit_launch(x)
        launch, grid, head, args, y = cached
        launch(*grid, self._get_stream(x.device.index), *head, (x, *args))
        return y

    def _jit_launch(self, x):
        spec, tokens = self.spec, x.shape[0]
        bt, pc, kc, ks, warps = self.config or default_config(spec, tokens)
        cps = triton.cdiv(triton.cdiv(spec.input_modes[2], kc), ks)  # k chunks per split
        consts = (*spec.input_modes, *spec.output_modes, spec.rank, *_pads(spec), bt, pc, kc, ks, cps)
        grid = (triton.cdiv(tokens, bt), spec.output_modes[1], triton.cdiv(spec.output_modes[0], pc) * ks)
        if tokens not in self._scratch:  # counters start at 0; the kernel resets them after use
            size = ks * tokens * spec.out_features if ks > 1 else 1
            self._scratch[tokens] = (torch.empty(tokens, spec.out_features, device=x.device, dtype=x.dtype),
                                     torch.empty(size, device=x.device, dtype=torch.float32),
                                     torch.zeros(grid[0] * grid[1] * grid[2] // ks, device=x.device, dtype=torch.int32))
        y, *scratch = self._scratch[tokens]
        args = (*self.cores, y, *scratch, tokens, *consts)
        kernel = _tr_fused_kernel[grid](x, *args, num_warps=warps)
        if x.is_cuda and x.data_ptr() % 16 == 0:  # the CPU interpreter has no compiled launcher
            cl = kernel.run  # Triton's Python launcher; its C launch function is cl.launch
            if cl.global_scratch_size or cl.profile_scratch_size or cl.gsan_enabled:
                launch = lambda *a: cl(*a[:-1], *a[-1])  # noqa: E731 (needs the Python launcher)
                head = (kernel.function, kernel.packed_metadata, None, None, None)
            else:
                launch = cl.launch
                head = (kernel.function, cl.launch_cooperative_grid, cl.launch_pdl, kernel.packed_metadata,
                        None, None, None, None, None, cl.arg_annotations, cl.kernel_signature)
            self._get_stream = triton.runtime.driver.active.get_current_stream
            self._launchers[tokens] = (launch, grid, head, args, y)
        return y


def tr_forward_fused(x: torch.Tensor, cores, spec, config=None) -> torch.Tensor:
    """One-off launch through Triton's JIT path (no launcher reuse)."""
    return FusedTR(cores, spec, config)(x)

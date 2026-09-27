"""Config sweep for the fused kernel on the five required cases (GPU).

Each config is checked against an FP32 dense oracle and timed like the harness: CUDA-event
stream latency, median of 5 blocks of 20 calls after 20 warm-up calls. Host dispatch time
per call is also recorded. With --write, the fastest correct config per case is written to
src/factorized_inference/tuned_configs.json.

    python benchmarks/tune.py --output results/tune.json --write
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from factorized_inference import (  # noqa: E402
    TRSpec, dense_forward, make_cores, materialize_dense_weight, tr_forward_reference,
)
from factorized_inference.fused_kernel import (  # noqa: E402
    TUNED_PATH, FusedTR, config_key, valid_config,
)

CASES = [(8, 1), (8, 8), (8, 32), (16, 1), (16, 32)]


def stream_ms(fn, block=20, samples=5, warmup=20):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(samples):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        for _ in range(block):
            fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) / block)
    return statistics.median(times)


def dispatch_us(fn, calls=200):
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(calls):
        fn()
    elapsed = time.perf_counter() - start
    torch.cuda.synchronize()
    return elapsed / calls * 1e6


def candidates(spec, tokens):
    """Stage 1: tile shapes with 3 pipeline stages (Triton's default).

    BT * PC is 16, or 32 from 8 tokens. R=16 with KC=4 or 4 warps is skipped because offline
    compilation for sm_80 and sm_90 showed register spills or too much shared memory.
    """
    for bt, pc, kc, ks, warps in itertools.product((1, 2, 4, 8), (2, 4, 8, 16), (1, 2, 4), (1, 2, 5, 10), (4, 8)):
        heavy = spec.rank > 8 and (kc > 2 or warps < 8)
        if (bt * pc in ((16, 32) if tokens >= 8 else (16,)) and bt <= max(4, tokens)
                and (tokens <= 8 or ks <= 2) and not heavy and valid_config(spec, (bt, pc, kc))):
            yield bt, pc, kc, ks, warps, 3


def refinements(best):
    """Stage 2: warps and pipeline stages for the six fastest tile shapes."""
    for bt, pc, kc, ks, _, _ in best:
        for warps, stages in itertools.product((2, 4, 8), (1, 2, 3, 4)):
            yield bt, pc, kc, ks, warps, stages


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=ROOT / "results/tune.json")
    parser.add_argument("--write", action="store_true", help="update tuned_configs.json")
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    report, tuned = {"device": torch.cuda.get_device_name(), "torch": torch.__version__, "cases": []}, {}
    for rank, tokens in CASES:
        spec = TRSpec(rank=rank)
        cores = make_cores(spec, device="cuda", dtype=torch.float16, seed=0)
        x = torch.randn(tokens, spec.in_features, device="cuda", dtype=torch.float16,
                        generator=torch.Generator(device="cuda").manual_seed(tokens))
        expected = dense_forward(x.float(), materialize_dense_weight([c.float() for c in cores], spec))
        weight = materialize_dense_weight(cores, spec)
        case = {"rank": rank, "tokens": tokens, "configs": [],
                "dense_ms": stream_ms(lambda: dense_forward(x, weight)),
                "reference_ms": stream_ms(lambda: tr_forward_reference(x, cores, spec))}
        del weight

        def measure(config):
            fused = FusedTR(cores, spec, config)
            run = lambda: fused(x)  # noqa: E731
            try:
                run()  # first call compiles, later calls use the direct launcher
                out = run().float()
                row = {"config": config, "max_abs_err": (out - expected).abs().max().item(),
                       "stream_ms": stream_ms(run), "dispatch_us": dispatch_us(run)}
                # checked again after the timed calls, which reuse the split-K scratch
                row["ok"] = all(torch.allclose(o, expected, rtol=2e-2, atol=2e-2) for o in (out, run().float()))
            except Exception as exc:  # for example not enough shared memory
                row = {"config": config, "ok": False, "error": repr(exc)[:300]}
            case["configs"].append(row)

        for config in candidates(spec, tokens):
            measure(config)
        seen = {row["config"] for row in case["configs"]}
        ranked = sorted((row for row in case["configs"] if row["ok"]), key=lambda row: row["stream_ms"])
        for config in refinements([row["config"] for row in ranked[:6]]):
            if config not in seen:
                seen.add(config)
                measure(config)
        report["cases"].append(case)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")  # saved after every case
        good = [row for row in case["configs"] if row["ok"]]
        if not good:
            print(f"R={rank:2d} T={tokens:2d}  no correct config (see {args.output})", flush=True)
            continue
        case["best"] = best = min(good, key=lambda row: row["stream_ms"])
        tuned[config_key(spec, tokens)] = best["config"]
        print(f"R={rank:2d} T={tokens:2d}  dense {case['dense_ms'] * 1e3:7.1f} us  reference "
              f"{case['reference_ms'] * 1e3:7.1f} us  fused {best['stream_ms'] * 1e3:7.1f} us "
              f"{tuple(best['config'])}  dispatch {best['dispatch_us']:.1f} us  "
              f"({len(good)}/{len(case['configs'])} configs ok)", flush=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if args.write:
        TUNED_PATH.write_text(json.dumps(tuned, indent=2) + "\n")
        print(f"Wrote {TUNED_PATH}")


if __name__ == "__main__":
    main()

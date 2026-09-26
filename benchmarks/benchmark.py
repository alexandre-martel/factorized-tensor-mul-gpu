from __future__ import annotations

import argparse
import gc
import json
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from factorized_inference import (  # noqa: E402
    TRSpec, dense_forward, make_cores, materialize_dense_weight,
    prepare_optimized, tr_forward_reference,
)

METHODS = ("dense", "factorized_reference", "factorized_optimized")


def triplet(value):
    values = tuple(int(part) for part in value.split(","))
    if len(values) != 3 or min(values) < 1:
        raise argparse.ArgumentTypeError("Use three positive integers")
    return values


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed_call(fn, device):
    synchronize(device)
    start = time.perf_counter_ns()
    value = fn()
    synchronize(device)
    return value, (time.perf_counter_ns() - start) / 1e6


def environment(device, args):
    try:
        driver = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total",
             "--format=csv,noheader"], text=True, timeout=10,
        ).strip() if device.type == "cuda" else None
    except (OSError, subprocess.SubprocessError):
        driver = "unavailable; record manually"
    return {
        "python": sys.version, "platform": platform.platform(),
        "cpu": platform.processor(), "torch": str(torch.__version__),
        "cuda_runtime": torch.version.cuda, "nvidia_smi": driver,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "dtype": args.dtype, "cpu_threads": torch.get_num_threads(),
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "fp16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
    }


def errors(actual, expected):
    difference = (actual.float() - expected.float()).abs()
    return {
        "max_absolute_error": difference.max().item(),
        "mean_absolute_error": difference.mean().item(),
        "relative_l2_error": (
            torch.linalg.vector_norm(difference)
            / torch.linalg.vector_norm(expected.float()).clamp_min(1e-12)
        ).item(),
    }


@torch.inference_mode()
def worker(args):
    device = torch.device(args.device)
    if device.type not in ("cpu", "cuda"):
        raise SystemExit("Supported benchmark devices: cpu, cuda, cuda:N")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    if device.type == "cpu" and args.dtype == "float16":
        raise SystemExit("Use float32 for the CPU smoke check")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.set_num_threads(args.cpu_threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    dtype = getattr(torch, args.dtype)
    spec = TRSpec(args.input_modes, args.output_modes, args.rank)
    tokens = int(args.tokens)

    # Initialize the device before measuring representation/preparation memory.
    probe = torch.empty(1, device=device)
    del probe
    synchronize(device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        baseline_bytes = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
    else:
        baseline_bytes = None

    cores = make_cores(spec, device=device, dtype=dtype, seed=args.seed)
    generator = torch.Generator(device=device).manual_seed(args.seed + tokens)
    x = torch.randn(tokens, spec.in_features, generator=generator, device=device, dtype=dtype)
    if args.worker == "dense":
        weight, preparation_ms = timed_call(lambda: materialize_dense_weight(cores, spec), device)
        run = lambda value: dense_forward(value, weight)
        representation_bytes = weight.numel() * weight.element_size()
        del cores
    elif args.worker == "factorized_reference":
        run = lambda value: tr_forward_reference(value, cores, spec)
        preparation_ms = 0.0
        representation_bytes = sum(c.numel() * c.element_size() for c in cores)
    else:
        run, preparation_ms = timed_call(lambda: prepare_optimized(cores, spec), device)
        representation_bytes = sum(c.numel() * c.element_size() for c in cores)

    first_output, first_call_ms = timed_call(lambda: run(x), device)
    del first_output
    for _ in range(args.warmup):
        run(x)
    synchronize(device)
    gc.collect()
    input_bytes = x.numel() * x.element_size()
    if device.type == "cuda":
        setup_peak = torch.cuda.max_memory_allocated(device)
        resident = torch.cuda.memory_allocated(device)
        # Warm-up allocations remain resident and are explicitly reported.
        torch.cuda.reset_peak_memory_stats(device)
    else:
        setup_peak = resident = None

    samples = []
    for _ in range(args.iterations):
        result, elapsed = timed_call(lambda: run(x), device)
        del result
        samples.append(elapsed)
    if device.type == "cuda":
        steady_peak = torch.cuda.max_memory_allocated(device)
        memory = {
            "context_baseline_allocated_bytes": baseline_bytes,
            "input_logical_bytes": input_bytes,
            "representation_logical_bytes": representation_bytes,
            "resident_after_warmup_allocated_bytes": resident,
            "extra_resident_allocated_bytes": max(0, resident - baseline_bytes - input_bytes - representation_bytes),
            "preparation_and_warmup_peak_allocated_bytes": setup_peak,
            "steady_peak_allocated_bytes": steady_peak,
            "incremental_workspace_and_output_bytes": steady_peak - resident,
            "steady_peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        }
    else:
        memory = {"input_logical_bytes": input_bytes,
                  "representation_logical_bytes": representation_bytes,
                  "cuda_memory": None}

    # Events bracket blocks to amortize event/synchronization overhead. Elapsed
    # stream time can still include host-submission gaps; it is not kernel-only.
    event_samples = []
    if device.type == "cuda":
        for _ in range(5):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            synchronize(device)
            start.record()
            for _ in range(args.event_block):
                run(x)
            end.record()
            end.synchronize()
            event_samples.append(start.elapsed_time(end) / args.event_block)

    # Validation comes AFTER measurement. Re-create pristine factors and use
    # FP32 dense arithmetic so validation tensors never contaminate memory data.
    check_cores = make_cores(spec, device=device, dtype=dtype, seed=args.seed)
    oracle_weight = materialize_dense_weight(tuple(c.float() for c in check_cores), spec)
    expected = dense_forward(x.float(), oracle_weight)
    actual = run(x)
    tolerance = 2e-2 if dtype == torch.float16 else 1e-4
    torch.testing.assert_close(actual.float(), expected, rtol=tolerance, atol=tolerance)
    correctness = {"rtol": tolerance, "atol": tolerance, **errors(actual, expected)}
    del actual, expected, oracle_weight, check_cores

    trace_path = None
    if args.profile_dir and tokens == 1:
        args.profile_dir.mkdir(parents=True, exist_ok=True)
        trace_path = args.profile_dir / f"{args.worker}_rank{args.rank}_tokens{tokens}.json"
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(activities=activities, record_shapes=True, profile_memory=True) as prof:
            for _ in range(3):
                with torch.profiler.record_function(args.worker):
                    run(x)
            synchronize(device)
        prof.export_chrome_trace(str(trace_path))

    return {
        "method": args.worker, "tokens": tokens,
        "environment": environment(device, args),
        "preparation_ms": preparation_ms, "first_call_ms": first_call_ms,
        "host_synchronized_median_ms": statistics.median(samples),
        "host_synchronized_samples_ms": samples,
        "cuda_event_stream_median_ms": statistics.median(event_samples) if event_samples else None,
        "cuda_event_stream_samples_ms": event_samples,
        "memory": memory, "correctness": correctness,
        "profiler_trace": str(trace_path) if trace_path else None,
    }


def main():
    parser = argparse.ArgumentParser(description="Isolated per-method factorized inference benchmark")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--input-modes", type=triplet, default=TRSpec().input_modes)
    parser.add_argument("--output-modes", type=triplet, default=TRSpec().output_modes)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--tokens", default="1,8,32")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--event-block", type=int, default=20)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("benchmark_results.json"))
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument("--worker", choices=METHODS, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.rank, args.warmup, args.iterations, args.event_block, args.cpu_threads) < 1:
        parser.error("Rank, warmup, iterations, event-block and cpu-threads must be positive")
    try:
        token_counts = [int(t) for t in args.tokens.split(",")]
        if not token_counts or min(token_counts) < 1:
            raise ValueError()
    except ValueError:
        parser.error("Tokens must be comma-separated positive integers")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.worker:
        if len(token_counts) != 1:
            parser.error("Internal worker requires exactly one token count")
        args.output.write_text(json.dumps(worker(args), indent=2) + "\n")
        return

    spec = TRSpec(args.input_modes, args.output_modes, args.rank)
    output = {
        "schema_version": 2,
        "spec": {"input_modes": spec.input_modes, "output_modes": spec.output_modes,
                 "rank": spec.rank, "dense_parameters": spec.dense_parameters,
                 "factor_parameters": spec.factor_parameters},
        "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k != "worker"},
        "measurement_notes": [
            "Each method/token case runs in a fresh child process; no baseline tensors coexist.",
            "Host timing includes dispatch and synchronization; CUDA events measure block stream elapsed time.",
            "Preparation and first call are separate; steady timing excludes compilation in warmup.",
            "Memory is PyTorch allocated/reserved memory, not total process VRAM or non-PyTorch allocations.",
            "Setup peak includes factor loading and dense materialization where applicable. Reserved memory may retain setup allocator blocks.",
            "Correctness oracle and profiler allocations are excluded from memory and latency results.",
        ],
        "cases": [],
    }
    with tempfile.TemporaryDirectory(prefix="factorized-benchmark-") as temporary:
        for tokens in token_counts:
            case = {"tokens": tokens, "methods": {}}
            for method in METHODS:
                result_path = Path(temporary) / f"{method}-{tokens}.json"
                command = [sys.executable, str(Path(__file__).resolve()), "--worker", method,
                           "--tokens", str(tokens), "--output", str(result_path)]
                for flag in ("device", "dtype", "rank", "warmup", "iterations", "event_block", "cpu_threads", "seed"):
                    command += ["--" + flag.replace("_", "-"), str(getattr(args, flag))]
                for flag in ("input_modes", "output_modes"):
                    command += ["--" + flag.replace("_", "-"), ",".join(map(str, getattr(args, flag)))]
                if args.profile_dir:
                    command += ["--profile-dir", str(args.profile_dir.resolve())]
                subprocess.run(command, check=True)
                result = json.loads(result_path.read_text())
                case["methods"][method] = result
                event = result["cuda_event_stream_median_ms"]
                peak = result["memory"].get("steady_peak_allocated_bytes")
                print(f"tokens={tokens} {method}: host={result['host_synchronized_median_ms']:.4f} ms; "
                      f"CUDA stream={event if event is not None else 'N/A'} ms; "
                      f"steady allocated peak={peak if peak is not None else 'N/A'} bytes", flush=True)
            output["cases"].append(case)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

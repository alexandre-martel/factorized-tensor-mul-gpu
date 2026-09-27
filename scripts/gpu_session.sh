#!/usr/bin/env bash
# Whole GPU session in one go: setup, tuning sweep, tests, the two required benchmark runs,
# profiler traces and environment. Run from the repo root on a Linux NVIDIA machine
# (sm_75+, driver >= 580 for torch 2.14 / CUDA 13):   bash scripts/gpu_session.sh
set -euo pipefail
mkdir -p results traces
date -u +"start %FT%TZ" > results/session_times.txt
nvidia-smi | tee results/nvidia-smi.txt

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
# uv-managed Python ships its headers (Python.h), which Triton needs to build its launcher.
rm -rf .venv && uv venv -q --seed --managed-python -p 3.12 .venv
source .venv/bin/activate
uv pip install -q -e '.[test]'
python -c "import torch, triton; assert torch.cuda.is_available(), 'CUDA unavailable: driver too old for CUDA 13?'; print(torch.__version__, triton.__version__, torch.cuda.get_device_name())"
python -c "
import torch
from factorized_inference import TRSpec, make_cores, prepare_optimized, tr_forward_reference
s = TRSpec(); c = make_cores(s, device='cuda', dtype=torch.float16)
x = torch.randn(1, s.in_features, device='cuda', dtype=torch.float16)
torch.testing.assert_close(prepare_optimized(c, s)(x).float(), tr_forward_reference(x, c, s).float(), rtol=2e-2, atol=2e-2)
print('fused kernel runs on the GPU and matches the reference')"

set +e  # from here on, run every step and always pack the artifacts
python benchmarks/tune.py --output results/tune.json --write 2>&1 | tee results/tune.txt
pytest -q 2>&1 | tee results/pytest.txt
# README commands with --device cuda:0 instead of cuda: with torch 2.14 the harness's
# torch.cuda.set_device(torch.device("cuda")) raises "Expected a torch.device with a specified index".
python benchmarks/benchmark.py --device cuda:0 --dtype float16 \
  --rank 8 --tokens 1,8,32 --output results/rank8.json --profile-dir traces/rank8
python benchmarks/benchmark.py --device cuda:0 --dtype float16 \
  --rank 16 --tokens 1,32 --output results/rank16.json
python -m pip freeze > results/environment.txt
date -u +"end %FT%TZ" >> results/session_times.txt

tar czf gpu_artifacts.tgz results traces $(ls src/factorized_inference/tuned_configs.json 2>/dev/null)
echo "Done: copy gpu_artifacts.tgz back, then stop AND delete the instance."

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
rm -rf .venv && uv venv -q --seed -p 3.12 .venv
source .venv/bin/activate
uv pip install -q -e '.[test]'
python -c "import torch, triton; assert torch.cuda.is_available(), 'CUDA unavailable: driver too old for CUDA 13?'; print(torch.__version__, triton.__version__, torch.cuda.get_device_name())"

set +e  # from here on, run every step and always pack the artifacts
python benchmarks/tune.py --output results/tune.json --write 2>&1 | tee results/tune.txt
pytest -q 2>&1 | tee results/pytest.txt
python benchmarks/benchmark.py --device cuda --dtype float16 \
  --rank 8 --tokens 1,8,32 --output results/rank8.json --profile-dir traces/rank8
python benchmarks/benchmark.py --device cuda --dtype float16 \
  --rank 16 --tokens 1,32 --output results/rank16.json
python -m pip freeze > results/environment.txt
date -u +"end %FT%TZ" >> results/session_times.txt

tar czf gpu_artifacts.tgz results traces $(ls src/factorized_inference/tuned_configs.json 2>/dev/null)
echo "Done: copy gpu_artifacts.tgz back, then stop AND delete the instance."

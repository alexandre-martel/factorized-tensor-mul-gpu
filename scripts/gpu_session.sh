#!/usr/bin/env bash
# GPU session: setup, checks, tests, the two required benchmark commands, repeats and extra
# measurements. Needs Linux, an NVIDIA GPU (tested on A100, sm_80) and driver >= 580.
# Usage from the repo root: bash scripts/gpu_session.sh
# TUNE=1 runs the config sweep first. REPEATS (default 3) sets how many times the required
# benchmarks run. The first run writes results/rank8.json, results/rank16.json and the traces.
set -euo pipefail
mkdir -p results/repeats traces
date -u +"start %FT%TZ" > results/session_times.txt
nvidia-smi | tee results/nvidia-smi.txt

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
# Triton needs Python.h to build its launcher. uv-managed Python includes it.
rm -rf .venv && uv venv -q --seed --managed-python -p 3.12 .venv
source .venv/bin/activate
uv pip install -q -e '.[test]'
python -c "import torch, triton; assert torch.cuda.is_available(); print(torch.__version__, triton.__version__, torch.cuda.get_device_name())"
python -c "
import torch
from factorized_inference import TRSpec, make_cores, prepare_optimized, tr_forward_reference
s = TRSpec(); c = make_cores(s, device='cuda', dtype=torch.float16)
x = torch.randn(1, s.in_features, device='cuda', dtype=torch.float16)
run, ref = prepare_optimized(c, s), tr_forward_reference(x, c, s).float()
for _ in range(2):  # JIT launch, then the direct launcher
    torch.testing.assert_close(run(x).float(), ref, rtol=2e-2, atol=2e-2)
print('fused kernel matches the reference on the GPU')"

set +e  # run every remaining step and always pack the results
if [ "${TUNE:-0}" = 1 ]; then
  python benchmarks/tune.py --output results/tune.json --write 2>&1 | tee results/tune.txt
fi
pytest -q 2>&1 | tee results/pytest.txt
# The README uses --device cuda. With torch 2.14 the harness call
# torch.cuda.set_device(torch.device("cuda")) fails, so cuda:0 is passed instead.
python benchmarks/benchmark.py --device cuda:0 --dtype float16 \
  --rank 8 --tokens 1,8,32 --output results/rank8.json --profile-dir traces/rank8
python benchmarks/benchmark.py --device cuda:0 --dtype float16 \
  --rank 16 --tokens 1,32 --output results/rank16.json
for n in $(seq 2 "${REPEATS:-3}"); do
  python benchmarks/benchmark.py --device cuda:0 --dtype float16 \
    --rank 8 --tokens 1,8,32 --output "results/repeats/rank8_run$n.json"
  python benchmarks/benchmark.py --device cuda:0 --dtype float16 \
    --rank 16 --tokens 1,32 --output "results/repeats/rank16_run$n.json"
done
python benchmarks/host_overhead.py 2>&1 | tee results/host_overhead.txt
python benchmarks/graphs.py 2>&1 | tee results/graphs.txt
python -m pip freeze > results/environment.txt
date -u +"end %FT%TZ" >> results/session_times.txt

tar czf gpu_artifacts.tgz results traces src/factorized_inference/tuned_configs.json
echo "Done. Copy gpu_artifacts.tgz back, then delete the instance."

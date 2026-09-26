from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_cpu_benchmark_runs_all_isolated_methods(tmp_path):
    script = Path(__file__).resolve().parents[1] / 'benchmarks/benchmark.py'
    output = tmp_path / 'results.json'
    subprocess.run([
        sys.executable, str(script), '--device', 'cpu', '--dtype', 'float32',
        '--input-modes', '2,2,2', '--output-modes', '2,3,2', '--rank', '2',
        '--tokens', '1', '--warmup', '1', '--iterations', '2', '--output', str(output),
    ], check=True, capture_output=True, text=True)
    results = json.loads(output.read_text())
    assert results['spec']['dense_parameters'] == 96
    methods = results['cases'][0]['methods']
    assert set(methods) == {'dense', 'factorized_reference', 'factorized_optimized'}
    for result in methods.values():
        assert len(result['host_synchronized_samples_ms']) == 2
        assert result['cuda_event_stream_median_ms'] is None
        assert result['memory']['cuda_memory'] is None
        assert result['correctness']['max_absolute_error'] < 1e-4
    assert methods['dense']['memory']['representation_logical_bytes'] == 96 * 4
    assert methods['factorized_reference']['memory']['representation_logical_bytes'] == 56 * 4

from __future__ import annotations

import pytest
import torch

from factorized_inference import (
    TRSpec, dense_forward, make_cores, materialize_dense_weight,
    prepare_optimized, tr_forward_reference,
)


def oracle(x, cores, spec):
    # Higher-precision oracle starts from the same quantized factor values.
    return dense_forward(x.double(), materialize_dense_weight(tuple(c.double() for c in cores), spec))


def check_result(actual, expected, dtype):
    tolerance = 2e-2 if dtype == torch.float16 else 1e-4
    assert actual.shape == expected.shape
    assert actual.dtype == dtype
    assert actual.device == expected.device
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual.double(), expected, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize('seed', [0, 19])
@pytest.mark.parametrize('spec', [
    TRSpec((2, 3, 2), (2, 4, 3), 2),
    TRSpec((4, 4, 4), (4, 4, 4), 8),
    TRSpec((4, 4, 4), (4, 4, 4), 16),
])
@pytest.mark.parametrize('method', ['reference', 'optimized'])
def test_cpu_numerics_and_changing_inputs(spec, seed, method):
    cores = make_cores(spec, seed=seed)
    original_cores = tuple(c.clone() for c in cores)
    run = (prepare_optimized(cores, spec) if method == 'optimized'
           else lambda x: tr_forward_reference(x, cores, spec))
    generator = torch.Generator().manual_seed(seed + 37)
    # One prepared object must handle changed values AND token counts.
    for tokens, scale in [(1, 1.0), (5, 0.0), (3, 0.1), (1, 2.0)]:
        x = torch.randn(tokens, spec.in_features, generator=generator) * scale
        original_x = x.clone()
        check_result(run(x), oracle(original_x, original_cores, spec), x.dtype)
        torch.testing.assert_close(x, original_x, rtol=0, atol=0)
        for core, original in zip(cores, original_cores):
            torch.testing.assert_close(core, original, rtol=0, atol=0)


@pytest.mark.parametrize('device,dtype', [
    ('cpu', torch.float32),
    pytest.param('cuda', torch.float16, marks=pytest.mark.skipif(
        not torch.cuda.is_available(), reason='CUDA required for FP16 checks')),
])
def test_prepared_weights_are_independent(device, dtype):
    spec = TRSpec((2, 3, 2), (2, 4, 3), 3)
    a = make_cores(spec, device=device, dtype=dtype, seed=1)
    b = make_cores(spec, device=device, dtype=dtype, seed=2)
    a_original = tuple(c.clone() for c in a)
    b_original = tuple(c.clone() for c in b)
    run_a = prepare_optimized(a, spec)
    run_b = prepare_optimized(b, spec)
    generator = torch.Generator(device=device).manual_seed(7)
    x = torch.randn(3, spec.in_features, generator=generator, device=device, dtype=dtype)
    for run, weights in [(run_a, a_original), (run_b, b_original), (run_a, a_original)]:
        check_result(run(x), oracle(x, weights, spec), dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required for final acceptance')
@pytest.mark.parametrize('rank,tokens', [(8, 1), (8, 8), (8, 32), (16, 1), (16, 32)])
def test_required_cuda_workloads(rank, tokens):
    spec = TRSpec(rank=rank)
    cores = make_cores(spec, device='cuda', dtype=torch.float16, seed=23)
    generator = torch.Generator(device='cuda').manual_seed(41 + tokens)
    x = torch.randn(tokens, spec.in_features, generator=generator, device='cuda', dtype=torch.float16)
    expected = oracle(x, cores, spec)
    check_result(tr_forward_reference(x, cores, spec), expected, x.dtype)
    check_result(prepare_optimized(cores, spec)(x), expected, x.dtype)


def test_tensor_ring_definition_independently():
    # Direct trace of selected core slices verifies index layout independently
    # of the contractions used by materialize_dense_weight and the reference.
    spec = TRSpec((2, 2, 2), (2, 2, 2), 2)
    cores = make_cores(spec, dtype=torch.float64, seed=11)
    weight = materialize_dense_weight(cores, spec)
    for output in range(8):
        p, q, r = output // 4, output // 2 % 2, output % 2
        for input_index in range(8):
            i, j, k = input_index // 4, input_index // 2 % 2, input_index % 2
            expected = torch.trace(cores[0][:, p, i, :] @ cores[1][:, q, j, :] @ cores[2][:, r, k, :])
            torch.testing.assert_close(weight[output, input_index], expected, rtol=1e-12, atol=1e-12)

# Copyright 2026 Reflective
# SPDX-License-Identifier: Apache-2.0
"""Native ordered redistribution, branch semantics and material budgets."""
import jax
import numpy as np
import pytest

from salsa_jax.grid import SizeGrid
from salsa_jax.constants import PI
from salsa_jax.processes.remapping import RemappingParameters, remap
from tools.remapping_oracle import build_oracle, reference_cases, run_oracle

jax.config.update('jax_enable_x64', True)
PROPERTY_NAMES = {'surface_tension': 'water_surface_tension', 'ion_factor': 'ion_count'}


def arguments(cases):
    low, high = cases['lower_volume'], cases['upper_volume']
    mid = (PI / 6.0) * cases['dry_diameter']**3
    grid = SizeGrid(low, high, cases['dry_diameter'], high/mid, low/mid)
    parameters = RemappingParameters(**{key: cases[PROPERTY_NAMES.get(key, key)]
                                       for key in RemappingParameters._fields})
    return (cases['number'], cases['volume'], cases['temperature'], grid), {
        'transfer_soluble': cases['soluble_transfer'], 'parameters': parameters}


def evaluate(cases, compiled=True):
    args, kwargs = arguments(cases)
    return (jax.jit(remap) if compiled else remap)(*args, **kwargs)


@pytest.fixture(scope='module')
def reference(tmp_path_factory):
    executable, provenance = build_oracle(tmp_path_factory.mktemp('remapping-jax'))
    cases = reference_cases()
    expected, _ = run_oracle(executable, cases)
    return executable, cases, expected, provenance


@pytest.mark.parametrize('compiled', [False, True])
def test_ordered_transfers_and_stopping_match_actual_fortran(reference, compiled):
    _, cases, expected, provenance = reference
    assert provenance['checked']
    result = evaluate(cases, compiled)
    for name in ('number', 'volume'):
        value = np.asarray(getattr(result, name))
        assert np.all(np.isfinite(value))
        np.testing.assert_allclose(value, expected[name], rtol=3e-13, atol=0)
    np.testing.assert_array_equal(result.iterations, expected['iterations'][:, 0])


def test_number_and_each_material_conserved_without_erasing_orphans(reference):
    _, cases, expected, _ = reference
    result = evaluate(cases)
    np.testing.assert_allclose(np.asarray(result.number).sum(-1), cases['number'].sum(-1), rtol=3e-14, atol=0)
    np.testing.assert_allclose(np.asarray(result.volume).sum(-2), cases['volume'].sum(-2), rtol=3e-14, atol=0)
    orphan = (expected['number'] == 0) & (expected['volume'].sum(-1) > 0)
    assert np.any(orphan), 'Native large-transfer residual material must remain visible'
    np.testing.assert_allclose(np.asarray(result.volume)[orphan], expected['volume'][orphan], rtol=3e-13, atol=0)


def test_two_calls_preserve_native_pass_order(reference):
    executable, cases, _, _ = reference
    expected, _ = run_oracle(executable, cases, passes=2)
    args, kwargs = arguments(cases)
    @jax.jit
    def twice(number, volume, temperature, grid, **kwargs):
        first = remap(number, volume, temperature, grid, **kwargs)
        second = remap(first.number, first.volume, temperature, grid, **kwargs)
        return second, first.iterations
    actual, first_counts = twice(*args, **kwargs)
    for name in ('number', 'volume'):
        np.testing.assert_allclose(getattr(actual, name), expected[name], rtol=3e-13, atol=0)
    np.testing.assert_array_equal(first_counts, expected['iterations'][:, 0])
    np.testing.assert_array_equal(actual.iterations, expected['iterations'][:, 1])


def test_scalar_and_multiaxis_batched_cells_match(reference):
    _, cases, _, _ = reference
    one = {key: value[0] for key, value in cases.items() if key != 'labels'}
    many = {key: np.broadcast_to(value, (2, 3) + np.shape(value)) for key, value in one.items()}
    scalar, batched = evaluate(one), evaluate(many)
    for name in ('number', 'volume', 'iterations'):
        np.testing.assert_array_equal(getattr(batched, name),
                                     np.broadcast_to(getattr(scalar, name), np.shape(getattr(batched, name))))


def test_invalid_shape_is_rejected(reference):
    _, cases, _, _ = reference
    args, kwargs = arguments(cases)
    with pytest.raises(ValueError, match='Expected'):
        remap(args[0][:, :-1], args[1], *args[2:], **kwargs)


@pytest.mark.parametrize('compiled', [False, True])
def test_invalid_soluble_zero_volume_signals_native_failure(reference, compiled):
    _, cases, _, _ = reference
    c = {key: value[:2].copy() for key, value in cases.items() if key != 'labels'}
    baseline = evaluate(c, compiled)
    c['number'][0] = 0
    c['volume'][0] = 0
    c['number'][0, 16] = 16
    c['soluble_transfer'][0] = True
    result = evaluate(c, compiled)
    assert np.all(np.isnan(result.number[0]))
    assert np.all(np.isnan(result.volume[0]))
    np.testing.assert_array_equal(result.number[1], baseline.number[1])
    np.testing.assert_array_equal(result.volume[1], baseline.volume[1])


@pytest.mark.parametrize('compiled', [False, True])
def test_strict_soluble_threshold_survives_scalar_and_batch_shapes(reference, compiled):
    _, cases, expected, _ = reference
    indices = [i for i, label in enumerate(cases['labels']) if 'soluble-fraction-threshold' in label]
    assert len(indices) == 3
    for i in indices:
        one = {key: value[i] for key, value in cases.items() if key != 'labels'}
        for shape in ((), (1,), (2, 3)):
            c = {key: np.broadcast_to(value, shape + np.shape(value)) for key, value in one.items()}
            result = evaluate(c, compiled)
            np.testing.assert_array_equal(result.number, np.broadcast_to(expected['number'][i], shape + (17,)))

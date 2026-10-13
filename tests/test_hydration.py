# Copyright 2026 Reflective
# SPDX-License-Identifier: Apache-2.0
"""Native water uptake, empty-bin policy and per-bin convergence acceptance."""
import jax
import numpy as np
import pytest

from salsa_jax.constants import PI
from salsa_jax.processes.hydration import HydrationParameters, equilibrate, solubility_mask
from tools.hydration_oracle import build_oracle, reference_cases, run_oracle

jax.config.update('jax_enable_x64', True)
INPUTS = ('number', 'volume', 'relative_humidity', 'temperature', 'dry_diameter')


def run_jax(cases, compiled=True):
    kernel = jax.jit(equilibrate) if compiled else equilibrate
    return kernel(*(cases[key] for key in INPUTS), soluble_mask=cases['soluble_mask'],
                  parameters=HydrationParameters(**{key: cases[key] for key in HydrationParameters._fields}))


@pytest.fixture(scope='module')
def reference(tmp_path_factory):
    executable, provenance = build_oracle(tmp_path_factory.mktemp('hydration-jax'))
    cases = reference_cases()
    expected, _ = run_oracle(executable, cases)
    return cases, expected, provenance


@pytest.mark.parametrize('compiled', [False, True])
def test_core_wet_diameter_and_iterations_match_fortran(reference, compiled):
    cases, expected, _ = reference
    result = run_jax(cases, compiled)
    for field in ('core', 'wet_diameter'):
        value = np.asarray(getattr(result, field))
        assert np.all(np.isfinite(value))
        assert np.all(value > 0)
        np.testing.assert_allclose(value, expected[field], rtol=3e-13, atol=0)
    np.testing.assert_array_equal(result.iterations, expected['iterations'])
    assert np.any(np.asarray(result.iterations) > 1)
    assert np.any(np.asarray(result.iterations) == 0)


def test_active_core_matches_material_volume_and_inactive_uses_midpoint(reference):
    cases, _, _ = reference
    result = run_jax(cases)
    active = cases['number'] > cases['number_threshold'][:, None]
    core = np.asarray(result.core)
    np.testing.assert_allclose((core * cases['number'])[active], cases['volume'].sum(-1)[active], rtol=3e-15, atol=0)
    empty_core = (PI / 6) * cases['dry_diameter']**3
    np.testing.assert_allclose(core[~active], empty_core[~active], rtol=3e-15, atol=0)
    np.testing.assert_array_equal(np.asarray(result.wet_diameter)[~active], cases['dry_diameter'][~active])
    insoluble = active & ~cases['soluble_mask'].astype(bool)
    np.testing.assert_allclose(np.asarray(result.wet_diameter)[insoluble]**3 * (PI/6), core[insoluble], rtol=4e-15, atol=0)
    np.testing.assert_array_equal(np.asarray(result.iterations)[~active | insoluble], 0)
    assert np.all(np.asarray(result.wet_diameter)**3 * (PI/6) >= core * (1-5e-15))


def test_native_solubility_initialization_for_both_hydration_switches():
    mask = np.asarray(solubility_mask(np.array([False, True])))
    np.testing.assert_array_equal(mask[0], np.arange(17) < 10)
    np.testing.assert_array_equal(mask[1], True)


def test_scalar_multiaxis_and_repeated_evaluation(reference):
    cases, expected, _ = reference
    one = {key: value[0] for key, value in cases.items() if key != 'labels'}
    scalar = run_jax(one)
    many = {key: np.broadcast_to(value, (2, 3) + np.shape(value)) for key, value in one.items()}
    result = run_jax(many)
    for field in ('core', 'wet_diameter', 'iterations'):
        np.testing.assert_allclose(getattr(scalar, field), expected[field][0], rtol=3e-13)
        np.testing.assert_array_equal(getattr(result, field), np.broadcast_to(getattr(scalar, field), (2, 3, 17)))
    repeated = run_jax(one)
    for before, after in zip(scalar, repeated):
        np.testing.assert_array_equal(before, after)


def test_invalid_shapes_rejected():
    with pytest.raises(ValueError, match='Expected'):
        equilibrate(np.zeros(16), np.zeros((16,5)), .5, 280., np.ones(16))


@pytest.mark.parametrize('compiled', [False, True])
def test_native_zero_volume_failure_is_nonfinite_not_an_infinite_jax_loop(reference, compiled):
    cases, _, _ = reference
    c = {key: value[:1].copy() for key, value in cases.items() if key != 'labels'}
    baseline = run_jax(c, compiled)
    c['number'][0, 0] = 1e8
    c['volume'][0, 0] = 0
    c['soluble_mask'][0, 0] = True
    result = run_jax(c, compiled)
    assert np.isnan(result.wet_diameter[0, 0])
    assert result.iterations[0, 0] == 0
    np.testing.assert_array_equal(result.wet_diameter[:, 1:], baseline.wet_diameter[:, 1:])

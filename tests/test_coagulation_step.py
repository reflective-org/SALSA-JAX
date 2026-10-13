# Copyright 2026 Reflective
# SPDX-License-Identifier: Apache-2.0
"""Complete coagulation parity uses verbatim Fortran, not a Python formula oracle."""
from pathlib import Path

import jax
import numpy as np
import pytest

from salsa_jax.processes.coagulation_step import (
    coefficient_matrix, coagulation_step, coagulation_update,
)
from tools.coagulation_process_oracle import build_oracle, reference_cases, run_oracle

jax.config.update("jax_enable_x64", True)
RTOL = 2e-13


def arguments(cases):
    return tuple(cases[key] for key in
                 ("number", "volume", "wet_diameter", "dry_diameter", "temperature", "pressure", "timestep"))


@pytest.fixture(scope="module")
def reference(tmp_path_factory):
    executable, provenance = build_oracle(tmp_path_factory.mktemp("full-coagulation"))
    cases = reference_cases()
    expected, timing = run_oracle(executable, cases)
    return executable, cases, expected, provenance, timing


@pytest.mark.parametrize("compiled", [False, True])
def test_complete_coagulation_fortran_parity(reference, compiled):
    _, cases, expected, _, _ = reference
    kernel = jax.jit(coagulation_step) if compiled else coagulation_step
    result = kernel(*arguments(cases))
    for field in ("number", "volume"):
        actual = np.asarray(getattr(result, field))
        assert actual.dtype == np.float64
        assert np.all(np.isfinite(actual))
        assert np.all(actual >= 0)
        np.testing.assert_allclose(actual, expected[field], rtol=RTOL, atol=0)


@pytest.mark.parametrize("compiled", [False, True])
def test_coefficient_matrix_matches_actual_fortran(reference, compiled):
    _, cases, expected, _, _ = reference
    kernel = jax.jit(coefficient_matrix) if compiled else coefficient_matrix
    actual = np.asarray(kernel(cases["dry_diameter"], cases["wet_diameter"],
                               cases["temperature"], cases["pressure"]))
    np.testing.assert_allclose(actual, expected["coefficients"], rtol=1e-13, atol=0)
    np.testing.assert_array_equal(actual, actual.swapaxes(-1, -2))
    assert np.all(actual > 0)


def test_transfer_with_fortran_coefficients(reference):
    _, cases, expected, _, _ = reference
    actual = jax.jit(coagulation_update)(cases["number"], cases["volume"], expected["coefficients"], cases["timestep"])
    np.testing.assert_allclose(actual.number, expected["number"], rtol=RTOL, atol=0)
    np.testing.assert_allclose(actual.volume, expected["volume"], rtol=RTOL, atol=0)


def test_disabled_process_preserves_state_exactly(reference):
    _, cases, _, _, _ = reference
    # A disabled caller skips the complete native process, including relabeling.
    volume = cases["volume"].copy()
    volume[:, 10:, 3] = 1e-12
    args = list(arguments(cases))
    args[1] = volume
    result = jax.jit(coagulation_step)(*args, enabled=False)
    np.testing.assert_array_equal(result.number, cases["number"])
    np.testing.assert_array_equal(result.volume, volume)


def test_scalar_and_multiple_leading_axes_match_batch(reference):
    _, cases, expected, _, _ = reference
    one = {key: value[0] for key, value in cases.items() if key != "labels"}
    scalar = coagulation_step(*arguments(one))
    assert scalar.number.shape == (17,)
    assert scalar.volume.shape == (17, 5)
    np.testing.assert_allclose(scalar.volume, expected["volume"][0], rtol=RTOL, atol=0)
    two = {key: np.stack([value[:2], value[:2]]) for key, value in cases.items() if key != "labels"}
    result = jax.jit(coagulation_step)(*arguments(two))
    assert result.number.shape == (2, 2, 17)
    np.testing.assert_allclose(result.volume, np.stack([expected["volume"][:2]]*2), rtol=RTOL, atol=0)


def test_fresh_fortran_process_repeats_exactly(reference):
    executable, cases, expected, _, _ = reference
    repeated, _ = run_oracle(executable, cases, repetitions=2)
    for field in expected:
        np.testing.assert_array_equal(repeated[field], expected[field])


def test_reject_incompatible_material_layout():
    with pytest.raises(ValueError, match="matching leading axes"):
        coagulation_update(np.zeros(17), np.zeros((17, 6)), np.zeros((17, 17)), 1.0)
    with pytest.raises(ValueError, match="17-bin"):
        coefficient_matrix(np.ones(16), np.ones(16), 280., 101325.)


def captured_case():
    from tools.coagulation_capture import load_fixture
    state = load_fixture(Path(__file__).parent / 'fixtures/coagulation-default')
    def record(stage, name):
        item = state[stage][name]
        return np.asarray(item['values']).reshape(item['shape'], order='F')
    np.testing.assert_array_equal(record('entry', 'boundaries'), [1, 3, 4, 10, 11, 17])
    case = {name: record('entry', native).reshape(shape) for name, native, shape in (
        ('number', 'pnaero', (1, 17)), ('volume', 'zvols', (1, 17, 5)),
        ('wet_diameter', 'zdwet', (1, 17)), ('dry_diameter', 'dpmid', (1, 17)),
        ('temperature', 'ptemp', (1,)), ('pressure', 'ppres', (1,)), ('timestep', 'ptstep', (1,)))}
    return case, {'number': record('exit', 'pnaero').reshape(1, 17),
                  'volume': record('exit', 'zvols').reshape(1, 17, 5)}


@pytest.mark.parametrize('compiled', [False, True])
def test_actual_post_hydration_process_capture(reference, compiled):
    executable, _, _, _, _ = reference
    cases, expected = captured_case()
    # First establish that extraction/stubbed globals reproduce the real model
    # boundary; then compare the JAX process with that same captured result.
    original, _ = run_oracle(executable, cases)
    kernel = jax.jit(coagulation_step) if compiled else coagulation_step
    actual = kernel(*arguments(cases))
    for field in expected:
        np.testing.assert_allclose(original[field], expected[field], rtol=RTOL, atol=0)
        np.testing.assert_allclose(getattr(actual, field), expected[field], rtol=RTOL, atol=0)
    before = cases['volume'].sum(axis=(1, 2))
    after = np.asarray(actual.volume).sum(axis=(1, 2))
    # Current captured case is dilute enough for near-roundoff closure. Stress
    # cases have a documented reference defect; this is not a global guarantee.
    np.testing.assert_allclose(after, before, rtol=2e-13, atol=0)


def test_reference_budget_drift_is_preserved_not_hidden(reference):
    _, cases, expected, _, _ = reference
    result = coagulation_step(*arguments(cases))
    initial = cases['volume'].sum(axis=(1, 2))
    reference_change = expected['volume'].sum(axis=(1, 2)) - initial
    actual_change = np.asarray(result.volume).sum(axis=(1, 2)) - initial
    nonempty = initial > 0
    np.testing.assert_allclose(actual_change[nonempty] / initial[nonempty],
                               reference_change[nonempty] / initial[nonempty], rtol=0, atol=3e-15)
    stressed = cases['labels'].index('mixed-dense-long-step')
    assert reference_change[stressed] / initial[stressed] > 0.8
    # Particle number is lost, including in cases with the volume-budget defect.
    assert np.all(np.asarray(result.number).sum(axis=1) <= cases['number'].sum(axis=1))


def test_repeated_process_calls_with_fixed_supplied_geometry(reference):
    executable, cases, _, _, _ = reference
    # This is an isolated-process sequence, not a hydrated full SALSA trajectory.
    chosen = [0, cases['labels'].index('unequal-hydration'), len(cases['labels']) - 1]
    original = {key: value[chosen].copy() for key, value in cases.items() if key != 'labels'}
    jax_state = (original['number'].copy(), original['volume'].copy())
    kernel = jax.jit(coagulation_step)
    for _ in range(20):
        expected, _ = run_oracle(executable, original)
        actual = kernel(*jax_state, original['wet_diameter'], original['dry_diameter'],
                        original['temperature'], original['pressure'], original['timestep'])
        for field in ('number', 'volume'):
            np.testing.assert_allclose(getattr(actual, field), expected[field], rtol=4e-13, atol=0)
        original['number'], original['volume'] = expected['number'], expected['volume']
        jax_state = actual


def test_enabled_zero_step_retains_native_salt_relabeling(reference):
    executable, cases, _, _, _ = reference
    case = {key: value[:1].copy() for key, value in cases.items() if key != 'labels'}
    case['volume'][:, 10:, 3] = 1e-12
    case['timestep'][:] = 0
    expected, _ = run_oracle(executable, case)
    actual = jax.jit(coagulation_step)(*arguments(case))
    np.testing.assert_array_equal(actual.number, case['number'])
    np.testing.assert_array_equal(actual.volume, expected['volume'])
    np.testing.assert_array_equal(actual.volume[:, 10:, 3], 0)
    np.testing.assert_array_equal(actual.volume[:, 10:, 0],
                                  case['volume'][:, 10:, 0] + case['volume'][:, 10:, 3])


def test_batched_state_shared_geometry_and_scalar_environment(reference):
    executable, cases, _, _, _ = reference
    cases = {key: value[:3].copy() for key, value in cases.items() if key != 'labels'}
    for name, value in (('temperature', 280.), ('pressure', 101325.), ('timestep', 60.)):
        cases[name][:] = value
    expected, _ = run_oracle(executable, cases)
    actual = jax.jit(coagulation_step)(cases['number'], cases['volume'], cases['wet_diameter'],
                                      cases['dry_diameter'][0], 280., 101325., 60.)
    np.testing.assert_allclose(actual.number, expected['number'], rtol=RTOL, atol=0)
    np.testing.assert_allclose(actual.volume, expected['volume'], rtol=RTOL, atol=0)

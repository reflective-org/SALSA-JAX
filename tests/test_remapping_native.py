"""Replay both actual native remapping/correction passes and their JAX chain."""
from copy import deepcopy
from pathlib import Path

import jax
import numpy as np
import pytest

from salsa_jax.grid import size_grid
from salsa_jax.processes.hydration import HydrationParameters
from salsa_jax.processes.remapping import RemappingParameters, remap
from salsa_jax.processes.remapping_correction import correction
from tools.remapping_fixture import adapt_capture, load_native_case
from tools.remapping_capture import load_fixture
from tools.remapping_oracle import build_oracle, run_oracle
from tools.remapping_correction_oracle import build_oracle as build_correction, run_oracle as run_correction
from tools.condensation_capture import load_fixture as load_condensation
from tools.hydration_fixture import record as process_record

jax.config.update('jax_enable_x64', True)
ROOT = Path(__file__).resolve().parents[1]
PROPERTY_NAMES = {'surface_tension': 'water_surface_tension', 'ion_factor': 'ion_count'}


def remapping_parameters(cases):
    return RemappingParameters(**{key: cases[PROPERTY_NAMES.get(key, key)] for key in RemappingParameters._fields})


def hydration_parameters(cases):
    return HydrationParameters(**{key: cases[key] for key in HydrationParameters._fields})


def assert_outputs(actual, expected, cases):
    for field, reference in expected.items():
        value = actual[field] if isinstance(actual, dict) else np.asarray(getattr(actual, field))
        if field == 'water_mass':
            # Subtracting wet and dry particle volumes is ill-conditioned near
            # zero water. Normalize cancellation error by native dry mass.
            scale = expected['core'] * cases['water_density'][:, None] * expected['number']
            assert np.all(np.abs(value-reference) <= 3e-13*np.abs(reference)+32*np.finfo(float).eps*scale)
        else:
            np.testing.assert_allclose(value, reference, rtol=3e-13, atol=0)


@pytest.fixture(scope='module')
def native(tmp_path_factory):
    stages, metadata = load_native_case()
    remapping, _ = build_oracle(tmp_path_factory.mktemp('native-remapping'))
    correction_exe, _ = build_correction(tmp_path_factory.mktemp('native-correction'))
    oracles = []
    for stage in stages:
        remapped, _ = run_oracle(remapping, stage['remapping_cases'])
        corrected = run_correction(correction_exe, stage['correction_cases'])
        oracles.append((remapped, corrected))
    return stages, metadata, oracles


def test_both_native_boundaries_match_independent_fortran(native):
    stages, _, oracles = native
    for stage, (remapped, corrected) in zip(stages, oracles):
        for name in ('number', 'volume'):
            np.testing.assert_allclose(remapped[name], stage['remapped'][name], rtol=3e-13, atol=0)
        assert_outputs(corrected, stage['corrected'], stage['correction_cases'])
        for name in ('number', 'volume', 'gas'):
            np.testing.assert_array_equal(corrected[name], stage['cleaned'][name])


@pytest.mark.parametrize('compiled', [False, True])
def test_both_native_passes_match_jax(native, compiled):
    stages, metadata, oracles = native
    for stage, (oracle_remap, oracle_correct) in zip(stages, oracles):
        cases, c = stage['remapping_cases'], stage['correction_cases']
        grid = size_grid(metadata['regime_limits'])
        for name, key in (('vlolim', 'lower_volume'), ('vhilim', 'upper_volume'), ('dpmid', 'dry_diameter')):
            np.testing.assert_array_equal(np.broadcast_to(getattr(grid, name), cases[key].shape), cases[key])
        kernel = jax.jit(remap) if compiled else remap
        remapped = kernel(cases['number'], cases['volume'], cases['temperature'], grid,
                          transfer_soluble=cases['soluble_transfer'], parameters=remapping_parameters(cases))
        for name in ('number', 'volume'):
            np.testing.assert_allclose(getattr(remapped, name), stage['remapped'][name], rtol=3e-13, atol=0)
        np.testing.assert_array_equal(remapped.iterations, oracle_remap['iterations'][:, 0])
        kernel = jax.jit(correction) if compiled else correction
        corrected = kernel(*(c[key] for key in ('number', 'volume', 'gas', 'relative_humidity', 'temperature', 'dry_diameter')),
                           soluble_mask=c['soluble_mask'], organic_enabled=c['organic_enabled'], parameters=hydration_parameters(c))
        assert np.all(corrected.valid)
        assert_outputs(corrected, stage['corrected'], c)
        np.testing.assert_array_equal(corrected.iterations, oracle_correct['iterations'])


@pytest.mark.parametrize('compiled', [False, True])
def test_native_two_pass_chain_starts_at_condensation_boundary(native, compiled):
    stages, metadata, _ = native
    first = stages[0]
    cases, c = first['remapping_cases'], first['correction_cases']
    condensation = load_condensation(ROOT/'tests/fixtures/condensation-default')
    n = len(cases['number'])
    for name, source in (('number', 'pnaero'), ('volume', 'zvols')):
        raw = process_record(condensation, 'exit', source)
        np.testing.assert_array_equal(cases[name], raw.reshape((n,)+raw.shape[2:]))
    raw_gas = process_record(condensation, 'exit', 'pc_gas').reshape(n, 2)
    gas = np.pad(raw_gas, ((0, 0), (0, 1)))
    np.testing.assert_array_equal(gas, c['gas'])

    def advance(number, volume, gas, rh, temperature, limits, rparams, hparams, soluble, transfer, organic):
        grid = size_grid(limits)
        results = []
        for _ in range(2):
            remapped = remap(number, volume, temperature, grid, transfer_soluble=transfer, parameters=rparams)
            corrected = correction(remapped.number, remapped.volume, gas, rh, temperature, grid.dpmid,
                                   soluble_mask=soluble, organic_enabled=organic, parameters=hparams)
            number, volume, gas = corrected.number, corrected.volume, corrected.gas
            results.append((remapped, corrected))
        return results

    kernel = jax.jit(advance) if compiled else advance
    results = kernel(cases['number'], cases['volume'], gas, c['relative_humidity'], c['temperature'],
                     metadata['regime_limits'], remapping_parameters(cases), hydration_parameters(c),
                     c['soluble_mask'], cases['soluble_transfer'], c['organic_enabled'])
    for stage, (remapped, corrected) in zip(stages, results):
        assert np.all(corrected.valid)
        for field in ('number', 'volume'):
            np.testing.assert_allclose(getattr(remapped, field), stage['remapped'][field], rtol=3e-13, atol=0)
        assert_outputs(corrected, stage['corrected'], stage['correction_cases'])


@pytest.mark.parametrize('field', ['argas', 'gas_indices', 'soluble'])
def test_adapter_rejects_incompatible_native_context(field):
    capture = deepcopy(load_fixture(ROOT/'tests/fixtures/remapping-default'))
    values = capture['context'][field]['values']
    values[0] = values[0]*1.01 if field == 'argas' else (0 if field == 'soluble' else 1)
    with pytest.raises(ValueError):
        adapt_capture(capture)

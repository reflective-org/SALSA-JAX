# Copyright 2026 Reflective
# SPDX-License-Identifier: Apache-2.0
"""Separate condensation acceptance against current repaired Fortran, nsnucl=0."""
from pathlib import Path

import jax
import numpy as np
import pytest

from salsa_jax.constants import AVOGADRO
from salsa_jax.processes.condensation import CondensationParameters, condensation_rates, condensation_step
from tools.condensation_process_oracle import build_oracle, reference_cases, run_oracle

jax.config.update('jax_enable_x64', True)
RTOL = 2e-13
INPUTS = ('number', 'volume', 'gas', 'wet_diameter', 'temperature', 'pressure', 'timestep')


def parameters(cases):
    return CondensationParameters(**{key: cases[key] for key in (
        'sulfate_molar_mass', 'sulfate_density', 'organic_molar_mass', 'organic_density', 'mass_accommodation')})


def run_jax(cases, compiled=False):
    kernel = jax.jit(condensation_step) if compiled else condensation_step
    return kernel(*(cases[key] for key in INPUTS), parameters=parameters(cases),
                  organic_enabled=cases['organic_enabled'])


@pytest.fixture(scope='module')
def reference(tmp_path_factory):
    executable, provenance = build_oracle(tmp_path_factory.mktemp('condensation-step'))
    cases = reference_cases()
    expected, timing = run_oracle(executable, cases)
    return executable, cases, expected, provenance, timing


@pytest.mark.parametrize('compiled', [False, True])
def test_condensation_matches_current_fortran(reference, compiled):
    _, cases, expected, _, _ = reference
    result = run_jax(cases, compiled)
    for field in ('number', 'volume', 'gas'):
        actual = np.asarray(getattr(result, field))
        assert actual.dtype == np.float64
        assert np.all(np.isfinite(actual))
        assert np.all(actual >= 0)
        np.testing.assert_allclose(actual, expected[field], rtol=RTOL, atol=0)
    np.testing.assert_array_equal(result.number, cases['number'])
    np.testing.assert_array_equal(result.gas[:, 0], cases['gas'][:, 0])
    np.testing.assert_array_equal(result.volume[..., 2:], cases['volume'][..., 2:])


@pytest.mark.parametrize('compiled', [False, True])
def test_collision_sinks_match_native_diagnostic(reference, compiled):
    _, cases, expected, _, _ = reference
    kernel = jax.jit(condensation_rates) if compiled else condensation_rates
    rates = np.asarray(kernel(cases['number'], cases['wet_diameter'], cases['temperature'],
                              cases['pressure'], parameters=parameters(cases)))
    np.testing.assert_allclose(rates, expected['rates'], rtol=1e-13, atol=0)
    assert np.all(np.isfinite(rates))
    assert np.all(rates >= 0)


def test_each_condensing_species_closes_gas_particle_budget(reference):
    _, cases, _, _, _ = reference
    actual = run_jax(cases, True)
    for gas_slot, volume_slot, prefix in ((1, 0, 'sulfate'), (2, 1, 'organic')):
        molecule_volume = cases[prefix + '_molar_mass'] / 1000 / AVOGADRO / cases[prefix + '_density']
        before = cases['gas'][:, gas_slot] + cases['volume'][:, :, volume_slot].sum(axis=1) / molecule_volume
        after = np.asarray(actual.gas)[:, gas_slot] + np.asarray(actual.volume)[:, :, volume_slot].sum(axis=1) / molecule_volume
        np.testing.assert_allclose(after, before, rtol=3e-14, atol=0)
        assert np.all(np.asarray(actual.gas)[:, gas_slot] <= cases['gas'][:, gas_slot])
        assert np.all(np.asarray(actual.volume)[:, :, volume_slot] >= cases['volume'][:, :, volume_slot])
    absent = ~cases['organic_enabled'].astype(bool)
    np.testing.assert_array_equal(np.asarray(actual.gas)[absent, 2], cases['gas'][absent, 2])
    np.testing.assert_array_equal(np.asarray(actual.volume)[absent, :, 1], cases['volume'][absent, :, 1])


def test_disabled_process_is_exact_identity(reference):
    _, cases, _, _, _ = reference
    actual = jax.jit(condensation_step)(*(cases[key] for key in INPUTS), enabled=False,
                                       organic_enabled=cases['organic_enabled'], parameters=parameters(cases))
    for field in ('number', 'volume', 'gas'):
        np.testing.assert_array_equal(getattr(actual, field), cases[field])


def test_scalar_and_multiaxis_pytree_contract(reference):
    _, cases, expected, _, _ = reference
    one = {key: value[0] for key, value in cases.items() if key != 'labels'}
    actual = run_jax(one, True)
    assert actual.number.shape == (17,)
    assert actual.volume.shape == (17, 5)
    assert actual.gas.shape == (3,)
    np.testing.assert_allclose(actual.volume, expected['volume'][0], rtol=RTOL, atol=0)
    multi = {key: np.stack([value[:2], value[:2]]) for key, value in cases.items() if key != 'labels'}
    actual = run_jax(multi, True)
    assert actual.volume.shape == (2, 2, 17, 5)
    np.testing.assert_allclose(actual.gas, np.stack([expected['gas'][:2]]*2), rtol=RTOL, atol=0)


def test_repeated_calls_match_fortran_without_number_creation(reference):
    executable, cases, _, _, _ = reference
    chosen = [0, len(cases['labels'])-1]
    original = {key: value[chosen].copy() for key, value in cases.items() if key != 'labels'}
    evolving = {key: value.copy() for key, value in original.items()}
    initial_number = original['number'].copy()
    for _ in range(20):
        expected, _ = run_oracle(executable, original)
        actual = run_jax(evolving, True)
        for field in ('number', 'volume', 'gas'):
            np.testing.assert_allclose(getattr(actual, field), expected[field], rtol=4e-13, atol=0)
            original[field] = expected[field]
            evolving[field] = np.asarray(getattr(actual, field))
    np.testing.assert_array_equal(actual.number, initial_number)


def test_wrong_state_shapes_are_rejected():
    with pytest.raises(ValueError, match='matching number'):
        condensation_step(np.zeros(17), np.zeros((17, 5)), np.zeros(2), np.ones(17), 280., 101325., 1.)
    with pytest.raises(ValueError, match='17-bin'):
        condensation_rates(np.zeros(16), np.ones(16), 280., 101325.)


def captured_case():
    from tools.condensation_capture import load_fixture
    state = load_fixture(Path(__file__).parent / 'fixtures/condensation-default')
    def record(stage, name):
        item = state[stage][name]
        return np.asarray(item['values']).reshape(item['shape'], order='F')
    # Default standalone mapping: SO2,H2SO4, with organic gas unregistered.
    np.testing.assert_array_equal(record('entry', 'gas_indices'), [2, 0])
    cases = {name: record('entry', native).reshape(shape) for name, native, shape in (
        ('number', 'pnaero', (1, 17)), ('volume', 'zvols', (1, 17, 5)),
        ('wet_diameter', 'zdwet', (1, 17)), ('temperature', 'ptemp', (1,)),
        ('pressure', 'ppres', (1,)), ('timestep', 'ptstep', (1,)),
        ('mass_accommodation', 'massacc', (1, 17)))}
    for index, prefix in enumerate(('sulfate', 'organic')):
        cases[prefix + '_molar_mass'] = record('entry', 'species_molar_mass')[index:index+1]
        cases[prefix + '_density'] = record('entry', 'species_density')[index:index+1]
    cases['organic_enabled'] = np.array([False])
    cases['gas'] = np.pad(record('entry', 'pc_gas').reshape(1, 2), ((0, 0), (0, 1)))
    expected = {'number': record('exit', 'pnaero').reshape(1, 17),
                'volume': record('exit', 'zvols').reshape(1, 17, 5),
                'gas': np.pad(record('exit', 'pc_gas').reshape(1, 2), ((0, 0), (0, 1)))}
    return cases, expected, state


@pytest.mark.parametrize('compiled', [False, True])
def test_actual_condensation_boundary_matches_extraction_and_jax(reference, compiled):
    executable, _, _, _, _ = reference
    cases, expected, _ = captured_case()
    original, _ = run_oracle(executable, cases)
    actual = run_jax(cases, compiled)
    for field in expected:
        np.testing.assert_allclose(original[field], expected[field], rtol=RTOL, atol=0)
        np.testing.assert_allclose(getattr(actual, field), expected[field], rtol=RTOL, atol=0)


def test_default_parameters_match_compiled_runtime_metadata():
    from salsa_jax.constants import BOLTZMANN, GAS_CONSTANT, PI, REFERENCE_PRESSURE, SULFURIC_ACID_DIAMETER
    _, _, state = captured_case()
    params = CondensationParameters()
    for name, value in [('avo', AVOGADRO), ('ak', BOLTZMANN), ('argas', GAS_CONSTANT),
                        ('pi', PI), ('p0sl_bg', REFERENCE_PRESSURE), ('d_sa', SULFURIC_ACID_DIAMETER)]:
        assert state['entry'][name]['values'] == [value]
    assert state['entry']['species_molar_mass']['values'] == [params.sulfate_molar_mass, params.organic_molar_mass]
    assert state['entry']['species_density']['values'] == [params.sulfate_density, params.organic_density]
    np.testing.assert_array_equal(state['entry']['massacc']['values'], np.full(17, params.mass_accommodation))


def test_shared_parameters_and_scalar_environment(reference):
    executable, cases, _, _, _ = reference
    chosen = {key: value[:3].copy() for key, value in cases.items() if key != 'labels'}
    params = CondensationParameters()
    for key in ('sulfate_molar_mass', 'sulfate_density', 'organic_molar_mass', 'organic_density', 'mass_accommodation'):
        chosen[key][...] = getattr(params, key)
    for key, value in (('temperature', 280.), ('pressure', 101325.), ('timestep', 60.)):
        chosen[key][...] = value
    expected, _ = run_oracle(executable, chosen)
    actual = jax.jit(condensation_step)(chosen['number'], chosen['volume'], chosen['gas'],
                                       chosen['wet_diameter'], 280., 101325., 60., organic_enabled=chosen['organic_enabled'])
    np.testing.assert_allclose(actual.volume, expected['volume'], rtol=RTOL, atol=0)
    np.testing.assert_allclose(actual.gas, expected['gas'], rtol=RTOL, atol=0)


def test_coagulation_then_condensation_matches_captured_process_boundaries():
    from salsa_jax.processes.coagulation_step import coagulation_step
    from tools.coagulation_capture import load_fixture
    coagulation = load_fixture(Path(__file__).parent / 'fixtures/coagulation-default')
    def record(name):
        item = coagulation['entry'][name]
        return np.asarray(item['values']).reshape(item['shape'], order='F')
    coagulated = jax.jit(coagulation_step)(record('pnaero').reshape(1, 17), record('zvols').reshape(1, 17, 5),
                                         record('zdwet').reshape(1, 17), record('dpmid'),
                                         record('ptemp').reshape(1), record('ppres').reshape(1), record('ptstep'))
    cases, expected, _ = captured_case()
    np.testing.assert_allclose(coagulated.number, cases['number'], rtol=RTOL, atol=0)
    np.testing.assert_allclose(coagulated.volume, cases['volume'], rtol=RTOL, atol=0)
    cases['number'], cases['volume'] = coagulated
    # Condensation still receives its captured hydrated diameter. This checks
    # two ported processes, not an implementation of intervening hydration.
    actual = run_jax(cases, True)
    for field in expected:
        np.testing.assert_allclose(getattr(actual, field), expected[field], rtol=RTOL, atol=0)

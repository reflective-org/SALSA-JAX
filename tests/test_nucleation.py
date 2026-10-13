# Copyright 2026 Reflective
# SPDX-License-Identifier: Apache-2.0
"""Independent source, growth and coupled insertion acceptance against Fortran."""
import jax
import numpy as np
import pytest

from salsa_jax.constants import AVOGADRO
from salsa_jax.processes.condensation import CondensationParameters, condensation_step
from salsa_jax.processes.condensation_nucleation import condensation_nucleation_step
from salsa_jax.processes.nucleation import nucleation_sources
from tools.nucleation_process_oracle import build_oracle, reference_cases, run_oracle

jax.config.update('jax_enable_x64', True)
SOURCE_INPUTS = ('number', 'gas', 'wet_diameter', 'temperature', 'relative_humidity', 'pressure', 'timestep')
COUPLED_INPUTS = ('number', 'volume', 'gas', 'wet_diameter', 'temperature', 'relative_humidity', 'pressure', 'timestep')


def subset(cases, index):
    return {key: value[index] for key, value in cases.items() if key != 'labels'}


def run_jax(cases, scheme, growth, compiled=True, coupled=False, **kwargs):
    function = condensation_nucleation_step if coupled else nucleation_sources
    kernel = jax.jit(function, static_argnames=('scheme', 'growth')) if compiled else function
    return kernel(*(cases[key] for key in (COUPLED_INPUTS if coupled else SOURCE_INPUTS)),
                  scheme=scheme, growth=growth, boundary_layer=cases['boundary_layer'],
                  organic_enabled=cases['organic_enabled'], parameters=CondensationParameters(
                      **{key: cases[key] for key in CondensationParameters._fields}), **kwargs)


@pytest.fixture(scope='module')
def reference(tmp_path_factory):
    executable, provenance = build_oracle(tmp_path_factory.mktemp('nucleation-jax'))
    cases = reference_cases()
    expected, _ = run_oracle(executable, cases)
    return executable, cases, expected, provenance


@pytest.mark.parametrize('scheme', [1, 2, 3])
@pytest.mark.parametrize('growth', [1, 2, 3])
@pytest.mark.parametrize('compiled', [False, True])
def test_coupled_state_matches_fortran(reference, scheme, growth, compiled):
    _, cases, expected, _ = reference
    selected = (cases['scheme'] == scheme) & (cases['growth'] == growth)
    inputs = subset(cases, selected)
    state = run_jax(inputs, scheme, growth, compiled, coupled=True)
    for field in ('number', 'volume', 'gas'):
        value = np.asarray(getattr(state, field))
        assert value.dtype == np.float64
        assert np.all(np.isfinite(value))
        assert np.all(value >= 0)
        np.testing.assert_allclose(value, expected[field][selected], rtol=3e-13, atol=0)
    np.testing.assert_array_equal(state.number[:, 1:], inputs['number'][:, 1:])
    np.testing.assert_array_equal(state.gas[:, 0], inputs['gas'][:, 0])
    np.testing.assert_array_equal(state.volume[..., 2:], inputs['volume'][..., 2:])
    for material, prefix in enumerate(('sulfate', 'organic')):
        molecular_volume = inputs[prefix + '_molar_mass'] / 1000 / AVOGADRO / inputs[prefix + '_density']
        before = inputs['gas'][:, material+1] + inputs['volume'][:, :, material].sum(axis=-1) / molecular_volume
        after = np.asarray(state.gas)[:, material+1] + np.asarray(state.volume)[:, :, material].sum(axis=-1) / molecular_volume
        np.testing.assert_allclose(after, before, rtol=3e-14, atol=0)
    assert np.any(expected['molecular_rate'][selected, 0] > 0), 'Must exercise active growth for each scheme/option'


def test_disabled_nucleation_and_disabled_process(reference):
    _, cases, _, _ = reference
    c = subset(cases, slice(0, 3))
    source = run_jax(c, 0, 1)
    assert not np.asarray(source.molecular_rate).any()
    assert not np.asarray(source.fractions).any()
    coupled = run_jax(c, 0, 1, coupled=True)
    plain = jax.jit(condensation_step)(*(c[k] for k in ('number', 'volume', 'gas', 'wet_diameter', 'temperature', 'pressure', 'timestep')),
                                      organic_enabled=c['organic_enabled'], parameters=CondensationParameters(
                                          **{k: c[k] for k in CondensationParameters._fields}))
    disabled = run_jax(c, 2, 3, coupled=True, enabled=False)
    for field in ('number', 'volume', 'gas'):
        np.testing.assert_array_equal(getattr(coupled, field), getattr(plain, field))
        np.testing.assert_array_equal(getattr(disabled, field), c[field])


def test_native_empty_background_gate_retained(reference):
    _, cases, expected, _ = reference
    index = cases['labels'].index('empty-scheme2-growth1')
    c = subset(cases, index)
    source = run_jax(c, 2, 1)
    assert source.molecular_rate[0] > 0
    state = run_jax(c, 2, 1, coupled=True)
    for field in ('number', 'volume', 'gas'):
        np.testing.assert_array_equal(getattr(state, field), c[field])
        np.testing.assert_array_equal(getattr(state, field), expected[field][index])


def test_multiaxis_and_scalar_states(reference):
    _, cases, _, _ = reference
    c = subset(cases, cases['labels'].index('scheme2-growth1-positive-bl'))
    scalar = run_jax(c, 2, 1, coupled=True)
    expanded = {k: np.broadcast_to(v, (2, 3) + np.shape(v)) for k, v in c.items()}
    many = run_jax(expanded, 2, 1, coupled=True)
    for field in ('number', 'volume', 'gas'):
        np.testing.assert_allclose(getattr(many, field), np.broadcast_to(getattr(scalar, field), getattr(many, field).shape), rtol=3e-14)


@pytest.mark.parametrize('growth', [1, 2, 3])
def test_repeated_coupled_calls_match_native_with_fixed_diameters(reference, growth):
    executable, cases, _, _ = reference
    indices = [cases['labels'].index(f'scheme{s}-growth{growth}-positive-bl') for s in (2, 3)]
    for index in indices:
        native = subset(cases, slice(index, index + 1))
        evolving = dict(native)
        scheme = int(native['scheme'][0])
        for _ in range(10):
            expected, _ = run_oracle(executable, native)
            result = run_jax(evolving, scheme, growth, coupled=True)
            for field in ('number', 'volume', 'gas'):
                actual = np.asarray(getattr(result, field))
                np.testing.assert_allclose(actual, expected[field], rtol=1e-12, atol=0)
                native[field] = expected[field]
                evolving[field] = actual


@pytest.mark.parametrize('scheme,growth', [(4, 1), (5, 2), (2, 0), (3, 4)])
def test_unsupported_options_rejected(scheme, growth):
    with pytest.raises(ValueError, match='Supported nucleation'):
        nucleation_sources(None, None, None, None, None, None, None, scheme=scheme, growth=growth)
    with pytest.raises(ValueError, match='Supported nucleation'):
        condensation_nucleation_step(None, None, None, None, None, None, None, None, scheme=scheme, growth=growth)


def test_tiny_source_uptake_does_not_insert_particles(reference):
    _, cases, expected, _ = reference
    index = cases['labels'].index('activation-cutoff-10000000000')
    c = subset(cases, index)
    assert 0 < expected['molecular_rate'][index, 0] <= 1
    result = run_jax(c, 2, 1, coupled=True)
    np.testing.assert_array_equal(result.number, c['number'])
    assert result.gas[1] < c['gas'][1]
    np.testing.assert_allclose(result.volume, expected['volume'][index], rtol=3e-13, atol=0)


def test_positive_scheme_with_zero_source_equals_plain_condensation(reference):
    _, cases, expected, _ = reference
    index = cases['labels'].index('activation-cutoff-100000000')
    c = subset(cases, index)
    assert not expected['molecular_rate'][index].any()
    result = run_jax(c, 2, 1, coupled=True)
    plain = run_jax(c, 0, 1, coupled=True)
    for field in ('number', 'volume', 'gas'):
        np.testing.assert_array_equal(getattr(result, field), getattr(plain, field))

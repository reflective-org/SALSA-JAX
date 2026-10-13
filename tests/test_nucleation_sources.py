# Copyright 2026 Reflective
# SPDX-License-Identifier: Apache-2.0
"""Isolated nucleation growth/source acceptance before condensation coupling."""
import jax
import numpy as np
import pytest

from salsa_jax.processes.condensation import CondensationParameters
from salsa_jax.processes.nucleation import nucleation_sources
from tools.nucleation_process_oracle import build_oracle, reference_cases, run_oracle

jax.config.update('jax_enable_x64', True)


@pytest.fixture(scope='module')
def reference(tmp_path_factory):
    executable, _ = build_oracle(tmp_path_factory.mktemp('nucleation-sources'))
    cases = reference_cases()
    expected, _ = run_oracle(executable, cases, operation='isolated')
    return cases, expected


@pytest.mark.parametrize('scheme', [1, 2, 3])
@pytest.mark.parametrize('growth', [1, 2, 3])
@pytest.mark.parametrize('compiled', [False, True])
def test_growth_sources_match_actual_fortran(reference, scheme, growth, compiled):
    cases, expected = reference
    index = (cases['scheme'] == scheme) & (cases['growth'] == growth)
    c = {key: value[index] for key, value in cases.items() if key != 'labels'}
    kernel = jax.jit(nucleation_sources, static_argnames=('scheme', 'growth')) if compiled else nucleation_sources
    result = kernel(*(c[k] for k in ('number', 'gas', 'wet_diameter', 'temperature', 'relative_humidity', 'pressure', 'timestep')),
                    scheme=scheme, growth=growth, boundary_layer=c['boundary_layer'],
                    organic_enabled=c['organic_enabled'], parameters=CondensationParameters(
                        **{k: c[k] for k in CondensationParameters._fields}))
    actual = np.asarray(result.molecular_rate)
    assert actual.dtype == np.float64
    assert np.all(np.isfinite(actual))
    assert np.all(actual >= 0)
    # Eager evaluation of the binary polynomial accumulates more roundoff.
    np.testing.assert_allclose(actual, expected['molecular_rate'][index], rtol=3e-13 if compiled else 5e-12, atol=0)
    np.testing.assert_array_equal(result.fractions, expected['fractions'][index])
    assert np.any(expected['molecular_rate'][index, 0] > 0)
    assert not actual[:, 1].any()


def test_zero_sulfate_and_unregistered_organic_produce_zero_source(reference):
    cases, _ = reference
    number = cases['number'][0]
    wet = cases['wet_diameter'][0]
    for scheme in (1, 2, 3):
        for growth in (1, 2, 3):
            result = nucleation_sources(number, np.array([1e13, 0., 1e16]), wet, 280., .5, 101325., 1.,
                                        scheme=scheme, growth=growth, organic_enabled=False)
            np.testing.assert_array_equal(result.molecular_rate, [0., 0.])

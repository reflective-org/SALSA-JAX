# Copyright 2026 Reflective
# SPDX-License-Identifier: Apache-2.0
"""Standalone five-material binary inverse molality leaf against native Fortran."""
import jax
import numpy as np
import pytest
from salsa_jax.processes.hydration import inverse_molality
from tools.hydration_oracle import build_oracle, run_molality

jax.config.update('jax_enable_x64', True)


@pytest.fixture(scope='module')
def oracle(tmp_path_factory):
    executable, _ = build_oracle(tmp_path_factory.mktemp('molality-leaf'))
    return executable


@pytest.mark.parametrize('compiled', [False, True])
def test_material_molalities_match_native(oracle, compiled):
    activity = np.array([1e-8, .001, .05, .3, .5, .8, .95, .99])[:, None]
    epsoc = np.array([0., .15, .8])[None, :]
    water_mass = np.array([18., 18.015, 20.])[None, :]
    expected = run_molality(oracle, activity, organic_soluble_fraction=epsoc, water_molar_mass=water_mass)
    kernel = jax.jit(inverse_molality) if compiled else inverse_molality
    actual = np.asarray(kernel(activity, organic_soluble_fraction=epsoc, water_molar_mass=water_mass))
    assert np.all(np.isfinite(actual))
    assert np.all(actual >= 0)
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=0)
    np.testing.assert_array_equal(actual[..., [2, 4]], 0)
    np.testing.assert_array_equal(actual[:, 0, 1], 0)


def test_scalar_and_species_axis_contract(oracle):
    actual = inverse_molality(.5)
    assert actual.shape == (5,)
    np.testing.assert_allclose(actual, run_molality(oracle, .5), rtol=2e-12, atol=0)

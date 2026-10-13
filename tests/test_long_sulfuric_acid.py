# Copyright 2026 Reflective
# SPDX-License-Identifier: Apache-2.0
"""Long-experiment initialization, reservoir forcing and observation contracts."""
from dataclasses import replace
from pathlib import Path

import jax
import numpy as np
import pytest

from salsa_jax.constants import PI
from salsa_jax.grid import size_grid
from salsa_jax.interface import mass_to_volume
from salsa_jax.model import STATIC_OPTIONS, salsa_step
from salsa_jax.sulfuric_acid import step_options, sulfur_inventory
from tools.long_sulfuric_acid import binned_lognormal, load_experiment, prepare, run_jax
from tools.salsa_oracle import OUTPUTS

jax.config.update('jax_enable_x64', True)
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def experiment():
    return load_experiment(ROOT/'configs/sulfuric-acid-six-hours.toml')


def test_requested_experiments_use_explicit_units_and_isolated_controls(experiment):
    assert experiment.duration_hours == 6
    assert experiment.h2so4_molecules_cm3 == 1e8
    assert experiment.constant_h2so4
    assert experiment.coagulation_enabled and experiment.condensation_enabled
    assert experiment.nucleation_scheme > 0
    coag = load_experiment(ROOT/'configs/coagulation-24-hours.toml')
    assert coag.duration_hours == 24
    assert coag.coagulation_enabled and not coag.condensation_enabled
    assert coag.nucleation_scheme == 0
    assert coag.h2so4_molecules_cm3 == 0
    assert not coag.constant_h2so4


@pytest.mark.parametrize('changes', [
    {'duration_hours': 0}, {'duration_hours': np.inf},
    {'timestep_s': 0}, {'timestep_s': np.nan},
    {'save_interval_s': 0}, {'save_interval_s': 1.5},
    {'duration_hours': 1/3600}, {'timestep_s': 7},
    {'initial_number_cm3': -1}, {'geometric_stddev': 1},
    {'geometric_mean_diameter_nm': 0}, {'relative_humidity': 1.1},
    {'h2so4_molecules_cm3': -1}, {'constant_h2so4': 1},
])
def test_invalid_experiments_rejected(experiment, changes):
    with pytest.raises(ValueError):
        replace(experiment, **changes)


def test_lognormal_integrates_native_bins_and_retains_small_tails(experiment):
    # Independent Gaussian-integral values, including positive remote tails.
    expected = np.array([
        32948.9572425296, 22971523.265223827, 476995526.7017758,
        419802312.3794894, 77701196.03475662, 2483792.339986913,
        12689.416076705513, 9.829658700373336,
        3.167660043265311e-5, 3.0883381268651605e-12,
    ])
    number = binned_lognormal(experiment, size_grid())
    np.testing.assert_allclose(number, expected, rtol=2e-13)
    assert np.all(number > 0)
    assert number.sum() < experiment.initial_number_cm3*1e6
    np.testing.assert_allclose(number.sum(), experiment.initial_number_cm3*1e6, rtol=2e-9)
    # An input centered at the lower edge keeps half its particles, rather
    # than silently renormalizing the represented domain to the stated total.
    edge = replace(experiment, geometric_mean_diameter_nm=3.)
    np.testing.assert_allclose(binned_lognormal(edge, size_grid()).sum(),
                               .5*edge.initial_number_cm3*1e6, rtol=2e-14)


def test_initial_mass_uses_native_midpoints_and_no_insoluble_material(experiment):
    _, cases, initial, mapping, grid = prepare(experiment)
    volume = np.asarray(mass_to_volume(initial['mass'], mapping))
    np.testing.assert_array_equal(initial['number'][0, :10],
                                  binned_lognormal(experiment, grid))
    np.testing.assert_array_equal(initial['number'][0, 10:], 0)
    np.testing.assert_array_equal(volume[..., 1:], 0)
    np.testing.assert_allclose(volume[..., 0], initial['number']*(PI/6)*grid.dpmid**3,
                               rtol=3e-15)
    np.testing.assert_array_equal(initial['gas'], [[0., 1e14, 0.]])
    for key in ('mass', 'number', 'gas'):
        np.testing.assert_array_equal(cases[key], initial[key])


def test_fixed_vapor_matches_manual_native_steps_and_accounts_for_reservoir(experiment):
    short = replace(experiment, duration_hours=2/3600, timestep_s=1., save_interval_s=1.)
    initial, actual, monitors, _, mapping, _ = run_jax(short)
    config, _, _, _, _ = prepare(short)
    advance = jax.jit(salsa_step, static_argnames=STATIC_OPTIONS)
    mass, number, gas = (initial[key] for key in ('mass', 'number', 'gas'))
    target = 1e14
    added = 0.
    for step in range(2):
        result = advance(mass, number, gas, short.temperature_k, short.relative_humidity,
                         short.pressure_pa, short.timestep_s, mapping, **step_options(config))
        result = jax.device_get(result)
        added += target-result.gas[0, 1]
        gas = result.gas.copy()
        gas[0, 1] = target
        for key in OUTPUTS:
            expected = gas if key == 'gas' else getattr(result, key)
            np.testing.assert_allclose(actual[key][step], expected, rtol=2e-13, atol=0)
        np.testing.assert_allclose(actual['h2so4_before_restore'][step], result.gas[:, 1], rtol=2e-13)
        np.testing.assert_allclose(actual['cumulative_reservoir_added'][step], added, rtol=2e-12)
        mass, number = result.mass, result.number
    np.testing.assert_array_equal(actual['gas'][..., 1], target)
    assert np.all(actual['h2so4_before_restore'] < target)
    assert np.all(np.diff(actual['cumulative_reservoir_added'][:, 0]) > 0)
    expected_sulfur = sulfur_inventory(initial['mass'], initial['gas'], mapping) + actual['cumulative_reservoir_added']
    np.testing.assert_allclose(sulfur_inventory(actual['mass'], actual['gas'], mapping),
                               expected_sulfur, rtol=3e-13)
    assert monitors['max_relative_sulfur_residual'][0] < 3e-13
    assert 0 < monitors['max_relative_h2so4_depletion'][0] < 1
    assert monitors['valid'] == [True]


def test_saving_stride_does_not_change_physics(experiment):
    short = replace(experiment, duration_hours=4/3600, timestep_s=1., save_interval_s=1.)
    _, frequent, _, _, _, _ = run_jax(short)
    _, sparse, _, _, _, _ = run_jax(replace(short, save_interval_s=2.))
    for key in frequent:
        np.testing.assert_allclose(sparse[key], frequent[key][1::2], rtol=2e-13, atol=0)


def test_closed_coagulation_reports_budget_without_reservoir():
    experiment = load_experiment(ROOT/'configs/coagulation-24-hours.toml')
    short = replace(experiment, duration_hours=2/3600, timestep_s=1., save_interval_s=1.)
    initial, actual, monitors, _, mapping, _ = run_jax(short)
    np.testing.assert_array_equal(actual['cumulative_reservoir_added'], 0.)
    np.testing.assert_array_equal(actual['gas'], 0.)
    expected = np.broadcast_to(sulfur_inventory(initial['mass'], initial['gas'], mapping), (2, 1))
    np.testing.assert_allclose(sulfur_inventory(actual['mass'], actual['gas'], mapping), expected, rtol=3e-13)
    assert np.all(np.diff(np.r_[initial['number'].sum(), actual['number'].sum(axis=(1, 2))]) < 0)
    assert monitors['max_relative_sulfur_residual'][0] < 3e-13
    assert monitors['max_relative_h2so4_depletion'] == [0.]
    assert all(monitors[key][0] >= 0 for key in ('minimum_number', 'minimum_mass', 'minimum_gas'))
    assert monitors['valid'] == [True]

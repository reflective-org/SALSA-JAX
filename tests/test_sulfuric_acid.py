# Copyright 2026 Reflective
# SPDX-License-Identifier: Apache-2.0
"""User-facing pure-sulfate experiment contracts and process isolation."""
from dataclasses import replace
from pathlib import Path

import jax
import numpy as np
import pytest

from salsa_jax.constants import AVOGADRO, PI, SULFATE_DENSITY, SULFATE_MOLAR_MASS
from salsa_jax.interface import mass_to_volume, validate_component_map
from salsa_jax.sulfuric_acid import initial_state, load_config, simulate, sulfur_inventory

jax.config.update('jax_enable_x64', True)
CONFIG = Path(__file__).resolve().parents[1] / 'configs/sulfuric-acid.toml'


@pytest.fixture
def config():
    return load_config(CONFIG)


@pytest.mark.parametrize('changes', [
    {'steps': 0}, {'steps': True}, {'timestep_s': 0.},
    {'temperature_k': -1.}, {'pressure_pa': np.inf},
    {'relative_humidity': 1.01}, {'relative_humidity': np.nan},
    {'initial_h2so4_molecules_m3': -1.}, {'h2so4_source_molecules_m3_s': -1.},
    {'initial_number_m3': (0.,) * 17},
    {'initial_number_m3': (-1.,) + (0.,) * 9},
    {'initial_number_m3': (np.nan,) + (0.,) * 9},
    {'nucleation_scheme': 4}, {'growth_option': 0},
    {'coagulation_enabled': 1}, {'condensation_enabled': False},
])
def test_invalid_experiment_settings_fail_before_simulation(config, changes):
    with pytest.raises(ValueError):
        replace(config, **changes)


@pytest.mark.parametrize('change', ['missing', 'unknown'])
def test_config_file_requires_explicit_known_settings(tmp_path, change):
    text = CONFIG.read_text()
    if change == 'missing':
        text = '\n'.join(line for line in text.splitlines() if not line.startswith('steps ='))
    else:
        text += '\nmisspelled_setting = 1\n'
    path = tmp_path / 'experiment.toml'
    path.write_text(text)
    with pytest.raises(ValueError, match='Configuration fields differ'):
        load_config(path)


def test_initialization_is_pure_sulfate_with_native_midpoint_sizes(config):
    config = replace(config, initial_number_m3=tuple(np.arange(1., 11.) * 1e7))
    mass, number, gas, mapping, grid = initial_state(config)
    validate_component_map(mapping)
    volume = np.asarray(mass_to_volume(mass, mapping))
    np.testing.assert_array_equal(number[0, :10], config.initial_number_m3)
    np.testing.assert_array_equal(number[0, 10:], 0.)
    np.testing.assert_array_equal(volume[..., 1:], 0.)
    np.testing.assert_array_equal(volume[:, 10:], 0.)
    np.testing.assert_allclose(volume[..., 0], number * (PI / 6) * grid.dpmid**3, rtol=3e-15)
    np.testing.assert_array_equal(gas, [[0., config.initial_h2so4_molecules_m3, 0.]])
    assert mass.dtype == number.dtype == gas.dtype == np.float64
    expected = (np.sum(np.asarray(number) * (PI / 6) * np.asarray(grid.dpmid)**3)
                * SULFATE_DENSITY / (SULFATE_MOLAR_MASS / 1000) * AVOGADRO)
    np.testing.assert_allclose(sulfur_inventory(mass, gas, mapping),
                               expected + config.initial_h2so4_molecules_m3, rtol=3e-15)


def test_float64_is_required_before_scientific_input_creation(config):
    with jax.enable_x64(False):
        with pytest.raises(ValueError, match='jax_enable_x64'):
            initial_state(config)


def test_inventory_uses_runtime_sulfate_map_and_cm3_conversion(config):
    _, _, _, mapping, _ = initial_state(config)
    # Permuting storage must not change which material counts as sulfur.
    mapping = np.where(mapping >= 0, 68 - mapping, -1)
    mass = np.full((2, 1, 69), 9e20)
    mass[..., mapping[:, 0]] = np.arange(1., 18.)
    gas = np.array([[[3e30, 7e6, 4e30]], [[3e30, 11e6, 4e30]]])
    expected = np.array([[160e6], [164e6]])
    np.testing.assert_array_equal(sulfur_inventory(mass, gas, mapping), expected)


@pytest.mark.parametrize('coagulation,condensation', [(False, False), (True, False), (False, True)])
def test_separate_process_controls_and_source_budget(config, coagulation, condensation):
    config = replace(config, steps=2, timestep_s=2., nucleation_scheme=0,
                     coagulation_enabled=coagulation, condensation_enabled=condensation)
    initial, trajectory, mapping, _ = simulate(config)
    injected = np.arange(1, config.steps + 1)[:, None] * config.timestep_s * config.h2so4_source_molecules_m3_s
    expected = sulfur_inventory(initial['mass'], initial['gas'], mapping) + injected
    np.testing.assert_allclose(sulfur_inventory(trajectory.mass, trajectory.gas, mapping), expected, rtol=3e-13)
    assert np.all(trajectory.valid)
    np.testing.assert_array_equal(trajectory.number[..., 10:], 0.)
    np.testing.assert_array_equal(trajectory.gas[..., (0, 2)], 0.)
    if not condensation:
        np.testing.assert_array_equal(trajectory.gas[..., 1], initial['gas'][..., 1] + injected)
    else:
        assert np.all(trajectory.gas[..., 1] < initial['gas'][..., 1] + injected)
    if coagulation:
        assert trajectory.number[-1].sum() < initial['number'].sum()
    else:
        # Condensation remaps particles between bins without creating particles.
        np.testing.assert_allclose(trajectory.number.sum(axis=-1), initial['number'].sum(), rtol=3e-13)


def test_closed_coupled_run_forms_particles_and_conserves_sulfur(config):
    config = replace(config, steps=3, h2so4_source_molecules_m3_s=0.)
    initial, trajectory, mapping, _ = simulate(config)
    inventory = sulfur_inventory(trajectory.mass, trajectory.gas, mapping)
    np.testing.assert_allclose(inventory,
        np.broadcast_to(sulfur_inventory(initial['mass'], initial['gas'], mapping), inventory.shape), rtol=3e-13)
    assert trajectory.number[-1].sum() > initial['number'].sum()
    assert trajectory.gas[-1, 0, 1] < initial['gas'][0, 1]
    non_sulfate = np.ones(69, dtype=bool)
    non_sulfate[mapping[:, 0]] = False
    np.testing.assert_array_equal(trajectory.mass[..., non_sulfate], 0.)

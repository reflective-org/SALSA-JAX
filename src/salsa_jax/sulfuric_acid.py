# Copyright 2026 Reflective
# SPDX-License-Identifier: Apache-2.0
"""A fixed-environment, pure-sulfate experiment using the accepted SALSA core.

This experiment has an additive H2SO4 source, not the legacy driver's prescribed
concentration overwrite. Initial aerosol occupies only soluble bins 1a/2a.
"""
from dataclasses import dataclass, fields
from pathlib import Path
import tomllib

import jax
import jax.numpy as jnp
import numpy as np

from salsa_jax.constants import PI
from salsa_jax.grid import size_grid
from salsa_jax.interface import validate_component_map, volume_to_mass
from salsa_jax.model import STATIC_OPTIONS, salsa_step


@dataclass(frozen=True)
class SulfuricAcidConfig:
    """Explicit experiment settings; scientific constants stay in constants.py."""
    steps: int
    timestep_s: float
    temperature_k: float
    pressure_pa: float
    relative_humidity: float
    initial_h2so4_molecules_m3: float
    h2so4_source_molecules_m3_s: float
    initial_number_m3: tuple
    nucleation_scheme: int
    growth_option: int
    coagulation_enabled: bool
    condensation_enabled: bool
    recalculate: bool

    def __post_init__(self):
        for name, allowed in (('steps', None), ('nucleation_scheme', (0, 1, 2, 3)),
                              ('growth_option', (1, 2, 3))):
            value = getattr(self, name)
            if type(value) is not int or (value < 1 if allowed is None else value not in allowed):
                raise ValueError(f'Invalid {name}')
        for name in ('coagulation_enabled', 'condensation_enabled', 'recalculate'):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f'{name} must be boolean')
        for name in ('timestep_s', 'temperature_k', 'pressure_pa', 'relative_humidity',
                     'initial_h2so4_molecules_m3', 'h2so4_source_molecules_m3_s'):
            value = getattr(self, name)
            if isinstance(value, bool) or not np.isscalar(value) or not np.isfinite(value):
                raise ValueError(f'{name} must be finite')
            if value < 0 or (name in ('timestep_s', 'temperature_k', 'pressure_pa') and value == 0):
                raise ValueError(f'Invalid {name}')
        if self.relative_humidity > 1:
            raise ValueError('relative_humidity must be in [0,1]')
        number = np.asarray(self.initial_number_m3, dtype=float)
        if number.shape != (10,) or not np.all(np.isfinite(number)) or np.any(number < 0):
            raise ValueError('initial_number_m3 must contain ten finite nonnegative soluble-bin counts')
        object.__setattr__(self, 'initial_number_m3', tuple(number.tolist()))
        if self.nucleation_scheme and not self.condensation_enabled:
            raise ValueError('Nucleation requires condensation_enabled=true; use scheme 0 to disable it')


def load_config(path):
    data = tomllib.loads(Path(path).read_text())
    expected = {field.name for field in fields(SulfuricAcidConfig)}
    if set(data) != expected:
        raise ValueError(f'Configuration fields differ: missing={sorted(expected-set(data))}, extra={sorted(set(data)-expected)}')
    return SulfuricAcidConfig(**data)


def initial_state(config):
    """Return mass, number, gas, map and grid; no native mixed-material recipe.

    Arrays have one case axis. Ten user counts populate soluble 1a/2a bins at
    native dry midpoint volumes; seven 2b bins remain empty. Sulfate uses native
    molecules/cm3, gas molecules/m3, and number particles/m3. Map order is an
    explicit bin-major permutation of the supported 69-component interface.
    """
    if not jax.config.x64_enabled:
        raise ValueError('Enable jax_enable_x64 before constructing scientific inputs')
    grid = size_grid()
    mapping = np.full((17, 5), -1, dtype=int)
    component = 0
    for bin_index in range(17):
        for species in range(5):
            if (bin_index >= 3 or species < 2) and (bin_index < 10 or species != 3):
                mapping[bin_index, species] = component
                component += 1
    validate_component_map(mapping)
    number = jnp.asarray([(*config.initial_number_m3, *([0.] * 7))], dtype=jnp.float64)
    volume = jnp.zeros((1, 17, 5), dtype=jnp.float64)
    volume = volume.at[..., 0].set(number * (PI / 6) * grid.dpmid**3)
    mass = volume_to_mass(volume, mapping)
    gas = jnp.asarray([[0., config.initial_h2so4_molecules_m3, 0.]], dtype=jnp.float64)
    return mass, number, gas, mapping, grid


def step_options(config):
    return dict(scheme=config.nucleation_scheme, growth=config.growth_option,
                coagulation_enabled=config.coagulation_enabled,
                condensation_enabled=config.condensation_enabled,
                recalculate=config.recalculate, organic_enabled=False,
                transfer_soluble=False, hydrate_insoluble=True, boundary_layer=True)


def simulate(config):
    """Return every post-step SalsaState; inject source*dt before each core call.

    A Python time loop around one compiled step keeps compilation independent
    of trajectory length and retains all diagnostics for validation. Returned
    arrays have (steps, 1, field width); initial inputs are returned separately.
    """
    mass, number, gas, mapping, grid = initial_state(config)
    initial = {'mass': np.asarray(mass), 'number': np.asarray(number), 'gas': np.asarray(gas)}
    advance = jax.jit(salsa_step, static_argnames=STATIC_OPTIONS)
    options = step_options(config)
    history = []
    for step in range(config.steps):
        gas = gas.at[..., 1].add(config.h2so4_source_molecules_m3_s * config.timestep_s)
        state = advance(mass, number, gas, config.temperature_k, config.relative_humidity,
                        config.pressure_pa, config.timestep_s, mapping, **options)
        state = jax.device_get(state)
        if not np.all(state.valid):
            raise ValueError(f'Invalid SALSA state at step {step+1}; no clipping or replacement applied')
        for name in ('mass', 'number', 'gas'):
            if np.any(getattr(state, name) < 0):
                raise ValueError(f'Negative {name} at step {step+1}')
        history.append(state)
        mass, number, gas = (jnp.asarray(getattr(state, key)) for key in ('mass', 'number', 'gas'))
    trajectory = type(state)(*(np.stack([getattr(row, field) for row in history]) for field in state._fields))
    return initial, trajectory, mapping, grid


def sulfur_inventory(mass, gas, component_map):
    """Sulfur atoms/m3: one per represented sulfate molecule and H2SO4 molecule.

    This pure-sulfate experiment has no SO2 chemistry or other sulfur species.
    Use molecule counts, not a 98 g/mol acid mass that would contradict the
    native aerosol SO4 species convention (96.0631 g/mol).
    """
    return np.asarray(gas)[..., 1] + 1e6 * np.asarray(mass)[..., np.asarray(component_map)[:, 0]].sum(-1)

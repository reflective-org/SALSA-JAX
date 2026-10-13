# Copyright 2026 Reflective
# SPDX-License-Identifier: Apache-2.0
"""Complete standalone native SALSA call on its mixed-unit mass interface."""
from typing import NamedTuple

import jax
import jax.numpy as jnp

from salsa_jax.constants import REMAPPING_ION_FACTOR, REMAPPING_SATURATION_LIMIT
from salsa_jax.grid import DEFAULT_REGIME_LIMITS, size_grid
from salsa_jax.interface import mass_to_volume, volume_to_mass, particle_density
from salsa_jax.processes.hydration import HydrationParameters, equilibrate, solubility_mask
from salsa_jax.processes.coagulation_step import CoagulationState, coagulation_step
from salsa_jax.processes.condensation import CondensationParameters, CondensationState
from salsa_jax.processes.condensation_nucleation import condensation_nucleation_step
from salsa_jax.processes.remapping import RemappingParameters, remap
from salsa_jax.processes.remapping_correction import cleanup, correction

STATIC_OPTIONS = ('scheme', 'growth', 'coagulation_enabled', 'condensation_enabled',
                  'recalculate', 'transfer_soluble', 'hydrate_insoluble', 'organic_enabled', 'return_trace')


class SalsaState(NamedTuple):
    """Native state/diagnostics; radii cm, density g/cm3, water kg/m3 air."""
    mass: jax.Array
    number: jax.Array
    gas: jax.Array
    volume: jax.Array
    core: jax.Array
    wet_diameter: jax.Array
    dry_radius_cm: jax.Array
    wet_radius_cm: jax.Array
    density: jax.Array
    water_mass: jax.Array
    valid: jax.Array


def salsa_step(mass, number, gas, temperature, relative_humidity, pressure, timestep,
               component_map, *, regime_limits=DEFAULT_REGIME_LIMITS,
               parameters=HydrationParameters(), mass_accommodation=1.0,
               ion_factor=REMAPPING_ION_FACTOR, saturation_limit=REMAPPING_SATURATION_LIMIT,
               boundary_layer=True, scheme=0, growth=1, coagulation_enabled=True,
               condensation_enabled=True, recalculate=False, transfer_soluble=False,
               hydrate_insoluble=True, organic_enabled=False, return_trace=False):
    """Advance one native SALSA call, including both remap/correction passes.

    Inputs have matching leading axes: mass (...,69) uses native component units
    (sulfate molecules/cm3, other materials ug/m3), number (...,17) particles/m3,
    gas (...,3) SO2/H2SO4/OC molecules/m3. component_map (...,17,5) is zero-based,
    with -1 for the native unrepresented bin/material slots; validate it on the
    host with interface.validate_component_map. Environment is K, RH fraction,
    Pa and seconds. For active nucleation timestep must be positive and the first
    regime diameter must be the supported native 3 nm. Enable float64.

    Parameters/geometry broadcast across leading axes. Scheme/growth and all
    switches in STATIC_OPTIONS must be static when JIT compiling; group batches
    by configuration. boundary_layer is the native jj >= INT(ppbl) condition.
    The default absent organic gas uses a padded third slot and retains D007
    failure signaling. No HAM wrapper, initialization, forcing or radiation is
    included. To repeat a call, feed back mass, number and gas, not internal
    volume: the native mixed-unit interface re-encodes represented materials.

    valid reports cleanup/nonfinite failures, not scientific budget validity.
    Known native volume inconsistencies are preserved. return_trace=True adds
    process-boundary pytrees for reference comparisons; default returns only
    SalsaState, allowing unused intermediate diagnostics to be eliminated.
    """
    if scheme not in (0, 1, 2, 3) or growth not in (1, 2, 3):
        raise ValueError('Supported nucleation schemes are 0/1/2/3 and growth options 1/2/3')
    mass, number, gas = (jnp.asarray(value, dtype=jnp.float64) for value in (mass, number, gas))
    if (number.shape[-1:] != (17,) or mass.shape != number.shape[:-1]+(69,) or
            gas.shape != number.shape[:-1]+(3,)):
        raise ValueError('Expected mass (...,69), number (...,17), gas (...,3)')
    grid = size_grid(regime_limits)
    mask = solubility_mask(hydrate_insoluble)
    volume = mass_to_volume(mass, component_map, parameters=parameters)
    initial = cleanup(number, volume, gas, parameters=parameters, organic_enabled=organic_enabled)
    valid = initial.valid
    number = jnp.where(valid[..., None], initial.number, 0.0)
    volume = jnp.where(valid[..., None, None], initial.volume, 0.0)
    gas = jnp.where(valid[..., None], initial.gas, 0.0)
    hydrated = equilibrate(number, volume, relative_humidity, temperature, grid.dpmid,
                           soluble_mask=mask, parameters=parameters)
    if coagulation_enabled:
        coagulated = coagulation_step(number, volume, hydrated.wet_diameter, grid.dpmid,
                                      temperature, pressure, timestep)
    else:
        coagulated = CoagulationState(number, volume)
    number, volume = coagulated
    sink_geometry = (equilibrate(number, volume, relative_humidity, temperature, grid.dpmid,
                                 soluble_mask=mask, parameters=parameters) if recalculate else hydrated)
    species_mass = jnp.asarray(parameters.species_molar_mass, dtype=jnp.float64)
    species_density = jnp.asarray(parameters.species_density, dtype=jnp.float64)
    cond_parameters = CondensationParameters(
        sulfate_molar_mass=species_mass[..., 0], organic_molar_mass=species_mass[..., 1],
        sulfate_density=species_density[..., 0], organic_density=species_density[..., 1],
        mass_accommodation=mass_accommodation)
    if condensation_enabled:
        condensed = condensation_nucleation_step(number, volume, gas, sink_geometry.wet_diameter,
            temperature, relative_humidity, pressure, timestep, scheme=scheme, growth=growth,
            boundary_layer=boundary_layer, organic_enabled=organic_enabled, parameters=cond_parameters)
    else:
        condensed = CondensationState(number, volume, gas)
    number, volume, gas = condensed
    remap_parameters = RemappingParameters(
        sulfate_molar_mass=species_mass[..., 0], sulfate_density=species_density[..., 0],
        water_molar_mass=parameters.water_molar_mass, water_density=parameters.water_density,
        surface_tension=parameters.water_surface_tension, organic_soluble_fraction=parameters.organic_soluble_fraction,
        saturation_limit=saturation_limit, ion_factor=ion_factor, number_threshold=parameters.number_threshold)
    remapped_passes, corrected_passes = [], []
    for _ in range(2):
        remapped = remap(number, volume, temperature, grid, transfer_soluble=transfer_soluble, parameters=remap_parameters)
        corrected = correction(remapped.number, remapped.volume, gas, relative_humidity, temperature, grid.dpmid,
                               soluble_mask=mask, parameters=parameters, organic_enabled=organic_enabled)
        number, volume, gas = corrected.number, corrected.volume, corrected.gas
        valid = valid & corrected.valid
        remapped_passes.append(remapped)
        corrected_passes.append(corrected)
    mass = volume_to_mass(volume, component_map, parameters=parameters)
    density = particle_density(volume, corrected.water_mass, parameters=parameters)
    arrays = (mass, number, gas, volume, corrected.core, corrected.wet_diameter,
              corrected.dry_radius_cm, corrected.wet_radius_cm, density, corrected.water_mass)
    for value in arrays:
        valid = valid & jnp.all(jnp.isfinite(value), axis=tuple(range(number.ndim-1, value.ndim)))
    arrays = tuple(jnp.where(valid.reshape(valid.shape+(1,)*(value.ndim-valid.ndim)), value, jnp.nan) for value in arrays)
    result = SalsaState(*arrays, valid)
    if return_trace:
        return result, {'grid': grid, 'initial_cleanup': initial, 'initial_hydration': hydrated,
                        'coagulation': coagulated, 'condensation_hydration': sink_geometry,
                        'condensation': condensed, 'remapping': tuple(remapped_passes), 'correction': tuple(corrected_passes)}
    return result

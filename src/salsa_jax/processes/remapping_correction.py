"""One native post-remapping cleanup/hydration/geometry correction pass.

This preserves the existing tiny-particle removal policy, including its loss of
nonvolatile volume and D007's invalid absent-organic-gas branch. It does not
perform redistribution, density calculation, or an entire SALSA timestep.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp

from salsa_jax.constants import AVOGADRO, PI, TINY_PARTICLE_DIAMETER
from salsa_jax.processes.hydration import HydrationParameters, equilibrate


class CorrectionState(NamedTuple):
    """Corrected state and native geometry; ``valid`` is false for D007 cells.

    Number is particles/m3, volume m3 material/m3 air, canonical gases
    SO2/H2SO4/OC molecules/m3, core m3/particle, wet diameter m, radii cm,
    water mass kg/m3 air. Iteration counts are per-bin, validity per-cell.
    """
    number: jax.Array
    volume: jax.Array
    gas: jax.Array
    core: jax.Array
    wet_diameter: jax.Array
    dry_radius_cm: jax.Array
    wet_radius_cm: jax.Array
    water_mass: jax.Array
    iterations: jax.Array
    valid: jax.Array


class CleanupState(NamedTuple):
    """Native tiny-particle cleanup; validity flags absent organic-gas access."""
    number: jax.Array
    volume: jax.Array
    gas: jax.Array
    valid: jax.Array


def cleanup(number, volume, gas, *, parameters=HydrationParameters(), organic_enabled=False):
    """Initial or post-remap native tiny cleanup in canonical SI particle/gas units.

    Includes number == threshold, uses only sulfate/OC size in 1a, restores only
    those two vapors, then clears every material. D007 cells return NaN physical
    arrays and valid=False. The native threshold must be positive.
    """
    number, volume, gas = (jnp.asarray(value, dtype=jnp.float64) for value in (number, volume, gas))
    if (number.shape[-1:] != (17,) or volume.shape != number.shape+(5,) or
            gas.shape != number.shape[:-1]+(3,)):
        raise ValueError('Expected number (...,17), volume (...,17,5), gas (...,3)')
    leading = number.shape[:-1]
    threshold = jnp.broadcast_to(jnp.asarray(parameters.number_threshold, dtype=jnp.float64), leading)
    organic = jnp.broadcast_to(jnp.asarray(organic_enabled, dtype=bool), leading)
    density = jnp.broadcast_to(jnp.asarray(parameters.species_density, dtype=jnp.float64), leading+(5,))
    mass = jnp.broadcast_to(jnp.asarray(parameters.species_molar_mass, dtype=jnp.float64), leading+(5,))
    total = volume[...,0]+volume[...,1]
    for species in range(2,5):
        total = total+volume[...,species]
    diameter_volume = jnp.where(jnp.arange(17)<3, volume[...,0]+volume[...,1], total)
    candidate = number >= threshold[...,None]
    diameter = (diameter_volume/jnp.where(candidate, number, 1.0)/(PI/6.0))**(1.0/3.0)
    tiny = candidate & (diameter < TINY_PARTICLE_DIAMETER)
    valid = ~jnp.any(tiny & ~organic[...,None], axis=-1)

    def restore_vapor(bin_index, restored):
        remove = tiny[...,bin_index] & valid
        for material, gas_slot in ((0,1),(1,2)):
            returned = (volume[...,bin_index,material]*density[...,material]/
                        (mass[...,material]/1000.0)*AVOGADRO)
            restored = restored.at[...,gas_slot].add(jnp.where(remove,returned,0.0))
        return restored

    restored_gas = jax.lax.fori_loop(0,17,restore_vapor,gas)
    new_number = jnp.where(tiny,0.0,number)
    new_volume = jnp.where(tiny[...,None],0.0,volume)
    return CleanupState(jnp.where(valid[...,None], new_number, jnp.nan),
                        jnp.where(valid[...,None,None], new_volume, jnp.nan),
                        jnp.where(valid[...,None], restored_gas, jnp.nan), valid)


def correction(number, volume, gas, relative_humidity, temperature, dry_diameter, *,
               soluble_mask=True, parameters=HydrationParameters(), organic_enabled=False):
    """Run one cleanup → hydration → radii/water pass, with native policy.

    State shapes are number (...,17), volume (...,17,5), gas (...,3), with
    matching leading axes; RH/T/grid/parameters follow ``equilibrate``. The
    number threshold must be positive. Cleanup includes N==threshold, while
    hydration only processes N>threshold. For 1a, cleanup diameter counts only
    sulfate+OC; for 2a/2b it counts every material. Removed material returns only
    SO4/OC to their gas slots, even though all five particle volumes are cleared.

    If cleanup executes without an organic gas mapping, native checked Fortran
    indexes gas zero (D007). Such cells return valid=False, NaN floating outputs
    and zero iteration counts. No repair or invented organic sink is applied.
    Valid cells retain native unclipped water. Near-zero water is a subtraction
    of nearly equal volumes, so roundoff signs can differ between backends.
    Enable JAX float64 before use. Inputs remain immutable.
    """
    cleaned = cleanup(number, volume, gas, parameters=parameters, organic_enabled=organic_enabled)
    new_number, new_volume, restored_gas, valid = cleaned
    leading = new_number.shape[:-1]
    # Invalid cells must not enter the native uncapped hydration loop with
    # partially failed cleanup state. Their public outputs are invalid below.
    safe_number = jnp.where(valid[...,None],new_number,0.0)
    safe_volume = jnp.where(valid[...,None,None],new_volume,0.0)
    geometry = equilibrate(safe_number,safe_volume,relative_humidity,temperature,dry_diameter,
                           soluble_mask=soluble_mask,parameters=parameters)
    dry_radius = (geometry.core/(PI/6.0))**(1.0/3.0)/2.0*100.0
    wet_radius = geometry.wet_diameter/2.0*100.0
    water_density = jnp.broadcast_to(jnp.asarray(parameters.water_density,dtype=jnp.float64),leading)
    water = ((PI/6.0)*geometry.wet_diameter**3-geometry.core)*water_density[...,None]*new_number

    def physical(value, dimensions):
        mask = valid.reshape(leading+(1,)*dimensions)
        return jnp.where(mask,value,jnp.nan)
    return CorrectionState(physical(new_number,1),physical(new_volume,2),physical(restored_gas,1),
                           physical(geometry.core,1),physical(geometry.wet_diameter,1),
                           physical(dry_radius,1),physical(wet_radius,1),physical(water,1),
                           geometry.iterations,valid)

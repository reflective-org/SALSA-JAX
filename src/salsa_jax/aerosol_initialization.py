"""Native SALSA material recipe and initial HAM tracer conversion."""
from typing import NamedTuple

import jax
import jax.numpy as jnp

from salsa_jax.constants import (AVOGADRO, AEROSOL_MOLECULAR_CONVERSION_FACTOR,
                                 INITIAL_MATERIAL_FRACTIONS, SULFATE_MOLAR_MASS)
from salsa_jax.processes.hydration import HydrationParameters


class AerosolInitialization(NamedTuple):
    """Number [particles/cm3], mixed native mass69, and updated HAM tracers."""
    number: jax.Array
    mass: jax.Array
    tracers: jax.Array


def initialize_aerosol(number, core, tracers, air_density, component_map,
                       component_tracer_indices, number_tracer_indices, component_unit_codes,
                       *, parameters=HydrationParameters(), mw_so4=SULFATE_MOLAR_MASS):
    """Port standalone SALSA conc2mmr, including the native number-unit mutation.

    Input number (...,17) is particles/m3; core is m3/particle; air_density
    kg/m3. Tracers (...,ntrac) share number's leading axes. Geometry, metadata
    and species properties broadcast over cells. component_map (...,17,5) maps
    SO4/OC/BC/SS/DU into69 zero-based native mass components, with -1 absent;
    validate it outside JIT with interface.validate_component_map. Component
    tracer indices/codes end in69, number tracer indices in17. Tracer indices
    are zero-based and must be in bounds; duplicate writes follow native order.

    Every material component is initialized from fixed native volume fractions.
    Returned mass uses sulfate molecules/cm3 and other materials ug/m3. Code2
    uses 6.022e20/mw_so4 for tracer conversion; ALL other codes use1e9, as native.
    mw_so4 [g/mol] is independent of the registry sulfate molar mass. Number
    tracers are written after mass tracers and therefore win mapping collisions.
    Returned number is particles/cm3; input arrays remain unchanged. Do not feed
    it back as particles/m3 or simplify its rounded intermediate conversion.

    Caller preconditions: finite inputs, nonnegative number/core, positive air
    density and species properties/mw_so4, valid integer maps. Enable float64.
    This is initialization, not a general conversion of existing material mass.
    """
    number, core, tracers = (jnp.asarray(value, dtype=jnp.float64) for value in (number, core, tracers))
    if (number.shape[-1:] != (17,) or tracers.ndim < 1 or not tracers.shape[-1] or
            tracers.shape[:-1] != number.shape[:-1] or core.shape[-1:] != (17,)):
        raise ValueError('Expected number/core (...,17) and tracers (...,ntrac) with matching case axes')
    leading = number.shape[:-1]
    core = jnp.broadcast_to(core, number.shape)
    def metadata(value, tail, name):
        value = jnp.asarray(value)
        if value.shape[-len(tail):] != tail or not jnp.issubdtype(value.dtype, jnp.integer):
            raise ValueError(f'{name} must have integer trailing shape {tail}')
        return jnp.broadcast_to(value, leading+tail)
    mapping = metadata(component_map, (17, 5), 'component_map')
    mass_indices = metadata(component_tracer_indices, (69,), 'component_tracer_indices')
    number_indices = metadata(number_tracer_indices, (17,), 'number_tracer_indices')
    codes = metadata(component_unit_codes, (69,), 'component_unit_codes')
    density = jnp.broadcast_to(jnp.asarray(parameters.species_density, dtype=jnp.float64), leading+(5,))
    molar_mass = jnp.broadcast_to(jnp.asarray(parameters.species_molar_mass, dtype=jnp.float64), leading+(5,))
    rho = jnp.broadcast_to(jnp.asarray(air_density, dtype=jnp.float64), leading)
    mw_so4 = jnp.broadcast_to(jnp.asarray(mw_so4, dtype=jnp.float64), leading)
    fractions = jnp.asarray(INITIAL_MATERIAL_FRACTIONS, dtype=jnp.float64)[jnp.where(jnp.arange(17)<3,0,jnp.where(jnp.arange(17)<10,1,2))]
    volume = fractions*number[..., None]*core[..., None]
    molecular_volume = (molar_mass[..., 0]/1000.0)/AVOGADRO/density[..., 0]
    canonical_mass = volume*density[..., None, :]/1e-9
    canonical_mass = canonical_mass.at[..., 0].set(volume[..., 0]/(1e6*molecular_volume[..., None]))
    # Validated maps contain sixteen absent slots followed by all69 component IDs.
    order = jnp.argsort(mapping.reshape(leading+(85,)), axis=-1)[..., -69:]
    mass = jnp.take_along_axis(canonical_mass.reshape(leading+(85,)), order, axis=-1)
    factor = jnp.where(codes == 2, (AEROSOL_MOLECULAR_CONVERSION_FACTOR/mw_so4)[..., None], 1e9)
    mass_tracers = mass/(factor*rho[..., None])
    number_cm3 = number*1e-6
    number_tracers = number_cm3/(1e-6*rho[..., None])
    values = jnp.concatenate((mass_tracers, number_tracers), axis=-1)
    indices = jnp.concatenate((mass_indices, number_indices), axis=-1)
    positions = jnp.arange(tracers.shape[-1])
    def write(index, state):
        return jnp.where(positions == indices[..., index, None], values[..., index, None], state)
    updated = jax.lax.fori_loop(0, 86, write, tracers)
    return AerosolInitialization(number_cm3, mass, updated)

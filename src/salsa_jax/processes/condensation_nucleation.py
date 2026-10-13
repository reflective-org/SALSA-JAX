"""Native condensation with supported sulfate nucleation and particle insertion."""
import jax
import jax.numpy as jnp

from salsa_jax.constants import NUCLEATION_ACTIVATION_COEFFICIENT
from salsa_jax.processes.condensation import (
    CondensationParameters, CondensationState, condensation_step, _condense,
)
from salsa_jax.processes.nucleation import nucleation_sources


def condensation_nucleation_step(number, volume, gas, wet_diameter, temperature,
                                 relative_humidity, pressure, timestep, *, scheme=2, growth=1,
                                 boundary_layer=True, organic_enabled=False, enabled=True,
                                 activation_coefficient=NUCLEATION_ACTIVATION_COEFFICIENT,
                                 parameters=CondensationParameters()):
    """Advance native uptake/insertion with fixed supplied hydrated diameters.

    State units/shapes follow condensation_step; RH is a fraction, timestep
    positive for active nucleation. scheme=0/1/2/3 and growth=1/2/3 are static
    JIT options. boundary_layer marks jj >= ppbl. Species are SO4/OC/BC/SS/DU;
    canonical gas is SO2/H2SO4/OC. Enable JAX float64. No VBS or NH3 chemistry.

    Nucleation sources are evaluated before condensation, as in Fortran. The
    native sink gate prevents insertion into a completely empty background.
    This is one condensation process call, not a complete SALSA timestep.
    """
    if scheme not in (0, 1, 2, 3) or growth not in (1, 2, 3):
        raise ValueError("Supported nucleation schemes are 0/1/2/3 and growth options 1/2/3")
    if scheme == 0:
        return condensation_step(number, volume, gas, wet_diameter, temperature, pressure, timestep,
                                 organic_enabled=organic_enabled, enabled=enabled, parameters=parameters)
    number, volume, gas = (jnp.asarray(v, dtype=jnp.float64) for v in (number, volume, gas))
    if (number.shape[-1:] != (17,) or volume.shape != number.shape + (5,) or
            gas.shape != number.shape[:-1] + (3,)):
        raise ValueError("Expected matching number (...,17), volume (...,17,5), gas (...,3)")
    timestep = jnp.asarray(timestep, dtype=jnp.float64)

    def advance():
        sources = nucleation_sources(number, gas, wet_diameter, temperature, relative_humidity,
                                     pressure, timestep, scheme=scheme, growth=growth,
                                     boundary_layer=boundary_layer, organic_enabled=organic_enabled,
                                     activation_coefficient=activation_coefficient, parameters=parameters)
        return _condense(number, volume, gas, wet_diameter, temperature, pressure, timestep,
                         organic_enabled, parameters, sources)

    return jax.lax.cond(enabled, advance, lambda: CondensationState(number, volume, gas))

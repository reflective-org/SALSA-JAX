"""Native HAM gas concentration-to-tracer conversion, independent of SALSA."""
import jax
import jax.numpy as jnp

from salsa_jax.constants import AVOGADRO, GAS_CONSTANT, GAS_MOLECULAR_CONVERSION_FACTOR


def gas_to_tracers(tracers, gas, air_density, pressure, temperature,
                   tracer_indices, molar_mass, unit_codes):
    """Return native gas conversions while preserving all unselected tracers.

    tracers (...,ntrac) and gas (...,ngas) have identical leading axes.
    Density [kg/m3], pressure [Pa] and temperature [K] broadcast over cases.
    Zero-based tracer indices, g/mol masses and integer conversion codes have
    final ngas axes and broadcast over cases. Code1 gas is ug/m3 -> kg/kg;
    code2 molecules/cm3 -> kg/kg; code3 molecules/cm3 -> volume mixing ratio.

    Native gas order is retained: duplicate tracer indices use the last known
    code's write. Unknown codes do nothing. Inputs remain unchanged. Indices
    must be in bounds; density, temperature and molecular masses must be
    positive and all numerical inputs finite. Validate data before JIT.
    Enable float64. This leaf does not apply driver molecules/m3 scaling,
    tendencies, negative-tracer borrowing or the complete HAM wrapper.
    """
    tracers, gas = (jnp.asarray(value, dtype=jnp.float64) for value in (tracers, gas))
    if tracers.ndim < 1 or gas.ndim < 1 or tracers.shape[:-1] != gas.shape[:-1] or tracers.shape[-1] == 0:
        raise ValueError("Expected tracers (...,ntrac) and gas (...,ngas) with matching case axes")
    leading = tracers.shape[:-1]
    def environment(value):
        return jnp.broadcast_to(jnp.asarray(value, dtype=jnp.float64), leading)
    density, pressure, temperature = map(environment, (air_density, pressure, temperature))
    indices, codes = (jnp.asarray(value) for value in (tracer_indices, unit_codes))
    if not jnp.issubdtype(indices.dtype, jnp.integer) or not jnp.issubdtype(codes.dtype, jnp.integer):
        raise ValueError("tracer_indices and unit_codes must be integer arrays")
    indices, codes = (jnp.broadcast_to(value, gas.shape) for value in (indices, codes))
    masses = jnp.broadcast_to(jnp.asarray(molar_mass, dtype=jnp.float64), gas.shape)
    positions = jnp.arange(tracers.shape[-1])
    def convert(index, state):
        code = codes[..., index]
        molecular_factor = GAS_MOLECULAR_CONVERSION_FACTOR * density * AVOGADRO / masses[..., index]
        volume_factor = (1e-6 * AVOGADRO / GAS_CONSTANT) * pressure / temperature
        factor = jnp.where(code == 1, 1e9 * density, jnp.where(code == 2, molecular_factor, volume_factor))
        converted = gas[..., index] / factor
        selected = (positions == indices[..., index, None]) & ((code >= 1) & (code <= 3))[..., None]
        return jnp.where(selected, converted[..., None], state)
    if gas.shape[-1] == 0:
        return tracers
    return jax.lax.fori_loop(0, gas.shape[-1], convert, tracers)

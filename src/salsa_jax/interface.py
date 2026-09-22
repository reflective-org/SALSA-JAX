"""Native SALSA/HAM aerosol conversions and final particle density.

The fixed 69-component interface represents SO4/OC in bins 1–3, all five
materials in bins 4–10, and SO4/OC/BC/DU in bins 11–17. Material volume is
m3 material/m3 air. HAM sulfate is molecules/cm3; other material slots are
micrograms/m3. These functions do not perform cleanup or hydration.
"""
import jax.numpy as jnp
import numpy as np

from salsa_jax.constants import AVOGADRO, DENSITY_VOLUME_THRESHOLD
from salsa_jax.processes.hydration import HydrationParameters

# Canonical volume order SO4/OC/BC/SS/DU; native one-based regions 1a/2a/2b.
_REPRESENTED = tuple(i * 5 + s for i in range(17) for s in range(5)
                     if (i >= 3 or s < 2) and (i < 10 or s != 3))


def validate_component_map(component_map):
    """Validate a host-side (...,17,5) map and return its integer NumPy array.

    Maps use zero-based component indices and -1 in unrepresented slots. Each
    cell must represent every index 0..68 exactly once. Validate when adapting
    input/configuration, before passing runtime arrays into compiled kernels.
    """
    mapping = np.asarray(component_map)
    if mapping.shape[-2:] != (17, 5) or not np.issubdtype(mapping.dtype, np.integer):
        raise ValueError('component_map must have integer shape (...,17,5)')
    flat = mapping.reshape(mapping.shape[:-2] + (85,))
    represented = np.zeros(85, dtype=bool)
    represented[list(_REPRESENTED)] = True
    if np.any(flat[..., ~represented] != -1):
        raise ValueError('Unrepresented native component slots must be -1')
    if not np.all(np.sort(flat[..., represented], axis=-1) == np.arange(69)):
        raise ValueError('Represented component slots must permute indices 0..68')
    return mapping


def _map(component_map, leading):
    mapping = jnp.asarray(component_map)
    if mapping.shape[-2:] != (17, 5) or not jnp.issubdtype(mapping.dtype, jnp.integer):
        raise ValueError('component_map must have integer shape (...,17,5)')
    return jnp.broadcast_to(mapping, leading + (17, 5))


def _properties(parameters, leading):
    density = jnp.broadcast_to(jnp.asarray(parameters.species_density, dtype=jnp.float64), leading + (5,))
    mass = jnp.broadcast_to(jnp.asarray(parameters.species_molar_mass, dtype=jnp.float64), leading + (5,))
    molecular_volume = (mass[..., 0] / 1000.0) / AVOGADRO / density[..., 0]
    return density, molecular_volume


def mass_to_volume(mass, component_map, parameters=HydrationParameters()):
    """Convert native mass (...,69) to volume (...,17,5), zeroing absent slots.

    Parameters broadcast over leading case axes. Supply a validated component
    map (also broadcastable over those axes); inputs are unchanged. Enable x64.
    """
    mass = jnp.asarray(mass, dtype=jnp.float64)
    if mass.shape[-1:] != (69,):
        raise ValueError('mass must have shape (...,69)')
    mapping = _map(component_map, mass.shape[:-1])
    density, molecular_volume = _properties(parameters, mass.shape[:-1])
    mapped = jnp.take_along_axis(mass[..., None, :], jnp.maximum(mapping, 0), axis=-1)
    volume = mapped / density[..., None, :] * 1e-9
    volume = volume.at[..., 0].set(mapped[..., 0] * 1e6 * molecular_volume[..., None])
    return jnp.where(mapping >= 0, volume, 0.0)


def volume_to_mass(volume, component_map, parameters=HydrationParameters()):
    """Convert volume (...,17,5) to all 69 native mass components.

    Material in unrepresented slots is omitted, exactly as in native output
    conversion; it is not moved to another component. Map validity is a caller
    precondition checked with validate_component_map outside JIT.
    """
    volume = jnp.asarray(volume, dtype=jnp.float64)
    if volume.shape[-2:] != (17, 5):
        raise ValueError('volume must have shape (...,17,5)')
    leading = volume.shape[:-2]
    mapping = _map(component_map, leading)
    density, molecular_volume = _properties(parameters, leading)
    mass = volume * density[..., None, :] * 1e9
    mass = mass.at[..., 0].set(volume[..., 0] / (1e6 * molecular_volume[..., None]))
    indices = jnp.asarray(_REPRESENTED)
    represented_map = jnp.take(mapping.reshape(leading + (85,)), indices, axis=-1)
    represented_mass = jnp.take(mass.reshape(leading + (85,)), indices, axis=-1)
    return jnp.take_along_axis(represented_mass, jnp.argsort(represented_map, axis=-1), axis=-1)


def particle_density(volume, water_mass, parameters=HydrationParameters()):
    """Return native mean particle density (...,17) [g/cm3].

    Volume (...,17,5) uses m3/m3 and water_mass (...,17) uses kg/m3 air.
    Water is signed and never clipped. Each region includes only its native
    represented materials. Total wet volume <= 1e-30 uses water density in
    1a/2a and BC density in 2b. The original sum/division order is retained.
    """
    volume = jnp.asarray(volume, dtype=jnp.float64)
    water = jnp.asarray(water_mass, dtype=jnp.float64)
    if volume.shape[-2:] != (17, 5) or water.shape != volume.shape[:-1]:
        raise ValueError('Expected volume (...,17,5) and water_mass (...,17)')
    leading = volume.shape[:-2]
    density, _ = _properties(parameters, leading)
    water_density = jnp.broadcast_to(jnp.asarray(parameters.water_density, dtype=jnp.float64), leading)[..., None]
    regions = []
    for start, end, species in ((0, 3, (0, 1)), (3, 10, (0, 1, 2, 3, 4)), (10, 17, (0, 1, 2, 4))):
        wet_volume = volume[..., start:end, 0]
        mass = wet_volume * density[..., 0, None]
        for s in species[1:]:
            wet_volume = wet_volume + volume[..., start:end, s]
            mass = mass + volume[..., start:end, s] * density[..., s, None]
        wet_volume = wet_volume + water[..., start:end] / water_density
        mass = mass + water[..., start:end]
        active = wet_volume > DENSITY_VOLUME_THRESHOLD
        fallback = water_density if start < 10 else density[..., 2, None]
        regions.append(jnp.where(active, mass / jnp.where(active, wet_volume, 1.0) / 1000.0,
                                 fallback / 1000.0))
    return jnp.concatenate(regions, axis=-1)

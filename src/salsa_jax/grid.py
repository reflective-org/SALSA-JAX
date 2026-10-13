"""Native SALSA dry size-bin geometry on the fixed 17-bin standalone layout."""
from typing import NamedTuple

import jax
import jax.numpy as jnp

from salsa_jax.constants import PI

# mo_ham_salsactl.F90: three diameter intervals, split into 3, 4 and 3 bins.
DEFAULT_REGIME_LIMITS = (3e-9, 5e-8, 7e-7, 1e-5)  # m
REGIME_BIN_COUNTS = (3, 4, 3)


class SizeGrid(NamedTuple):
    """Dry volume bounds [m3/particle], diameter [m], and bound/mid volume ratios."""

    vlolim: jax.Array
    vhilim: jax.Array
    dpmid: jax.Array
    vratiohi: jax.Array
    vratiolo: jax.Array


def size_grid(regime_limits=DEFAULT_REGIME_LIMITS):
    """Port of set_sizebins; return fields (...,17) for diameter limits (...,4).

    Limits must be finite, positive, strictly increasing, in meters. Enable
    JAX float64 before use. The 2b bins duplicate all seven 2a bins exactly;
    diameter ordering therefore restarts at index10. This is the fixed native
    3/7/7 layout, not an arbitrary-bin generator. Mid-diameter corresponds to
    the arithmetic mean of lower and upper particle volume, not diameter.
    """
    limits = jnp.asarray(regime_limits, dtype=jnp.float64)
    if limits.shape[-1:] != (4,):
        raise ValueError('Expected regime diameter limits (...,4)')
    lower, upper = [], []
    for interval, count in enumerate(REGIME_BIN_COUNTS):
        ratio = limits[..., interval+1] / limits[..., interval]
        # Native REAL(ii)/REAL(nbin) is evaluated in default real (binary32),
        # then promoted for the double-precision power. Using float64 division
        # changes the one-third/two-thirds edges by about 1e-7 relative.
        exponents = (jnp.arange(count+1, dtype=jnp.float32) /
                     jnp.asarray(count, dtype=jnp.float32)).astype(jnp.float64)
        diameters = limits[..., interval, None] * ratio[..., None]**exponents
        volumes = (PI / 6.0) * diameters**3
        lower.append(volumes[..., :-1])
        upper.append(volumes[..., 1:])
    vlolim = jnp.concatenate((*lower, *lower[1:]), axis=-1)
    vhilim = jnp.concatenate((*upper, *upper[1:]), axis=-1)
    dpmid = ((vhilim + vlolim) / (2.0 * (PI / 6.0)))**(1.0 / 3.0)
    middle_volume = (PI / 6.0) * dpmid**3
    return SizeGrid(vlolim, vhilim, dpmid, vhilim / middle_volume, vlolim / middle_volume)

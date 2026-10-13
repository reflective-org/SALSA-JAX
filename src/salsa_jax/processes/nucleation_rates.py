"""Native SALSA binary, activation and kinetic critical-cluster rates.

The binary polynomial and its collision fallback are transcribed from
mo_ham_salsa_nucleation.F90:binnucl, preserving the historical constants.
This module does not grow clusters to 3 nm or update aerosol/gas state.
"""
from typing import NamedTuple
import struct

import jax
import jax.numpy as jnp

from salsa_jax.constants import (
    AVOGADRO, GAS_CONSTANT, PI, NUCLEATION_ACTIVATION_COEFFICIENT,
    NUCLEATION_CRITICAL_DIAMETER,
)


class CriticalCluster(NamedTuple):
    """Rate [m-3 s-1], sulfate molecules per critical cluster, diameter [m]."""

    rate: jax.Array
    sulfate_molecules: jax.Array
    diameter: jax.Array


def _single(value):
    """Promote the source's default-real binary32 literal, without re-rounding."""
    return struct.unpack('f', struct.pack('f', value))[0]


def binary_nucleation(sulfate_cm3, temperature, relative_humidity):
    """Native binary H2SO4/H2O parameterization, with broadcast input axes.

    Input sulfate is molecules/cm3, temperature K, relative humidity a fraction.
    The historical fit clamps these inputs to [1e4,1e9], [190.15,300.15], and
    [1e-4,1], respectively. Inputs must be finite; enable JAX float64 first.
    Critical sizes are retained even when the fitted rate is set to zero.
    """
    sulfate, temperature, humidity = jnp.broadcast_arrays(*[
        jnp.asarray(value, dtype=jnp.float64)
        for value in (sulfate_cm3, temperature, relative_humidity)
    ])
    zt = jnp.clip(temperature, 190.15, 300.15)
    zpcsa = jnp.clip(sulfate, 1e4, 1e9)
    zrh = jnp.clip(humidity, 1e-4, 1.0)
    log_rh, log_sulfate = jnp.log(zrh), jnp.log(zpcsa)
    zx = (
        0.7409967177282139
        - 0.002663785665140117*zt
        + 0.002010478847383187*log_rh
        - 0.0001832894131464668*zt*log_rh
        + 0.001574072538464286*log_rh**2
        - 0.00001790589121766952*zt*log_rh**2
        + 0.0001844027436573778*log_rh**3
        - 1.503452308794887e-6*zt*log_rh**3
        - 0.003499978417957668*log_sulfate
        + 0.0000504021689382576*zt*log_sulfate
    )
    log_rate = (
        0.1430901615568665
        + 2.219563673425199*zt
        - 0.02739106114964264*zt**2
        + 0.00007228107239317088*zt**3
        + 5.91822263375044/zx
        + 0.1174886643003278*log_rh
        + 0.4625315047693772*zt*log_rh
        - 0.01180591129059253*zt**2*log_rh
        + 0.0000404196487152575*zt**3*log_rh
        + (15.79628615047088*log_rh)/zx
        - 0.215553951893509*log_rh**2
        - 0.0810269192332194*zt*log_rh**2
        + 0.001435808434184642*zt**2*log_rh**2
        - 4.775796947178588e-6*zt**3*log_rh**2
        - (2.912974063702185*log_rh**2)/zx
        - 3.588557942822751*log_rh**3
        + 0.04950795302831703*zt*log_rh**3
        - 0.0002138195118737068*zt**2*log_rh**3
        + 3.108005107949533e-7*zt**3*log_rh**3
        - (0.02933332747098296*log_rh**3)/zx
        + 1.145983818561277*log_sulfate
        - 0.6007956227856778*zt*log_sulfate
        + 0.00864244733283759*zt**2*log_sulfate
        - 0.00002289467254710888*zt**3*log_sulfate
        - (8.44984513869014*log_sulfate)/zx
        + 2.158548369286559*log_rh*log_sulfate
        + 0.0808121412840917*zt*log_rh*log_sulfate
        - 0.0004073815255395214*zt**2*log_rh*log_sulfate
        - 4.019572560156515e-7*zt**3*log_rh*log_sulfate
        + (0.7213255852557236*log_rh*log_sulfate)/zx
        + 1.62409850488771*log_rh**2*log_sulfate
        - 0.01601062035325362*zt*log_rh**2*log_sulfate
        + 0.00003771238979714162*zt**2*log_rh**2*log_sulfate
        + 3.217942606371182e-8*zt**3*log_rh**2*log_sulfate
        - (0.01132550810022116*log_rh**2*log_sulfate)/zx
        + 9.71681713056504*log_sulfate**2
        - 0.1150478558347306*zt*log_sulfate**2
        + 0.0001570982486038294*zt**2*log_sulfate**2
        + 4.009144680125015e-7*zt**3*log_sulfate**2
        + (0.7118597859976135*log_sulfate**2)/zx
        - 1.056105824379897*log_rh*log_sulfate**2
        + 0.00903377584628419*zt*log_rh*log_sulfate**2
        - 0.00001984167387090606*zt**2*log_rh*log_sulfate**2
        + 2.460478196482179e-8*zt**3*log_rh*log_sulfate**2
        - (0.05790872906645181*log_rh*log_sulfate**2)/zx
        - 0.1487119673397459*log_sulfate**3
        + 0.002835082097822667*zt*log_sulfate**3
        - 9.24618825471694e-6*zt**2*log_sulfate**3
        + 5.004267665960894e-9*zt**3*log_sulfate**3
        - (0.01270805101481648*log_sulfate**3)/zx
    )
    fitted_rate = jnp.exp(log_rate)
    fitted_rate = jnp.where(fitted_rate < 1e-7, 0.0, fitted_rate)
    log_total_molecules = (
        - 0.002954125078716302
        - 0.0976834264241286*zt
        + 0.001024847927067835*zt**2
        - 2.186459697726116e-6*zt**3
        - 0.1017165718716887/zx
        - 0.002050640345231486*log_rh
        - 0.007585041382707174*zt*log_rh
        + 0.0001926539658089536*zt**2*log_rh
        - 6.70429719683894e-7*zt**3*log_rh
        - (0.2557744774673163*log_rh)/zx
        + 0.003223076552477191*log_rh**2
        + 0.000852636632240633*zt*log_rh**2
        - 0.00001547571354871789*zt**2*log_rh**2
        + 5.666608424980593e-8*zt**3*log_rh**2
        + (0.03384437400744206*log_rh**2)/zx
        + 0.04743226764572505*log_rh**3
        - 0.0006251042204583412*zt*log_rh**3
        + 2.650663328519478e-6*zt**2*log_rh**3
        - 3.674710848763778e-9*zt**3*log_rh**3
        - (0.0002672510825259393*log_rh**3)/zx
        - 0.01252108546759328*log_sulfate
        + 0.005806550506277202*zt*log_sulfate
        - 0.0001016735312443444*zt**2*log_sulfate
        + 2.881946187214505e-7*zt**3*log_sulfate
        + (0.0942243379396279*log_sulfate)/zx
        - 0.0385459592773097*log_rh*log_sulfate
        - 0.0006723156277391984*zt*log_rh*log_sulfate
        + 2.602884877659698e-6*zt**2*log_rh*log_sulfate
        + 1.194163699688297e-8*zt**3*log_rh*log_sulfate
        - (0.00851515345806281*log_rh*log_sulfate)/zx
        - 0.01837488495738111*log_rh**2*log_sulfate
        + 0.0001720723574407498*zt*log_rh**2*log_sulfate
        - 3.717657974086814e-7*zt**2*log_rh**2*log_sulfate
        - 5.148746022615196e-10*zt**3*log_rh**2*log_sulfate
        + (0.0002686602132926594*log_rh**2*log_sulfate)/zx
        - 0.06199739728812199*log_sulfate**2
        + 0.000906958053583576*zt*log_sulfate**2
        - 9.11727926129757e-7*zt**2*log_sulfate**2
        - 5.367963396508457e-9*zt**3*log_sulfate**2
        - (0.007742343393937707*log_sulfate**2)/zx
        + 0.0121827103101659*log_rh*log_sulfate**2
        - 0.0001066499571188091*zt*log_rh*log_sulfate**2
        + 2.534598655067518e-7*zt**2*log_rh*log_sulfate**2
        - 3.635186504599571e-10*zt**3*log_rh*log_sulfate**2
        + (0.0006100650851863252*log_rh*log_sulfate**2)/zx
        + 0.0003201836700403512*log_sulfate**3
        - 0.0000174761713262546*zt*log_sulfate**3
        + 6.065037668052182e-8*zt**2*log_sulfate**3
        - 1.421771723004557e-11*zt**3*log_sulfate**3
        + (0.0001357509859501723*log_sulfate**3)/zx
    )
    total_molecules = jnp.exp(log_total_molecules)
    sulfate_molecules = zx * total_molecules
    # The original overwrites its earlier 1e-9 zero-rate diameter here.
    diameter = 2e-9 * jnp.exp(-1.6524245 + 0.42316402 * zx + 0.33466487 * jnp.log(total_molecules))

    # Pure-sulfuric-acid density fit. All decimal coefficients below lack _dp
    # in Fortran; they enter this arithmetic as binary32 values promoted to dp.
    # Retain nested order even though zxmass is always exactly one.
    mass_fraction = 1.0
    a = (_single(0.7681724) + mass_fraction * (_single(2.1847140) + mass_fraction *
         (_single(7.1630022) + mass_fraction * (_single(-44.31447) + mass_fraction *
         (_single(88.75606) + mass_fraction * (_single(-75.73729) + mass_fraction * _single(23.43228)))))))
    b = (_single(1.808225e-3) + mass_fraction * (_single(-9.294656e-3) + mass_fraction *
         (_single(-0.03742148) + mass_fraction * (_single(0.2565321) + mass_fraction *
         (_single(-0.5362872) + mass_fraction * (_single(0.4857736) - mass_fraction * _single(0.1629592)))))))
    c = (_single(-3.478524e-6) + mass_fraction * (_single(1.335867e-5) + mass_fraction *
         (_single(5.195706e-5) + mass_fraction * (_single(-3.717636e-4) + mass_fraction *
         (_single(7.990811e-4) + mass_fraction * (_single(-7.458060e-4) + mass_fraction * _single(2.58139e-4)))))))
    density = (a + zt * (b + c * zt)) * 1e3
    mass1 = _single(0.098)
    mass2 = mass1
    volume1 = mass1 / AVOGADRO / density
    volume2 = volume1
    collision_rate = (zpcsa * zpcsa * (3.0 * PI / 4.0)**(1.0 / 6.0) *
                      jnp.sqrt(6.0 * GAS_CONSTANT * zt / mass1 + 6.0 * GAS_CONSTANT * zt / mass2) *
                      (volume1**(1.0 / 3.0) + volume2**(1.0 / 3.0))**2.0 * 1e6)
    rate = jnp.minimum(jnp.where(sulfate_molecules < 4.0, collision_rate, fitted_rate), 1e10) * 1e6
    return CriticalCluster(rate, sulfate_molecules, diameter)


def nucleation_rates(sulfate_m3, temperature, relative_humidity, *, scheme=2,
                     boundary_layer=True, activation_coefficient=NUCLEATION_ACTIVATION_COEFFICIENT):
    """Return native scheme 1/2/3 critical-cluster properties, before growth.

    Sulfate is molecules/m3, temperature K, RH a fraction. Inputs must be
    finite; sulfate and activation coefficient nonnegative. ``scheme`` is a
    static Python integer (use static_argnames with jax.jit). Scheme 1 uses
    binary nucleation everywhere. Schemes 2/3 use activation/kinetic rates
    where boundary_layer is True and binary rates above it. The caller maps
    native inclusive ppbl level indices to this boolean mask. Arrays broadcast.
    Native schemes 2/3 overwrite diameter in ALL cells, including above the
    boundary layer; their activation/kinetic concentrations are not clamped.
    """
    if scheme not in (1, 2, 3):
        raise ValueError('Supported nucleation rate schemes are 1, 2 and 3')
    sulfate = jnp.asarray(sulfate_m3, dtype=jnp.float64)
    binary = binary_nucleation(sulfate * 1e-6, temperature, relative_humidity)
    if scheme == 1:
        return binary
    if scheme == 2:
        boundary_rate = jnp.asarray(activation_coefficient, dtype=jnp.float64) * sulfate
    else:
        boundary_rate = 5.0e-13 * (sulfate * 1e-6)**2 * 1e6
    rate = jnp.where(boundary_layer, boundary_rate, binary.rate)
    molecules = jnp.where(boundary_layer, 2.0, binary.sulfate_molecules)
    rate, molecules = jnp.broadcast_arrays(rate, molecules)
    diameter = jnp.full_like(rate, NUCLEATION_CRITICAL_DIAMETER)
    return CriticalCluster(rate, molecules, diameter)

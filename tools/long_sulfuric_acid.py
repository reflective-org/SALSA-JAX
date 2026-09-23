"""Long controlled sulfate experiments with fixed vapor or coagulation only."""
from dataclasses import dataclass, asdict, replace
import math
from pathlib import Path
import tomllib

import jax
import jax.numpy as jnp
import numpy as np

from salsa_jax.grid import size_grid
from salsa_jax.model import salsa_step
from salsa_jax.sulfuric_acid import SulfuricAcidConfig, initial_state, step_options
from tools.sulfuric_acid_validation import reference_case
from tools.salsa_oracle import OUTPUTS


@dataclass(frozen=True)
class Experiment:
    label: str
    duration_hours: float
    timestep_s: float
    save_interval_s: float
    temperature_k: float
    pressure_pa: float
    relative_humidity: float
    initial_number_cm3: float
    geometric_mean_diameter_nm: float
    geometric_stddev: float
    h2so4_molecules_cm3: float
    constant_h2so4: bool
    nucleation_scheme: int
    growth_option: int
    coagulation_enabled: bool
    condensation_enabled: bool
    recalculate: bool

    def __post_init__(self):
        for key in ('duration_hours', 'timestep_s', 'save_interval_s', 'temperature_k',
                    'pressure_pa', 'initial_number_cm3', 'geometric_mean_diameter_nm', 'geometric_stddev'):
            value = getattr(self, key)
            if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
                raise ValueError(f'{key} must be finite and positive')
        if self.geometric_stddev <= 1 or not 0 <= self.relative_humidity <= 1:
            raise ValueError('Require geometric_stddev>1 and RH in [0,1]')
        if not np.isfinite(self.h2so4_molecules_cm3) or self.h2so4_molecules_cm3 < 0:
            raise ValueError('H2SO4 must be finite and nonnegative')
        if type(self.constant_h2so4) is not bool:
            raise ValueError('constant_h2so4 must be boolean')
        ratios = (self.duration_hours*3600/self.timestep_s, self.save_interval_s/self.timestep_s,
                  self.duration_hours*3600/self.save_interval_s)
        if any(not np.isfinite(x) or x < 1 or not np.isclose(x, round(x), rtol=0, atol=1e-9) for x in ratios):
            raise ValueError('Duration and save interval must be exact positive multiples of timestep')

    @property
    def steps(self):
        return round(self.duration_hours*3600/self.timestep_s)

    @property
    def save_every(self):
        return round(self.save_interval_s/self.timestep_s)


def load_experiment(path):
    return Experiment(**tomllib.loads(Path(path).read_text()))


def binned_lognormal(experiment, grid):
    """Integrate number exactly over ten soluble-bin edges; do not renormalize.

    Stable erfc tails avoid cancellation. Aerosol mass will use native midpoint
    volumes, so this defines a discrete lognormal-number initialization rather
    than preserving the analytic lognormal third moment.
    """
    lower = np.cbrt(np.asarray(grid.vlolim)[:10]/(np.pi/6))
    upper = np.cbrt(np.asarray(grid.vhilim)[:10]/(np.pi/6))
    mean = experiment.geometric_mean_diameter_nm*1e-9
    sigma = math.log(experiment.geometric_stddev)
    def interval(lo, hi):
        a, b = math.log(lo/mean)/(sigma*math.sqrt(2)), math.log(hi/mean)/(sigma*math.sqrt(2))
        if a >= 0:
            return .5*(math.erfc(a)-math.erfc(b))
        if b <= 0:
            return .5*(math.erfc(-b)-math.erfc(-a))
        return .5*(math.erf(b)-math.erf(a))
    return np.array([interval(lo, hi) for lo, hi in zip(lower, upper)])*experiment.initial_number_cm3*1e6


def prepare(experiment):
    grid = size_grid()
    number = binned_lognormal(experiment, grid)
    config = SulfuricAcidConfig(steps=experiment.steps, timestep_s=experiment.timestep_s,
        temperature_k=experiment.temperature_k, pressure_pa=experiment.pressure_pa,
        relative_humidity=experiment.relative_humidity,
        initial_h2so4_molecules_m3=experiment.h2so4_molecules_cm3*1e6,
        h2so4_source_molecules_m3_s=0., initial_number_m3=tuple(number),
        nucleation_scheme=experiment.nucleation_scheme, growth_option=experiment.growth_option,
        coagulation_enabled=experiment.coagulation_enabled, condensation_enabled=experiment.condensation_enabled,
        recalculate=experiment.recalculate)
    mass, number, gas, mapping, grid = initial_state(config)
    return config, reference_case(config), {'mass': np.asarray(mass), 'number': np.asarray(number),
                                          'gas': np.asarray(gas)}, mapping, grid


def run_jax(experiment):
    """JIT blocks of native core steps; retain snapshots and every-step monitors.

    Constant-vapor mode restores gas AFTER each core call, accounting for the
    actual makeup as an external sulfur reservoir. Internal vapor depletion
    within a native step is retained; refine dt to assess that approximation.
    """
    config, cases, initial, mapping, grid = prepare(experiment)
    sulfate_ids = jnp.asarray(mapping[:, 0])
    initial_sulfur = jnp.asarray(initial['gas'][:, 1] + 1e6*initial['mass'][:, mapping[:, 0]].sum(-1))
    target = jnp.asarray(initial['gas'][:, 1])
    options = step_options(config)
    # Seven scientific outputs, reservoir amount, pre-restore gas, budget,
    # relative depletion, minimum number/mass/gas, all-step validity.
    carry = {key: jnp.asarray(initial[key]) for key in ('mass', 'number', 'gas')}
    carry.update({key: jnp.zeros((1, width), dtype=jnp.float64)
                  for key, width in OUTPUTS.items() if key not in carry})
    carry.update(cumulative_reservoir_added=jnp.zeros(1), h2so4_before_restore=target,
                 max_relative_sulfur_residual=jnp.zeros(1), max_relative_h2so4_depletion=jnp.zeros(1),
                 minimum_number=jnp.min(carry['number'], axis=-1), minimum_mass=jnp.min(carry['mass'], axis=-1),
                 minimum_gas=jnp.min(carry['gas'], axis=-1), valid=jnp.ones(1, dtype=bool))
    def step(_, state):
        result = salsa_step(state['mass'], state['number'], state['gas'], experiment.temperature_k,
            experiment.relative_humidity, experiment.pressure_pa, experiment.timestep_s, mapping, **options)
        gas_before = result.gas[:, 1]
        makeup = target-gas_before if experiment.constant_h2so4 else jnp.zeros_like(target)
        gas = result.gas.at[:, 1].set(target) if experiment.constant_h2so4 else result.gas
        supplied = state['cumulative_reservoir_added']+makeup
        inventory = gas[:, 1]+1e6*jnp.sum(result.mass[:, sulfate_ids], axis=-1)
        expected = initial_sulfur+supplied
        error = jnp.abs(inventory-expected)/jnp.where(expected != 0, jnp.abs(expected), 1.)
        depleted = (target-gas_before)/jnp.where(target > 0, target, 1.) if experiment.constant_h2so4 else jnp.zeros_like(target)
        return {**{key: getattr(result, key) for key in OUTPUTS}, 'gas': gas,
            'cumulative_reservoir_added': supplied, 'h2so4_before_restore': gas_before,
            'max_relative_sulfur_residual': jnp.maximum(state['max_relative_sulfur_residual'], error),
            'max_relative_h2so4_depletion': jnp.maximum(state['max_relative_h2so4_depletion'], depleted),
            'minimum_number': jnp.minimum(state['minimum_number'], jnp.min(result.number, axis=-1)),
            'minimum_mass': jnp.minimum(state['minimum_mass'], jnp.min(result.mass, axis=-1)),
            'minimum_gas': jnp.minimum(state['minimum_gas'], jnp.min(result.gas, axis=-1)),
            'valid': state['valid'] & result.valid}
    advance = jax.jit(lambda state: jax.lax.fori_loop(0, experiment.save_every, step, state))
    saved = []
    for index in range(experiment.steps//experiment.save_every):
        carry = advance(carry)
        host = jax.device_get(carry)
        if not np.all(host['valid']):
            raise ValueError(f'Invalid JAX state by step {(index+1)*experiment.save_every}')
        saved.append(host)
    snapshots = {key: np.stack([row[key] for row in saved])
                 for key in (*OUTPUTS, 'cumulative_reservoir_added', 'h2so4_before_restore')}
    monitors = {key: np.asarray(host[key]).tolist() for key in host if key not in snapshots}
    return initial, snapshots, monitors, cases, mapping, grid

"""Fortran case adapter and explicit gates for the pure-sulfate experiment."""
from dataclasses import asdict
import numpy as np

from salsa_jax import constants
from salsa_jax.grid import DEFAULT_REGIME_LIMITS
from salsa_jax.processes.hydration import HydrationParameters
from salsa_jax.sulfuric_acid import initial_state, step_options, sulfur_inventory
from tools.salsa_oracle import OUTPUTS

# Long trajectories accumulate rounding. Reference zeros are exact except for
# water, which retains the existing wet-minus-dry cancellation bound (also at zero).
TRAJECTORY_RTOL = 3e-11
SULFUR_BUDGET_RTOL = 1e-10


def reference_case(config):
    """Adapt the exact experiment initial arrays to the original SALSA oracle."""
    mass, number, gas, mapping, _ = initial_state(config)
    props = HydrationParameters()
    row = {**dict(props._asdict()), **step_options(config),
           'temperature': config.temperature_k, 'pressure': config.pressure_pa,
           'relative_humidity': config.relative_humidity, 'timestep': config.timestep_s,
           'regime_limits': DEFAULT_REGIME_LIMITS, 'mass_accommodation': np.ones(17),
           'cluster_molecules': constants.NUCLEATION_CLUSTER_MOLECULES,
           'acid_diameter': constants.SULFURIC_ACID_DIAMETER,
           'activation_coefficient': constants.NUCLEATION_ACTIVATION_COEFFICIENT,
           'saturation_limit': constants.REMAPPING_SATURATION_LIMIT,
           'ion_count': constants.REMAPPING_ION_FACTOR, 'component_map': mapping}
    return {'labels': ['pure-sulfate'], **{key: np.asarray(value)[None] for key, value in row.items()},
            'mass': np.asarray(mass), 'number': np.asarray(number), 'gas': np.asarray(gas)}


def summarize(config, initial, outputs, mapping):
    """Report sulfur closure independently of agreement between implementations."""
    values = {key: np.asarray(outputs[key] if isinstance(outputs, dict) else getattr(outputs, key)) for key in OUTPUTS}
    if any(not np.all(np.isfinite(value)) for value in values.values()):
        raise ValueError('Nonfinite trajectory output')
    for key in ('mass', 'number', 'gas'):
        if np.any(values[key] < 0):
            raise ValueError(f'Negative trajectory {key}')
    inventory = sulfur_inventory(values['mass'], values['gas'], mapping)
    start = sulfur_inventory(initial['mass'], initial['gas'], mapping)
    supplied = np.arange(1, config.steps+1)[:, None] * config.timestep_s * config.h2so4_source_molecules_m3_s
    expected = start + supplied
    residual = inventory - expected
    relative = np.divide(residual, expected, out=np.zeros_like(residual), where=expected != 0)
    budget_ok = bool(np.all(np.abs(residual) <= SULFUR_BUDGET_RTOL * expected))
    sulfate_indices = mapping[:, 0]
    other_indices = np.setdiff1d(np.arange(69), sulfate_indices)
    purity_ok = bool(np.all(values['mass'][..., other_indices] == 0) and
                     np.all(values['gas'][..., [0, 2]] == 0) and np.all(values['number'][..., 10:] == 0) and
                     np.all(values['mass'][..., mapping[10:, 0]] == 0))
    return {
        'sulfur_budget_passed': budget_ok, 'sulfur_budget_relative_tolerance': SULFUR_BUDGET_RTOL,
        'maximum_relative_sulfur_residual': float(np.max(np.abs(relative))),
        'maximum_absolute_sulfur_residual_molecules_m3': float(np.max(np.abs(residual))),
        'initial_sulfur_molecules_m3': float(start[0]),
        'supplied_sulfur_molecules_m3': float(supplied[-1, 0]),
        'final_sulfur_molecules_m3': float(inventory[-1, 0]),
        'initial_number_m3': float(initial['number'].sum()),
        'final_number_m3': float(values['number'][-1].sum()),
        'final_h2so4_molecules_m3': float(values['gas'][-1, 0, 1]),
        'pure_sulfate_soluble_only': purity_ok,
        'negative_water_values': int(np.count_nonzero(values['water_mass'] < 0)),
        'all_state_fields_finite': True,
        'nonnegative_mass_number_gas': True,
        'configuration': asdict(config)}


def compare(actual, expected):
    """Compare every bin/component at every step, including signed water."""
    metrics = {}
    for key in OUTPUTS:
        value = np.asarray(actual[key] if isinstance(actual, dict) else getattr(actual, key))
        reference = np.asarray(expected[key])
        if value.shape != reference.shape or not np.all(np.isfinite(value)) or not np.all(np.isfinite(reference)):
            raise ValueError(f'Invalid trajectory shape/nonfinite {key}')
        error = np.abs(value-reference)
        bound = TRAJECTORY_RTOL * np.abs(reference)
        if key == 'water_mass':
            dry_scale = (constants.PI/6)*(np.asarray(expected['dry_radius_cm'])*.02)**3 * constants.WATER_DENSITY * expected['number']
            bound = bound + 32*np.finfo(float).eps*dry_scale
        relative = np.divide(error, np.abs(reference), out=np.zeros_like(error), where=reference != 0)
        location = np.unravel_index(np.argmax(error), error.shape)
        metrics[key] = {'passed': bool(np.all(error <= bound)),
                        'max_absolute_error': float(error.max()),
                        'max_relative_error_nonzero': float(relative.max()),
                        'nonzero_at_zero_reference': int(np.count_nonzero(value[reference == 0])),
                        'largest_absolute_error_index': list(map(int, location))}
    return {'passed': all(value['passed'] for value in metrics.values()),
            'relative_tolerance': TRAJECTORY_RTOL, 'absolute_tolerance': 0,
            'water_cancellation_allowance': '32*eps*dry_particle_volume*water_density*number, per bin and step',
            'fields': metrics}

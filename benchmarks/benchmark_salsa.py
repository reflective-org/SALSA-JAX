"""Matched CPU timing of the complete native SALSA call, excluding the HAM driver."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import tempfile
import time

import jax
import jaxlib
import numpy as np

from salsa_jax.constants import PI
from salsa_jax.model import STATIC_OPTIONS, salsa_step
from tools.salsa_oracle import OUTPUTS, build_oracle, reference_cases, run_oracle

ROOT = Path(__file__).resolve().parents[1]
RTOL = 3e-13
WATER_ROUNDOFF_ULPS = 32
DEFAULT_REPETITIONS = 500
DEFAULT_TRIALS = 5


def physical_outputs(*args, **kwargs):
    """Return exactly the seven native caller outputs, without internal traces."""
    state = salsa_step(*args, **kwargs)
    return tuple(getattr(state, name) for name in OUTPUTS)


def errors(actual, expected):
    delta = np.abs(actual - expected)
    nonzero = expected != 0
    return {'max_absolute': float(np.max(delta)),
            'max_relative_nonzero': float(np.max(delta[nonzero] / np.abs(expected[nonzero]), initial=0)),
            'max_absolute_at_reference_zero': float(np.max(delta[~nonzero], initial=0))}


def validate_outputs(actual, expected, cases):
    """Accept all physical outputs before timing; water uses a dry-mass scale.

    Arrays include both native levels: (cases,2,field width). Native signed
    water is never clipped. Its cancellation tolerance has no absolute floor.
    """
    metrics = {}
    for name in OUTPUTS:
        value, target = np.asarray(actual[name]), np.asarray(expected[name])
        if value.shape != target.shape or value.dtype != np.float64 or not np.all(np.isfinite(value)):
            raise AssertionError(f'Invalid physical output: {name}')
        if name == 'water_mass':
            # Native dry radius is cm; convert to volume then dry-core water mass.
            scale = ((4.0 * PI / 3.0) * (np.asarray(expected['dry_radius_cm']) * 1e-2)**3 *
                     np.asarray(cases['water_density'])[:, None, None] * np.asarray(expected['number']))
            delta = np.abs(value - target)
            bound = WATER_ROUNDOFF_ULPS * np.finfo(float).eps * np.abs(scale) + RTOL * np.abs(target)
            if not np.all(delta <= bound):
                raise AssertionError('water_mass exceeds dry-mass-scaled cancellation tolerance')
            normalized = np.divide(delta, np.abs(scale), out=np.zeros_like(delta), where=scale != 0)
            metrics[name] = {**errors(value, target), 'max_absolute_over_dry_water_mass_scale': float(normalized.max()),
                             'negative_fortran_values': int(np.sum(target < 0)), 'negative_jax_values': int(np.sum(value < 0)),
                             'cancellation_ulps': WATER_ROUNDOFF_ULPS}
        else:
            np.testing.assert_allclose(value, target, rtol=RTOL, atol=0, err_msg=name)
            metrics[name] = errors(value, target)
    return metrics


def _checksums(actual, expected_timing):
    checksums = {name: float(np.asarray(value).sum()) for name, value in actual.items()}
    for name in ('mass', 'number', 'gas'):
        np.testing.assert_allclose(checksums[name], expected_timing[name + '_checksum'], rtol=RTOL, atol=0)
    return checksums


def _distribution(samples):
    return {'minimum': min(samples), 'median': statistics.median(samples),
            'maximum': max(samples), 'samples': samples}


def _collect(groups, results, case_count):
    output = {name: np.empty((case_count, 2, width)) for name, width in OUTPUTS.items()}
    for group, result in zip(groups, results):
        for name, value in zip(OUTPUTS, result):
            output[name][group['indices']] = np.asarray(value).reshape(len(group['indices']), 2, OUTPUTS[name])
    return output


def benchmark(repetitions=DEFAULT_REPETITIONS, trials=DEFAULT_TRIALS, single_case=False):
    if (isinstance(repetitions, bool) or isinstance(trials, bool) or
            not isinstance(repetitions, int) or not isinstance(trials, int) or min(repetitions, trials) < 1):
        raise ValueError('repetitions and trials must be positive integers')
    # This adapter is shared with full-call validation, not a second input model.
    from tools.salsa_inputs import arguments, select_cases, static_groups
    jax.config.update('jax_enable_x64', True)
    jax.config.update('jax_enable_compilation_cache', False)
    cases = reference_cases()
    if single_case:
        cases = {name: value[:1] for name, value in cases.items()}
    count = len(cases['labels'])
    device = jax.devices('cpu')[0]
    groups = []
    compile_started = time.perf_counter()
    for indices in static_groups(cases):
        args, kwargs = arguments(select_cases(cases, indices), all_levels=True)
        args = jax.device_put(args, device)
        dynamic = {name: value for name, value in kwargs.items() if name not in STATIC_OPTIONS}
        static = {name: value for name, value in kwargs.items() if name in STATIC_OPTIONS}
        dynamic = jax.device_put(dynamic, device)
        kwargs = {**dynamic, **static}
        jax.block_until_ready((args, kwargs))
        started = time.perf_counter()
        compiled = jax.jit(physical_outputs, static_argnames=STATIC_OPTIONS).lower(*args, **kwargs).compile()
        compile_seconds = time.perf_counter() - started
        started = time.perf_counter()
        result = jax.block_until_ready(compiled(*args, **dynamic))
        first_execution_seconds = time.perf_counter() - started
        diagnostic = jax.block_until_ready(jax.jit(salsa_step, static_argnames=STATIC_OPTIONS)(*args, **kwargs))
        if not np.all(np.asarray(diagnostic.valid)):
            raise AssertionError('Native/JAX workload contains an invalid SALSA state')
        groups.append({'indices': np.asarray(indices), 'args': args, 'kwargs': dynamic, 'compiled': compiled,
                       'initial_result': result, 'compile_seconds': compile_seconds,
                       'first_execution_seconds': first_execution_seconds, 'options': static})
    preparation_seconds = time.perf_counter() - compile_started
    actual = _collect(groups, [group['initial_result'] for group in groups], count)
    jax_trials, fortran_trials, cpu_trials, orders = [], [], [], []
    with tempfile.TemporaryDirectory(prefix='salsa-full-call-benchmark-') as directory:
        checked, checked_provenance = build_oracle(Path(directory) / 'checked', checked=True)
        checked_state, _ = run_oracle(checked, cases, all_levels=True)
        executable, provenance = build_oracle(Path(directory) / 'optimized', checked=False)
        expected, timing = run_oracle(executable, cases, all_levels=True)
        for name in OUTPUTS:
            np.testing.assert_array_equal(expected[name], checked_state[name], err_msg=name)
        metrics = validate_outputs(actual, expected, cases)
        _checksums(actual, timing)
        # Every configuration has already executed and synchronized outside timing.
        for trial in range(trials):
            order = ('jax', 'fortran') if trial % 2 == 0 else ('fortran', 'jax')
            orders.append(order)
            for implementation in order:
                if implementation == 'jax':
                    started = time.perf_counter()
                    for _ in range(repetitions):
                        results = [group['compiled'](*group['args'], **group['kwargs']) for group in groups]
                        jax.block_until_ready(results)
                    jax_trials.append((time.perf_counter() - started) / repetitions)
                else:
                    repeated, timing = run_oracle(executable, cases, repetitions=repetitions,
                                                   steps=1, all_levels=True, timeout=300)
                    for name in OUTPUTS:
                        np.testing.assert_array_equal(repeated[name], expected[name], err_msg=name)
                    fortran_trials.append(timing['kernel_wall_seconds'] / repetitions)
                    cpu_trials.append(timing['kernel_cpu_seconds'] / repetitions)
    actual = _collect(groups, results, count)
    metrics = validate_outputs(actual, expected, cases)
    checksums = _checksums(actual, timing)
    return {
        'scope': 'complete original native SALSA call; 17 bins, 5 materials, 69 mass components; no HAM driver',
        'scientific_limit': 'Numerical compatibility retains native defects. No initialization, forcing, radiation, full box-model trajectory, accelerator or gradient claim.',
        'platform': platform.platform(), 'device': str(device), 'dtype': 'float64',
        'versions': {'jax': jax.__version__, 'jaxlib': jaxlib.__version__, 'numpy': np.__version__},
        'thread_environment': {key: os.environ.get(key) for key in (
            'XLA_FLAGS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'JAX_NUM_THREADS', 'TF_NUM_INTRAOP_THREADS', 'TF_NUM_INTEROP_THREADS')},
        'case_count': count, 'native_levels_per_case': 2, 'physical_cells_per_batch': 2 * count,
        'case_labels': cases['labels'], 'case_selection': 'captured default only' if single_case else 'complete reference matrix',
        'boundary_layer_by_native_level': [False, True],
        'case_sha256': {name: hashlib.sha256(np.asarray(value, dtype='<f8').tobytes(order='C')).hexdigest()
                        for name, value in cases.items() if name != 'labels'},
        'case_hash_encoding': 'named arrays, C-order IEEE float64 little endian; booleans/maps encoded numerically',
        'implementation_sha256': {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                  for path in [*sorted((ROOT / 'src/salsa_jax').rglob('*.py')),
                                               ROOT / 'tools/salsa_inputs.py', ROOT / 'tools/salsa_oracle.py',
                                               ROOT / 'benchmarks/benchmark_salsa.py']},
        'checked_fortran_provenance': checked_provenance, 'fortran_provenance': provenance,
        'checked_optimized_exact_match': True, 'parity': metrics, 'rtol': RTOL, 'atol': 0,
        'output_fields': list(OUTPUTS), 'jax_output_checksums': checksums,
        'fortran_output_checksums': {name: float(value.sum()) for name, value in expected.items()},
        'repetitions': repetitions, 'steps_per_repetition': 1, 'input_reset_each_repetition': True,
        'trials': trials, 'trial_orders': orders,
        'jax_static_groups': [{'case_indices': group['indices'].tolist(), 'options': group['options'],
                               'physical_cells': 2 * len(group['indices']),
                               'compile_seconds': group['compile_seconds'],
                               'first_execution_seconds': group['first_execution_seconds']} for group in groups],
        'jax_compile_seconds': sum(group['compile_seconds'] for group in groups),
        'jax_preparation_seconds_including_diagnostic_compiles': preparation_seconds,
        'jax_persistent_compilation_cache': False,
        'jax_warm_wall_seconds_per_batch': _distribution(jax_trials),
        'fortran_warm_wall_seconds_per_batch': _distribution(fortran_trials),
        'fortran_cpu_seconds_per_batch': _distribution(cpu_trials),
        'fortran_clock_ticks_per_second': timing['clock_ticks_per_second'],
        'median_fortran_over_jax': statistics.median(fortran_trials) / statistics.median(jax_trials),
        'timing_notes': 'Each trial repeats the same complete case batch, resetting inputs for one native call; '
                        'no trajectory is advanced between repetitions. Both implementations evaluate two levels '
                        'per case, above and within the boundary layer. Fortran includes copying fixed inputs, '
                        'setting species/flags, native grid setup, both correction passes and final output copies. '
                        'JAX computes the grid and all native outputs in the compiled call, dispatches each static '
                        'configuration group, then synchronizes every physical output once per complete batch. '
                        'Immutable JAX inputs require no reset copies. The timed JAX wrapper returns exactly the '
                        'seven native caller fields; internal volume/core/diameter/valid diagnostics are checked '
                        'outside timing and unused outputs may be eliminated. Compilation, initial transfers, '
                        'native process startup/I/O, checksum and comparison work are excluded from warm timing. '
                        'Fortran uses separately compiled O2 source without fast-math/LTO, runtime checks disabled; '
                        'checked and optimized outputs must agree exactly before any trials. Trial order alternates. '
                        'Thread settings are recorded, not standardized. This is composed microphysics performance, '
                        'not complete HAM box-model performance.',
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repetitions', type=int, default=DEFAULT_REPETITIONS)
    parser.add_argument('--trials', type=int, default=DEFAULT_TRIALS)
    parser.add_argument('--single-case', action='store_true', help='use captured default; both native levels remain included')
    args = parser.parse_args()
    if min(args.repetitions, args.trials) < 1:
        parser.error('repetitions and trials must be positive')
    print(json.dumps(benchmark(args.repetitions, args.trials, args.single_case), indent=2))

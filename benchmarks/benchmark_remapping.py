"""CPU float64 native remapping parity and warmed timing on identical case cells."""
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

from salsa_jax.grid import SizeGrid
from salsa_jax.constants import PI
from salsa_jax.processes.remapping import RemappingParameters, remap
from tools.remapping_oracle import build_oracle, reference_cases, run_oracle

ROOT = Path(__file__).resolve().parents[1]
INPUTS = ('number', 'volume', 'temperature', 'dry_diameter', 'lower_volume', 'upper_volume')
PROPERTY_NAMES = {'surface_tension': 'water_surface_tension', 'ion_factor': 'ion_count'}
PROPERTIES = RemappingParameters._fields
OUTPUTS = ('number', 'volume')
RTOL = 3e-13


def errors(actual, expected):
    delta = np.abs(actual - expected)
    nonzero = expected != 0
    return {'max_absolute': float(delta.max()),
            'max_relative_nonzero': float(np.max(delta[nonzero] / np.abs(expected[nonzero]), initial=0)),
            'max_absolute_at_reference_zero': float(np.max(delta[~nonzero], initial=0))}


def physical_outputs(*args, **kwargs):
    """Leave iteration diagnostics unused so the timed compilation can drop them."""
    result = remap(*args, **kwargs)
    return result.number, result.volume


def benchmark(repetitions=500, trials=5, single_case=False):
    if not isinstance(repetitions, int) or not isinstance(trials, int) or min(repetitions, trials) < 1:
        raise ValueError('repetitions and trials must be positive integers')
    jax.config.update('jax_enable_x64', True)
    jax.config.update('jax_enable_compilation_cache', False)
    cases = reference_cases()
    if single_case:
        cases = {key: value[:1] for key, value in cases.items()}
    device = jax.devices('cpu')[0]
    mid_volume = (PI / 6.0) * cases['dry_diameter']**3
    grid = SizeGrid(cases['lower_volume'], cases['upper_volume'], cases['dry_diameter'],
                    cases['upper_volume'] / mid_volume, cases['lower_volume'] / mid_volume)
    inputs = tuple(jax.device_put(cases[key], device) for key in ('number', 'volume', 'temperature')) + (jax.device_put(grid, device),)
    parameters = RemappingParameters(**{key: jax.device_put(cases[PROPERTY_NAMES.get(key, key)], device) for key in PROPERTIES})
    kwargs = {'parameters': parameters, 'transfer_soluble': jax.device_put(cases['soluble_transfer'], device)}
    jax.block_until_ready((inputs, kwargs))
    started = time.perf_counter()
    compiled = jax.jit(physical_outputs).lower(*inputs, **kwargs).compile()
    compile_seconds = time.perf_counter() - started
    started = time.perf_counter()
    actual = jax.block_until_ready(compiled(*inputs, **kwargs))
    first_execution_seconds = time.perf_counter() - started
    # Separate untimed diagnostic execution validates the exact stopping paths.
    diagnostic = jax.block_until_ready(jax.jit(remap)(*inputs, **kwargs))
    jax_trials, fortran_trials, cpu_trials, orders = [], [], [], []
    with tempfile.TemporaryDirectory(prefix='salsa-remapping-benchmark-') as directory:
        checked, checked_provenance = build_oracle(Path(directory)/'checked', checked=True)
        checked_state, _ = run_oracle(checked, cases)
        executable, provenance = build_oracle(Path(directory)/'optimized', checked=False)
        expected, timing = run_oracle(executable, cases)
        # Timing never substitutes optimized output for checked scientific acceptance.
        for name in (*OUTPUTS, 'iterations'):
            np.testing.assert_array_equal(expected[name], checked_state[name])
        for name, value in zip(OUTPUTS, actual):
            np.testing.assert_allclose(np.asarray(value), expected[name], rtol=RTOL, atol=0)
            np.testing.assert_allclose(np.asarray(getattr(diagnostic, name)), expected[name], rtol=RTOL, atol=0)
        np.testing.assert_array_equal(np.asarray(diagnostic.iterations), expected['iterations'][:, 0])
        for trial in range(trials):
            order = ('jax', 'fortran') if trial % 2 == 0 else ('fortran', 'jax')
            orders.append(order)
            for implementation in order:
                if implementation == 'jax':
                    started = time.perf_counter()
                    for _ in range(repetitions):
                        actual = jax.block_until_ready(compiled(*inputs, **kwargs))
                    jax_trials.append((time.perf_counter() - started) / repetitions)
                else:
                    repeated, timing = run_oracle(executable, cases, repetitions=repetitions)
                    for name in (*OUTPUTS, 'iterations'):
                        np.testing.assert_array_equal(repeated[name], expected[name])
                    fortran_trials.append(timing['kernel_wall_seconds'] / repetitions)
                    cpu_trials.append(timing['kernel_cpu_seconds'] / repetitions)
    metrics = {}
    for name, value in zip(OUTPUTS, actual):
        value = np.asarray(value)
        assert value.dtype == np.float64
        assert np.all(np.isfinite(value)) and np.all(value >= 0)
        np.testing.assert_allclose(value, expected[name], rtol=RTOL, atol=0)
        np.testing.assert_allclose(value.sum(), timing[name+'_checksum'], rtol=RTOL, atol=0)
        metrics[name] = errors(value, expected[name])
    iterations = expected['iterations']
    return {
        'scope': 'one isolated native distr_update call; 17 bins, 5 materials, optional lsol2b transfer',
        'scientific_limit': 'Fixed case inputs; excludes cleanup, hydration/corrections and second native pass. Preserves terminal bins, descending transfers and orphan material; no full SALSA or trajectory claim.',
        'platform': platform.platform(), 'device': str(device), 'dtype': str(actual[0].dtype),
        'versions': {'jax': jax.__version__, 'jaxlib': jaxlib.__version__, 'numpy': np.__version__},
        'thread_environment': {key: os.environ.get(key) for key in (
            'XLA_FLAGS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'JAX_NUM_THREADS', 'TF_NUM_INTRAOP_THREADS', 'TF_NUM_INTEROP_THREADS')},
        'implementation_sha256': {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in (
            'src/salsa_jax/constants.py', 'src/salsa_jax/processes/remapping.py',
            'tools/remapping_oracle.py', 'tools/coagulation_process_oracle.py',
            'benchmarks/benchmark_remapping.py')},
        'case_count': len(cases['labels']), 'physical_cells_per_batch': len(cases['labels']),
        'bins_per_cell': 17, 'case_labels': cases['labels'],
        'case_selection': 'first reference_cases entry only' if single_case else 'all reference_cases entries',
        'case_sha256': {key: hashlib.sha256(np.asarray(cases[key], dtype='<f8').tobytes(order='C')).hexdigest()
                        for key in (*INPUTS, *[PROPERTY_NAMES.get(name, name) for name in PROPERTIES], 'soluble_transfer')},
        'case_hash_encoding': 'named arrays, C-order IEEE float64 little endian; boolean mask encoded as 0/1 float64',
        'soluble_transfer': np.asarray(cases['soluble_transfer'], dtype=bool).tolist(),
        'checked_fortran_provenance': checked_provenance,
        'fortran_provenance': provenance, 'checked_optimized_exact_match': True,
        'parity': metrics,
        'iterations': {'jax_fortran_exact_match': True, 'timed_diagnostics': False,
                       'maximum_scans_per_call': int(iterations.max()),
                       'total_per_batch': int(iterations.sum()),
                       'per_case_total': iterations.sum(axis=-1).tolist(),
                       'per_case_per_pass': iterations.tolist()},
        'repetitions': repetitions, 'trials': trials, 'trial_orders': orders,
        'jax_compile_seconds': compile_seconds, 'jax_persistent_compilation_cache': False,
        'jax_first_execution_seconds': first_execution_seconds,
        'jax_warm_wall_seconds_per_batch': jax_trials,
        'fortran_warm_wall_seconds_per_batch': fortran_trials,
        'fortran_cpu_seconds_per_batch': cpu_trials,
        'fortran_clock_ticks_per_second': timing['clock_ticks_per_second'],
        'median_fortran_over_jax': statistics.median(fortran_trials) / statistics.median(jax_trials),
        'timing_notes': 'Alternating warmed wall-time trials on the same physical cells. Fortran uses '
                        'separately compiled original arithmetic with O2, no fast-math/LTO, and runtime '
                        'checks disabled. Timed iteration capture is off; its conditional branch remains. '
                        'Fortran includes copying fixed number/volume/environment/geometry, setting species '
                        'properties and the soluble-transfer switch, and copying physical outputs. Startup, '
                        'I/O, iteration diagnostics and checksums are excluded. Checked and optimized outputs '
                        'and iteration counts are first verified identical outside timing. The timed JAX '
                        'wrapper returns only number and material volume, leaving its iteration counter unused '
                        'for compilation to eliminate; exact counts are validated in a separate untimed call. '
                        'JAX includes dispatch plus synchronization of both outputs, excluding transfer and '
                        'compilation. Fortran loops over cells/bins; JAX batches them and retains masked bins '
                        'until every native descending scan finishes. Thread settings are recorded, not '
                        'normalized. Neither path advances a trajectory across timing repetitions. These '
                        'measurements are CPU process timings, not full-model or GPU performance.',
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repetitions', type=int, default=500)
    parser.add_argument('--trials', type=int, default=5)
    parser.add_argument('--single-case', action='store_true', help='time first curated reference case; exact label recorded')
    args = parser.parse_args()
    if min(args.repetitions, args.trials) < 1:
        parser.error('repetitions and trials must be positive')
    print(json.dumps(benchmark(args.repetitions, args.trials, args.single_case), indent=2))

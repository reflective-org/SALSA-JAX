# Copyright 2026 Reflective
# SPDX-License-Identifier: Apache-2.0
"""Equal-work CPU timing of native seven-mode size initialization."""
import argparse
import hashlib
import json
import os
import platform
import statistics
import tempfile
import time
from pathlib import Path

import jax
import jaxlib
import numpy as np

from salsa_jax.constants import PI
from salsa_jax.grid import SizeGrid
from salsa_jax.initialization import size_distribution
from tools.initialization_oracle import build_oracle, reference_cases, run_oracle

ROOT = Path(__file__).resolve().parents[1]


def benchmark(repetitions=500, trials=5, single_case=False):
    if min(repetitions, trials) < 1:
        raise ValueError('Positive repetitions and trials required')
    jax.config.update('jax_enable_x64', True)
    jax.config.update('jax_enable_compilation_cache', False)
    cases = reference_cases()
    if single_case:
        cases = {key: value[:1] for key, value in cases.items()}
    d = cases['dry_diameter']
    middle = (PI/6.0)*d**3
    grid = SizeGrid(cases['lower_volume'], cases['upper_volume'], d,
                    cases['upper_volume']/middle, cases['lower_volume']/middle)
    args = jax.device_put((cases['mode_number'], cases['dpg'], cases['sigma'], grid), jax.devices('cpu')[0])
    jax.block_until_ready(args)
    started = time.perf_counter()
    compiled = jax.jit(size_distribution).lower(*args).compile()
    compilation = time.perf_counter()-started
    actual = np.asarray(jax.block_until_ready(compiled(*args)))
    samples = {'jax': [], 'fortran': []}
    with tempfile.TemporaryDirectory(prefix='salsa-initialization-benchmark-') as directory:
        checked, checked_provenance = build_oracle(Path(directory)/'checked')
        optimized, provenance = build_oracle(Path(directory)/'optimized', checked=False)
        checked_state, _ = run_oracle(checked, cases)
        expected, timing = run_oracle(optimized, cases)
        np.testing.assert_array_equal(expected['number'], checked_state['number'])
        np.testing.assert_allclose(actual, expected['number'], rtol=3e-13, atol=0)
        for trial in range(trials):
            for kind in (('jax', 'fortran') if trial % 2 == 0 else ('fortran', 'jax')):
                if kind == 'jax':
                    started = time.perf_counter()
                    for _ in range(repetitions):
                        result = jax.block_until_ready(compiled(*args))
                    samples[kind].append((time.perf_counter()-started)/repetitions)
                else:
                    repeated, timing = run_oracle(optimized, cases, repetitions=repetitions)
                    np.testing.assert_array_equal(repeated['number'], expected['number'])
                    samples[kind].append(timing['kernel_wall_seconds']/repetitions)
    np.testing.assert_allclose(result, expected['number'], rtol=3e-13, atol=0)
    difference = np.abs(actual-expected['number'])
    relative = np.divide(difference, np.abs(expected['number']), out=np.zeros_like(difference), where=expected['number']!=0)
    files = ('src/salsa_jax/initialization.py','src/salsa_jax/constants.py',
             'tools/initialization_oracle.py','benchmarks/benchmark_initialization.py')
    return {'scope': 'Seven-mode size_distribution only, one native cell per case; no composition, HAM conversion or grid generation',
            'platform': platform.platform(), 'jax_version': jax.__version__, 'jaxlib_version': jaxlib.__version__, 'numpy_version': np.__version__,
            'thread_environment': {key: os.environ.get(key) for key in
                ('XLA_FLAGS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
                 'VECLIB_MAXIMUM_THREADS','JAX_NUM_THREADS','TF_NUM_INTRAOP_THREADS','TF_NUM_INTEROP_THREADS')},
            'device': str(jax.devices('cpu')[0]), 'dtype': 'float64', 'cases': len(cases['labels']),
            'case_labels': cases['labels'], 'repetitions': repetitions, 'trials': trials,
            'jax_compile_seconds': compilation, 'samples_seconds_per_batch': samples,
            'median_seconds_per_batch': {key: statistics.median(value) for key, value in samples.items()},
            'median_fortran_over_jax': statistics.median(samples['fortran'])/statistics.median(samples['jax']),
            'parity': {'rtol': 3e-13, 'atol': 0, 'max_absolute': float(difference.max()),
                       'max_relative_nonzero': float(relative.max()), 'checked_optimized_exact': True},
            'number_checksums': {'jax': float(actual.sum()), 'fortran': float(expected['number'].sum())},
            'checked_provenance': checked_provenance, 'optimized_provenance': provenance,
            'implementation_sha256': {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in files},
            'case_sha256': {key: hashlib.sha256(np.asarray(value).tobytes()).hexdigest() for key,value in cases.items() if key!='labels'},
            'timing_notes': 'Native timing includes geometry assignment and output zeroing; modal arrays are passed as existing slices. JAX inputs are immutable. '
                            'JAX dispatch and all output synchronization included. Compile/transfers, native startup/I/O and '
                            'comparison work excluded. Alternating trial order; every repetition evaluates fixed inputs.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repetitions', type=int, default=500)
    parser.add_argument('--trials', type=int, default=5)
    parser.add_argument('--single-case', action='store_true')
    args = parser.parse_args()
    if min(args.repetitions, args.trials) < 1:
        parser.error('Positive repetitions and trials required')
    print(json.dumps(benchmark(args.repetitions, args.trials, args.single_case), indent=2))

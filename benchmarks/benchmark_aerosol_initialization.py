# Copyright 2026 Reflective
# SPDX-License-Identifier: Apache-2.0
"""Equal-work CPU timing of native SALSA conc2mmr aerosol initialization."""
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

from salsa_jax.aerosol_initialization import initialize_aerosol
from tools.aerosol_initialization_oracle import build_oracle,reference_cases,run_oracle
from tools.ham_conversion_fixture import aerosol_arguments

ROOT=Path(__file__).resolve().parents[1]
OUTPUTS=('number','mass','tracers')


def validate_outputs(actual,expected):
    """Accept every native output, including untouched and duplicate-target tracers."""
    metrics={}
    for field in OUTPUTS:
        value=np.asarray(actual[field] if isinstance(actual,dict) else getattr(actual,field))
        reference=np.asarray(expected[field])
        assert value.shape==reference.shape,field
        assert value.dtype==np.float64,field
        assert np.all(np.isfinite(value)),field
        np.testing.assert_allclose(value,reference,rtol=3e-13,atol=0,err_msg=field)
        difference=np.abs(value-reference)
        relative=np.divide(difference,np.abs(reference),out=np.zeros_like(difference),where=reference!=0)
        metrics[field]={'max_absolute':float(difference.max()),'max_relative_nonzero':float(relative.max()),
                        'nonzero_at_zero_reference':int(np.count_nonzero(value[reference==0])),
                        'negative_jax_values':int(np.count_nonzero(value<0)),
                        'negative_fortran_values':int(np.count_nonzero(reference<0))}
    return metrics


def _checksums(expected,timing):
    result={}
    for field,key in (('number','number_checksum'),('mass','mass_checksum'),('tracers','tracer_checksum')):
        values=np.asarray(expected[field]);native=float(timing[key]);host=float(values.sum())
        # Native SUM is sequential in column order; NumPy uses pairwise sums.
        bound=values.size*np.finfo(float).eps*float(np.abs(values).sum())
        assert abs(native-host)<=bound,f'{field} checksum does not cover the output workload'
        result[field]={'native_reported':native,'numpy_sum':host}
    return result


def benchmark(repetitions=500,trials=5,single_case=False):
    for name,value in (('repetitions',repetitions),('trials',trials)):
        if isinstance(value,bool) or not isinstance(value,(int,np.integer)) or value<1:
            raise ValueError(f'{name} must be a positive integer')
    jax.config.update('jax_enable_x64',True)
    jax.config.update('jax_enable_compilation_cache',False)
    cases=reference_cases()
    if single_case:cases={key:value[:1] for key,value in cases.items()}
    args,kwargs=aerosol_arguments(cases)
    device=jax.devices('cpu')[0]
    args,kwargs=jax.device_put((args,kwargs),device)
    jax.block_until_ready((args,kwargs))
    started=time.perf_counter()
    compiled=jax.jit(initialize_aerosol).lower(*args,**kwargs).compile()
    compilation=time.perf_counter()-started
    actual=jax.block_until_ready(compiled(*args,**kwargs))
    samples={'jax':[],'fortran':[]}
    with tempfile.TemporaryDirectory(prefix='salsa-aerosol-initialization-benchmark-') as directory:
        checked,checked_provenance=build_oracle(Path(directory)/'checked')
        optimized,optimized_provenance=build_oracle(Path(directory)/'optimized',checked=False)
        checked_state,_=run_oracle(checked,cases)
        expected,timing=run_oracle(optimized,cases)
        for field in OUTPUTS:np.testing.assert_array_equal(expected[field],checked_state[field])
        parity=validate_outputs(actual,expected)
        checksums=_checksums(expected,timing)
        for trial in range(trials):
            for kind in (('jax','fortran') if trial%2==0 else ('fortran','jax')):
                if kind=='jax':
                    started=time.perf_counter()
                    for _ in range(repetitions):
                        result=jax.block_until_ready(compiled(*args,**kwargs))
                    samples[kind].append((time.perf_counter()-started)/repetitions)
                else:
                    repeated,timing=run_oracle(optimized,cases,repetitions=repetitions)
                    for field in OUTPUTS:np.testing.assert_array_equal(repeated[field],expected[field])
                    _checksums(repeated,timing)
                    samples[kind].append(timing['kernel_wall_seconds']/repetitions)
    validate_outputs(result,expected)
    files=('src/salsa_jax/aerosol_initialization.py','src/salsa_jax/constants.py',
           'src/salsa_jax/processes/hydration.py','tools/aerosol_initialization_oracle.py',
           'tools/ham_conversion_fixture.py','benchmarks/benchmark_aerosol_initialization.py')
    return {'scope':'SALSA conc2mmr aerosol initialization only; one native cell per case; all number, mass and tracer outputs retained. No size-distribution generation, gas conversion or full-model timing.',
            'case_selection':('First curated case: native number/core/maps/properties with synthetic air density 1.2 kg/m3 and zero initial tracers.' if single_case else 'Full curated matrix, including synthetic properties, fallback unit codes, runtime maps, duplicate targets and sentinel tracers.'),
            'platform':platform.platform(),'python_version':platform.python_version(),'jax_version':jax.__version__,
            'jaxlib_version':jaxlib.__version__,'numpy_version':np.__version__,'device':str(device),'dtype':'float64',
            'thread_environment':{key:os.environ.get(key) for key in ('XLA_FLAGS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','JAX_NUM_THREADS','TF_NUM_INTRAOP_THREADS','TF_NUM_INTEROP_THREADS')},
            'cases':len(cases['labels']),'case_labels':cases['labels'],'native_cells_per_case':1,
            'tracer_width':cases['tracers'].shape[-1],'repetitions':int(repetitions),'trials':int(trials),
            'jax_compile_seconds':compilation,'samples_seconds_per_batch':samples,
            'median_seconds_per_batch':{key:statistics.median(value) for key,value in samples.items()},
            'median_fortran_over_jax':statistics.median(samples['fortran'])/statistics.median(samples['jax']),
            'parity':{'relative_tolerance':3e-13,'absolute_tolerance':0,'checked_optimized_exact':True,'fields':parity},
            'checksums':{field:{**checksums[field],'jax_numpy_sum':float(np.asarray(getattr(actual,field)).sum())} for field in OUTPUTS},
            'units':{'number':'particles/cm3','mass':'SO4 molecules/cm3; other materials ug/m3','tracers':'Native mass/number mixing ratios; untouched sentinel slots preserved'},
            'checked_provenance':checked_provenance,'optimized_provenance':optimized_provenance,
            'implementation_sha256':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in files},
            'case_sha256':{key:hashlib.sha256(np.asarray(value).tobytes()).hexdigest() for key,value in cases.items() if key!='labels'},
            'case_shapes':{key:list(np.shape(value)) for key,value in cases.items() if key!='labels'},
            'timing_notes':'Native timed calls include registry/map/property setup, mutable input copies and mass-output initialization. JAX inputs are immutable, and ordered tracer scatters are part of its kernel. Each repetition evaluates the same initial state; it does not feed particles/cm3 back as particles/m3. All three JAX outputs are synchronized, including dispatch cost. Compilation and initial device transfers, native startup/parsing/output, checksums and comparisons are excluded. Native kernels are compiled separately without LTO. Trial order alternates; Fortran loops serially over cases while JAX batches the same cases.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repetitions',type=int,default=500)
    parser.add_argument('--trials',type=int,default=5)
    parser.add_argument('--single-case',action='store_true')
    options=parser.parse_args()
    if min(options.repetitions,options.trials)<1:parser.error('Positive repetitions and trials required')
    print(json.dumps(benchmark(options.repetitions,options.trials,options.single_case),indent=2))

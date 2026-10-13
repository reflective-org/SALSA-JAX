"""Lightweight output/accounting checks; final timing runs are separate."""
import numpy as np
import pytest
from benchmarks.benchmark_aerosol_initialization import OUTPUTS,_checksums,benchmark,validate_outputs


def state():
    return {key:np.ones((2,width),dtype=np.float64) for key,width in (('number',17),('mass',69),('tracers',109))}


@pytest.mark.parametrize('field',OUTPUTS)
def test_parity_checks_every_output(field):
    expected=state();actual={key:value.copy() for key,value in expected.items()}
    expected[field][0,0]=0.;actual[field][0,0]=1e-100
    with pytest.raises(AssertionError):validate_outputs(actual,expected)


def test_negative_untouched_tracers_are_preserved():
    expected=state();expected['tracers'][0,0]=-7.
    metrics=validate_outputs(expected,expected)
    assert metrics['tracers']['negative_jax_values']==1
    assert metrics['tracers']['negative_fortran_values']==1
    assert set(metrics)==set(OUTPUTS)


def test_native_checksum_covers_all_cases_and_outputs():
    expected=state();timing={'number_checksum':34.,'mass_checksum':138.,'tracer_checksum':218.}
    assert set(_checksums(expected,timing))==set(OUTPUTS)
    timing['tracer_checksum']/=2
    with pytest.raises(AssertionError,match='tracers checksum'):_checksums(expected,timing)


@pytest.mark.parametrize('repetitions,trials',[(0,1),(1,0),(1.5,1),(True,1),(1,False)])
def test_invalid_counts_fail_before_compilation(repetitions,trials):
    with pytest.raises(ValueError):benchmark(repetitions,trials)

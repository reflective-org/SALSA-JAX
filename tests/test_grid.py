# Copyright 2026 Reflective
# SPDX-License-Identifier: Apache-2.0
"""Fixed native grid, including default-real spacing exponents, against Fortran."""
import hashlib
from pathlib import Path

import jax
import numpy as np
import pytest

from salsa_jax.constants import PI
from salsa_jax.grid import DEFAULT_REGIME_LIMITS, SizeGrid, size_grid
from tools.grid_oracle import FIELDS, build_oracle, run_oracle
from tools.state_capture import load_fixture

jax.config.update('jax_enable_x64', True)
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def reference(tmp_path_factory):
    directory = tmp_path_factory.mktemp('native-grid')
    executable,provenance = build_oracle(directory)
    limits = np.array([DEFAULT_REGIME_LIMITS, [2e-9,4e-8,1e-6,2e-5],
                       [4e-9,8e-8,8e-7,8e-6], [3e-9,3.1e-9,3.2e-9,3.3e-9]])
    return directory,executable,provenance,limits,run_oracle(executable,limits)


def test_oracle_uses_actual_routine_configuration_and_checked_build(reference):
    directory,_,provenance,_,_ = reference
    routine = (directory/'set_sizebins_verbatim.inc').read_text().rstrip('\n')
    assert routine in (ROOT/'mo_ham_salsa_init.F90').read_text()
    assert hashlib.sha256(routine.encode()).hexdigest() == provenance['routine_sha256']
    config = (directory/'grid_configuration_verbatim.inc').read_text().rstrip('\n')
    assert config in (ROOT/'mo_ham_salsactl.F90').read_text()
    assert hashlib.sha256(config.encode()).hexdigest() == provenance['native_configuration_sha256']
    assert '-fcheck=all' in provenance['flags']
    assert '-ffpe-trap=invalid,zero,overflow' in provenance['flags']
    assert provenance['configuration'] == {
        'nreg':2,'nbin':[3,7],'nbin2':4,'nbin3':3,'boundaries':[1,3,4,10,11,17],
        'default_real_bits':32,'dp_bits':64,'reglim':list(DEFAULT_REGIME_LIMITS),'pi':PI}


@pytest.mark.parametrize('compiled',[False,True])
def test_grid_matches_checked_fortran(reference,compiled):
    _,_,_,limits,expected = reference
    actual = (jax.jit(size_grid) if compiled else size_grid)(limits)
    assert isinstance(actual,SizeGrid)
    for field in FIELDS:
        values=np.asarray(getattr(actual,field))
        assert values.shape == (4,17)
        assert values.dtype == np.float64
        assert np.all(np.isfinite(values))
        np.testing.assert_allclose(values,expected[field],rtol=3e-14,atol=0)


def test_default_grid_matches_actual_full_model_capture(reference):
    _,_,_,_,expected = reference
    fixture = load_fixture(ROOT/'tests/fixtures/native-default')['entry']
    np.testing.assert_array_equal(fixture['reglim']['values'],DEFAULT_REGIME_LIMITS)
    actual = jax.jit(size_grid)()
    for field in FIELDS:
        captured = np.asarray(fixture[field]['values'])
        np.testing.assert_array_equal(expected[field][0],captured)
        np.testing.assert_allclose(getattr(actual,field),captured,rtol=3e-14,atol=0)


def test_native_bounds_midpoint_and_duplicate_regimes(reference):
    _,_,_,limits,_ = reference
    actual = size_grid(limits)
    assert np.all(np.asarray(actual.vhilim) > np.asarray(actual.vlolim))
    for field in actual:
        np.testing.assert_array_equal(np.asarray(field)[...,3:10],np.asarray(field)[...,10:17])
    # Native independent interval formulas leave floating-point roundoff gaps
    # at shared regime boundaries; preserve them instead of snapping endpoints.
    np.testing.assert_allclose(actual.vhilim[...,:9],actual.vlolim[...,1:10],rtol=1e-15,atol=0)
    assert np.all(np.diff(np.asarray(actual.dpmid)[...,:10],axis=-1)>0)
    assert np.all(np.diff(np.asarray(actual.dpmid)[...,10:17],axis=-1)>0)
    center = PI/6*np.asarray(actual.dpmid)**3
    np.testing.assert_allclose(center,(actual.vlolim+actual.vhilim)/2,rtol=1e-14,atol=0)
    np.testing.assert_allclose(actual.vratiohi+actual.vratiolo,2.,rtol=1e-14,atol=0)
    assert np.all(np.asarray(actual.vratiolo)<1)
    assert np.all(np.asarray(actual.vratiohi)>1)


def test_default_real_exponent_rounding_is_observable(reference):
    _,_,_,limits,expected = reference
    # A float64-only transcription would silently shift 1a's one-third edge.
    ratio=limits[0,1]/limits[0,0]
    wrong=PI/6*(limits[0,0]*ratio**(1./3.))**3
    correct=expected['vhilim'][0,0]
    assert abs(wrong/correct-1)>1e-8
    np.testing.assert_allclose(size_grid().vhilim[0],correct,rtol=3e-14,atol=0)


def test_multiple_leading_axes_and_fresh_reference_repeat(reference):
    _,executable,_,limits,expected=reference
    many=limits.reshape(2,2,4)
    actual=jax.jit(size_grid)(many)
    repeated=run_oracle(executable,many)
    for field in FIELDS:
        assert getattr(actual,field).shape==(2,2,17)
        np.testing.assert_array_equal(repeated[field].reshape(4,17),expected[field])
        np.testing.assert_allclose(getattr(actual,field),repeated[field],rtol=3e-14,atol=0)


@pytest.mark.parametrize('limits',[[],[1,2,3],np.ones((0,4)),[0,1,2,3],[1,1,2,3],
                                    [4,3,2,1],[1,2,np.nan,4],[1,2,3,np.inf]])
def test_oracle_rejects_invalid_grid_before_execution(limits):
    with pytest.raises(ValueError,match='regime_limits|Regime limits'):
        run_oracle('must-not-run',limits)


def test_jax_rejects_wrong_regime_limit_shape():
    with pytest.raises(ValueError,match='regime diameter limits'):
        size_grid([1,2,3])

# Copyright 2026 Reflective
# SPDX-License-Identifier: Apache-2.0
"""Independent native seven-mode midpoint quadrature and initialization loops."""
import numpy as np
import pytest
from tools.initialization_oracle import (ROOT,build_oracle,driver_defaults,
                                         reference_cases,run_oracle,load_fixture)


@pytest.fixture(scope='module')
def oracle(tmp_path_factory):
    directory=tmp_path_factory.mktemp('initialization-native')
    executable,provenance=build_oracle(directory)
    cases=reference_cases();outputs,timing=run_oracle(executable,cases)
    return directory,executable,provenance,cases,outputs,timing


def test_scientific_module_byte_identical_and_native_defaults(oracle):
    directory,_,provenance,_,_,_=oracle
    name='mo_ham_salsa_sizedist.F90'
    assert (directory/name).read_bytes()==(ROOT/name).read_bytes()
    assert provenance['scientific_module_byte_identical']==name
    assert '-fcheck=all' in provenance['flags']
    assert '-ffpe-trap=invalid,zero,overflow' in provenance['flags']
    for key,value in driver_defaults().items():
        np.testing.assert_array_equal(value,provenance['compiled_defaults'][key])
    assert driver_defaults()['sigma'][0]==float(np.float32(1.59))
    assert driver_defaults()['sigma'][0]!=1.59
    assert driver_defaults()['dpg'][0]==float(np.float32(.01))*1e-6
    assert driver_defaults()['mode_number'][3]==float(np.float32(.001))*1e6


def test_checked_optimized_repetitions_and_timing(oracle,tmp_path):
    _,executable,_,cases,expected,_=oracle
    repeated,timing=run_oracle(executable,cases,repetitions=4)
    fast,provenance=build_oracle(tmp_path,checked=False)
    optimized,_=run_oracle(fast,cases)
    assert not provenance['checked']
    assert '-fcheck=all' not in provenance['flags']
    np.testing.assert_array_equal(repeated['number'],expected['number'])
    np.testing.assert_array_equal(optimized['number'],expected['number'])
    assert np.all(np.isfinite(expected['number']))
    assert np.all(expected['number']>=0)
    np.testing.assert_allclose(timing['number_checksum'],expected['number'].sum(),rtol=3e-15,atol=0)
    assert timing['kernel_wall_seconds']>0
    assert timing['input_reset_included'] and timing['geometry_setup_included']


def test_default_matches_actual_initial_native_number(oracle):
    _,_,_,_,outputs,_=oracle
    fixture=load_fixture(ROOT/'tests/fixtures/native-default')
    raw=fixture['entry']['pnaero']
    number=np.asarray(raw['values']).reshape(raw['shape'],order='F')[0,0]
    np.testing.assert_allclose(outputs['number'][0],number,rtol=3e-15,atol=0)


def test_empty_modes_superposition_and_count_scaling(oracle):
    _,_,_,cases,outputs,_=oracle
    n=outputs['number'];index=cases['labels'].index
    np.testing.assert_array_equal(n[index('empty')],0.)
    isolated=np.stack([n[index(f'isolated-mode{i+1}')] for i in range(7)])
    np.testing.assert_allclose(isolated.sum(0),n[0],rtol=3e-15,atol=0)
    for name,scale in (('tiny-mode-counts',1e-12),('large-mode-counts',1e8)):
        np.testing.assert_allclose(n[index(name)],n[0]*scale,rtol=3e-15,atol=0)
    # Native uses all seven modes for both composition regimes with equal
    # geometry. Do not impose a modal-composition partition or normalization.
    np.testing.assert_array_equal(n[0,3:10],n[0,10:17])


def test_active_horizontal_and_vertical_loops_leave_inactive_cells_zero(oracle):
    _,executable,_,cases,expected,_=oracle
    all_cells,timing=run_oracle(executable,cases,kproma=2,kbdim=3,klev=2,all_cells=True)
    number=all_cells['number']
    assert number.shape==(len(cases['labels']),3,2,17)
    for horizontal in (0,1):
        for level in (0,1):np.testing.assert_array_equal(number[:,horizontal,level],expected['number'])
    np.testing.assert_array_equal(number[:,2],0.)
    np.testing.assert_allclose(timing['number_checksum'],4*expected['number'].sum(),rtol=3e-15,atol=0)
    inactive,_=run_oracle(executable,cases,kproma=0,kbdim=3,klev=2,all_cells=True)
    np.testing.assert_array_equal(inactive['number'],0.)


def test_scalar_and_broadcast_batch_layout(oracle):
    _,executable,_,cases,expected,_=oracle
    scalar={key:value[0] for key,value in cases.items() if key!='labels'}
    out,_=run_oracle(executable,scalar)
    assert out['number'].shape==(17,)
    np.testing.assert_array_equal(out['number'],expected['number'][0])
    batched={**scalar,'mode_number':np.broadcast_to(scalar['mode_number'],(2,3,7))}
    out,_=run_oracle(executable,batched)
    assert out['number'].shape==(2,3,17)
    np.testing.assert_array_equal(out['number'],np.broadcast_to(expected['number'][0],(2,3,17)))


@pytest.mark.parametrize('field',['mode_number','dpg','sigma','lower_volume','upper_volume','dry_diameter'])
def test_invalid_physical_inputs_rejected(field):
    cases=reference_cases();cases[field]=np.full_like(cases[field],-1.)
    with pytest.raises(ValueError,match=field):run_oracle('must-not-run',cases)


def test_zero_width_and_unit_sigma_rejected():
    cases=reference_cases()
    with pytest.raises(ValueError,match='Volume limits'):
        run_oracle('must-not-run',{**cases,'upper_volume':cases['lower_volume']})
    with pytest.raises(ValueError,match='sigma'):
        run_oracle('must-not-run',{**cases,'sigma':np.ones_like(cases['sigma'])})


@pytest.mark.parametrize('options',[{'kproma':2},{'kbdim':0},{'klev':0},{'repetitions':0},{'kproma':0}])
def test_invalid_loop_dimensions_rejected(options):
    with pytest.raises(ValueError):run_oracle('must-not-run',reference_cases(),**options)

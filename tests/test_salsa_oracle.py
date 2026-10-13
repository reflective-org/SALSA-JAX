# Copyright 2026 Reflective
# SPDX-License-Identifier: Apache-2.0
"""Whole original SALSA call independently validates its native process order."""
import subprocess
import numpy as np
import pytest
from tools.salsa_oracle import (ROOT,PROCESS_SOURCES,OUTPUTS,build_oracle,native_case,
                                reference_cases,record,run_oracle,load_fixture)


@pytest.fixture(scope='module')
def oracle(tmp_path_factory):
    directory=tmp_path_factory.mktemp('whole-salsa')
    executable,provenance=build_oracle(directory)
    cases=reference_cases()
    outputs,timing=run_oracle(executable,cases,all_levels=True)
    return directory,executable,provenance,cases,outputs,timing


def test_entire_scientific_modules_are_unmodified(oracle):
    directory,_,provenance,_,_,_=oracle
    for name in PROCESS_SOURCES:
        assert (directory/name).read_bytes()==(ROOT/name).read_bytes()
    assert provenance['scientific_modules_byte_identical']==list(PROCESS_SOURCES)
    assert '-fcheck=all' in provenance['flags']
    assert provenance['compiled_constants']['gas_constant']==8.314472


def test_complete_call_matches_actual_captured_default(oracle):
    _,_,_,_,outputs,_=oracle
    fixture=load_fixture(ROOT/'tests/fixtures/native-default')
    names=('paerml','pnaero','pc_gas','pm6rp','pm6dry','prhop','pww')
    for key,name in zip(OUTPUTS,names):
        expected=record(fixture,'exit',name)[0,0]
        for level in (0,1):
            actual=outputs[key][0,level,:len(expected)]
            np.testing.assert_allclose(actual,expected,rtol=3e-13,atol=0)


def test_checked_optimized_and_repeated_resets_are_identical(oracle,tmp_path):
    _,exe,_,cases,expected,timing=oracle
    repeat,_=run_oracle(exe,cases,repetitions=3,all_levels=True)
    optimized,provenance=build_oracle(tmp_path,checked=False)
    fast,_=run_oracle(optimized,cases,all_levels=True)
    assert not provenance['checked']
    for key in OUTPUTS:
        np.testing.assert_array_equal(repeat[key],expected[key])
        np.testing.assert_array_equal(fast[key],expected[key])
    for key in ('mass','number','gas'):
        np.testing.assert_allclose(timing[key+'_checksum'],expected[key].sum(),rtol=3e-15,atol=0)
    assert timing['native_cells_per_case']==2
    assert timing['boundary_layer_by_native_level']==[False,True]
    assert timing['input_reset_included'] and timing['grid_setup_included']


def test_selected_level_and_two_step_state_propagation(oracle):
    _,exe,_,cases,all_outputs,_=oracle
    selected,_=run_oracle(exe,cases)
    levels=cases['boundary_layer'].astype(int)
    for key in OUTPUTS:
        np.testing.assert_array_equal(selected[key],all_outputs[key][np.arange(len(levels)),levels])
    # Scheme0 is independent of the other level; verify actual chained state,
    # separately from the driver's repetition/reset timing loop.
    i=cases['labels'].index('captured-default')
    one={k:v[i:i+1] for k,v in cases.items() if k!='labels'}
    first,_=run_oracle(exe,one)
    twice,_=run_oracle(exe,one,steps=2)
    advanced={**one,**{k:first[k] for k in ('mass','number','gas')}}
    sequential,_=run_oracle(exe,advanced)
    for key in OUTPUTS:np.testing.assert_array_equal(twice[key],sequential[key])
    assert not np.array_equal(first['number'],twice['number'])


def test_runtime_component_permutation_preserves_physical_result(oracle):
    _,exe,_,cases,_,_=oracle
    original={k:v[:1].copy() for k,v in cases.items() if k!='labels'}
    expected,_=run_oracle(exe,original)
    permutation=np.random.default_rng(187).permutation(69)
    permuted={k:v.copy() for k,v in original.items()}
    present=permuted['component_map']>=0
    permuted['component_map'][present]=permutation[permuted['component_map'][present]]
    permuted['mass'][:,permutation]=original['mass']
    actual,_=run_oracle(exe,permuted)
    np.testing.assert_array_equal(actual['mass'][:,permutation],expected['mass'])
    for key in OUTPUTS:
        if key!='mass':np.testing.assert_array_equal(actual[key],expected[key])


def test_flags_and_nucleation_have_observable_effects(oracle):
    _,_,_,cases,output,_=oracle
    get=lambda label:cases['labels'].index(label)
    assert not np.array_equal(output['number'][get('flags-coag0-cond0-recalc0')],output['number'][get('flags-coag1-cond0-recalc0')])
    assert not np.array_equal(output['mass'][get('flags-coag0-cond0-recalc0')],output['mass'][get('flags-coag0-cond1-recalc0')])
    assert not np.array_equal(output['gas'][get('flags-coag1-cond1-recalc0')],output['gas'][get('flags-coag1-cond1-recalc1')])
    assert not np.array_equal(output['wet_radius_cm'][get('captured-default')],output['wet_radius_cm'][get('unhydrated-insoluble')])
    i=get('active-scheme2-growth1')
    assert output['number'][i,1,0]>output['number'][i,0,0]


def test_initial_tiny_cleanup_restores_registered_vapors(oracle):
    _,_,_,cases,output,_=oracle
    for label in ('initial-tiny-cleanup-bin1','initial-tiny-cleanup-bin4','initial-tiny-cleanup-bin11'):
        i=cases['labels'].index(label)
        assert not output['number'][i].any()
        assert not output['mass'][i].any()
        assert np.all(output['gas'][i,:,1:]>0)
        sulfate=cases['mass'][i,:17].sum()*1e6
        np.testing.assert_allclose(output['gas'][i,:,1],sulfate,rtol=3e-15,atol=0)


def test_initial_cleanup_retains_native_D007_abort(oracle):
    _,exe,_,cases,_,_=oracle
    i=cases['labels'].index('initial-tiny-cleanup-bin1')
    one={k:v[i:i+1].copy() for k,v in cases.items() if k!='labels'}
    one['organic_enabled'][:]=False
    with pytest.raises(subprocess.CalledProcessError) as exc:
        run_oracle(exe,one)
    assert 'below lower bound of 1' in exc.value.stderr


@pytest.mark.parametrize('field',['mass','number','gas','temperature','scheme','component_map'])
def test_invalid_inputs_rejected(field):
    cases=reference_cases();cases[field]=np.full_like(cases[field],-1)
    with pytest.raises(ValueError):run_oracle('must-not-execute',cases)

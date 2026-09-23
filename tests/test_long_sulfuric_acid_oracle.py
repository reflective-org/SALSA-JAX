"""Native checkpoints retain complete scientific calls and explicit reservoirs."""
import numpy as np
import pytest

from tools import salsa_oracle as salsa
from tools import long_sulfuric_acid_oracle as long


@pytest.fixture(scope='module')
def native(tmp_path_factory):
    path = tmp_path_factory.mktemp('long-native')
    executable, provenance = long.build_oracle(path/'checked')
    optimized, _ = long.build_oracle(path/'optimized', checked=False)
    cases = salsa.reference_cases()
    indexes = [cases['labels'].index(name) for name in
               ('captured-default', 'active-scheme2-growth1')]
    cases = {key: np.asarray(value)[indexes].copy() for key,value in cases.items() if key != 'labels'}
    return executable, optimized, provenance, cases, path/'checked'/'salsa_oracle'


def test_closed_checkpoints_match_complete_original_calls(native):
    executable, _, provenance, cases, parent = native
    history, timing = long.run_oracle(executable, cases, steps=6, save_every=2, all_levels=True)
    assert provenance['scientific_modules_byte_identical'] == list(salsa.PROCESS_SOURCES)
    assert '-fcheck=all' in provenance['flags']
    assert timing['saved_steps'] == [2,4,6]
    for index,steps in enumerate((2,4,6)):
        expected,_ = salsa.run_oracle(parent,cases,steps=steps,all_levels=True)
        for key in salsa.OUTPUTS:
            np.testing.assert_array_equal(history[key][index],expected[key])
    np.testing.assert_array_equal(history['cumulative_reservoir_added'],0.)
    np.testing.assert_array_equal(history['h2so4_before_restore'],history['gas'][...,1])


def test_constant_reservoir_matches_manual_original_replay(native):
    executable, _, _, cases, parent = native
    cases = {key:value[1:2].copy() for key,value in cases.items()}
    history,timing = long.run_oracle(executable,cases,steps=2,constant_h2so4=True)
    state = {key:value.copy() for key,value in cases.items()}
    added = np.zeros(1)
    for index in range(2):
        expected,_ = salsa.run_oracle(parent,state)
        before = expected['gas'][:,1].copy()
        added += cases['gas'][:,1]-before
        expected['gas'][:,1] = cases['gas'][:,1]
        for key in salsa.OUTPUTS:
            np.testing.assert_array_equal(history[key][index],expected[key])
        np.testing.assert_array_equal(history['h2so4_before_restore'][index],before)
        np.testing.assert_array_equal(history['cumulative_reservoir_added'][index],added)
        state.update({key:expected[key] for key in ('mass','number','gas')})
    assert timing['every_step_finite']
    assert timing['max_relative_h2so4_depletion'][0] > 0


@pytest.mark.parametrize('constant',[False,True])
def test_stride_and_checked_optimized_are_exact(native,constant):
    executable, optimized, _, cases, _ = native
    full,_ = long.run_oracle(executable,cases,steps=8,constant_h2so4=constant,all_levels=True)
    sparse,checked_timing = long.run_oracle(executable,cases,steps=8,save_every=4,constant_h2so4=constant,all_levels=True)
    fast,fast_timing = long.run_oracle(optimized,cases,steps=8,save_every=4,constant_h2so4=constant,all_levels=True)
    selected,_ = long.run_oracle(executable,cases,steps=8,save_every=4,constant_h2so4=constant)
    for key in full:
        np.testing.assert_array_equal(sparse[key],full[key][3::4])
        np.testing.assert_array_equal(sparse[key],fast[key])
        np.testing.assert_array_equal(selected[key],sparse[key][:,np.arange(2),cases['boundary_layer'].astype(int)])
    for key in long.DIAGNOSTICS:
        np.testing.assert_array_equal(checked_timing[key],fast_timing[key])


@pytest.mark.parametrize('changes',[
    {'steps':0}, {'save_every':0}, {'save_every':True}, {'save_every':1.5},
    {'steps':5,'save_every':2}, {'constant_h2so4':1},
])
def test_invalid_checkpoint_configuration_rejected(changes):
    cases = salsa.reference_cases()
    args = {'steps':4,'save_every':2,**changes}
    with pytest.raises(ValueError):
        long.run_oracle('must-not-execute',cases,**args)


def test_reservoir_removal_is_signed_and_budgeted(native):
    executable, _, _, _, parent = native
    cases = salsa.reference_cases()
    index = cases['labels'].index('initial-tiny-cleanup-bin1')
    cases = {key:np.asarray(value)[index:index+1].copy()
             for key,value in cases.items() if key != 'labels'}
    native_output,_ = salsa.run_oracle(parent,cases)
    output,timing = long.run_oracle(executable,cases,steps=1,constant_h2so4=True)
    assert native_output['gas'][0,1] > 0
    np.testing.assert_array_equal(output['cumulative_reservoir_added'][0],-native_output['gas'][:,1])
    np.testing.assert_array_equal(output['gas'][...,1],0.)
    assert timing['max_relative_sulfur_residual'][0] < 1e-14

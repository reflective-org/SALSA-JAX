"""Actual distribution-update behavior, including source artifacts and failures."""
import subprocess

import numpy as np
import pytest

from tools.remapping_oracle import ROOT, _extract, build_oracle, reference_cases, run_oracle


@pytest.fixture(scope='module')
def oracle(tmp_path_factory):
    directory=tmp_path_factory.mktemp('remapping-native')
    executable,provenance=build_oracle(directory)
    cases=reference_cases()
    outputs,timing=run_oracle(executable,cases)
    return directory,executable,provenance,cases,outputs,timing


def test_actual_extraction_and_checked_build(oracle):
    directory,_,provenance,_,_,_=oracle
    expected=_extract((ROOT/'mo_ham_salsa_update.F90').read_text(),'SUBROUTINE','distr_update')
    assert (directory/'distr_update_verbatim.inc').read_text()==expected+'\n'
    assert '-fcheck=all' in provenance['flags']
    assert provenance['separate_compilation_without_lto']
    assert provenance['configuration']['native_loop_uncapped']
    assert provenance['compiled_constants']['gas_constant']==8.314472


def test_checked_optimized_repetitions_and_conservation(oracle,tmp_path):
    _,executable,_,cases,expected,_=oracle
    repeated,timing=run_oracle(executable,cases,repetitions=3)
    optimized,provenance=build_oracle(tmp_path,checked=False)
    fast,_=run_oracle(optimized,cases)
    assert not provenance['checked']
    for field in expected:
        np.testing.assert_array_equal(repeated[field],expected[field])
        np.testing.assert_array_equal(fast[field],expected[field])
    np.testing.assert_allclose(expected['number'].sum(axis=1),cases['number'].sum(axis=1),rtol=3e-15,atol=0)
    np.testing.assert_allclose(expected['volume'].sum(axis=1),cases['volume'].sum(axis=1),rtol=3e-15,atol=0)
    for field in ('number','volume'):
        np.testing.assert_allclose(timing[field+'_checksum'],expected[field].sum(),rtol=3e-15,atol=0)
    assert not timing['iteration_capture_included']
    assert timing['kernel_wall_seconds']>0


def test_terminal_bins_and_number_threshold_are_inactive(oracle):
    _,_,_,cases,outputs,_=oracle
    labels=['ignored-terminal-small-bin10','ignored-terminal-large-bin10','ignored-terminal-small-bin17',
            'ignored-terminal-large-bin17','below-number-threshold','at-number-threshold',
            'inactive-number-positive-volume','zero-volume-first-bin']
    for label in labels:
        index=cases['labels'].index(label)
        for field in ('number','volume'):
            np.testing.assert_array_equal(outputs[field][index],cases[field][index])
    index=cases['labels'].index('above-number-threshold')
    assert outputs['number'][index,4]>0


def test_below_first_insoluble_bin_moves_to_last_regime1_bin(oracle):
    _,_,_,cases,outputs,_=oracle
    index=cases['labels'].index('below2b-to-regime1-no-regrowth')
    assert outputs['number'][index,2]==cases['number'][index,10]
    assert not outputs['number'][index,10]
    assert not outputs['number'][index,9]
    np.testing.assert_array_equal(outputs['volume'][index,2],cases['volume'][index,10])


def test_optional_soluble_transfer_preserves_whole_bin(oracle):
    _,_,_,cases,outputs,_=oracle
    for label in ('soluble-transfer-sulfate','soluble-transfer-organic'):
        index=cases['labels'].index(label)
        assert outputs['number'][index,9]==cases['number'][index,16]
        assert outputs['number'][index,16]==0
        np.testing.assert_array_equal(outputs['volume'][index,9],cases['volume'][index,16])
    for label in ('soluble-transfer-black-carbon','soluble-transfer-disabled'):
        index=cases['labels'].index(label)
        for field in ('number','volume'):
            np.testing.assert_array_equal(outputs[field][index],cases[field][index])


def test_soluble_fraction_threshold_is_strict_at_adjacent_floats(oracle):
    _,_,_,cases,outputs,_=oracle
    for prefix in ('below','at','above'):
        index=cases['labels'].index(prefix+'-soluble-fraction-threshold')
        if prefix=='above':
            assert outputs['number'][index,16]==0
            assert outputs['number'][index,9]==16
        else:
            for field in ('number','volume'):
                np.testing.assert_array_equal(outputs[field][index],cases[field][index])


def test_native_full_number_transfer_leaves_orphan_volume_and_oversized_receiver(oracle):
    _,_,_,cases,outputs,_=oracle
    index=cases['labels'].index('full-number-transfer-orphan-volume')
    assert outputs['number'][index,3]==0
    assert outputs['volume'][index,3].sum()>0
    assert outputs['number'][index,4]==cases['number'][index,3]
    mean=outputs['volume'][index,4].sum()/outputs['number'][index,4]
    assert mean>cases['upper_volume'][index,4]
    assert outputs['iterations'][index,0]==1
    orphan_fraction=outputs['volume'][index,3].sum()/cases['volume'][index].sum()
    assert .07<orphan_fraction<.08


def test_explicit_passes_repeat_original_routine_not_independent_inputs(oracle):
    _,executable,_,cases,first,_=oracle
    twice,_=run_oracle(executable,cases,passes=2)
    advanced={**cases,'number':first['number'],'volume':first['volume']}
    sequential,_=run_oracle(executable,advanced)
    for field in ('number','volume'):
        np.testing.assert_array_equal(twice[field],sequential[field])
    index=cases['labels'].index('full-number-transfer-orphan-volume')
    assert twice['number'][index,5]>0
    assert first['number'][index,5]==0
    assert twice['volume'][index,3].sum()==first['volume'][index,3].sum()
    assert twice['iterations'].shape==(len(cases['labels']),2)


def test_soluble_active_zero_volume_terminal_bin_reproduces_native_failure(oracle):
    _,executable,_,cases,_,_=oracle
    case={key:value[:1].copy() for key,value in cases.items() if key!='labels'}
    case['number'][:]=0
    case['number'][0,16]=16
    case['volume'][:]=0
    case['soluble_transfer'][:]=True
    with pytest.raises(subprocess.CalledProcessError) as failure:
        run_oracle(executable,case)
    assert failure.value.returncode<0
    assert 'Program received signal' in failure.value.stderr


@pytest.mark.parametrize('field',['number','volume','temperature','lower_volume','ion_count'])
def test_invalid_values_rejected(field):
    cases=reference_cases()
    with pytest.raises(ValueError,match=field):
        run_oracle('must-not-run',{**cases,field:np.full_like(cases[field],-1.)})

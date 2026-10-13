# Copyright 2026 Reflective
# SPDX-License-Identifier: Apache-2.0
"""Complete JAX SALSA composition against the whole original Fortran call."""
import jax
import numpy as np
import pytest
from salsa_jax.constants import PI
from salsa_jax.model import STATIC_OPTIONS,SalsaState,salsa_step
from tools.salsa_inputs import arguments,select_cases,static_groups
from tools.salsa_oracle import OUTPUTS,build_oracle,reference_cases,run_oracle
from tools.salsa_fixture import load_native_case

jax.config.update('jax_enable_x64',True)
COMPILED=jax.jit(salsa_step,static_argnames=STATIC_OPTIONS)


def assert_outputs(actual,expected,cases,*,all_levels=False):
    assert np.all(np.asarray(actual.valid))
    for field in OUTPUTS:
        value=np.asarray(getattr(actual,field));reference=expected[field]
        assert value.dtype==np.float64
        assert np.all(np.isfinite(value))
        if field=='water_mass':
            density=cases['water_density'][:,None,None] if all_levels else cases['water_density'][:,None]
            scale=(PI/6)*(expected['dry_radius_cm']*.02)**3*density*expected['number']
            assert np.all(np.abs(value-reference)<=3e-13*np.abs(reference)+32*np.finfo(float).eps*scale)
        else:
            np.testing.assert_allclose(value,reference,rtol=3e-13,atol=0)


@pytest.fixture(scope='module')
def reference(tmp_path_factory):
    executable,provenance=build_oracle(tmp_path_factory.mktemp('whole-salsa-jax'))
    cases=reference_cases()
    expected,_=run_oracle(executable,cases,all_levels=True)
    return executable,provenance,cases,expected


@pytest.mark.parametrize('compiled',[False,True])
def test_complete_matrix_both_native_levels(reference,compiled):
    _,provenance,cases,expected=reference
    assert provenance['checked']
    for indices in static_groups(cases):
        subset=select_cases(cases,indices)
        args,kwargs=arguments(subset,all_levels=True)
        result=(COMPILED if compiled else salsa_step)(*args,**kwargs)
        assert_outputs(result,{key:value[indices] for key,value in expected.items()},subset,all_levels=True)


def test_actual_native_default_boundary_and_trace(reference):
    executable,_,_,_=reference
    cases,expected,_=load_native_case()
    native,_=run_oracle(executable,cases)
    args,kwargs=arguments(cases,return_trace=True)
    actual,trace=COMPILED(*args,**kwargs)
    assert_outputs(actual,native,cases)
    assert_outputs(actual,expected,cases)
    for field in ('volume','core','wet_diameter'):
        np.testing.assert_allclose(getattr(actual,field),expected[field],rtol=3e-13,atol=0)
    assert len(trace['remapping'])==len(trace['correction'])==2
    np.testing.assert_array_equal(trace['correction'][-1].number,actual.number)


@pytest.mark.parametrize('compiled',[False,True])
def test_three_calls_feed_back_native_mass_interface(reference,compiled):
    executable,_,cases,_=reference
    labels=('captured-default','active-scheme2-growth1','flags-coag1-cond1-recalc1',
            'flags-coag0-cond0-recalc0','unhydrated-insoluble')
    for label in labels:
        case=select_cases(cases,[cases['labels'].index(label)])
        expected,_=run_oracle(executable,case,steps=3,all_levels=True)
        args,kwargs=arguments(case,all_levels=True)
        kernel=COMPILED if compiled else salsa_step
        for _ in range(3):
            actual=kernel(*args,**kwargs)
            args=(actual.mass,actual.number,actual.gas,*args[3:])
        assert_outputs(actual,expected,case,all_levels=True)


@pytest.mark.parametrize('compiled',[False,True])
def test_soluble_transfer_depleted_bin_has_native_roundoff_sensitivity(reference,compiled):
    executable,_,cases,_=reference
    case=select_cases(cases,[cases['labels'].index('soluble-transfer')])
    args,kwargs=arguments(case,all_levels=True)
    kernel=COMPILED if compiled else salsa_step
    components=case['component_map'][0,12]
    components=components[components>=0]
    for step in range(1,4):
        actual=kernel(*args,**kwargs)
        args=(actual.mass,actual.number,actual.gas,*args[3:])
        expected,_=run_oracle(executable,case,steps=step,all_levels=True)
        if step==1:
            assert_outputs(actual,expected,case,all_levels=True)
            continue
        assert np.all((expected['number'][...,12]>0)&(expected['number'][...,12]<=case['number_threshold']))
        # Newly repopulated bin13 inherits cancellation in the native transfer
        # fraction. Bound only these five depleted entries by the initial same-
        # component scale: 32 epsilon allows accumulated rounding across calls,
        # as for wet-minus-dry water; every other entry retains strict rtol.
        strict={key:value.copy() for key,value in expected.items()}
        for field,indices in (('number',[12]),('mass',components)):
            observed=np.asarray(getattr(actual,field))[...,indices]
            reference_values=expected[field][...,indices]
            scale=case[field][:,None,indices]
            bound=3e-13*np.abs(reference_values)+32*np.finfo(float).eps*scale
            assert np.all(np.abs(observed-reference_values)<=bound)
            strict[field][...,indices]=observed
        assert_outputs(actual,strict,case,all_levels=True)
    # Independently demonstrate that the native trajectory itself is sensitive
    # to one representable step in its populated initial mass values.
    perturbed={**case,'mass':np.where(case['mass']>0,np.nextafter(case['mass'],0.),0.)}
    perturbed_output,_=run_oracle(executable,perturbed,steps=3,all_levels=True)
    delta=np.abs(perturbed_output['number'][...,12]-expected['number'][...,12])
    assert np.all(delta>1e-9)
    assert np.all(delta<=32*np.finfo(float).eps*case['number'][0,12])


def test_runtime_component_map_permutation(reference):
    executable,_,cases,_=reference
    case=select_cases(cases,[0])
    permutation=np.random.default_rng(852).permutation(69)
    old_mass=case['mass'].copy();present=case['component_map']>=0
    case['component_map'][present]=permutation[case['component_map'][present]]
    case['mass'][:,permutation]=old_mass
    expected,_=run_oracle(executable,case)
    args,kwargs=arguments(case)
    assert_outputs(COMPILED(*args,**kwargs),expected,case)


def test_D007_invalid_tiny_cell_does_not_poison_valid_neighbor(reference):
    _,_,cases,_=reference
    tiny=cases['labels'].index('initial-tiny-cleanup-bin1')
    mixed=select_cases(cases,[0,tiny])
    for key in STATIC_OPTIONS:
        if key!='return_trace':mixed[key][:]=mixed[key][0]
    args,kwargs=arguments(mixed)
    actual=COMPILED(*args,**kwargs)
    np.testing.assert_array_equal(actual.valid,[True,False])
    for field in SalsaState._fields:
        if field=='valid':continue
        assert np.all(np.isfinite(np.asarray(getattr(actual,field))[0]))
        assert np.all(np.isnan(np.asarray(getattr(actual,field))[1]))


def test_scalar_and_extra_batch_axes(reference):
    _,_,cases,_=reference
    args,kwargs=arguments(select_cases(cases,[0]),all_levels=True)
    expected=COMPILED(*args,**kwargs)
    scalar_args=tuple(value[0,1] for value in args)
    scalar_kwargs=jax.tree.map(lambda value:value[0,1] if isinstance(value,np.ndarray) else value,kwargs)
    scalar=COMPILED(*scalar_args,**scalar_kwargs)
    assert scalar.valid.shape==()
    for field in SalsaState._fields:
        np.testing.assert_array_equal(getattr(scalar,field),getattr(expected,field)[0,1])


def test_host_adapter_rejects_mixed_static_options(reference):
    _,_,cases,_=reference
    with pytest.raises(ValueError,match='static option'):
        arguments(cases)


@pytest.mark.parametrize('field,shape',[('mass',(68,)),('number',(16,)),('gas',(2,))])
def test_wrong_state_layout_rejected(field,shape,reference):
    _,_,cases,_=reference
    args,kwargs=arguments(select_cases(cases,[0]))
    args=list(args);args[('mass','number','gas').index(field)]=np.zeros(shape)
    with pytest.raises(ValueError,match='Expected mass'):salsa_step(*args,**kwargs)


def test_host_adapter_rejects_nonintegral_scheme():
    cases=select_cases(reference_cases(),[0])
    cases['scheme']=np.array([1.5])
    with pytest.raises(ValueError,match='Unsupported static option scheme'):
        arguments(cases)


def test_host_adapter_rejects_nonboolean_switch():
    cases=select_cases(reference_cases(),[0])
    cases['coagulation_enabled']=np.array([2])
    with pytest.raises(ValueError,match='Unsupported static option coagulation_enabled'):
        arguments(cases)


@pytest.mark.parametrize('field',['cluster_molecules','acid_diameter','activation_coefficient'])
def test_host_adapter_rejects_changed_fixed_nucleation_constant(field):
    cases=select_cases(reference_cases(),[0])
    cases[field]*=1.01
    with pytest.raises(ValueError,match='Unsupported fixed nucleation constant'):
        arguments(cases)

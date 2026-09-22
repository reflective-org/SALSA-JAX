"""Actual conc2mmr fractions, units, registry lookup and write ordering."""
import numpy as np
import pytest
from tools.aerosol_initialization_oracle import (ROOT,_extract,build_oracle,
                                                 reference_cases,run_oracle)


@pytest.fixture(scope='module')
def oracle(tmp_path_factory):
    directory=tmp_path_factory.mktemp('aerosol-initialization-native')
    executable,provenance=build_oracle(directory)
    cases=reference_cases();outputs,timing=run_oracle(executable,cases)
    return directory,executable,provenance,cases,outputs,timing


def test_entire_original_conc2mmr_and_constants(oracle):
    directory,_,provenance,_,_,_=oracle
    source=_extract((ROOT/'mo_convert_concentrations.F90').read_text(),'SUBROUTINE','conc2mmr')
    assert (directory/'conc2mmr_verbatim.inc').read_text()==source+'\n'
    assert 'CASE(HAM_M7)' in source and 'CASE(HAM_SALSA)' in source
    assert '-fcheck=all' in provenance['flags']
    assert provenance['compiled_constants']=={'avogadro':6.02214179e23,'mw_so4':96.0631,'legacy_sulfate_prefactor':6.022e20}
    assert provenance['configuration']['mw_so4_units']=='g/mol'


def test_checked_optimized_repetitions_exact(oracle,tmp_path):
    _,executable,_,cases,expected,_=oracle
    repeated,timing=run_oracle(executable,cases,repetitions=3)
    optimized,provenance=build_oracle(tmp_path,checked=False)
    fast,_=run_oracle(optimized,cases)
    assert not provenance['checked']
    for field in ('number','mass','tracers'):
        np.testing.assert_array_equal(repeated[field],expected[field])
        np.testing.assert_array_equal(fast[field],expected[field])
    np.testing.assert_allclose(timing['number_checksum'],expected['number'].sum(),rtol=3e-15,atol=0)
    np.testing.assert_allclose(timing['mass_checksum'],expected['mass'].sum(),rtol=3e-15,atol=0)
    # Match the native column-order checksum; NumPy's pairwise sum differs
    # on the heterogeneous 1e15-particle stress case plus sentinel values.
    assert timing['tracer_checksum']==sum(expected['tracers'].ravel())
    assert timing['input_reset_included'] and timing['registry_setup_included']


def test_number_conversion_and_material_volume_partition(oracle):
    _,_,_,cases,output,_=oracle
    np.testing.assert_array_equal(output['number'],cases['number']*1e-6)
    fractions=np.zeros((17,5));fractions[:3,:2]=[.2,.8];fractions[3:10]=[.5,.05,.05,.1,.3];fractions[10:]=[.01,.01,.01,0,.97]
    for i,mapping in enumerate(cases['component_map']):
        mw=cases['species_molar_mass'][i];rho=cases['species_density'][i]
        mass=output['mass'][i][np.maximum(mapping,0)]
        volume=mass/rho[None,:]*1e-9
        volume[:,0]=mass[:,0]*1e6*(mw[0]/1000/6.02214179e23/rho[0])
        volume=np.where(mapping>=0,volume,0.)
        np.testing.assert_allclose(volume,(cases['number'][i]*cases['core'][i])[:,None]*fractions,rtol=3e-15,atol=0)
        np.testing.assert_allclose(volume.sum(-1),cases['number'][i]*cases['core'][i],rtol=3e-15,atol=0)


def test_all_non2_codes_take_native_microgram_fallback(oracle):
    _,_,_,cases,output,_=oracle
    first=cases['labels'].index('all-code1');fallback=cases['labels'].index('fallback-unit-codes')
    for field in ('number','mass','tracers'):
        np.testing.assert_array_equal(output[field][first],output[field][fallback])
    two=cases['labels'].index('all-code2')
    target=cases['component_tracer_indices'][two]
    np.testing.assert_allclose(output['tracers'][two,target],output['mass'][two]/((6.022e20/cases['mw_so4'][two])*cases['air_density'][two]),rtol=3e-15,atol=0)


def test_independent_legacy_molecular_weight(oracle):
    _,_,_,cases,output,_=oracle
    other=cases['labels'].index('independent-mw-so4')
    np.testing.assert_array_equal(output['mass'][other],output['mass'][0])
    np.testing.assert_array_equal(output['number'][other],output['number'][0])
    selected=cases['component_unit_codes'][0]==2
    targets=cases['component_tracer_indices'][0,selected]
    np.testing.assert_allclose(output['tracers'][other,targets],output['tracers'][0,targets]*(cases['mw_so4'][other]/cases['mw_so4'][0]),rtol=3e-15,atol=0)


def test_duplicate_targets_follow_component_then_number_last_write(oracle):
    _,_,_,cases,output,_=oracle
    i=cases['labels'].index('duplicate-component-targets')
    target=cases['component_tracer_indices'][i,0]
    factor=6.022e20/cases['mw_so4'][i] if cases['component_unit_codes'][i,-1]==2 else 1e9
    assert output['tracers'][i,target]==output['mass'][i,-1]/(factor*cases['air_density'][i])
    for label in ('duplicate-number-targets','number-overwrites-mass'):
        i=cases['labels'].index(label);target=cases['number_tracer_indices'][i,-1]
        assert output['tracers'][i,target]==output['number'][i,-1]/(1e-6*cases['air_density'][i])


def test_untouched_tracers_preserve_sentinels(oracle):
    _,_,_,cases,output,_=oracle
    for i in range(len(cases['labels'])):
        touched=np.r_[cases['component_tracer_indices'][i],cases['number_tracer_indices'][i]]
        untouched=np.ones(cases['tracers'].shape[-1],dtype=bool);untouched[touched]=False
        np.testing.assert_array_equal(output['tracers'][i,untouched],cases['tracers'][i,untouched])


def test_two_native_calls_apply_number_unit_conversion_twice(oracle):
    _,executable,_,cases,first,_=oracle
    second,_=run_oracle(executable,cases,steps=2)
    sequential,_=run_oracle(executable,{**cases,'number':first['number'],'tracers':first['tracers']})
    for field in ('number','mass','tracers'):np.testing.assert_array_equal(second[field],sequential[field])
    np.testing.assert_array_equal(second['number'],first['number']*1e-6)


def test_scalar_and_broadcast_layout(oracle):
    _,exe,_,cases,expected,_=oracle
    scalar={key:value[0] for key,value in cases.items() if key!='labels'}
    output,_=run_oracle(exe,scalar)
    for key in output:np.testing.assert_array_equal(output[key],expected[key][0])
    broad={**scalar,'number':np.broadcast_to(scalar['number'],(2,3,17))}
    output,_=run_oracle(exe,broad)
    for key in output:np.testing.assert_array_equal(output[key],np.broadcast_to(expected[key][0],output[key].shape))


@pytest.mark.parametrize('field',['number','core','air_density','mw_so4','species_molar_mass','species_density','component_map','component_tracer_indices','number_tracer_indices'])
def test_invalid_input_rejected(field):
    cases=reference_cases();cases[field]=np.full_like(cases[field],-1)
    with pytest.raises(ValueError,match=field):run_oracle('must-not-run',cases)

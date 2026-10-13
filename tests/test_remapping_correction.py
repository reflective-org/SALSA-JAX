# Copyright 2026 Reflective
# SPDX-License-Identifier: Apache-2.0
"""One actual native post-remap correction, including its unrepaired D007 branch."""
import hashlib
import subprocess

import jax
import numpy as np
import pytest

from salsa_jax.constants import AVOGADRO,PI
from salsa_jax.processes.hydration import HydrationParameters
from salsa_jax.processes.remapping_correction import correction
from tools.hydration_oracle import reference_cases as hydration_cases
from tools.remapping_correction_oracle import FLOAT_FIELDS,build_oracle,correction_block,run_oracle

jax.config.update('jax_enable_x64',True)
INPUTS=('number','volume','gas','relative_humidity','temperature','dry_diameter')


def cases():
    base=hydration_cases()
    rows=[{key:value[i].copy() for key,value in base.items() if key!='labels'} for i in range(len(base['labels']))]
    labels=list(base['labels'])
    for row in rows:
        row['gas']=np.array([1e13,1e12,777.])
        row['organic_enabled']=np.array(True)
    rows[0]['organic_enabled']=np.array(False)
    for bin_index in (0,2,3,9,10,16):
        for scale in (.999999,1.,1.000001):
            row={key:value.copy() for key,value in rows[0].items()}
            row['number']=np.zeros(17)
            row['number'][bin_index]=1e6
            row['volume']=np.zeros((17,5))
            fractions=np.array([.6,.4,0.,0.,0.])
            row['volume'][bin_index]=row['number'][bin_index]*(PI/6)*(scale*1e-10)**3*fractions
            row['organic_enabled']=np.array(True)
            rows.append(row)
            labels.append(f'tiny-bin{bin_index+1}-scale{scale}')
    for number in (np.nextafter(1.,0.),1.,np.nextafter(1.,np.inf)):
        row={key:value.copy() for key,value in rows[0].items()}
        row['number']=np.zeros(17)
        row['number'][0]=number
        row['volume']=np.zeros((17,5))
        row['volume'][0,0]=number*(PI/6)*(1e-11)**3
        row['organic_enabled']=np.array(True)
        rows.append(row)
        labels.append(f'number-threshold-{number:.17g}')
    row={key:value.copy() for key,value in rows[0].items()}
    row['number']=np.zeros(17)
    row['volume']=np.full((17,5),1e-20)
    rows.append(row)
    labels.append('orphan-volume-zero-number')
    row={key:value.copy() for key,value in rows[0].items()}
    row['number']=np.full(17,1e6)
    row['volume']=np.zeros((17,5))
    row['volume'][:,2]=1e-30
    row['organic_enabled']=np.array(True)
    rows.append(row)
    labels.append('tiny-nonvolatile-loss')
    return {'labels':labels,**{key:np.stack([row[key] for row in rows]) for key in rows[0]}}


def run_jax(c,compiled=False):
    kernel=jax.jit(correction) if compiled else correction
    return kernel(*(c[key] for key in INPUTS),soluble_mask=c['soluble_mask'],organic_enabled=c['organic_enabled'],
                  parameters=HydrationParameters(**{key:c[key] for key in HydrationParameters._fields}))


@pytest.fixture(scope='module')
def reference(tmp_path_factory):
    directory=tmp_path_factory.mktemp('remapping-correction')
    executable,provenance=build_oracle(directory)
    c=cases()
    expected=run_oracle(executable,c)
    return directory,executable,provenance,c,expected


def test_reference_is_verbatim_native_block_with_checked_hydration(reference):
    directory,_,provenance,_,_=reference
    block=correction_block()
    assert (directory/'correction_verbatim.inc').read_text()==block+'\n'
    assert provenance['correction_block_sha256']==hashlib.sha256(block.encode()).hexdigest()
    assert '-fcheck=all' in provenance['flags']
    assert '-ffpe-trap=invalid,zero,overflow' in provenance['flags']
    assert provenance['hydration_provenance']['checked']
    assert 'pc_gas(ii,jj,isubm_ocnv)' in block


@pytest.mark.parametrize('compiled',[False,True])
def test_full_correction_matches_original_fortran(reference,compiled):
    _,_,_,c,expected=reference
    actual=run_jax(c,compiled)
    assert np.all(actual.valid)
    for field in FLOAT_FIELDS:
        value=np.asarray(getattr(actual,field))
        assert value.dtype==np.float64
        assert np.all(np.isfinite(value))
        if field=='water_mass':
            # Water subtracts almost equal wet/dry volumes. No fixed absolute
            # floor or clipping: calibrate cancellation error to each bin's
            # dry-volume water-mass scale, plus relative hydrated-water error.
            scale=expected['core']*c['water_density'][:,None]*expected['number']
            bound=32*np.finfo(float).eps*scale+3e-13*np.abs(expected[field])
            assert np.all(np.abs(value-expected[field])<=bound)
        else:
            np.testing.assert_allclose(value,expected[field],rtol=3e-13,atol=0)
    np.testing.assert_array_equal(actual.iterations,expected['iterations'])


def test_threshold_policy_nonvolatile_loss_and_orphan_volume(reference):
    _,_,_,c,expected=reference
    below=c['labels'].index(f'number-threshold-{np.nextafter(1.,0.):.17g}')
    equal=c['labels'].index('number-threshold-1')
    assert expected['number'][below,0]==c['number'][below,0]
    assert expected['number'][equal,0]==0
    np.testing.assert_array_equal(expected['iterations'][below],0)
    lost=c['labels'].index('tiny-nonvolatile-loss')
    assert np.sum(c['volume'][lost,:,2])>0
    assert not expected['number'][lost].any()
    assert not expected['volume'][lost].any()
    np.testing.assert_array_equal(expected['gas'][lost],c['gas'][lost])
    orphan=c['labels'].index('orphan-volume-zero-number')
    np.testing.assert_array_equal(expected['volume'][orphan],c['volume'][orphan])
    np.testing.assert_array_equal(expected['wet_diameter'][orphan],c['dry_diameter'][orphan])
    np.testing.assert_allclose(expected['core'][orphan],PI/6*c['dry_diameter'][orphan]**3,rtol=3e-15,atol=0)


def test_cleanup_restores_only_registered_sulfate_and_organic_budgets(reference):
    _,_,_,c,expected=reference
    selected=np.array(['tiny-bin' in label for label in c['labels']])
    for material,gas_slot in ((0,1),(1,2)):
        molecule_volume=c['species_molar_mass'][:,material]/1000/AVOGADRO/c['species_density'][:,material]
        before=c['gas'][:,gas_slot]+c['volume'][:,:,material].sum(-1)/molecule_volume
        after=expected['gas'][:,gas_slot]+expected['volume'][:,:,material].sum(-1)/molecule_volume
        np.testing.assert_allclose(after[selected],before[selected],rtol=3e-15,atol=0)
    np.testing.assert_array_equal(expected['gas'][:,0],c['gas'][:,0])


def test_original_D007_abort_and_explicit_jax_invalid_cells(reference):
    _,executable,_,c,_=reference
    index=c['labels'].index('tiny-bin1-scale0.999999')
    bad={key:value[index:index+1].copy() for key,value in c.items() if key!='labels'}
    bad['organic_enabled'][...]=False
    with pytest.raises(subprocess.CalledProcessError) as error:
        run_oracle(executable,bad)
    assert 'below lower bound of 1' in error.value.stderr
    # One valid neighbor must remain usable while the D007 cell is invalid.
    mixed={key:np.concatenate([value[:1],bad[key]],axis=0) for key,value in c.items() if key!='labels'}
    result=run_jax(mixed,True)
    np.testing.assert_array_equal(result.valid,[True,False])
    for field in FLOAT_FIELDS:
        assert np.all(np.isfinite(np.asarray(getattr(result,field))[0]))
        assert np.all(np.isnan(np.asarray(getattr(result,field))[1]))
    np.testing.assert_array_equal(result.iterations[1],0)


def test_scalar_and_multiple_leading_axes(reference):
    _,_,_,c,expected=reference
    one={key:value[0] for key,value in c.items() if key!='labels'}
    scalar=run_jax(one,True)
    assert scalar.valid.shape==()
    assert scalar.number.shape==(17,)
    multi={key:np.broadcast_to(value,(2,3)+np.shape(value)) for key,value in one.items()}
    actual=run_jax(multi,True)
    assert actual.number.shape==(2,3,17)
    for field in FLOAT_FIELDS:
        np.testing.assert_allclose(getattr(actual,field),np.broadcast_to(expected[field][0],getattr(actual,field).shape),
                                   rtol=3e-13,atol=0)


def test_wrong_state_layout_rejected():
    with pytest.raises(ValueError,match='Expected number'):
        correction(np.zeros(17),np.zeros((17,5)),np.zeros(2),.5,280.,np.ones(17))


def test_native_roundoff_water_is_not_clipped(reference):
    _,_,_,c,expected=reference
    assert np.any(expected['water_mass']<0)
    # Native and JIT fused arithmetic both retain tiny negative water here;
    # eager evaluation can yield exact zero instead, covered by the mass-scaled
    # roundoff bound in the complete field comparison.
    actual=run_jax(c,True)
    assert np.any(np.asarray(actual.water_mass)<0)

# Copyright 2026 Reflective
# SPDX-License-Identifier: Apache-2.0
"""Critical-cluster rates checked against extracted actual SALSA Fortran leaves."""
from functools import partial
from pathlib import Path
import re
import subprocess

import jax
import numpy as np
import pytest

from salsa_jax.processes.nucleation_rates import binary_nucleation, nucleation_rates

jax.config.update('jax_enable_x64', True)
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def leaf_oracle(tmp_path_factory):
    directory = tmp_path_factory.mktemp('nucleation-rate-leaves')
    source = (ROOT / 'mo_ham_salsa_nucleation.F90').read_text()
    leaves = []
    for name in ('binnucl', 'actnucl', 'kinnucl'):
        match = re.search(rf'^\s*SUBROUTINE {name}\(.*?^\s*END SUBROUTINE {name}', source, re.M | re.S | re.I)
        assert match is not None
        leaves.append(match.group(0))
    (directory / 'leaves.f90').write_text('module native_leaves\ncontains\n' + '\n'.join(leaves) + '\nend module\n')
    (directory / 'driver.f90').write_text('''program oracle
 use mo_kind, only: dp
 use native_leaves
 implicit none
 integer :: n,scheme,i,mask
 integer, allocatable :: pbl(:)
 real(dp), allocatable :: gas(:,:),temp(:,:),rh(:,:),rate(:,:),nsa(:,:),noc(:,:),diam(:,:),ksa(:,:),koc(:,:)
 real(dp) :: coefficient
 read(*,*) n,scheme,coefficient
 allocate(pbl(n),gas(n,2),temp(n,2),rh(n,2),rate(n,2),nsa(n,2),noc(n,2),diam(n,2),ksa(n,2),koc(n,2))
 do i=1,n
   read(*,*) gas(i,1),temp(i,1),rh(i,1),mask
   gas(i,2)=gas(i,1)
   temp(i,2)=temp(i,1)
   rh(i,2)=rh(i,1)
   pbl(i)=2-mask
 end do
 rate=0._dp
 nsa=0._dp
 noc=0._dp
 diam=0._dp
 ksa=0._dp
 koc=0._dp
 if(scheme==1) pbl=2
 call binnucl(n,n,2,gas*1.e-6_dp,temp,rh,rate,nsa,noc,diam,pbl,ksa,koc)
 if(scheme==2) call actnucl(n,n,2,gas,rate,diam,pbl,nsa,noc,ksa,koc,coefficient)
 if(scheme==3) call kinnucl(n,n,2,gas*1.e-6_dp,temp,rate,diam,pbl,nsa,noc,ksa,koc)
 do i=1,n
   write(*,'(3(ES26.17E3,1X))') rate(i,1),nsa(i,1),diam(i,1)
 end do
end program
''')
    sources = [str(ROOT / name) for name in ('parkind1.F90', 'mo_kind.F90', 'mo_math_constants.F90', 'mo_physical_constants.F90')]
    executable = directory / 'oracle'
    subprocess.run(['gfortran', '-O2', '-fno-fast-math', '-cpp', '-ffree-line-length-none', '-fcheck=all',
                    '-ffpe-trap=invalid,zero,overflow', *sources, 'leaves.f90', 'driver.f90', '-o', str(executable)],
                   cwd=directory, check=True, capture_output=True, text=True)
    return executable


def reference(executable, sulfate, temperature, humidity, scheme=1, boundary_layer=True, activation=1e-7):
    values = np.broadcast_arrays(sulfate, temperature, humidity, boundary_layer)
    lines = [f'{values[0].size} {scheme} {activation:.17e}']
    for gas, temp, rh, bl in zip(*(value.ravel() for value in values)):
        lines.append(f'{gas:.17e} {temp:.17e} {rh:.17e} {int(bl)}')
    output = subprocess.check_output([str(executable)], input='\n'.join(lines)+'\n', text=True)
    return np.fromstring(output, sep=' ').reshape(*values[0].shape, 3)


def cases():
    # Include all validity clamps, low/high fits and the collision fallback.
    gas, temp, rh = np.meshgrid([1e2,1e4,1e6,1e8,1e9,1e11], [180.,190.15,230.,270.,300.15,320.],
                               [0.,1e-4,.1,.8,1.,1.2], indexing='ij')
    return gas.ravel()*1e6, temp.ravel(), rh.ravel()


@pytest.mark.parametrize('compiled', [False, True])
def test_binary_polynomial_and_collision_fallback_match_native(leaf_oracle, compiled):
    gas,temp,rh = cases()
    expected = reference(leaf_oracle,gas,temp,rh)
    kernel = jax.jit(binary_nucleation) if compiled else binary_nucleation
    actual = kernel(gas*1e-6,temp,rh)
    assert np.all(np.isfinite(np.stack(actual)))
    # Large fitted log-polynomials cancel many terms; exp amplifies their
    # last-bit backend/reduction differences. Structural zeros stay exact.
    for field, index in zip(actual, range(3)):
        np.testing.assert_allclose(field, expected[:,index], rtol=1e-11, atol=0)
    assert np.any(expected[:,0]==0)
    assert np.any(expected[expected[:,0]==0,2] != 1e-9)
    assert np.any(expected[:,1]<4)
    assert np.any(expected[:,1]>=4)


@pytest.mark.parametrize('scheme', [1,2,3])
@pytest.mark.parametrize('compiled', [False, True])
def test_scheme_and_boundary_layer_behavior_match_native(leaf_oracle, scheme, compiled):
    gas,temp,rh = cases()
    boundary = np.arange(gas.size)%2==0
    expected = reference(leaf_oracle,gas,temp,rh,scheme,boundary,activation=3e-7)
    kernel = partial(nucleation_rates,scheme=scheme)
    if compiled:
        kernel = jax.jit(kernel)
    actual = kernel(gas,temp,rh,boundary_layer=boundary,activation_coefficient=3e-7)
    for field, index in zip(actual,range(3)):
        np.testing.assert_allclose(field,expected[:,index],rtol=1e-11,atol=0)
    if scheme != 1:
        np.testing.assert_array_equal(actual.diameter,np.full(gas.shape,7.9375e-10))
        np.testing.assert_array_equal(np.asarray(actual.sulfate_molecules)[boundary],2.)


def test_binary_clamps_and_scalar_broadcast():
    actual = jax.jit(binary_nucleation)(np.array([[0.],[1e11]]),np.array([180.,320.]),1.2)
    bounded = binary_nucleation(np.array([[1e4],[1e9]]),np.array([190.15,300.15]),1.)
    for left,right in zip(actual,bounded):
        assert left.shape == (2,2)
        np.testing.assert_allclose(left,right,rtol=1e-11,atol=0)
    scalar = binary_nucleation(1e6,280.,.5)
    assert scalar.rate.shape == ()


def test_unsupported_scheme_rejected():
    with pytest.raises(ValueError,match='schemes'):
        nucleation_rates(1e12,280.,.5,scheme=4)


def test_activation_coefficient_axes_broadcast_all_outputs():
    coefficients = np.array([1e-7, 3e-7])
    actual = jax.jit(nucleation_rates)(1e12, 280., .5, activation_coefficient=coefficients)
    for field in actual:
        assert field.shape == (2,)
    np.testing.assert_array_equal(actual.rate, coefficients * 1e12)
    np.testing.assert_array_equal(actual.sulfate_molecules, [2., 2.])

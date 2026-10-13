# Copyright 2026 Reflective
# SPDX-License-Identifier: Apache-2.0
"""Checkpointed long trajectories through byte-identical native SALSA modules.

A constant H2SO4 reservoir restores the initial concentration after every native
call. Its signed addition is recorded explicitly; native process formulas and
within-step gas depletion are unchanged. Closed runs perform no restoration.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess

import numpy as np

from tools import salsa_oracle as salsa
from tools.sulfuric_acid_oracle import _input_text, _replace_once, trajectory_driver

OUTPUTS = salsa.OUTPUTS
DIAGNOSTICS = ('max_relative_sulfur_residual', 'max_relative_h2so4_depletion',
               'minimum_number', 'minimum_mass', 'minimum_gas')


def checkpoint_driver():
    """Modify the observation wrapper only, retaining both native model levels."""
    text = trajectory_driver()
    text = _replace_once(text, ' use iso_fortran_env,only:int64',
                        ' use iso_fortran_env,only:int64\n use, intrinsic :: ieee_arithmetic,only:ieee_is_finite')
    text = _replace_once(text, 'integer::n,repetitions,steps,i,j,k,level,ngas,step_index',
                        'integer::n,repetitions,steps,i,j,k,level,ngas,step_index,save_every,nsave,constant_h2so4')
    text = _replace_once(text, 'real(dp),allocatable::source_rate(:)',
                        'real(dp),allocatable::source_rate(:),added_out(:,:,:),before_out(:,:,:),diagnostics(:,:,:)')
    text = _replace_once(text, 'read(*,*) n,repetitions,steps',
                        'read(*,*) n,repetitions,steps,save_every,constant_h2so4\n nsave=steps/save_every')
    text = _replace_once(text, 'allocate(mapping(17,5,n),source_rate(n))',
                        'allocate(mapping(17,5,n),source_rate(n),added_out(2,n,nsave),before_out(2,n,nsave),diagnostics(5,2,n))')
    text = text.replace(',2,n,steps)', ',2,n,nsave)')
    text = _replace_once(text, 'do step_index=1,steps', 'do step_index=1,nsave')
    text = _replace_once(text, 'rho_out(:,level,i,step_index),water_out(:,level,i,step_index)',
                        'rho_out(:,level,i,step_index),water_out(:,level,i,step_index),added_out(level,i,step_index),before_out(level,i,step_index)')
    text = _replace_once(text, ' end do\ncontains', ''' end do
 do i=1,n
 do level=1,2
 write(*,'(ES26.17E3)') diagnostics(:,level,i)
 end do
 end do
contains''')
    text = _replace_once(text, 'integer::s,bin,lev', '''integer::s,bin,lev,checkpoint
 real(dp)::added(2),before(2),inventory(2),initial_inventory(2),target,relative_residual(2),depletion(2)
 real(dp)::diag(5,2)''')
    old = ''' do s=1,steps
 gas(1,:,isubm_so4g)=gas(1,:,isubm_so4g)+source_rate(index)*environment(4,index)
 call salsa(1,1,2,1,pressure,rh,temperature,environment(4,index),gas,mass,number,wet,dry,rho,water,pbl)
 mass_out(:,:,index,s)=transpose(mass(1,:,:));number_out(:,:,index,s)=transpose(number(1,:,:));gas_out(:,:,index,s)=transpose(gas(1,:,:))
 wet_out(:,:,index,s)=transpose(wet(1,:,:));dry_out(:,:,index,s)=transpose(dry(1,:,:));rho_out(:,:,index,s)=transpose(rho(1,:,:));water_out(:,:,index,s)=transpose(water(1,:,:))
 end do'''
    new = ''' target=initial_gas(isubm_so4g,index)
 added=0._dp;diag(1:2,:)=0._dp
 do lev=1,2
 initial_inventory(lev)=sum(mass(1,lev,iso4b))*1.e6_dp+gas(1,lev,isubm_so4g)
 diag(3,lev)=minval(number(1,lev,:));diag(4,lev)=minval(mass(1,lev,:));diag(5,lev)=minval(gas(1,lev,:))
 end do
 do s=1,steps
 call salsa(1,1,2,1,pressure,rh,temperature,environment(4,index),gas,mass,number,wet,dry,rho,water,pbl)
 if(.not.(all(ieee_is_finite(mass)).and.all(ieee_is_finite(number)).and.all(ieee_is_finite(gas)).and. &
 all(ieee_is_finite(wet)).and.all(ieee_is_finite(dry)).and.all(ieee_is_finite(rho)).and.all(ieee_is_finite(water)))) &
 error stop 'Nonfinite native trajectory state'
 before=gas(1,:,isubm_so4g)
 depletion=abs(target-before)/max(abs(target),1._dp)
 if(constant_h2so4==1) then
 added=added+(target-before)
 gas(1,:,isubm_so4g)=target
 end if
 do lev=1,2
 inventory(lev)=sum(mass(1,lev,iso4b))*1.e6_dp+gas(1,lev,isubm_so4g)
 relative_residual(lev)=abs(inventory(lev)-initial_inventory(lev)-added(lev))/max(abs(initial_inventory(lev)+added(lev)),1._dp)
 diag(1,lev)=max(diag(1,lev),relative_residual(lev))
 if(constant_h2so4==1) diag(2,lev)=max(diag(2,lev),depletion(lev))
 diag(3,lev)=min(diag(3,lev),minval(number(1,lev,:)))
 diag(4,lev)=min(diag(4,lev),minval(mass(1,lev,:)))
 diag(5,lev)=min(diag(5,lev),minval(gas(1,lev,:)),before(lev))
 end do
 if(mod(s,save_every)==0) then
 checkpoint=s/save_every
 mass_out(:,:,index,checkpoint)=transpose(mass(1,:,:));number_out(:,:,index,checkpoint)=transpose(number(1,:,:));gas_out(:,:,index,checkpoint)=transpose(gas(1,:,:))
 wet_out(:,:,index,checkpoint)=transpose(wet(1,:,:));dry_out(:,:,index,checkpoint)=transpose(dry(1,:,:));rho_out(:,:,index,checkpoint)=transpose(rho(1,:,:));water_out(:,:,index,checkpoint)=transpose(water(1,:,:))
 added_out(:,index,checkpoint)=added;before_out(:,index,checkpoint)=before
 end if
 end do
 diagnostics(:,:,index)=diag'''
    text = _replace_once(text, old, new)
    # This runner measures one trajectory; no untimed duplicate long integration.
    text = _replace_once(text, ' do i=1,n\n call run_case(i)\n end do\n call cpu_time(started)',
                        ' call cpu_time(started)')
    return text


def build_oracle(build_directory, compiler='gfortran', *, checked=True):
    directory = Path(build_directory).resolve()
    _, provenance = salsa.build_oracle(directory, compiler, checked=checked)
    driver = checkpoint_driver()
    (directory/'driver.f90').write_text(driver)
    flags = provenance['flags']
    subprocess.run([compiler, *flags, '-c', 'driver.f90', '-o', 'driver.o'],
                   cwd=directory, check=True, capture_output=True, text=True)
    sources = (*salsa.BASE_SOURCES, 'stubs.f90', 'init.F90', *salsa.PROCESS_SOURCES, 'driver.f90')
    executable = directory/'long_sulfuric_acid_oracle'
    subprocess.run([compiler, *flags, *(str(directory/(Path(name).stem+'.o')) for name in sources),
                    '-o', str(executable)], cwd=directory, check=True, capture_output=True, text=True)
    provenance['parent_driver_sha256'] = provenance['wrapper_sha256']['driver.f90']
    provenance['wrapper_sha256']['driver.f90'] = hashlib.sha256(driver.encode()).hexdigest()
    provenance['scope'] += '; strided checkpoints; optional post-call constant H2SO4 reservoir restoration; every-step finite and sulfur checks'
    provenance['reservoir_units'] = 'molecules/m3'
    provenance['output_order'] = 'checkpoint, case, native level, field component'
    return executable, provenance


def run_oracle(executable, cases, *, steps, save_every=1, constant_h2so4=False,
               all_levels=False, timeout=120):
    """Return seven checkpoint fields plus reservoir addition and pre-restore gas.

    Normal field shape is (steps/save_every,cases,width). The two extra outputs
    ``cumulative_reservoir_added`` and ``h2so4_before_restore`` omit width. With
    all_levels=True a native level axis follows cases. Initial state is omitted.
    Reservoir additions are signed and may be negative. Timing diagnostics are
    per case (or case,level); sulfur checks count sulfate plus gas H2SO4 only.
    """
    if isinstance(save_every, (bool, np.bool_)) or not isinstance(save_every, (int, np.integer)) or save_every < 1:
        raise ValueError('save_every must be a positive integer')
    text, values = _input_text(cases, 0., steps, 1)
    if steps % save_every:
        raise ValueError('steps must be divisible by save_every')
    if not isinstance(constant_h2so4, (bool, np.bool_)):
        raise ValueError('constant_h2so4 must be boolean')
    n = len(values['number'])
    text = text.replace(f'{n} 1 {steps}\n', f'{n} 1 {steps} {save_every} {int(constant_h2so4)}\n', 1)
    result = subprocess.run([str(executable)], input=text, capture_output=True,
                            text=True, check=True, timeout=timeout)
    data = np.fromstring(result.stdout, sep=' ')
    count = steps//save_every
    width = sum(OUTPUTS.values())+2
    end = 6+count*n*2*width
    if data.size != end+n*2*len(DIAGNOSTICS) or not np.isfinite(data).all():
        raise ValueError('Unexpected or nonfinite native checkpoint output')
    rows = data[6:end].reshape(count,n,2,width)
    diagnostics = data[end:].reshape(n,2,len(DIAGNOSTICS))
    if not all_levels:
        levels = values['boundary_layer'].astype(int)
        rows = rows[:,np.arange(n),levels]
        diagnostics = diagnostics[np.arange(n),levels]
    outputs = {}; offset = 0
    for key,size in OUTPUTS.items():
        outputs[key] = rows[...,offset:offset+size]
        offset += size
    outputs['cumulative_reservoir_added'] = rows[...,offset]
    outputs['h2so4_before_restore'] = rows[...,offset+1]
    timing = dict(zip(('kernel_wall_seconds','kernel_cpu_seconds','mass_checksum',
                       'number_checksum','gas_checksum','clock_ticks_per_second'),data[:6].tolist()),
                  steps=int(steps),save_every=int(save_every),saved_steps=list(range(save_every,steps+1,save_every)),
                  cases=n,native_cells_per_case=2,all_levels=bool(all_levels),
                  constant_h2so4=bool(constant_h2so4),every_step_finite=True,
                  input_reset_included=True,grid_setup_included=True,
                  forcing_and_diagnostics_included=True,checksums_include_all_saved_steps_and_levels=True)
    timing.update({key:diagnostics[...,i].tolist() for i,key in enumerate(DIAGNOSTICS)})
    return outputs,timing

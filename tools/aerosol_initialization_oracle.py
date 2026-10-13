# Copyright 2026 Reflective
# SPDX-License-Identifier: Apache-2.0
"""Original conc2mmr aerosol initialization with data-only registry stubs.

The complete routine, including its unused M7 branch, is extracted verbatim.
SALSA material slots are permutable. Effective unit codes are represented by
virtual registry entries so every component retains native lookup/ELSE behavior.
Duplicate tracer destinations keep the source's sequential last-write semantics.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import re
import subprocess
import numpy as np
from tools.coagulation_process_oracle import FLAGS,CHECK_FLAGS,_extract
from tools.state_capture import load_fixture
from tools.initialization_capture import load_fixture as load_initialization

ROOT=Path(__file__).resolve().parents[1]
SOURCES=('parkind1.F90','mo_kind.F90','mo_math_constants.F90','mo_physical_constants.F90',
         'mo_convert_concentrations.F90','mo_ham.F90','mo_ham_subm_species.F90')
STUBS='''module mo_species
 use mo_kind,only:dp
 type species_type
 real(dp)::moleweight,density
 end type
 type(species_type)::speclist(5)
end module
module mo_ham_species
 integer,parameter::id_so4=1,id_oc=2,id_bc=3,id_ss=4,id_du=5
end module
module mo_ham
 use mo_kind,only:dp
 integer,parameter::HAM_M7=1,HAM_SALSA=2,naerocomp=69,nclass=17,immr2molec=2
 integer::nham_subm=HAM_SALSA,subm_aerounitconv(69),subm_aero_idx(74)
 real(dp)::mw_so4
 type component_type
 integer::idt,spid
 end type
 type bin_type
 integer::idt_no
 end type
 type(component_type)::aerocomp(69)
 type(bin_type)::sizeclass(17)
end module
module mo_ham_m7ctl
 integer,parameter::inucs=1,iaits=2,iaccs=3,icoas=4,iaiti=5,iacci=6,icoai=7
 integer,parameter::iso4ns=1,iso4ks=2,iso4as=3,iso4cs=4,ibcks=5,ibcas=6,ibccs=7,ibcki=8
 integer,parameter::iocks=9,iocas=10,ioccs=11,iocki=12,issas=13,isscs=14,iduas=15,iducs=16,iduai=17,iduci=18
end module
module mo_ham_salsactl
 integer,parameter::in1a=1,fn1a=3,in2a=4,fn2a=10,in2b=11,fn2b=17
 integer::iso4b(17),iocb(17),ibcb(17),issb(17),idub(17)
end module
module mo_exception
 character(len=512)::message_text
contains
 subroutine finish(where,text)
 character(len=*),intent(in)::where,text
 write(*,*) where,': ',text
 error stop 'Native registry mapping failure'
 end subroutine
 subroutine message(where,text)
 character(len=*),intent(in)::where,text
 write(*,*) where,': ',text
 end subroutine
end module
'''
DRIVER='''program aerosol_initialization_reference
 use iso_fortran_env,only:int64
 use mo_kind,only:dp
 use mo_species
 use mo_ham
 use mo_ham_salsactl
 use mo_physical_constants,only:avo
 use extracted_aerosol_initialization,only:conc2mmr
 implicit none
 integer::n,ntrac,repetitions,steps,i,j
 integer(int64)::started,finished,clock_rate
 integer,allocatable::mapping(:,:,:),mass_tracers(:,:),number_tracers(:,:),codes(:,:)
 real(dp),allocatable::numbers(:,:),cores(:,:),initial_tracers(:,:),density(:),sulfate_mass(:),masses(:,:),densities(:,:)
 real(dp),allocatable::number_out(:,:),mass_out(:,:),tracer_out(:,:),tracers(:,:,:)
 real(dp)::number(1,1,17),mass(1,1,69),air(1,1),cpu_started,cpu_finished
 character(len=32)::command
 call get_command_argument(1,command)
 if(trim(command)=='--constants') then
__MW_SO4__
 write(*,'(ES26.17E3)') avo,mw_so4,6.022e20_dp
 stop
 end if
 read(*,*) n,ntrac,repetitions,steps
 allocate(mapping(17,5,n),mass_tracers(69,n),number_tracers(17,n),codes(69,n))
 allocate(numbers(17,n),cores(17,n),initial_tracers(ntrac,n),density(n),sulfate_mass(n),masses(5,n),densities(5,n))
 allocate(number_out(17,n),mass_out(69,n),tracer_out(ntrac,n),tracers(1,1,ntrac))
 do i=1,n
 read(*,*) density(i),sulfate_mass(i)
 read(*,*) mapping(:,:,i)
 read(*,*) mass_tracers(:,i)
 read(*,*) number_tracers(:,i)
 read(*,*) codes(:,i)
 read(*,*) masses(:,i)
 read(*,*) densities(:,i)
 read(*,*) numbers(:,i)
 read(*,*) cores(:,i)
 read(*,*) initial_tracers(:,i)
 end do
 do i=1,n
 call run_case(i)
 end do
 call cpu_time(cpu_started)
 call system_clock(started,clock_rate)
 do j=1,repetitions
 do i=1,n
 call run_case(i)
 end do
 end do
 call system_clock(finished)
 call cpu_time(cpu_finished)
 write(*,'(6ES26.17E3)') real(finished-started,dp)/real(clock_rate,dp),cpu_finished-cpu_started,sum(number_out),sum(mass_out),sum(tracer_out),real(clock_rate,dp)
 do i=1,n
 write(*,'(ES26.17E3)') number_out(:,i),mass_out(:,i),tracer_out(:,i)
 end do
contains
 subroutine run_case(index)
 integer,intent(in)::index
 integer::k,step
 iso4b=mapping(:,1,index);iocb=mapping(:,2,index);ibcb=mapping(:,3,index);issb=mapping(:,4,index);idub=mapping(:,5,index)
 subm_aero_idx=0
 do k=1,69
 aerocomp(k)%idt=mass_tracers(k,index)
 aerocomp(k)%spid=k+5
 subm_aero_idx(k+5)=k
 end do
 subm_aerounitconv=codes(:,index)
 do k=1,17
 sizeclass(k)%idt_no=number_tracers(k,index)
 end do
 do k=1,5
 speclist(k)=species_type(masses(k,index),densities(k,index))
 end do
 mw_so4=sulfate_mass(index);air=density(index)
 number(1,1,:)=numbers(:,index);mass=0._dp;tracers(1,1,:)=initial_tracers(:,index)
 do step=1,steps
 call conc2mmr(1,1,1,ntrac,tracers,mass,number,cores(:,index),air)
 end do
 number_out(:,index)=number(1,1,:);mass_out(:,index)=mass(1,1,:);tracer_out(:,index)=tracers(1,1,:)
 end subroutine
end program
'''


def native_sulfate_mass():
    source=(ROOT/'mo_ham.F90').read_text()
    match=re.search(r'\bmw_so4\s*=\s*([\d.eE+-]+)_dp',source)
    if match is None:raise ValueError('Missing native mw_so4 data assignment')
    return float(match[1]),'mw_so4 = '+match[1]+'_dp'


def build_oracle(build_directory,compiler='gfortran',*,checked=True):
    directory=Path(build_directory).resolve();directory.mkdir(parents=True,exist_ok=True)
    routine=_extract((ROOT/'mo_convert_concentrations.F90').read_text(),'SUBROUTINE','conc2mmr')
    (directory/'conc2mmr_verbatim.inc').write_text(routine+'\n')
    _,assignment=native_sulfate_mass()
    texts={'stubs.f90':STUBS,'conc2mmr.F90':'module extracted_aerosol_initialization\ncontains\n'+routine+'\nend module\n','driver.f90':DRIVER.replace('__MW_SO4__',assignment)}
    for name,text in texts.items():(directory/name).write_text(text)
    flags=FLAGS+CHECK_FLAGS if checked else FLAGS;objects=[]
    for path in [*[ROOT/name for name in SOURCES[:4]],*[directory/name for name in texts]]:
        obj=directory/(path.stem+'.o')
        subprocess.run([compiler,*flags,'-c',str(path),'-o',str(obj)],cwd=directory,check=True,capture_output=True,text=True)
        objects.append(str(obj))
    exe=directory/'aerosol_initialization_oracle'
    subprocess.run([compiler,*flags,*objects,'-o',str(exe)],cwd=directory,check=True,capture_output=True,text=True)
    constants=np.fromstring(subprocess.check_output([str(exe),'--constants'],text=True),sep=' ')
    sha=lambda b:hashlib.sha256(b).hexdigest()
    return exe,{'source_sha256':{name:sha((ROOT/name).read_bytes()) for name in SOURCES},'routine_sha256':sha(routine.encode()),'wrapper_sha256':{name:sha(text.encode()) for name,text in texts.items()},'flags':list(flags),'checked':checked,'compiler':subprocess.check_output([compiler,'--version'],text=True).splitlines()[0],'compiled_constants':dict(zip(('avogadro','mw_so4','legacy_sulfate_prefactor'),constants.tolist())),'configuration':{'bins':17,'components':69,'nonwater_species':5,'nham_subm':'HAM_SALSA','mw_so4_units':'g/mol','number_input_units':'particles/m3','number_output_units':'particles/cm3','M7_branch_retained_but_untested':True},'registry_stub':'Per-component effective unit codes are represented by virtual species IDs6:74 and indices1:69; physical masses/densities remain canonical SO4/OC/BC/SS/DU entries1:5. Native lookup/assignment loops unchanged.','separate_compilation_without_lto':True}


def reference_cases():
    f=load_fixture(ROOT/'tests/fixtures/native-default');initial=load_initialization(ROOT/'tests/fixtures/initialization-default')
    def rec(capture,stage,key):
        item=capture[stage][key];return np.asarray(item['values']).reshape(item['shape'],order='F')
    mapping=rec(f,'entry','material_map').astype(int)-1
    species=rec(f,'entry','volume_species').astype(int)-1
    codes=np.ones(69,dtype=int);codes[mapping[:,0]]=2
    base={'number':rec(initial,'exit','naero')[0,0],'core':rec(initial,'entry','core'),
          'tracers':np.zeros(f['dimensions']['capture_ntrac']),'air_density':np.asarray(1.2),
          'component_map':mapping,'component_tracer_indices':rec(f,'entry','component_tracer').astype(int)-1,
          'number_tracer_indices':rec(f,'entry','number_tracer').astype(int)-1,'component_unit_codes':codes,
          'species_molar_mass':rec(f,'entry','species_molar_mass')[species],
          'species_density':rec(f,'entry','species_density')[species],'mw_so4':np.asarray(native_sulfate_mass()[0])}
    labels=[];rows=[]
    def add(label,**changes):
        labels.append(label);rows.append({**{key:np.array(value,copy=True) for key,value in base.items()},**changes})
    add('native-layout-density1.2')
    add('empty-number',number=np.zeros(17),tracers=np.linspace(-5,5,len(base['tracers'])))
    add('heterogeneous-number',number=np.geomspace(1e-6,1e15,17))
    add('zero-core',core=np.zeros(17))
    add('scaled-core',core=base['core']*3.)
    add('low-air-density',air_density=np.asarray(.01))
    add('high-air-density',air_density=np.asarray(3.))
    add('heterogeneous-properties',species_molar_mass=np.array([100.,200.,14.,60.,270.]),species_density=np.array([1900.,1800.,2100.,2200.,2600.]),mw_so4=np.asarray(98.))
    add('independent-mw-so4',mw_so4=np.asarray(120.))
    add('all-code2',component_unit_codes=np.full(69,2))
    add('all-code1',component_unit_codes=np.ones(69,dtype=int))
    add('fallback-unit-codes',component_unit_codes=np.resize([0,3,99,-1],69))
    add('unrelated-tracer-sentinels',tracers=np.linspace(-7,9,len(base['tracers'])))
    permutation=np.random.default_rng(9203).permutation(69);permuted=mapping.copy();present=permuted>=0;permuted[present]=permutation[permuted[present]]
    permuted_codes=np.empty(69,dtype=int);permuted_codes[permutation]=codes
    permuted_tracers=np.empty(69,dtype=int);permuted_tracers[permutation]=base['component_tracer_indices']
    add('component-permutation',component_map=permuted,component_unit_codes=permuted_codes,component_tracer_indices=permuted_tracers)
    permutation=np.random.default_rng(112).permutation(len(base['tracers']))
    add('tracer-permutation',component_tracer_indices=permutation[base['component_tracer_indices']],number_tracer_indices=permutation[base['number_tracer_indices']])
    duplicate=base['component_tracer_indices'].copy();duplicate[:]=duplicate[0]
    add('duplicate-component-targets',component_tracer_indices=duplicate)
    duplicate=base['number_tracer_indices'].copy();duplicate[:]=duplicate[0]
    add('duplicate-number-targets',number_tracer_indices=duplicate)
    duplicate=np.full(17,base['component_tracer_indices'][0])
    add('number-overwrites-mass',number_tracer_indices=duplicate)
    return {'labels':labels,**{key:np.stack([row[key] for row in rows]) for key in base}}


def run_oracle(executable,cases,repetitions=1,*,steps=1,timeout=10):
    if any(not isinstance(v,(int,np.integer)) or v<1 for v in (repetitions,steps)):raise ValueError('Positive integer repetitions and steps required')
    raw={key:np.asarray(value) for key,value in cases.items() if key!='labels'}
    if raw['tracers'].ndim<1 or raw['tracers'].shape[-1]<1:raise ValueError('Invalid tracers shape')
    ntrac=raw['tracers'].shape[-1]
    shapes={'number':(17,),'core':(17,),'tracers':(ntrac,),'air_density':(), 'component_map':(17,5),'component_tracer_indices':(69,),'number_tracer_indices':(17,),'component_unit_codes':(69,),'species_molar_mass':(5,),'species_density':(5,),'mw_so4':()}
    leading=[]
    for key,shape in shapes.items():
        a=raw[key]
        if shape and a.shape[-len(shape):]!=shape:raise ValueError(f'Invalid {key} shape')
        if not np.all(np.isfinite(a)):raise ValueError(f'Nonfinite {key}')
        leading.append(a.shape[:-len(shape)] if shape else a.shape)
    leading=np.broadcast_shapes(*leading);count=int(np.prod(leading)) if leading else 1
    if not count:raise ValueError('Empty cases')
    values={key:np.broadcast_to(raw[key],leading+shape).reshape((count,)+shape) for key,shape in shapes.items()}
    for key in ('number','core'):
        if np.any(values[key]<0):raise ValueError(f'Negative {key}')
    for key in ('air_density','species_molar_mass','species_density','mw_so4'):
        if np.any(values[key]<=0):raise ValueError(f'Nonpositive {key}')
    for key in ('component_map','component_tracer_indices','number_tracer_indices','component_unit_codes'):
        if np.any(values[key]!=values[key].astype(int)):raise ValueError(f'Noninteger {key}')
    present=np.ones((17,5),bool);present[:3,2:]=False;present[10:,3]=False
    for mapping in values['component_map']:
        if np.any(mapping[~present]!=-1) or not np.array_equal(np.sort(mapping[present]),np.arange(69)):raise ValueError('Invalid component_map')
    for key in ('component_tracer_indices','number_tracer_indices'):
        if np.any((values[key]<0)|(values[key]>=ntrac)):raise ValueError(f'Out-of-range {key}')
    lines=[f'{count} {ntrac} {repetitions} {steps}']
    for i in range(count):
        lines.append(f"{values['air_density'][i]:.17e} {values['mw_so4'][i]:.17e}")
        for key in ('component_map','component_tracer_indices','number_tracer_indices','component_unit_codes'):
            offset=0 if key=='component_unit_codes' else 1
            lines.append(' '.join(str(int(v)+offset) for v in values[key][i].ravel(order='F')))
        for key in ('species_molar_mass','species_density','number','core','tracers'):
            lines.append(' '.join(f'{v:.17e}' for v in values[key][i]))
    result=subprocess.run([str(executable)],input='\n'.join(lines)+'\n',text=True,capture_output=True,check=True,timeout=timeout)
    output=np.fromstring(result.stdout,sep=' ');width=17+69+ntrac
    if output.size!=6+count*width or not np.all(np.isfinite(output)):raise ValueError('Unexpected or nonfinite native output')
    rows=output[6:].reshape(count,width)
    arrays={'number':rows[:,:17].reshape(leading+(17,)),'mass':rows[:,17:86].reshape(leading+(69,)),'tracers':rows[:,86:].reshape(leading+(ntrac,))}
    timing=dict(zip(('kernel_wall_seconds','kernel_cpu_seconds','number_checksum','mass_checksum','tracer_checksum','clock_ticks_per_second'),output[:6].tolist()))
    timing.update(repetitions=int(repetitions),steps=int(steps),cases=count,registry_setup_included=True,input_reset_included=True)
    return arrays,timing

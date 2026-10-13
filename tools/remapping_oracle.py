# Copyright 2026 Reflective
# SPDX-License-Identifier: Apache-2.0
"""Verbatim standalone SALSA distribution-update oracle, including native quirks.

Only data modules are stubbed. Descending updates, repeated donor checks and
optional soluble transfer retain their original order and arithmetic. Native
loop counts are captured outside timing; all runs have a subprocess timeout.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import subprocess

import numpy as np

from tools.coagulation_process_oracle import _extract, FLAGS, CHECK_FLAGS
from tools.state_capture import load_fixture
from tools.condensation_capture import load_fixture as load_condensation_fixture

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ('parkind1.F90', 'mo_kind.F90', 'mo_math_constants.F90', 'mo_physical_constants.F90',
           'mo_ham_salsa_update.F90', 'mo_ham_salsactl.F90', 'mo_ham_species.F90', 'mo_ham.F90')
PARAMETERS = ('number_threshold', 'organic_soluble_fraction', 'water_surface_tension', 'ion_count', 'saturation_limit',
              'sulfate_molar_mass', 'sulfate_density', 'water_molar_mass', 'water_density')
STUBS = '''module mo_ham_salsactl
 use mo_kind,only: dp
 implicit none
 integer,parameter :: in1a=1,fn1a=3,in2a=4,fn2a=10,in2b=11,fn2b=17
 real(dp) :: nlim=1._dp,epsoc=.15_dp,surfw0=.073_dp,ions=3._dp,slim=1.005_dp
 real(dp) :: dpmid(17),vlolim(17),vhilim(17)
 logical :: lsol2b=.false.
end module
module mo_ham
 implicit none
 integer,parameter :: subm_naerospec_nowat=5
end module
module mo_species
 use mo_kind,only: dp
 implicit none
 type species_type
   real(dp) :: moleweight,density
 end type
 type(species_type) :: speclist(2)
end module
module mo_ham_species
 implicit none
 integer,parameter :: id_so4=1,id_wat=2
end module
'''
DRIVER = '''program remapping_reference
 use iso_fortran_env,only: int64
 use mo_kind,only: dp
 use mo_ham_salsactl
 use mo_species,only: speclist,species_type
 use mo_math_constants,only: pi
 use mo_physical_constants,only: argas
 use extracted_remapping,only: distr_update,capture_iterations,last_iterations
 implicit none
 integer :: n,repetitions,passes,i,j,k
 integer(int64) :: wall_start,wall_end,clock_rate
 integer,allocatable :: transfer(:),iterations(:,:)
 real(dp),allocatable :: initial_number(:,:),initial_volume(:,:,:),dry(:,:),lower(:,:),upper(:,:),temperature(:),properties(:,:)
 real(dp),allocatable :: number_out(:,:),volume_out(:,:,:)
 real(dp) :: number(1,1,17),volume(1,1,17,5),temp(1,1),started,finished
 character(len=32) :: command
 call get_command_argument(1,command)
 if(trim(command)=='--constants') then
   write(*,'(ES26.17E3)') pi,argas
   stop
 end if
 read(*,*) n,repetitions,passes
 allocate(transfer(n),iterations(passes,n),initial_number(17,n),initial_volume(17,5,n),dry(17,n),lower(17,n),upper(17,n))
 allocate(temperature(n),properties(9,n),number_out(17,n),volume_out(17,5,n))
 do i=1,n
   read(*,*) temperature(i),transfer(i)
   read(*,*) properties(:,i)
   read(*,*) dry(:,i)
   read(*,*) lower(:,i)
   read(*,*) upper(:,i)
   read(*,*) initial_number(:,i)
   read(*,*) initial_volume(:,:,i)
 end do
 capture_iterations=.true.
 do i=1,n
   call run_case(i)
 end do
 capture_iterations=.false.
 call cpu_time(started)
 call system_clock(wall_start,clock_rate)
 do j=1,repetitions
   do i=1,n
     call run_case(i)
   end do
 end do
 call system_clock(wall_end)
 call cpu_time(finished)
 write(*,'(5(ES26.17E3,1X))') real(wall_end-wall_start,dp)/real(clock_rate,dp),finished-started, &
                            sum(number_out),sum(volume_out),real(clock_rate,dp)
 do i=1,n
   write(*,'(ES26.17E3)') number_out(:,i),volume_out(:,:,i)
   write(*,'(I0)') iterations(:,i)
 end do
contains
 subroutine run_case(index)
   integer,intent(in) :: index
   integer :: pass
   number(1,1,:)=initial_number(:,index)
   volume(1,1,:,:)=initial_volume(:,:,index)
   temp=temperature(index)
   dpmid=dry(:,index)
   vlolim=lower(:,index)
   vhilim=upper(:,index)
   nlim=properties(1,index)
   epsoc=properties(2,index)
   surfw0=properties(3,index)
   ions=properties(4,index)
   slim=properties(5,index)
   speclist(1)=species_type(properties(6,index),properties(7,index))
   speclist(2)=species_type(properties(8,index),properties(9,index))
   lsol2b=transfer(index)==1
   do pass=1,passes
     call distr_update(1,1,1,number,volume,temp)
     if(capture_iterations) iterations(pass,index)=last_iterations
   end do
   number_out(:,index)=number(1,1,:)
   volume_out(:,:,index)=volume(1,1,:,:)
 end subroutine
end program
'''


def build_oracle(build_directory, compiler='gfortran', *, checked=True):
    directory = Path(build_directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    routine = _extract((ROOT / 'mo_ham_salsa_update.F90').read_text(), 'SUBROUTINE', 'distr_update')
    (directory / 'distr_update_verbatim.inc').write_text(routine+'\n')
    initialization = '\n    DO jj = 1,klev\n'
    anchor = '             within_bins = .TRUE.\n'
    if routine.count(initialization) != 1 or routine.count(anchor) != 1:
        raise ValueError('Missing or ambiguous remapping diagnostic anchor')
    diagnostic = '             IF (capture_iterations) last_iterations=last_iterations+1\n'
    instrumented = routine.replace(initialization, '    IF (capture_iterations) last_iterations=0\n'+initialization).replace(anchor, anchor+diagnostic)
    wrappers = {'stubs.f90': STUBS, 'remapping.F90': 'module extracted_remapping\n logical :: capture_iterations=.false.\n integer :: last_iterations\ncontains\n'+instrumented+'\nend module\n', 'driver.f90': DRIVER}
    for name,text in wrappers.items():
        (directory/name).write_text(text)
    flags = FLAGS + CHECK_FLAGS if checked else FLAGS
    objects=[]
    for path in [*[ROOT/name for name in SOURCES[:4]], *[directory/name for name in wrappers]]:
        obj=directory/(path.stem+'.o')
        subprocess.run([compiler,*flags,'-c',str(path),'-o',str(obj)],cwd=directory,check=True,capture_output=True,text=True)
        objects.append(str(obj))
    executable=directory/'remapping_oracle'
    subprocess.run([compiler,*flags,*objects,'-o',str(executable)],cwd=directory,check=True,capture_output=True,text=True)
    constants=np.fromstring(subprocess.check_output([str(executable),'--constants'],text=True),sep=' ')
    sha=lambda content:hashlib.sha256(content).hexdigest()
    provenance={'source_sha256':{name:sha((ROOT/name).read_bytes()) for name in SOURCES},
                'routine_sha256':sha(routine.encode()),'wrapper_sha256':{name:sha(text.encode()) for name,text in wrappers.items()},
                'diagnostic_overlay':diagnostic.strip(),'flags':list(flags),'checked':checked,
                'compiled_constants':dict(zip(('pi','gas_constant'),constants.tolist())),
                'configuration':{'bins':17,'species':5,'boundaries':[1,3,4,10,11,17],'native_loop_uncapped':True},
                'separate_compilation_without_lto':True,
                'compiler':subprocess.check_output([compiler,'--version'],text=True).splitlines()[0]}
    return executable,provenance


def reference_cases():
    native=load_fixture(ROOT/'tests/fixtures/native-default')
    def rec(capture,stage,name):
        item=capture[stage][name];return np.array(item['values']).reshape(item['shape'],order='F')
    dry=rec(native,'entry','dpmid');lower=rec(native,'entry','vlolim');upper=rec(native,'entry','vhilim')
    middle=np.pi/6*dry**3
    labels,records=[],[]
    def add(label,number=None,volume=None,*,soluble_transfer=False,temperature=280.,**parameters):
        number=np.zeros(17) if number is None else np.asarray(number,dtype=float)
        volume=np.zeros((17,5)) if volume is None else np.asarray(volume,dtype=float)
        labels.append(label)
        records.append({'number':number,'volume':volume,'temperature':temperature,'dry_diameter':dry.copy(),
                        'lower_volume':lower.copy(),'upper_volume':upper.copy(),'soluble_transfer':soluble_transfer,
                        'number_threshold':1.,'organic_soluble_fraction':.15,'water_surface_tension':.073,
                        'ion_count':3.,'saturation_limit':1.005,'sulfate_molar_mass':96.0631,'sulfate_density':1841.,
                        'water_molar_mass':18.,'water_density':1000.,**parameters})
    condensation=load_condensation_fixture(ROOT/'tests/fixtures/condensation-default')
    add('native-condensation-exit',rec(condensation,'exit','pnaero')[0,0],rec(condensation,'exit','zvols')[0,0],
        temperature=float(rec(condensation,'entry','ptemp')[0,0]))
    add('empty')
    def single(label,bin_index,mean,*,number_value=16.,material=0,**kwargs):
        number=np.zeros(17);number[bin_index]=number_value
        volume=np.zeros((17,5));volume[bin_index,material]=number_value*mean
        add(label,number,volume,**kwargs)
    for bin_index in (0,2,3,10,11,15):
        single(f'below-lower-bin{bin_index+1}',bin_index,.8*lower[bin_index])
        single(f'partial-growth-bin{bin_index+1}',bin_index,1.2*middle[bin_index])
    for label,mean in (('exact-lower',lower[3]),('below-midpoint',np.nextafter(middle[3],0.)),
                       ('exact-midpoint',middle[3]),('above-midpoint',np.nextafter(middle[3],np.inf)),
                       ('exact-upper',upper[3])):
        single(label,3,mean)
    for bin_index in (9,16):
        single(f'ignored-terminal-small-bin{bin_index+1}',bin_index,.1*lower[bin_index])
        single(f'ignored-terminal-large-bin{bin_index+1}',bin_index,100*upper[bin_index])
    for label,value in (('below-number-threshold',np.nextafter(1.,0.)),('at-number-threshold',1.),('above-number-threshold',np.nextafter(1.,np.inf))):
        single(label,3,1.2*middle[3],number_value=value)
    single('full-number-transfer-orphan-volume',3,10*upper[3])
    single('descending-downward-cascade',15,.8*lower[10])
    single('below2b-to-regime1-no-regrowth',10,.8*middle[2])
    for material,name in ((0,'sulfate'),(1,'organic'),(2,'black-carbon')):
        single('soluble-transfer-'+name,16,middle[16],material=material,soluble_transfer=True)
    single('soluble-transfer-disabled',16,middle[16])
    single('soluble-transfer-small-bin',10,middle[10],soluble_transfer=True)
    # Measured from the actual zepsv/T**3 diagnostic at default properties,
    # T=280 K and native bin17 geometry. Adjacent-float branch tests establish
    # that this is the compiled reference threshold, rather than a fit tolerance.
    native_threshold=1.23784536088485407e-7
    total_volume=2.**-48
    for label,fraction in (('below',np.nextafter(native_threshold,0.)),
                           ('at',native_threshold),('above',np.nextafter(native_threshold,np.inf))):
        number=np.zeros(17);number[16]=16.
        volume=np.zeros((17,5))
        volume[16,0]=fraction*total_volume
        volume[16,2]=total_volume-volume[16,0]
        add(f'{label}-soluble-fraction-threshold',number,volume,soluble_transfer=True)
    single('zero-volume-first-bin',0,0.)
    single('zero-volume-downward-bin',3,0.)
    single('inactive-number-positive-volume',3,middle[3],number_value=.5)
    single('heterogeneous-solubility-parameters',16,middle[16],material=1,soluble_transfer=True,
           organic_soluble_fraction=.3,water_surface_tension=.08,ion_count=2.,saturation_limit=1.01,
           sulfate_molar_mass=98.,sulfate_density=1900.,water_molar_mass=18.015,water_density=995.,temperature=240.)
    rng=np.random.default_rng(3829)
    for index in range(6):
        number=10**rng.uniform(2,8,17)
        fraction=rng.dirichlet(np.ones(5),17)
        volume=(number*middle*rng.uniform(.9,1.2,17))[:,None]*fraction
        add(f'mixed-{index}',number,volume,soluble_transfer=bool(index%2),temperature=rng.uniform(240,310))
    return {'labels':labels,**{key:np.stack([case[key] for case in records]) for key in records[0]}}


def run_oracle(executable,cases,repetitions=1,*,passes=1,timeout=10):
    number=np.asarray(cases['number'],dtype=float)
    if number.ndim!=2 or number.shape[1]!=17 or not len(number):
        raise ValueError('number must have nonempty shape (cases,17)')
    if any(not isinstance(value,(int,np.integer)) or value<1 for value in (repetitions,passes)):
        raise ValueError('repetitions and passes must be positive integers')
    n=len(number);values={'number':number}
    shapes={'volume':(n,17,5),**{name:(n,17) for name in ('dry_diameter','lower_volume','upper_volume')},
            **{name:(n,) for name in ('temperature','soluble_transfer',*PARAMETERS)}}
    for name,shape in shapes.items():
        value=np.asarray(cases[name],dtype=float)
        if name in ('dry_diameter','lower_volume','upper_volume') and value.shape==(17,):
            value=np.broadcast_to(value,shape)
        if value.shape!=shape:
            raise ValueError(f'{name} must have shape {shape}')
        values[name]=value
    for name,value in values.items():
        nonnegative=name in ('number','volume','number_threshold','organic_soluble_fraction','soluble_transfer')
        if not np.all(np.isfinite(value)) or np.any(value<0 if nonnegative else value<=0):
            raise ValueError(f'Invalid {name}')
    if np.any(values['lower_volume']>=values['upper_volume']):
        raise ValueError('Volume bounds must increase')
    if not np.all(np.isin(values['soluble_transfer'],[0,1])):
        raise ValueError('soluble_transfer must contain booleans')
    lines=[f'{n} {repetitions} {passes}']
    for i in range(n):
        lines.append(f"{values['temperature'][i]:.17e} {int(values['soluble_transfer'][i])}")
        for row in ([values[name][i] for name in PARAMETERS],values['dry_diameter'][i],values['lower_volume'][i],
                    values['upper_volume'][i],number[i],values['volume'][i].ravel(order='F')):
            lines.append(' '.join(f'{value:.17e}' for value in row))
    result=subprocess.run([str(executable)],input='\n'.join(lines)+'\n',capture_output=True,text=True,check=True,timeout=timeout)
    output=np.fromstring(result.stdout,sep=' ')
    if output.size!=5+n*(102+passes) or not np.all(np.isfinite(output)):
        raise ValueError('Unexpected or nonfinite native remapping output')
    rows=output[5:].reshape(n,102+passes)
    state={'number':rows[:,:17],'volume':np.stack([row[17:102].reshape(17,5,order='F') for row in rows]),'iterations':rows[:,102:].astype(int)}
    timing={'kernel_wall_seconds':float(output[0]),'kernel_cpu_seconds':float(output[1]),'number_checksum':float(output[2]),
            'volume_checksum':float(output[3]),'clock_ticks_per_second':int(output[4]),'repetitions':int(repetitions),
            'passes':int(passes),'cases':n,'input_reset_included':True,'iteration_capture_included':False}
    return state,timing

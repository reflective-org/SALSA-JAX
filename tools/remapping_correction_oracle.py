# Copyright 2026 Reflective
# SPDX-License-Identifier: Apache-2.0
"""Execute the actual native post-remap cleanup/hydration/radii-water block."""
from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess

import numpy as np

from tools import hydration_oracle

ROOT = Path(__file__).resolve().parents[1]
FLOAT_FIELDS = ('number','volume','gas','core','wet_diameter','dry_radius_cm','wet_radius_cm','water_mass')
DRIVER = '''program correction_reference
 use mo_kind, only: dp
 use mo_species, only: speclist,species_type
 use mo_ham, only: sizeclass
 use mo_ham_salsactl, only: dpmid,epsoc,nlim,surfw0
 use extracted_hydration, only: capture_iterations,last_iterations
 use extracted_correction, only: correction,isubm_ocnv
 implicit none
 integer :: n,i,k,organic,soluble(17)
 real(dp) :: properties(5),masses(5),densities(5)
 real(dp) :: number(1,1,17),volume(1,1,17,5),gas(1,1,3),rh(1,1),temperature(1,1)
 real(dp) :: core(1,1,17),wet(1,1,17),dry_radius(1,1,17),wet_radius(1,1,17),water(1,1,17)
 read(*,*) n
 capture_iterations=.true.
 do i=1,n
   read(*,*) rh,temperature,organic
   read(*,*) properties
   read(*,*) masses
   read(*,*) densities
   read(*,*) soluble
   read(*,*) dpmid
   read(*,*) number
   read(*,*) volume
   read(*,*) gas
   epsoc=properties(1)
   surfw0=properties(2)
   nlim=properties(3)
   speclist(6)=species_type(properties(4),properties(5))
   do k=1,5
     speclist(k)=species_type(masses(k),densities(k))
   end do
   sizeclass(:)%lsoluble=soluble==1
   isubm_ocnv=merge(3,0,organic==1)
   call correction(1,1,1,number,volume,gas,rh,temperature,core,wet,dry_radius,wet_radius,water)
   write(*,'(ES26.17E3)') number,volume,gas,core,wet,dry_radius,wet_radius,water
   write(*,'(I0)') last_iterations
 end do
end program
'''
HEADER = '''module extracted_correction
 use mo_kind, only: dp
 use mo_species, only: speclist
 use mo_ham_species, only: id_so4,id_oc,id_wat
 use mo_ham_salsactl
 use mo_math_constants, only: pi
 use mo_physical_constants, only: avo
 use extracted_hydration, only: equilibration
 implicit none
 integer,parameter :: isubm_so4g=2
 integer :: isubm_ocnv=0
contains
 subroutine correction(kproma,kbdim,klev,pnaero,zvols,pc_gas,prh,ptemp,zcore,zdwet,pm6dry,pm6rp,pww)
 integer,intent(in) :: kproma,kbdim,klev
 real(dp),intent(inout) :: pnaero(kbdim,klev,17),zvols(kbdim,klev,17,5),pc_gas(kbdim,klev,3)
 real(dp),intent(in) :: prh(kbdim,klev),ptemp(kbdim,klev)
 real(dp),intent(out) :: zcore(kbdim,klev,17),zdwet(kbdim,klev,17),pm6dry(kbdim,klev,17), &
                         pm6rp(kbdim,klev,17),pww(kbdim,klev,17)
 real(dp) :: zddry
 integer :: ii,jj,kk
'''


def correction_block():
    """Extract the entire native block after one distr_update, without repairs."""
    source = (ROOT/'mo_ham_salsa.F90').read_text()
    loop = source.index('    DO nn = 1, 2\n')
    start = source.index('       DO kk = in1a,fn1a',loop)
    end = source.index('\n    END DO\n\n    !---------------------------------------------------------------',start)
    return source[start:end]


def build_oracle(build_directory,compiler='gfortran'):
    directory=Path(build_directory).resolve()
    directory.mkdir(parents=True,exist_ok=True)
    hydration_directory=directory/'hydration'
    _,hydration_provenance=hydration_oracle.build_oracle(hydration_directory,compiler,checked=True)
    block=correction_block()
    (directory/'correction_verbatim.inc').write_text(block+'\n')
    wrapper=HEADER+block+'\n end subroutine\nend module\n'
    texts={'correction.f90':wrapper,'driver.f90':DRIVER}
    objects=[str(hydration_directory/(Path(name).stem+'.o')) for name in hydration_oracle.SOURCES[:4]]
    objects += [str(hydration_directory/name) for name in ('stubs.o','hydration.o')]
    flags=tuple(hydration_provenance['flags'])
    for name,text in texts.items():
        source=directory/name
        source.write_text(text)
        obj=source.with_suffix('.o')
        subprocess.run([compiler,*flags,'-I',str(hydration_directory),'-c',str(source),'-o',str(obj)],
                       cwd=directory,check=True,capture_output=True,text=True)
        objects.append(str(obj))
    executable=directory/'remapping_correction_oracle'
    subprocess.run([compiler,*flags,*objects,'-o',str(executable)],cwd=directory,
                   check=True,capture_output=True,text=True)
    sha=lambda text:hashlib.sha256(text.encode()).hexdigest()
    provenance={'source_sha256':{'mo_ham_salsa.F90':hashlib.sha256((ROOT/'mo_ham_salsa.F90').read_bytes()).hexdigest()},
                'correction_block_sha256':sha(block),'wrapper_sha256':{name:sha(text) for name,text in texts.items()},
                'hydration_provenance':hydration_provenance,'flags':list(flags),'checked':True,
                'scope':'one post-distr_update cleanup/hydration/radii-water pass; no redistribution or final density',
                'gas_layout':['SO2','H2SO4','OC'],'organic_absent_index':0,
                'D007':'original unguarded organic gas access retained; checked execution fails if cleanup reaches it',
                'compiler':hydration_provenance['compiler']}
    return executable,provenance


def run_oracle(executable,cases):
    """Execute checked native block; D007 failure propagates as CalledProcessError."""
    number=np.asarray(cases['number'],dtype=float)
    if number.ndim!=2 or number.shape[1]!=17 or not len(number):
        raise ValueError('number must have nonempty shape (cases,17)')
    n=len(number)
    values={'number':number}
    shapes={'volume':(n,17,5),'gas':(n,3),'organic_enabled':(n,),
            'dry_diameter':(n,17),'soluble_mask':(n,17),'species_molar_mass':(n,5),'species_density':(n,5),
            **{key:(n,) for key in ('relative_humidity','temperature',*hydration_oracle.SCALAR_PROPERTIES)}}
    for key,shape in shapes.items():
        value=np.asarray(cases[key],dtype=float)
        if value.shape!=shape:
            raise ValueError(f'{key} must have shape {shape}')
        if not np.all(np.isfinite(value)):
            raise ValueError(f'{key} must be finite')
        if key!='relative_humidity':
            nonnegative=key in ('number','volume','gas','organic_enabled','soluble_mask','organic_soluble_fraction')
            if np.any(value<0 if nonnegative else value<=0):
                raise ValueError(f'Invalid {key}')
        values[key]=value
    if not np.all(np.isfinite(number)) or np.any(number<0):
        raise ValueError('number must be finite and nonnegative')
    for key in ('soluble_mask','organic_enabled'):
        if not np.all(np.isin(values[key],[0,1])):
            raise ValueError(f'{key} must contain booleans')
    lines=[str(n)]
    for i in range(n):
        lines.append(f"{values['relative_humidity'][i]:.17e} {values['temperature'][i]:.17e} {int(values['organic_enabled'][i])}")
        for row in ([values[key][i] for key in hydration_oracle.SCALAR_PROPERTIES],
                    values['species_molar_mass'][i],values['species_density'][i]):
            lines.append(' '.join(f'{value:.17e}' for value in row))
        lines.append(' '.join(str(int(value)) for value in values['soluble_mask'][i]))
        for row in (values['dry_diameter'][i],number[i],values['volume'][i].ravel(order='F'),values['gas'][i]):
            lines.append(' '.join(f'{value:.17e}' for value in row))
    result=subprocess.run([str(executable)],input='\n'.join(lines)+'\n',capture_output=True,text=True,
                          check=True,timeout=30)
    output=np.fromstring(result.stdout,sep=' ')
    if output.size!=n*207 or not np.all(np.isfinite(output)):
        raise ValueError('Unexpected native correction output')
    rows=output.reshape(n,207)
    result={'number':rows[:,:17], 'volume':np.stack([row[17:102].reshape(17,5,order='F') for row in rows]),
            'gas':rows[:,102:105]}
    for i,field in enumerate(FLOAT_FIELDS[3:]):
        result[field]=rows[:,105+i*17:122+i*17]
    result['iterations']=rows[:,190:207].astype(int)
    return result

"""Actual SALSA initialization branch must preserve HAM's registered components."""
from pathlib import Path
import re
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


def compile_initialization(directory, source):
    branch = re.search(r"CASE\(HAM_SALSA\)(.*?)END SELECT", source, re.S)[1]
    program = '''PROGRAM check_components
USE iso_fortran_env, ONLY: dp=>real64
IMPLICIT NONE
INTEGER, PARAMETER :: kproma=1,in1a=1,fn1a=3,in2a=4,fn2a=10,in2b=11,fn2b=17
INTEGER, PARAMETER :: id_so4=1,id_oc=2,id_bc=3,id_ss=4,id_du=5
TYPE species_type
  REAL(dp) :: density
END TYPE
TYPE(species_type) :: speclist(5)
INTEGER :: ii,jj,i,iso4b(17),iocb(17),ibcb(17),issb(17),idub(17),saved(17,5),maps(17,5)
REAL(dp) :: paerml(1,1,69),paernl(1,1,17),core(17),zmvsu,volume(17,5),fractions(17,5)
speclist%density=[1841._dp,2000._dp,2000._dp,2165._dp,2650._dp]
zmvsu=98.e-3_dp/6.02214179e23_dp/1841._dp
iso4b=[(i,i=1,17)]
iocb=[(i,i=18,34)]
ibcb=0; issb=0; idub=0
ibcb(4:17)=[(i,i=35,48)]
idub(4:17)=[(i,i=49,62)]
issb(4:10)=[(i,i=63,69)]
saved(:,1)=iso4b; saved(:,2)=iocb; saved(:,3)=ibcb; saved(:,4)=issb; saved(:,5)=idub
core=[(REAL(i,dp)**3*1.e-27_dp,i=1,17)]
paernl=1.e6_dp
paerml=0._dp
''' + branch + '''
maps(:,1)=iso4b; maps(:,2)=iocb; maps(:,3)=ibcb; maps(:,4)=issb; maps(:,5)=idub
IF (ANY(maps/=saved)) ERROR STOP 'registered component maps were overwritten'
volume=0._dp
fractions=0._dp
fractions(1:3,1)=0.2_dp; fractions(1:3,2)=0.8_dp
fractions(4:10,1)=0.5_dp; fractions(4:10,2:3)=0.05_dp
fractions(4:10,4)=0.1_dp; fractions(4:10,5)=0.3_dp
fractions(11:17,1:3)=0.01_dp; fractions(11:17,5)=0.97_dp
DO ii=1,17
  volume(ii,1)=paerml(1,1,iso4b(ii))*1.e6_dp*zmvsu
  DO jj=2,5
    IF (maps(ii,jj)>0) volume(ii,jj)=paerml(1,1,maps(ii,jj))*1.e-9_dp/speclist(jj)%density
  END DO
  IF (ANY(ABS(volume(ii,:)/(paernl(1,1,ii)*core(ii))-fractions(ii,:))>2.e-15_dp)) &
    ERROR STOP 'wrong species volume fraction'
END DO
IF (ANY(paerml<=0._dp)) ERROR STOP 'unfilled component'
PRINT *, 'registered maps and species fractions preserved'
END PROGRAM
'''
    path = directory / 'mapping.f90'
    path.write_text(program)
    executable = directory / 'mapping'
    subprocess.run(['gfortran', '-O0', '-fcheck=all', '-ffpe-trap=invalid,zero,overflow',
                    '-ffree-line-length-none', str(path), '-o', str(executable)],
                   cwd=directory, check=True, capture_output=True, text=True)
    return subprocess.run([str(executable)], capture_output=True, text=True)


def test_original_initialization_overwrites_registered_maps(tmp_path):
    source = subprocess.check_output(['git', 'show', '5b37e45:mo_convert_concentrations.F90'], cwd=ROOT, text=True)
    result = compile_initialization(tmp_path, source)
    assert result.returncode != 0
    assert 'registered component maps were overwritten' in result.stderr


def test_current_initialization_preserves_maps_and_fractions(tmp_path):
    result = compile_initialization(tmp_path, (ROOT / 'mo_convert_concentrations.F90').read_text())
    assert result.returncode == 0, result.stderr
    assert 'species fractions preserved' in result.stdout

"""Checked oracle executing verbatim native set_sizebins with its actual layout."""
from __future__ import annotations

import hashlib
from pathlib import Path
import re
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ('parkind1.F90', 'mo_kind.F90', 'mo_math_constants.F90',
           'mo_ham_salsactl.F90', 'mo_ham_salsa_init.F90')
FIELDS = ('vlolim', 'vhilim', 'dpmid', 'vratiohi', 'vratiolo')
FLAGS = ('-O2', '-fno-fast-math', '-fno-lto', '-cpp', '-ffree-line-length-none',
         '-fcheck=all', '-ffpe-trap=invalid,zero,overflow', '-fbacktrace')
DRIVER = '''program grid_reference
 use mo_kind, only: dp
 use mo_math_constants, only: pi
 use mo_ham_salsactl
 use extracted_grid, only: set_sizebins
 implicit none
 integer :: n,i
 character(len=20) :: argument
 call get_command_argument(1,argument)
 if (trim(argument)=='--configuration') then
   write(*,'(I0)') nreg,nbin,nbin2,nbin3,in1a,fn1a,in2a,fn2a,in2b,fn2b,storage_size(1.),storage_size(1._dp)
   write(*,'(ES26.17E3)') reglim,pi
   stop
 end if
 read(*,*) n
 do i=1,n
   read(*,*) reglim
   call set_sizebins()
   write(*,'(ES26.17E3)') vlolim,vhilim,dpmid,vratiohi,vratiolo
 end do
end program
'''


def build_oracle(build_directory, compiler='gfortran'):
    """Compile original arithmetic/layout; only expose reglim as test input."""
    directory = Path(build_directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    init = (ROOT / SOURCES[-1]).read_text()
    match = re.search(r'^  SUBROUTINE set_sizebins\(\).*?^  END SUBROUTINE set_sizebins', init, re.M | re.S)
    if match is None:
        raise ValueError('Could not find original set_sizebins')
    routine = match.group(0)
    control = (ROOT / 'mo_ham_salsactl.F90').read_text()
    match = re.search(r'^  INTEGER, PARAMETER ::[^\n]*\n\s*nreg = 2\b.*?^\s*nbins = fn2b[^\n]*', control, re.M | re.S)
    if match is None:
        raise ValueError('Could not find native grid configuration declarations')
    declarations = match.group(0)
    if declarations.count('REAL(dp), PARAMETER ::') != 1:
        raise ValueError('Expected one regime-limit declaration')
    # All counts, bounds and default diameters remain actual source text.
    # Removing PARAMETER only from reglim lets the driver test other limits.
    test_declarations = declarations.replace('REAL(dp), PARAMETER ::', 'REAL(dp) ::', 1)
    wrapper = ('module mo_ham_salsactl\n use mo_kind, only: dp\n implicit none\n' + test_declarations +
               '\n real(dp) :: ' + ','.join(f'{name}(fn2b)' for name in FIELDS) + '\nend module\n')
    texts = {'grid_configuration.f90': wrapper,
             'grid.f90': 'module extracted_grid\ncontains\n'+routine+'\nend module\n',
             'driver.f90': DRIVER}
    (directory / 'set_sizebins_verbatim.inc').write_text(routine+'\n')
    (directory / 'grid_configuration_verbatim.inc').write_text(declarations+'\n')
    for name, text in texts.items():
        (directory / name).write_text(text)
    objects = []
    for path in [*[ROOT / name for name in SOURCES[:3]], *[directory/name for name in texts]]:
        output = directory/(path.stem+'.o')
        subprocess.run([compiler,*FLAGS,'-c',str(path),'-o',str(output)],cwd=directory,
                       check=True,capture_output=True,text=True)
        objects.append(str(output))
    executable = directory/'grid_oracle'
    subprocess.run([compiler,*FLAGS,*objects,'-o',str(executable)],cwd=directory,
                   check=True,capture_output=True,text=True)
    raw = np.fromstring(subprocess.check_output([str(executable),'--configuration'],text=True),sep=' ')
    if raw.shape != (18,):
        raise ValueError('Unexpected compiled grid configuration')
    configuration = {'nreg':int(raw[0]),'nbin':raw[1:3].astype(int).tolist(),
                     'nbin2':int(raw[3]),'nbin3':int(raw[4]),'boundaries':raw[5:11].astype(int).tolist(),
                     'default_real_bits':int(raw[11]),'dp_bits':int(raw[12]),
                     'reglim':raw[13:17].tolist(),'pi':float(raw[17])}
    sha = lambda text: hashlib.sha256(text.encode()).hexdigest()
    provenance = {'source_sha256':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in SOURCES},
                  'routine_sha256':sha(routine),'native_configuration_sha256':sha(declarations),
                  'wrapper_sha256':{name:sha(text) for name,text in texts.items()},
                  'configuration_overlay':'reglim PARAMETER removed only to accept test inputs; all other native declarations unchanged',
                  'configuration':configuration,'flags':list(FLAGS),'checked':True,
                  'compiler':subprocess.check_output([compiler,'--version'],text=True).splitlines()[0]}
    return executable,provenance


def run_oracle(executable, regime_limits):
    """Return five grid fields for finite increasing inputs (...,4), in SI units."""
    limits = np.asarray(regime_limits,dtype=np.float64)
    if limits.shape[-1:] != (4,) or limits.size == 0:
        raise ValueError('Expected nonempty regime_limits (...,4)')
    if not np.all(np.isfinite(limits)) or np.any(limits<=0) or np.any(np.diff(limits,axis=-1)<=0):
        raise ValueError('Regime limits must be finite, positive and strictly increasing')
    rows = limits.reshape(-1,4)
    payload = str(len(rows))+'\n'+'\n'.join(' '.join(f'{v:.17e}' for v in row) for row in rows)+'\n'
    result = subprocess.run([str(executable)],input=payload,capture_output=True,text=True,check=True)
    values = np.fromstring(result.stdout,sep=' ')
    if values.size != len(rows)*len(FIELDS)*17 or not np.all(np.isfinite(values)):
        raise ValueError('Unexpected Fortran grid output')
    values = values.reshape(len(rows),len(FIELDS),17)
    return {name:values[:,i,:].reshape(limits.shape[:-1]+(17,)) for i,name in enumerate(FIELDS)}

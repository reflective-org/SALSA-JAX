"""Execute the byte-identical native seven-mode size-distribution initializer.

Only geometry data are stubbed. The driver-default mode assignments are extracted
verbatim, preserving default-real literal rounding before conversion to REAL(dp).
The native midpoint quadrature evaluates every mode in every sectional bin;
its overlapping 2a/2b bins are not partitioned by modal composition.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import re
import subprocess
import numpy as np
from tools.coagulation_process_oracle import FLAGS,CHECK_FLAGS
from tools.state_capture import load_fixture

ROOT=Path(__file__).resolve().parents[1]
SOURCES=('parkind1.F90','mo_kind.F90','mo_math_constants.F90','mo_ham_salsa_sizedist.F90')
FIELDS={'mode_number':7,'dpg':7,'sigma':7,'lower_volume':17,'upper_volume':17,'dry_diameter':17}
STUB='''module mo_ham_salsactl
 use mo_kind,only:dp
 implicit none
 integer,parameter::nreg=2,in1a=1,fn2b=17
 real(dp)::vlolim(17),vhilim(17),dpmid(17)
end module
'''
DRIVER='''program initialization_reference
 use iso_fortran_env,only:int64
 use mo_kind,only:dp
 use mo_math_constants,only:pi
 use mo_ham_salsactl
 use mo_ham_salsa_sizedist,only:size_distribution
 implicit none
 integer::ncases,repetitions,kproma,kbdim,klev,i,j
 integer(int64)::started,finished,clock_rate
 real(dp)::n(7),dpg(7),sigmag(7),cpu_start,cpu_end
 real(dp),allocatable::modes(:,:,:),geometry(:,:,:),number(:,:,:,:)
 character(len=32)::command
 call get_command_argument(1,command)
 if(trim(command)=='--defaults') then
__DEFAULTS__
 write(*,'(ES26.17E3)') n,dpg,sigmag
 stop
 end if
 if(trim(command)=='--constants') then
 write(*,'(ES26.17E3)') pi
 stop
 end if
 read(*,*) ncases,repetitions,kproma,kbdim,klev
 allocate(modes(7,3,ncases),geometry(17,3,ncases),number(kbdim,klev,17,ncases))
 do i=1,ncases
 read(*,*) modes(:,:,i)
 read(*,*) geometry(:,:,i)
 end do
 do i=1,ncases
 call run_case(i)
 end do
 call cpu_time(cpu_start)
 call system_clock(started,clock_rate)
 do j=1,repetitions
 do i=1,ncases
 call run_case(i)
 end do
 end do
 call system_clock(finished)
 call cpu_time(cpu_end)
 write(*,'(4ES26.17E3)') real(finished-started,dp)/real(clock_rate,dp),cpu_end-cpu_start,sum(number),real(clock_rate,dp)
 do i=1,ncases
 write(*,'(ES26.17E3)') number(:,:,:,i)
 end do
contains
 subroutine run_case(index)
 integer,intent(in)::index
 vlolim=geometry(:,1,index);vhilim=geometry(:,2,index);dpmid=geometry(:,3,index)
 call size_distribution(kproma,kbdim,klev,modes(:,1,index),modes(:,2,index),modes(:,3,index),number(:,:,:,index))
 end subroutine
end program
'''


def default_assignments():
    source=(ROOT/'driver.F90').read_text()
    expressions={}
    for name in ('n','dpg','sigmag'):
        matches=re.findall(r'^\s*'+name+r'\s*=\s*\(/[^\n]+$',source,re.M|re.I)
        if len(matches)!=1:raise ValueError(f'Missing or ambiguous native default assignment: {name}')
        expressions[name]=matches[0].strip()
    return expressions


def driver_defaults():
    """Parse only the native literal data, retaining their Fortran precision."""
    result={}
    for name,expression in default_assignments().items():
        match=re.fullmatch(r'\w+\s*=\s*\(/([^/]+?)/\)\s*(?:\*\s*([\d.eE+-]+)_dp)?',expression)
        if match is None:raise ValueError(f'Unsupported native data expression: {expression}')
        # Each unsuffixed literal is default REAL; multiplication is REAL(dp).
        data=np.asarray([float(value) for value in match[1].split(',')],dtype=np.float32).astype(np.float64)
        if data.shape!=(7,):raise ValueError('Native mode count changed')
        result[{'n':'mode_number','dpg':'dpg','sigmag':'sigma'}[name]]=data*(float(match[2]) if match[2] else 1.)
    return result


def build_oracle(build_directory,compiler='gfortran',*,checked=True):
    directory=Path(build_directory).resolve();directory.mkdir(parents=True,exist_ok=True)
    source=ROOT/'mo_ham_salsa_sizedist.F90'
    (directory/source.name).write_bytes(source.read_bytes())
    assignments=default_assignments()
    texts={'stubs.f90':STUB,'driver.f90':DRIVER.replace('__DEFAULTS__','\n'.join(assignments.values()))}
    for name,text in texts.items():(directory/name).write_text(text)
    flags=FLAGS+CHECK_FLAGS if checked else FLAGS
    objects=[]
    for path in [*[ROOT/name for name in SOURCES[:3]],directory/'stubs.f90',directory/source.name,directory/'driver.f90']:
        obj=directory/(path.stem+'.o')
        subprocess.run([compiler,*flags,'-c',str(path),'-o',str(obj)],cwd=directory,check=True,capture_output=True,text=True)
        objects.append(str(obj))
    executable=directory/'initialization_oracle'
    subprocess.run([compiler,*flags,*objects,'-o',str(executable)],cwd=directory,check=True,capture_output=True,text=True)
    compiled=np.fromstring(subprocess.check_output([str(executable),'--defaults'],text=True),sep=' ').reshape(3,7)
    defaults=driver_defaults()
    if not np.array_equal(compiled,np.stack([defaults[key] for key in ('mode_number','dpg','sigma')])):
        raise ValueError('Parsed driver defaults differ from actual native default-real arithmetic')
    sha=lambda content:hashlib.sha256(content).hexdigest()
    return executable,{'source_sha256':{name:sha((ROOT/name).read_bytes()) for name in (*SOURCES,'driver.F90')},
        'wrapper_sha256':{name:sha(text.encode()) for name,text in texts.items()},'flags':list(flags),'checked':checked,
        'scientific_module_byte_identical':source.name,'native_default_assignments':assignments,
        'compiled_defaults':{key:compiled[i].tolist() for i,key in enumerate(('mode_number','dpg','sigma'))},
        'compiled_pi':float(subprocess.check_output([str(executable),'--constants'],text=True)),
        'compiler':subprocess.check_output([compiler,'--version'],text=True).splitlines()[0],
        'configuration':{'modes':7,'bins':17,'native_default_literal_bits':32,'arithmetic_bits':64},
        'separate_compilation_without_lto':True}


def reference_cases():
    fixture=load_fixture(ROOT/'tests/fixtures/native-default')
    def record(name):
        value=fixture['entry'][name];return np.asarray(value['values']).reshape(value['shape'],order='F')
    base={**driver_defaults(),'lower_volume':record('vlolim'),'upper_volume':record('vhilim'),'dry_diameter':record('dpmid')}
    records=[];labels=[]
    def add(label,**updates):
        records.append({**{key:value.copy() for key,value in base.items()},**updates});labels.append(label)
    add('driver-default')
    add('empty',mode_number=np.zeros(7))
    for mode in range(7):
        number=np.zeros(7);number[mode]=base['mode_number'][mode]
        add(f'isolated-mode{mode+1}',mode_number=number)
    add('all-modes-equal',mode_number=np.full(7,1e8))
    add('narrow-modes',sigma=np.full(7,1.1))
    add('broad-modes',sigma=np.full(7,3.))
    add('shifted-smaller',dpg=base['dpg']*.3)
    add('shifted-larger',dpg=base['dpg']*3.)
    add('tiny-mode-counts',mode_number=base['mode_number']*1e-12)
    add('large-mode-counts',mode_number=base['mode_number']*1e8)
    scale=np.geomspace(.4,2.4,17)
    add('nonuniform-grid',lower_volume=base['lower_volume']*scale**3,upper_volume=base['upper_volume']*scale**3,dry_diameter=base['dry_diameter']*scale)
    add('narrow-bin-widths',lower_volume=base['lower_volume'],upper_volume=base['lower_volume']*1.0001)
    rng=np.random.default_rng(703)
    for i in range(5):
        add(f'mixed-{i}',mode_number=10**rng.uniform(1,10,7),dpg=10**rng.uniform(-9,-5,7),sigma=rng.uniform(1.2,2.8,7))
    return {'labels':labels,**{key:np.stack([row[key] for row in records]) for key in base}}


def run_oracle(executable,cases,repetitions=1,*,kproma=1,kbdim=1,klev=1,all_cells=False,timeout=10):
    """Return ({number}, timing), preserving arbitrary common leading case axes.

    With all_cells=True append (kbdim,klev,17); inactive horizontal cells remain
    native zeros. Default returns the first active cell and requires kproma>0.
    Timings exclude I/O/startup and include geometry assignment and native reset.
    """
    for name,value in (('repetitions',repetitions),('kbdim',kbdim),('klev',klev)):
        if not isinstance(value,(int,np.integer)) or value<1:raise ValueError(f'{name} must be a positive integer')
    if not isinstance(kproma,(int,np.integer)) or not 0<=kproma<=kbdim:raise ValueError('kproma must be between zero and kbdim')
    if not all_cells and kproma==0:raise ValueError('all_cells is required for zero active cells')
    arrays={key:np.asarray(cases[key],dtype=float) for key in FIELDS}
    for key,value in arrays.items():
        if value.shape[-1:]!=(FIELDS[key],) or not np.all(np.isfinite(value)):raise ValueError(f'Invalid {key} shape or nonfinite value')
        if np.any(value<0 if key=='mode_number' else value<=1 if key=='sigma' else value<=0):raise ValueError(f'Invalid {key} domain')
    leading=np.broadcast_shapes(*(array.shape[:-1] for array in arrays.values()))
    count=int(np.prod(leading)) if leading else 1
    if count==0:raise ValueError('Cases must be nonempty')
    arrays={key:np.broadcast_to(value,leading+(FIELDS[key],)).reshape(count,FIELDS[key]) for key,value in arrays.items()}
    if np.any(arrays['lower_volume']>=arrays['upper_volume']):raise ValueError('Volume limits must increase')
    lines=[f'{count} {repetitions} {kproma} {kbdim} {klev}']
    for i in range(count):
        for keys in (('mode_number','dpg','sigma'),('lower_volume','upper_volume','dry_diameter')):
            lines.append(' '.join(f'{v:.17e}' for key in keys for v in arrays[key][i]))
    process=subprocess.run([str(executable)],input='\n'.join(lines)+'\n',text=True,capture_output=True,check=True,timeout=timeout)
    result=np.fromstring(process.stdout,sep=' ')
    if result.size!=4+count*kbdim*klev*17 or not np.all(np.isfinite(result)):raise ValueError('Unexpected or nonfinite native initialization output')
    cells=np.stack([row.reshape(kbdim,klev,17,order='F') for row in result[4:].reshape(count,-1)])
    number=cells.reshape(leading+(kbdim,klev,17)) if all_cells else cells[:,0,0,:].reshape(leading+(17,))
    timing=dict(zip(('kernel_wall_seconds','kernel_cpu_seconds','number_checksum','clock_ticks_per_second'),result[:4].tolist()))
    timing.update(repetitions=int(repetitions),cases=count,kproma=int(kproma),kbdim=int(kbdim),klev=int(klev),input_reset_included=True,geometry_setup_included=True)
    return {'number':number},timing

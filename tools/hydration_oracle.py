"""Verbatim SALSA equilibrium hydration and inverse-molality Fortran oracle.

The native loop has no iteration cap. Copy-only diagnostics count its actual
iterations, outside timed calls. Supported standalone composition is SO4/OC/BC/
SS/DU, with native species properties and configurable solubility flags.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import subprocess

import numpy as np

from tools.coagulation_process_oracle import _extract, FLAGS, CHECK_FLAGS
from tools.state_capture import load_fixture

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ('parkind1.F90', 'mo_kind.F90', 'mo_math_constants.F90', 'mo_physical_constants.F90',
           'mo_ham_salsa_properties.F90', 'mo_ham_salsa_init.F90', 'mo_ham_salsactl.F90', 'mo_ham_species.F90', 'mo_ham.F90')
STUBS = '''module mo_species
  use mo_kind, only: dp
  implicit none
  type species_type
    real(dp) :: moleweight,density
  end type
  type(species_type) :: speclist(6)
end module
module mo_ham_species
  implicit none
  integer,parameter :: id_so4=1,id_oc=2,id_ss=4,id_wat=6
end module
module mo_ham_salsactl
  use mo_kind, only: dp
  implicit none
  integer,parameter :: in1a=1,fn1a=3,in2a=4,fn2a=10,in2b=11,fn2b=17
  real(dp) :: dpmid(17),epsoc=.15_dp,nlim=1._dp,surfw0=.073_dp
  logical :: lhydr2b=.true.
end module
module mo_ham
  implicit none
  integer,parameter :: subm_naerospec_nowat=5,subm_aerospec_nowat(5)=[1,2,3,4,5],nsoa=0
  type bin_type
    logical :: lsoluble
  end type
  type(bin_type) :: sizeclass(17)
end module
'''
DRIVER = '''program hydration_reference
  use iso_fortran_env, only: int64
  use mo_kind, only: dp
  use mo_species, only: speclist,species_type
  use mo_ham, only: sizeclass
  use mo_ham_salsactl, only: dpmid,epsoc,nlim,surfw0
  use mo_physical_constants, only: avo,ak
  use mo_math_constants, only: pi
  use extracted_hydration, only: equilibration,zbinmol_inv,capture_iterations,last_iterations
  implicit none
  integer :: n,repetitions,i,j,k
  integer(int64) :: wall_start,wall_end,clock_rate
  integer,allocatable :: soluble(:,:),iterations(:,:)
  real(dp),allocatable :: numbers(:,:),volumes(:,:,:),dry(:,:),environment(:,:),properties(:,:),masses(:,:),densities(:,:)
  real(dp),allocatable :: core_out(:,:),wet_out(:,:)
  real(dp) :: number(1,1,17),volume(1,1,17,5),rh(1,1),temperature(1,1),core(1,1,17),wet(1,1,17)
  real(dp) :: started,finished,activity,water_mass
  character(len=32) :: command
  call get_command_argument(1,command)
  if(trim(command)=='--constants') then
    write(*,'(ES26.17E3)') avo,ak,pi
    stop
  end if
  if(trim(command)=='--molality') then
    read(*,*) n
    do i=1,n
      read(*,*) activity,water_mass,epsoc
      speclist(6)=species_type(water_mass,1000._dp)
      do k=1,5
        write(*,'(ES26.17E3)') zbinmol_inv(k,activity)
      end do
    end do
    stop
  end if
  read(*,*) n,repetitions
  allocate(numbers(17,n),volumes(17,5,n),dry(17,n),environment(2,n),properties(5,n),masses(5,n),densities(5,n))
  allocate(soluble(17,n),iterations(17,n),core_out(17,n),wet_out(17,n))
  do i=1,n
    read(*,*) environment(:,i)
    read(*,*) properties(:,i)
    read(*,*) masses(:,i)
    read(*,*) densities(:,i)
    read(*,*) soluble(:,i)
    read(*,*) dry(:,i)
    read(*,*) numbers(:,i)
    read(*,*) volumes(:,:,i)
  end do
  capture_iterations=.true.
  do i=1,n
    call run_case(i)
    iterations(:,i)=last_iterations(1,1,:)
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
                              sum(core_out),sum(wet_out),real(clock_rate,dp)
  do i=1,n
    write(*,'(ES26.17E3)') core_out(:,i),wet_out(:,i)
    write(*,'(I0)') iterations(:,i)
  end do
contains
  subroutine run_case(index)
    integer,intent(in) :: index
    integer :: material
    number(1,1,:)=numbers(:,index)
    volume(1,1,:,:)=volumes(:,:,index)
    rh=environment(1,index)
    temperature=environment(2,index)
    epsoc=properties(1,index)
    surfw0=properties(2,index)
    nlim=properties(3,index)
    speclist(6)=species_type(properties(4,index),properties(5,index))
    do material=1,5
      speclist(material)=species_type(masses(material,index),densities(material,index))
    end do
    dpmid=dry(:,index)
    sizeclass(:)%lsoluble=soluble(:,index)==1
    call equilibration(1,1,1,number,volume,rh,temperature,core,wet)
    core_out(:,index)=core(1,1,:)
    wet_out(:,index)=wet(1,1,:)
  end subroutine
end program
'''
SCALAR_PROPERTIES = ('organic_soluble_fraction', 'water_surface_tension', 'number_threshold', 'water_molar_mass', 'water_density')


def build_oracle(build_directory, compiler='gfortran', *, checked=True):
    directory = Path(build_directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    source = (ROOT / 'mo_ham_salsa_properties.F90').read_text()
    molality = _extract(source, 'FUNCTION', 'zbinmol_inv')
    equilibration = _extract(source, 'SUBROUTINE', 'equilibration')
    (directory / 'equilibration_verbatim.inc').write_text(equilibration + '\n')
    (directory / 'molality_verbatim.inc').write_text(molality + '\n')
    anchor = '                      zdold = max(pdwet(ii,jj,kk),1.e-20_dp)'
    initialization = '    pdwet(1:kproma,:,:) = 0._dp'
    if equilibration.count(anchor) != 1 or equilibration.count(initialization) != 1:
        raise ValueError('Missing or ambiguous hydration diagnostic anchors')
    diagnostic = '                      IF (capture_iterations) last_iterations(ii,jj,kk)=last_iterations(ii,jj,kk)+1\n'
    instrumented = equilibration.replace(anchor, diagnostic + anchor).replace(initialization, '    IF (capture_iterations) last_iterations=0\n' + initialization)
    wrappers = {'stubs.f90': STUBS, 'hydration.F90': 'module extracted_hydration\n  logical :: capture_iterations=.false.\n  integer :: last_iterations(1,1,17)\ncontains\n' + molality + '\n' + instrumented + '\nend module\n', 'driver.f90': DRIVER}
    for name, text in wrappers.items():
        (directory / name).write_text(text)
    flags = FLAGS + CHECK_FLAGS if checked else FLAGS
    objects = []
    for path in [*[ROOT / name for name in SOURCES[:4]], *[directory / name for name in wrappers]]:
        obj = directory / (path.stem + '.o')
        subprocess.run([compiler, *flags, '-c', str(path), '-o', str(obj)], cwd=directory, check=True, capture_output=True, text=True)
        objects.append(str(obj))
    executable = directory / 'hydration_oracle'
    subprocess.run([compiler, *flags, *objects, '-o', str(executable)], cwd=directory, check=True, capture_output=True, text=True)
    constants = np.fromstring(subprocess.check_output([str(executable), '--constants'], text=True), sep=' ')
    sha = lambda content: hashlib.sha256(content).hexdigest()
    provenance = {'source_sha256': {name: sha((ROOT / name).read_bytes()) for name in SOURCES},
                  'equilibration_sha256': sha(equilibration.encode()), 'molality_sha256': sha(molality.encode()),
                  'wrapper_sha256': {name: sha(text.encode()) for name, text in wrappers.items()},
                  'diagnostic_overlay': diagnostic.strip(), 'flags': list(flags), 'checked': checked,
                  'compiled_constants': dict(zip(('avogadro', 'boltzmann', 'pi'), constants.tolist())),
                  'configuration': {'nsoa': 0, 'HAMMOZ': False, 'bins': 17, 'material_order': ['SO4', 'OC', 'BC', 'SS', 'DU'],
                                    'lhydr2b_directly_used': False, 'soluble_mask_controls_hydration': True},
                  'separate_compilation_without_lto': True,
                  'compiler': subprocess.check_output([compiler, '--version'], text=True).splitlines()[0]}
    return executable, provenance


def reference_cases():
    native = load_fixture(ROOT / 'tests/fixtures/native-default')
    def record(stage, name):
        item = native[stage][name]
        return np.array(item['values']).reshape(item['shape'], order='F')
    dry = record('entry', 'dpmid')
    species = record('entry', 'volume_species').astype(int) - 1
    masses = record('entry', 'species_molar_mass')[species]
    densities = record('entry', 'species_density')[species]
    # mo_ham_salsa_init: first ten bins soluble; last seven follow lhydr2b.
    default_mask = np.r_[np.ones(10, dtype=bool), np.full(7, bool(native['options']['lhydr2b']))]
    labels, records = [], []
    def add(label, *, number=None, volume=None, relative_humidity=.5, temperature=280., soluble_mask=None, **properties):
        number = np.full(17, 1e8) if number is None else np.broadcast_to(number, (17,)).copy()
        if volume is None:
            volume = (number * np.pi / 6 * dry**3)[:, None] * np.array([.4, .3, .1, .1, .1])
        labels.append(label)
        records.append({'number': number, 'volume': volume, 'relative_humidity': relative_humidity, 'temperature': temperature,
                        'dry_diameter': dry.copy(), 'soluble_mask': default_mask.copy() if soluble_mask is None else np.array(soluble_mask),
                        'species_molar_mass': masses.copy(), 'species_density': densities.copy(),
                        'organic_soluble_fraction': .15, 'water_surface_tension': .073, 'number_threshold': 1.,
                        'water_molar_mass': 18., 'water_density': 1000., **properties})
    add('native-final-state', number=record('exit', 'pnaero')[0, 0], volume=record('exit', 'zvols')[0, 0],
        relative_humidity=float(record('entry', 'prh')[0, 0]), temperature=float(record('entry', 'ptemp')[0, 0]))
    for material, name in enumerate(('SO4', 'OC', 'BC', 'SS', 'DU')):
        volume = np.zeros((17, 5))
        volume[:, material] = 1e8 * np.pi / 6 * dry**3
        add(f'pure-{name}-soluble', volume=volume)
    add('mixed-soluble')
    add('all-insoluble', soluble_mask=np.zeros(17, dtype=bool))
    add('region2b-insoluble', soluble_mask=np.arange(17) < 10)
    add('empty', number=0.)
    for label, number in (('below-count-threshold', np.nextafter(1., 0.)), ('at-count-threshold', 1.), ('above-count-threshold', np.nextafter(1., np.inf))):
        add(label, number=number)
    for rh in (-.1, 0., .05, .3, .8, .95, 1., 1.2):
        add(f'rh-{rh}', relative_humidity=rh)
    for temperature in (230., 310.):
        add(f'temperature-{temperature}', temperature=temperature)
    add('zero-organic-solubility', organic_soluble_fraction=0.)
    add('heterogeneous-material-properties', species_molar_mass=masses * np.array([1.02, .9, 1.1, 1.03, .8]),
        species_density=densities * np.linspace(.9, 1.1, 5), water_molar_mass=18.015, water_density=995.,
        organic_soluble_fraction=.3, water_surface_tension=.067)
    rng = np.random.default_rng(883)
    for index in range(6):
        number = 10**rng.uniform(2, 10, 17)
        fractions = rng.dirichlet(np.ones(5), 17)
        add(f'random-{index}', number=number, volume=(number * np.pi / 6 * dry**3)[:, None] * fractions,
            temperature=rng.uniform(230, 310), relative_humidity=rng.uniform(.05, .95), soluble_mask=rng.random(17) > .25)
    return {'labels': labels, **{key: np.stack([case[key] for case in records]) for key in records[0]}}


def run_oracle(executable, cases, repetitions=1):
    number = np.asarray(cases['number'], dtype=float)
    if number.ndim != 2 or number.shape[1] != 17 or not len(number):
        raise ValueError('number must have nonempty shape (cases,17)')
    if not isinstance(repetitions, (int, np.integer)) or repetitions < 1:
        raise ValueError('repetitions must be a positive integer')
    n = len(number)
    values = {'number': number}
    for name, shape in {'volume': (n,17,5), 'dry_diameter': (n,17), 'soluble_mask': (n,17),
                        'species_molar_mass': (n,5), 'species_density': (n,5),
                        **{key: (n,) for key in ('temperature', 'relative_humidity', *SCALAR_PROPERTIES)}}.items():
        value = np.asarray(cases[name], dtype=float)
        if name in ('dry_diameter','soluble_mask') and value.shape == (17,):
            value = np.broadcast_to(value, shape)
        if value.shape != shape:
            raise ValueError(f'{name} must have shape {shape}')
        values[name] = value
    for name, value in values.items():
        if not np.all(np.isfinite(value)):
            raise ValueError(f'{name} must be finite')
        if name == 'relative_humidity':
            continue  # native code clamps even negative RH
        nonnegative = name in ('number','volume','soluble_mask','organic_soluble_fraction','number_threshold')
        if np.any(value < 0 if nonnegative else value <= 0):
            raise ValueError(f'Invalid {name}')
    if not np.all(np.isin(values['soluble_mask'], [0,1])):
        raise ValueError('soluble_mask must contain booleans')
    lines = [f'{n} {repetitions}']
    for i in range(n):
        for row in ([values[key][i] for key in ('relative_humidity','temperature')],
                    [values[key][i] for key in SCALAR_PROPERTIES], values['species_molar_mass'][i], values['species_density'][i]):
            lines.append(' '.join(f'{value:.17e}' for value in row))
        lines.append(' '.join(str(int(value)) for value in values['soluble_mask'][i]))
        for row in (values['dry_diameter'][i], number[i], values['volume'][i].ravel(order='F')):
            lines.append(' '.join(f'{value:.17e}' for value in row))
    result = subprocess.run([str(executable)], input='\n'.join(lines)+'\n', capture_output=True, text=True, check=True, timeout=30)
    output = np.fromstring(result.stdout, sep=' ')
    if output.size != 5 + n*51 or not np.all(np.isfinite(output)):
        raise ValueError('Unexpected or nonfinite native hydration output')
    rows = output[5:].reshape(n,51)
    state = {'core': rows[:,:17], 'wet_diameter': rows[:,17:34], 'iterations': rows[:,34:].astype(int)}
    timing = {'kernel_wall_seconds': float(output[0]), 'kernel_cpu_seconds': float(output[1]),
              'core_checksum': float(output[2]), 'wet_diameter_checksum': float(output[3]),
              'clock_ticks_per_second': int(output[4]), 'repetitions': int(repetitions), 'cases': n,
              'input_setup_included': True, 'iteration_capture_included': False}
    return state,timing


def run_molality(executable, activity, *, water_molar_mass=18., organic_soluble_fraction=.15):
    """Run the native inverse-binary-molality leaf for all five materials."""
    activity, water_mass, epsoc = np.broadcast_arrays(activity, water_molar_mass, organic_soluble_fraction)
    if (not np.all(np.isfinite(activity)) or np.any((activity <= 0) | (activity >= 1)) or
            not np.all(np.isfinite(water_mass)) or np.any(water_mass <= 0) or
            not np.all(np.isfinite(epsoc)) or np.any(epsoc < 0)):
        raise ValueError('Expected activity in (0,1), positive water mass and nonnegative organic solubility')
    lines = [str(activity.size)]
    for values in zip(activity.ravel(), water_mass.ravel(), epsoc.ravel()):
        lines.append(' '.join(f'{value:.17e}' for value in values))
    result = subprocess.run([str(executable),'--molality'], input='\n'.join(lines)+'\n', capture_output=True, text=True, check=True)
    values = np.fromstring(result.stdout, sep=' ')
    if values.size != activity.size*5 or not np.all(np.isfinite(values)):
        raise ValueError('Unexpected native molality output')
    return values.reshape(activity.shape+(5,))

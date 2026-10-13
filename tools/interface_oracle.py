# Copyright 2026 Reflective
# SPDX-License-Identifier: Apache-2.0
"""Checked verbatim native HAM/SALSA conversions and final density oracle."""
from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess

import numpy as np

from tools.coagulation_process_oracle import FLAGS, CHECK_FLAGS
from tools.state_capture import load_fixture

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ('parkind1.F90', 'mo_kind.F90', 'mo_math_constants.F90', 'mo_physical_constants.F90')
HEADER = '''module extracted_interface
 use mo_kind, only: dp
 use mo_physical_constants, only: avo
 implicit none
 integer,parameter :: in1a=1,fn1a=3,in2a=4,fn2a=10,in2b=11,fn2b=17
 integer,parameter :: id_so4=1,id_oc=2,id_bc=3,id_ss=4,id_du=5,id_wat=6
 type species_type
   real(dp) :: moleweight,density
 end type
 type(species_type) :: speclist(6)
contains
'''
DECLARATIONS = ''' integer,parameter :: kproma=1,kbdim=1,klev=1
 integer :: ii,jj,kk
 integer,intent(in) :: mapping(17,5)
 integer :: iso4b(17),iocb(17),ibcb(17),issb(17),idub(17)
 real(dp) :: zmvsu,zvq(1,1,17)
 iso4b=mapping(:,1)
 iocb=mapping(:,2)
 ibcb=mapping(:,3)
 issb=mapping(:,4)
 idub=mapping(:,5)
'''
DRIVER = '''program interface_reference
 use extracted_interface
 implicit none
 integer :: n,i,k,mapping(17,5)
 real(dp) :: masses(5),densities(5),water_density
 real(dp) :: mass(1,1,69),volume(1,1,17,5),water(1,1,17)
 real(dp) :: volume_out(1,1,17,5),mass_out(1,1,69),density_out(1,1,17)
 character(len=32) :: command
 call get_command_argument(1,command)
 if(trim(command)=='--constants') then
   write(*,'(ES26.17E3)') avo
   stop
 end if
 read(*,*) n
 do i=1,n
   read(*,*) masses
   read(*,*) densities
   read(*,*) water_density
   read(*,*) mapping
   read(*,*) mass
   read(*,*) volume
   read(*,*) water
   do k=1,5
     speclist(k)=species_type(masses(k),densities(k))
   end do
   speclist(6)=species_type(18._dp,water_density)
   call convert_in(mass,volume_out,mapping)
   call convert_out(volume,mass_out,mapping)
   call final_density(volume,water,density_out,mapping)
   write(*,'(ES26.17E3)') volume_out,mass_out,density_out
 end do
end program
'''


def native_blocks():
    """Extract complete native nsoa=0 blocks; no formulas are rewritten."""
    source = (ROOT / 'mo_ham_salsa.F90').read_text()
    blocks = {}
    for name, marker in (('mass_to_volume', '!>> convert mass concentrations to volume concentrations for SALSA'),
                         ('volume_to_mass', '!>> convert volume concentrations to mass concentrations for HAM')):
        start = source.index('    DO ii = in1a, fn1a', source.index(marker))
        end = source.index('#ifdef HAMMOZ', start)
        blocks[name] = source[start:end]
    end = source.index('    !>> convert volume concentrations to mass concentrations for HAM')
    start = source.rindex('\n    DO jj = 1,klev      ! vertical grid', 0, end) + 1
    blocks['particle_density'] = source[start:end]
    blocks['molecular_volume'] = next(line for line in source.splitlines() if line.startswith('    zmvsu ='))
    return blocks


def build_oracle(build_directory, compiler='gfortran', *, checked=True):
    directory = Path(build_directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    blocks = native_blocks()
    for name, block in blocks.items():
        (directory / f'{name}_verbatim.inc').write_text(block)
    wrapper = HEADER
    for routine, key, declarations in (
        ('convert_in(paerml,zvols,mapping)', 'mass_to_volume',
         ' real(dp),intent(in) :: paerml(1,1,69)\n real(dp),intent(out) :: zvols(1,1,17,5)\n'),
        ('convert_out(zvols,paerml,mapping)', 'volume_to_mass',
         ' real(dp),intent(in) :: zvols(1,1,17,5)\n real(dp),intent(out) :: paerml(1,1,69)\n'),
        ('final_density(zvols,pww,prhop,mapping)', 'particle_density',
         ' real(dp),intent(in) :: zvols(1,1,17,5),pww(1,1,17)\n real(dp),intent(out) :: prhop(1,1,17)\n')):
        wrapper += ' subroutine ' + routine + '\n' + declarations + DECLARATIONS
        wrapper += blocks['molecular_volume'] + '\n'
        if key == 'mass_to_volume':
            wrapper += ' zvols=0._dp\n'
        wrapper += blocks[key] + '\n end subroutine\n'
    wrapper += 'end module\n'
    texts = {'interface.f90': wrapper, 'driver.f90': DRIVER}
    for name, text in texts.items():
        (directory / name).write_text(text)
    flags = FLAGS + CHECK_FLAGS if checked else FLAGS
    objects = []
    for path in [*[ROOT / name for name in SOURCES], *[directory / name for name in texts]]:
        obj = directory / (path.stem + '.o')
        subprocess.run([compiler, *flags, '-c', str(path), '-o', str(obj)], cwd=directory,
                       check=True, capture_output=True, text=True)
        objects.append(str(obj))
    executable = directory / 'interface_oracle'
    subprocess.run([compiler, *flags, *objects, '-o', str(executable)], cwd=directory,
                   check=True, capture_output=True, text=True)
    sha = lambda content: hashlib.sha256(content).hexdigest()
    provenance = {
        'source_sha256': {name: sha((ROOT / name).read_bytes()) for name in (*SOURCES, 'mo_ham_salsa.F90')},
        'block_sha256': {name: sha(block.encode()) for name, block in blocks.items()},
        'wrapper_sha256': {name: sha(text.encode()) for name, text in texts.items()},
        'compiled_constants': {'avogadro': float(subprocess.check_output([str(executable), '--constants'], text=True))},
        'flags': list(flags), 'checked': checked,
        'compiler': subprocess.check_output([compiler, '--version'], text=True).splitlines()[0],
        'scope': 'nsoa=0, 17 bins, five materials, 69 native components; no microphysics, cleanup or hydration',
    }
    return executable, provenance


def run_oracle(executable, cases):
    """Evaluate independently supplied mass, volume and signed water case arrays.

    Required shapes: mass (n,69), volume (n,17,5), water_mass (n,17),
    component_map (n,17,5), species_molar_mass/species_density (n,5),
    water_density (n,). Maps are zero-based with absent slots -1.
    """
    from salsa_jax.interface import validate_component_map
    mass = np.asarray(cases['mass'], dtype=float)
    if mass.ndim != 2 or mass.shape[1] != 69 or not len(mass):
        raise ValueError('mass must have nonempty shape (cases,69)')
    n = len(mass)
    values = {'mass': mass}
    for key, shape in {'volume': (n, 17, 5), 'water_mass': (n, 17), 'component_map': (n, 17, 5),
                       'species_molar_mass': (n, 5), 'species_density': (n, 5), 'water_density': (n,)}.items():
        value = np.asarray(cases[key])
        if value.shape != shape:
            raise ValueError(f'{key} must have shape {shape}')
        values[key] = value
    validate_component_map(values['component_map'])
    for key, value in values.items():
        if not np.all(np.isfinite(value)):
            raise ValueError(f'{key} must be finite')
        if key in ('species_molar_mass', 'species_density', 'water_density') and np.any(value <= 0):
            raise ValueError(f'{key} must be positive')
        if key in ('mass', 'volume') and np.any(value < 0):
            raise ValueError(f'{key} must be nonnegative')
    lines = [str(n)]
    for i in range(n):
        for key in ('species_molar_mass', 'species_density', 'water_density'):
            lines.append(' '.join(f'{v:.17e}' for v in np.atleast_1d(values[key][i])))
        lines.append(' '.join(str(int(v + 1)) for v in values['component_map'][i].ravel(order='F')))
        for key in ('mass', 'volume', 'water_mass'):
            lines.append(' '.join(f'{v:.17e}' for v in values[key][i].ravel(order='F')))
    completed = subprocess.run([str(executable)], input='\n'.join(lines) + '\n', text=True,
                               capture_output=True, check=True, timeout=30)
    output = np.fromstring(completed.stdout, sep=' ')
    if output.size != n * 171 or not np.all(np.isfinite(output)):
        raise ValueError('Unexpected native interface output')
    output = output.reshape(n, 171)
    return {'volume_from_mass': np.stack([row[:85].reshape(17, 5, order='F') for row in output]),
            'mass_from_volume': output[:, 85:154], 'density': output[:, 154:]}


def native_cases(fixture=None):
    """Return actual native entry mass and exit volume/water/property inputs."""
    fixture = Path(fixture) if fixture is not None else ROOT / 'tests/fixtures/native-default'
    state = load_fixture(fixture)
    def record(stage, key):
        item = state[stage][key]
        return np.asarray(item['values']).reshape(item['shape'], order='F')
    mass = record('entry', 'paerml').reshape(-1, 69)
    n = len(mass)
    species = record('entry', 'volume_species').astype(int) - 1
    water_index = state['mapping']['species_names'].index('WAT')
    mapping = record('entry', 'material_map').astype(int) - 1
    return {'labels': [f'{fixture.name}-cell{i}' for i in range(n)], 'mass': mass,
            'volume': record('exit', 'zvols').reshape(n, 17, 5),
            'water_mass': record('exit', 'pww').reshape(n, 17),
            'component_map': np.broadcast_to(mapping, (n, 17, 5)).copy(),
            'species_molar_mass': np.broadcast_to(record('entry', 'species_molar_mass')[species], (n, 5)).copy(),
            'species_density': np.broadcast_to(record('entry', 'species_density')[species], (n, 5)).copy(),
            'water_density': np.full(n, record('entry', 'species_density')[water_index])}


def reference_cases():
    """Native data plus density thresholds, material masks and map permutations."""
    native = native_cases()
    base = {key: value[0].copy() for key, value in native.items() if key != 'labels'}
    rows = [base]
    labels = ['native-entry-mass-exit-volume']
    def add(label, **updates):
        rows.append({**{key: value.copy() for key, value in base.items()}, **updates})
        labels.append(label)
    add('empty', mass=np.zeros(69), volume=np.zeros((17, 5)), water_mass=np.zeros(17))
    for material in range(5):
        volume = np.zeros((17, 5))
        volume[:, material] = 1e-12
        add(f'pure-material-{material}', volume=volume, water_mass=np.zeros(17))
    for value, label in ((np.nextafter(1e-30, 0.), 'below'), (1e-30, 'equal'),
                         (np.nextafter(1e-30, np.inf), 'above')):
        volume = np.zeros((17, 5))
        volume[:, 0] = value
        add(f'density-threshold-{label}', volume=volume, water_mass=np.zeros(17))
    add('water-only', volume=np.zeros((17, 5)), water_mass=np.full(17, 1e-9))
    add('signed-water-small', water_mass=-np.full(17, 1e-20))
    add('signed-water-negative-total', volume=np.zeros((17, 5)), water_mass=-np.full(17, 1e-20))
    add('all-materials-with-unrepresented-volume', volume=np.full((17, 5), 1e-12), water_mass=np.zeros(17))
    add('heterogeneous-properties', species_molar_mass=base['species_molar_mass'] * np.linspace(.9, 1.1, 5),
        species_density=base['species_density'] * np.linspace(.8, 1.2, 5), water_density=np.array(997.))
    rng = np.random.default_rng(18)
    for permutation, label in ((np.arange(68, -1, -1), 'reversed-map'), (rng.permutation(69), 'permuted-map')):
        mapping = base['component_map'].copy()
        mask = mapping >= 0
        mapping[mask] = permutation[mapping[mask]]
        mass = np.zeros(69)
        mass[permutation] = base['mass']
        add(label, component_map=mapping, mass=mass)
    return {'labels': labels, **{key: np.stack([row[key] for row in rows]) for key in base}}

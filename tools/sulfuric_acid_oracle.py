# Copyright 2026 Reflective
# SPDX-License-Identifier: Apache-2.0
"""Forced trajectories through unchanged native SALSA scientific modules.

The source adds H2SO4 molecules/m3/s before each complete SALSA call. Every
post-step output is saved; the input cases retain the initial mass/number/gas.
Both native levels run, preserving SALSA's boundary-layer nucleation behavior.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess

import numpy as np

from tools import salsa_oracle as salsa

OUTPUTS = salsa.OUTPUTS


def _replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f'Expected exactly one native wrapper anchor: {old!r}')
    return text.replace(old, new, 1)


def trajectory_driver():
    """Adapt only the wrapper: additive source and post-step observations."""
    text = salsa.DRIVER
    text = _replace_once(text, 'integer::n,repetitions,steps,i,j,k,level,ngas',
                         'integer::n,repetitions,steps,i,j,k,level,ngas,step_index')
    text = _replace_once(text, 'real(dp),allocatable::mass_out(:,:,:),number_out(:,:,:),gas_out(:,:,:),wet_out(:,:,:),dry_out(:,:,:),rho_out(:,:,:),water_out(:,:,:)',
                         'real(dp),allocatable::mass_out(:,:,:,:),number_out(:,:,:,:),gas_out(:,:,:,:),wet_out(:,:,:,:),dry_out(:,:,:,:),rho_out(:,:,:,:),water_out(:,:,:,:)\n real(dp),allocatable::source_rate(:)')
    text = _replace_once(text, 'allocate(mapping(17,5,n))',
                         'allocate(mapping(17,5,n),source_rate(n))')
    old = 'allocate(mass_out(69,2,n),number_out(17,2,n),gas_out(3,2,n),wet_out(17,2,n),dry_out(17,2,n),rho_out(17,2,n),water_out(17,2,n))'
    text = _replace_once(text, old, old.replace(',2,n)', ',2,n,steps)'))
    text = _replace_once(text, 'read(*,*) initial_gas(:,i)',
                         'read(*,*) initial_gas(:,i)\n read(*,*) source_rate(i)')
    old = ''' do i=1,n
 do level=1,2
 write(*,'(ES26.17E3)') mass_out(:,level,i),number_out(:,level,i),gas_out(:,level,i),wet_out(:,level,i),dry_out(:,level,i),rho_out(:,level,i),water_out(:,level,i)
 end do
 end do
contains'''
    new = ''' do step_index=1,steps
 do i=1,n
 do level=1,2
 write(*,'(ES26.17E3)') mass_out(:,level,i,step_index),number_out(:,level,i,step_index),gas_out(:,level,i,step_index),wet_out(:,level,i,step_index),dry_out(:,level,i,step_index),rho_out(:,level,i,step_index),water_out(:,level,i,step_index)
 end do
 end do
 end do
contains'''
    text = _replace_once(text, old, new)
    old = ''' do s=1,steps
 call salsa(1,1,2,1,pressure,rh,temperature,environment(4,index),gas,mass,number,wet,dry,rho,water,pbl)
 end do
 mass_out(:,:,index)=transpose(mass(1,:,:));number_out(:,:,index)=transpose(number(1,:,:));gas_out(:,:,index)=transpose(gas(1,:,:))
 wet_out(:,:,index)=transpose(wet(1,:,:));dry_out(:,:,index)=transpose(dry(1,:,:));rho_out(:,:,index)=transpose(rho(1,:,:));water_out(:,:,index)=transpose(water(1,:,:))'''
    new = ''' do s=1,steps
 gas(1,:,isubm_so4g)=gas(1,:,isubm_so4g)+source_rate(index)*environment(4,index)
 call salsa(1,1,2,1,pressure,rh,temperature,environment(4,index),gas,mass,number,wet,dry,rho,water,pbl)
 mass_out(:,:,index,s)=transpose(mass(1,:,:));number_out(:,:,index,s)=transpose(number(1,:,:));gas_out(:,:,index,s)=transpose(gas(1,:,:))
 wet_out(:,:,index,s)=transpose(wet(1,:,:));dry_out(:,:,index,s)=transpose(dry(1,:,:));rho_out(:,:,index,s)=transpose(rho(1,:,:));water_out(:,:,index,s)=transpose(water(1,:,:))
 end do'''
    return _replace_once(text, old, new)


def build_oracle(build_directory, compiler='gfortran', *, checked=True):
    """Reuse native compilation, replacing only its observation/forcing driver."""
    directory = Path(build_directory).resolve()
    _, provenance = salsa.build_oracle(directory, compiler, checked=checked)
    driver = trajectory_driver()
    (directory / 'driver.f90').write_text(driver)
    flags = provenance['flags']
    subprocess.run([compiler, *flags, '-c', 'driver.f90', '-o', 'driver.o'],
                   cwd=directory, check=True, capture_output=True, text=True)
    sources = (*salsa.BASE_SOURCES, 'stubs.f90', 'init.F90',
               *salsa.PROCESS_SOURCES, 'driver.f90')
    executable = directory / 'sulfuric_acid_oracle'
    subprocess.run([compiler, *flags,
                    *(str(directory / (Path(name).stem + '.o')) for name in sources),
                    '-o', str(executable)], cwd=directory, check=True,
                   capture_output=True, text=True)
    provenance['parent_driver_sha256'] = provenance['wrapper_sha256']['driver.f90']
    provenance['wrapper_sha256']['driver.f90'] = hashlib.sha256(driver.encode()).hexdigest()
    provenance['scope'] += '; constant additive H2SO4 source before every call; seven post-step fields at both native levels'
    provenance['source_rate_units'] = 'molecules/m3/s'
    provenance['output_order'] = 'step, case, native level, field component'
    return executable, provenance


def _input_text(cases, source_rate, steps, repetitions):
    number = np.asarray(cases['number'], dtype=float)
    if number.ndim != 2 or number.shape[1] != 17 or not len(number):
        raise ValueError('number must have shape (cases,17)')
    if any(isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer)) or x < 1
           for x in (steps, repetitions)):
        raise ValueError('steps and repetitions must be positive integers')
    n = len(number)
    rate = np.asarray(source_rate, dtype=float)
    if rate.ndim == 0:
        rate = np.full(n, rate)
    if rate.shape != (n,) or not np.isfinite(rate).all() or np.any(rate < 0):
        raise ValueError('source_rate must be nonnegative finite scalar or shape (cases,)')
    base, native_map = salsa.native_case()
    mapping = np.asarray(cases['component_map'])
    if mapping.shape != (n, 17, 5) or not np.isfinite(mapping).all() or np.any(mapping != mapping.astype(int)):
        raise ValueError('Invalid component_map')
    for row in mapping:
        if not np.array_equal(row >= 0, native_map > 0) or not np.array_equal(np.sort(row[row >= 0]), np.arange(69)):
            raise ValueError('Unsupported component_map')
    cases = {**{key: np.full(n, base[key]) for key in
                ('cluster_molecules', 'acid_diameter', 'activation_coefficient')}, **cases}
    shapes = {'mass': (n, 69), 'number': (n, 17), 'gas': (n, 3),
              'species_molar_mass': (n, 5), 'species_density': (n, 5),
              'water_molar_mass': (n,), 'water_density': (n,),
              'regime_limits': (n, 4), 'mass_accommodation': (n, 17),
              **{key: (n,) for key in (*salsa.OPTIONS, *salsa.PROPERTIES,
                 'temperature', 'pressure', 'relative_humidity', 'timestep')}}
    values = {}
    for key, shape in shapes.items():
        value = np.asarray(cases[key], dtype=float)
        if value.shape != shape or not np.isfinite(value).all() or np.any(value < 0):
            raise ValueError(f'Invalid {key} shape, sign or nonfinite value')
        values[key] = value
    if np.any(~np.isin(values['scheme'], [0, 1, 2, 3])) or np.any(~np.isin(values['growth'], [1, 2, 3])):
        raise ValueError('Unsupported scheme or growth')
    for key in salsa.OPTIONS:
        if key not in ('scheme', 'growth') and np.any(~np.isin(values[key], [0, 1])):
            raise ValueError(f'Invalid {key}')
    for key in ('temperature', 'pressure', 'species_molar_mass', 'species_density',
                'regime_limits', 'acid_diameter', 'saturation_limit', 'ion_count',
                'water_surface_tension', 'cluster_molecules', 'water_molar_mass', 'water_density'):
        if np.any(values[key] <= 0):
            raise ValueError(f'Nonpositive {key}')
    lines = [f'{n} {repetitions} {steps}']
    for i in range(n):
        lines.append(' '.join(str(int(v) + 1) for v in mapping[i].ravel(order='F')))
        lines.append(' '.join(str(int(values[key][i])) for key in salsa.OPTIONS))
        rows = ([values[key][i] for key in ('temperature', 'pressure', 'relative_humidity', 'timestep')],
                [values[key][i] for key in salsa.PROPERTIES],
                np.r_[values['species_molar_mass'][i], values['water_molar_mass'][i]],
                np.r_[values['species_density'][i], values['water_density'][i]],
                *(values[key][i] for key in ('regime_limits', 'mass_accommodation', 'mass', 'number', 'gas')),
                [rate[i]])
        lines.extend(' '.join(f'{value:.17e}' for value in row) for row in rows)
    return '\n'.join(lines) + '\n', values


def run_oracle(executable, cases, source_rate=0., *, steps=1,
               repetitions=1, all_levels=False, timeout=30):
    """Return seven post-step arrays, normally shaped (steps, cases, width).

    With all_levels=True retain both native levels as (steps, cases, 2, width).
    Otherwise each case selects its boundary_layer level. Timed repetitions reset
    input state and include grid/registry setup, forcing and trajectory recording;
    checksums span every saved step and both levels, not just the selected output.
    """
    text, values = _input_text(cases, source_rate, steps, repetitions)
    n = len(values['number'])
    result = subprocess.run([str(executable)], input=text, capture_output=True,
                            text=True, check=True, timeout=timeout)
    data = np.fromstring(result.stdout, sep=' ')
    width = sum(OUTPUTS.values())
    if data.size != 6 + steps * n * 2 * width or not np.isfinite(data).all():
        raise ValueError('Unexpected or nonfinite SALSA trajectory output')
    rows = data[6:].reshape(steps, n, 2, width)
    if not all_levels:
        rows = rows[:, np.arange(n), values['boundary_layer'].astype(int)]
    outputs = {}
    offset = 0
    for key, size in OUTPUTS.items():
        outputs[key] = rows[..., offset:offset + size]
        offset += size
    timing = dict(zip(('kernel_wall_seconds', 'kernel_cpu_seconds', 'mass_checksum',
                       'number_checksum', 'gas_checksum', 'clock_ticks_per_second'), data[:6].tolist()),
                  repetitions=repetitions, steps=steps, cases=n,
                  input_reset_included=True, grid_setup_included=True,
                  forcing_included=True, trajectory_recording_included=True,
                  checksums_include_all_steps_and_levels=True,
                  native_cells_per_case=2, boundary_layer_by_native_level=[False, True],
                  all_levels=bool(all_levels))
    return outputs, timing

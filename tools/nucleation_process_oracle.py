# Copyright 2026 Reflective
# SPDX-License-Identifier: Apache-2.0
"""Actual SALSA nucleation and condensation oracle for schemes/growth modes 1–3.

The complete native nucleation module is copied byte-for-byte before a mechanical
critical-state diagnostic overlay. Two equal cells straddle the boundary-layer
index; the requested cell is reported. Timings include both native cells.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import subprocess

import numpy as np

from tools.condensation_process_oracle import DEFAULT_PARAMETERS, STUBS as CONDENSATION_STUBS
from tools.coagulation_process_oracle import _extract, FLAGS, CHECK_FLAGS
from tools.state_capture import load_fixture

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ("parkind1.F90", "mo_kind.F90", "mo_math_constants.F90", "mo_physical_constants.F90",
           "mo_ham_salsa_nucleation.F90", "mo_ham_salsa_dynamics.F90", "mo_ham_salsactl.F90",
           "mo_ham_species.F90", "mo_ham.F90")
STUBS = CONDENSATION_STUBS.split("module mo_ham_salsa_nucleation")[0].replace(
    "integer :: nsnucl=0", "integer :: nsnucl=1, nj3=1\n  real(dp) :: act_coeff=1.e-7_dp\n"
    "  real(dp), parameter :: reglim(4)=[3.e-9_dp,5.e-8_dp,7.e-7_dp,1.e-5_dp]") + """module mo_exception
contains
  subroutine finish(where,message)
    character(len=*), intent(in) :: where,message
    write(*,*) where, ': ', message
    error stop 'Unsupported nucleation configuration'
  end subroutine
end module
"""
DRIVER = """program reference_nucleation
  use iso_fortran_env, only: int64
  use mo_kind, only: dp
  use mo_species, only: speclist, species_type
  use mo_ham, only: subm_ngasspec
  use mo_ham_salsactl, only: massacc,dpmid,d_sa,n3,reglim,nsnucl,nj3,act_coeff
  use mo_ham_subm_species, only: isubm_ocnv
  use mo_physical_constants, only: avo,argas,ak,p0sl_bg
  use mo_math_constants, only: pi
  use mo_ham_salsa_nucleation, only: nucleation,capture_critical,last_critical
  use extracted_condensation, only: condensation
  implicit none
  integer :: n,repetitions,operation,i,j,level,pbl(1)
  integer, allocatable :: options(:,:)
  integer(int64) :: wall_started,wall_finished,clock_rate
  character(len=32) :: command
  real(dp), allocatable :: initial_number(:,:),initial_volume(:,:,:),initial_gas(:,:)
  real(dp), allocatable :: wet(:,:),accommodation(:,:),environment(:,:),parameters(:,:)
  real(dp), allocatable :: number_out(:,:),volume_out(:,:,:),gas_out(:,:),critical(:,:),molecular(:,:),fractions(:,:)
  real(dp) :: number(1,2,17),volume(1,2,17,5),gas(1,2,3),diameter(1,2,17),temperature(1,2),pressure(1,2),rh(1,2)
  real(dp) :: rate(1,2,2),sa(1,2),oc(1,2),started,finished
  call get_command_argument(1,command)
  if(trim(command)=='--constants') then
    write(*,'(ES26.17E3)') avo,argas,ak,p0sl_bg,pi,d_sa,n3,reglim(1),act_coeff
    stop
  end if
  read(*,*) n,repetitions,operation
  allocate(options(4,n),initial_number(17,n),initial_volume(17,5,n),initial_gas(3,n))
  allocate(wet(17,n),accommodation(17,n),environment(4,n),parameters(4,n))
  allocate(number_out(17,n),volume_out(17,5,n),gas_out(3,n),critical(4,n),molecular(2,n),fractions(2,n))
  pbl=2
  do i=1,n
    read(*,*) options(:,i)
    read(*,*) environment(:,i)
    read(*,*) parameters(:,i)
    read(*,*) accommodation(:,i)
    read(*,*) wet(:,i)
    read(*,*) initial_number(:,i)
    read(*,*) initial_volume(:,:,i)
    read(*,*) initial_gas(:,i)
  end do
  capture_critical=.true.
  do i=1,n
    call run_case(i,0)
    level=1+options(4,i)
    critical(:,i)=last_critical(1,level,:)
    molecular(:,i)=rate(1,level,:)
    fractions(:,i)=[sa(1,level),oc(1,level)]
  end do
  capture_critical=.false.
  do i=1,n
    call run_case(i,operation)
  end do
  call cpu_time(started)
  call system_clock(wall_started,clock_rate)
  do j=1,repetitions
    do i=1,n
      call run_case(i,operation)
    end do
  end do
  call system_clock(wall_finished)
  call cpu_time(finished)
  write(*,'(6(ES26.17E3,1X))') real(wall_finished-wall_started,dp)/real(clock_rate,dp), &
       finished-started,sum(number_out),sum(volume_out),sum(gas_out),real(clock_rate,dp)
  do i=1,n
    write(*,'(ES26.17E3)') critical(:,i),molecular(:,i),fractions(:,i)
    write(*,'(ES26.17E3)') number_out(:,i),volume_out(:,:,i),gas_out(:,i)
  end do
contains
  subroutine run_case(index,mode)
    integer,intent(in) :: index,mode
    integer :: lev
    nsnucl=options(1,index)
    nj3=options(2,index)
    subm_ngasspec=2+options(3,index)
    isubm_ocnv=3*options(3,index)
    speclist(1)=species_type(parameters(1,index),parameters(3,index))
    speclist(2)=species_type(parameters(2,index),parameters(4,index))
    speclist(3)=species_type(58.443_dp,2165._dp)
    speclist(4)=species_type(18._dp,1000._dp)
    massacc=accommodation(:,index)
    dpmid=wet(:,index)
    do lev=1,2
      number(1,lev,:)=initial_number(:,index)
      volume(1,lev,:,:)=initial_volume(:,:,index)
      gas(1,lev,:)=initial_gas(:,index)
      diameter(1,lev,:)=wet(:,index)
    end do
    temperature=environment(1,index)
    pressure=environment(2,index)
    rh=environment(3,index)
    rate=0._dp
    if(mode==0) then
      call nucleation(1,1,2,1,number,diameter,temperature,rh,pressure,environment(4,index),gas,rate,sa,oc,pbl)
    else
      call condensation(1,1,2,1,number,volume,diameter,gas,rh,temperature,pressure,environment(4,index),pbl)
    end if
    lev=1+options(4,index)
    if(mode==0) then
      molecular(:,index)=rate(1,lev,:)
      fractions(:,index)=[sa(1,lev),oc(1,lev)]
    end if
    number_out(:,index)=number(1,lev,:)
    volume_out(:,:,index)=volume(1,lev,:,:)
    gas_out(:,index)=gas(1,lev,:)
  end subroutine
end program
"""


def build_oracle(build_directory, compiler="gfortran", *, checked=True):
    flags = FLAGS + CHECK_FLAGS if checked else FLAGS
    directory = Path(build_directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    original = (ROOT / "mo_ham_salsa_nucleation.F90").read_bytes()
    (directory / "nucleation_verbatim.F90").write_bytes(original)
    header = b"\nMODULE mo_ham_salsa_nucleation\n"
    anchor = b"    zcsa_local(1:kproma,:) = pcsa(1:kproma,:)\n"
    declarations = b"  USE mo_kind, ONLY: dp\n  LOGICAL :: capture_critical=.false.\n  REAL(dp) :: last_critical(1,2,4)\n"
    diagnostic = b"    IF (capture_critical) THEN\n      last_critical(:,:,1)=zjnuc\n      last_critical(:,:,2)=znsa\n      last_critical(:,:,3)=znoc\n      last_critical(:,:,4)=zdcrit\n    END IF\n"
    if original.count(header) != 1 or original.count(anchor) != 1:
        raise ValueError("Missing or ambiguous nucleation diagnostic anchors")
    instrumented = original.replace(header, header + declarations).replace(anchor, diagnostic + anchor)
    condensation = _extract((ROOT / "mo_ham_salsa_dynamics.F90").read_text(), "SUBROUTINE", "condensation")
    wrappers = {"stubs.f90": STUBS.encode(), "nucleation.F90": instrumented,
                "condensation.F90": ("module extracted_condensation\ncontains\n" + condensation + "\nend module\n").encode(),
                "driver.f90": DRIVER.encode()}
    for name, content in wrappers.items():
        (directory / name).write_bytes(content)
    objects = []
    for path in [*[ROOT / name for name in SOURCES[:4]], *[directory / name for name in wrappers]]:
        obj = directory / (path.stem + ".o")
        subprocess.run([compiler, *flags, "-c", str(path), "-o", str(obj)], cwd=directory,
                       check=True, capture_output=True, text=True)
        objects.append(str(obj))
    executable = directory / "nucleation_oracle"
    subprocess.run([compiler, *flags, *objects, "-o", str(executable)], cwd=directory,
                   check=True, capture_output=True, text=True)
    constants = np.fromstring(subprocess.check_output([str(executable), "--constants"], text=True), sep=" ")
    sha = lambda content: hashlib.sha256(content).hexdigest()
    provenance = {"source_sha256": {name: sha((ROOT / name).read_bytes()) for name in SOURCES},
                  "wrapper_sha256": {name: sha(content) for name, content in wrappers.items()},
                  "diagnostic_overlay": diagnostic.decode(), "checked": checked, "flags": list(flags),
                  "compiled_constants": dict(zip(("avogadro", "gas_constant", "boltzmann", "reference_pressure", "pi", "molecule_diameter", "cluster_molecules", "minimum_diameter", "activation_coefficient"), constants.tolist())),
                  "configuration": {"schemes": [1, 2, 3], "growth": [1, 2, 3], "nsoa": 0, "HAMMOZ": False,
                                    "bins": 17, "materials": 5, "native_levels_per_case": 2, "pbl_index": 2,
                                    "species_parameters_supplied_per_case": True, "default_species_parameters": DEFAULT_PARAMETERS},
                  "separate_compilation_without_lto": True,
                  "compiler": subprocess.check_output([compiler, "--version"], text=True).splitlines()[0]}
    return executable, provenance


def reference_cases():
    dry = np.array(load_fixture(ROOT / "tests/fixtures/native-default")["entry"]["dpmid"]["values"])
    labels, records = [], []
    def add(label, *, scheme=1, growth=1, boundary_layer=True, organic_enabled=False,
            temperature=280., pressure=101325., relative_humidity=.5, timestep=1.,
            number=None, gas=None, wet=None, **parameters):
        number = np.full(17, 1e6) if number is None else np.broadcast_to(number, (17,)).copy()
        labels.append(label)
        records.append({"number": number, "volume": np.zeros((17, 5)),
                        "gas": np.array([1e13, 1e14, 2e12] if gas is None else gas),
                        "wet_diameter": dry.copy() if wet is None else np.array(wet),
                        "mass_accommodation": np.ones(17), "organic_enabled": organic_enabled,
                        "scheme": scheme, "growth": growth, "boundary_layer": boundary_layer,
                        "temperature": temperature, "pressure": pressure, "relative_humidity": relative_humidity,
                        "timestep": timestep, **DEFAULT_PARAMETERS, **parameters})
    for scheme in (1, 2, 3):
        for growth in (1, 2, 3):
            for organic in ("absent", "zero", "positive"):
                for boundary_layer in (False, True):
                    add(f"scheme{scheme}-growth{growth}-{organic}-{'bl' if boundary_layer else 'above'}",
                        scheme=scheme, growth=growth, organic_enabled=organic != "absent",
                        gas=[1e13, 1e14, 0. if organic == "zero" else 2e12], boundary_layer=boundary_layer)
    for scheme in (1, 2, 3):
        for growth in (1, 2, 3):
            add(f"empty-scheme{scheme}-growth{growth}", scheme=scheme, growth=growth, number=0.)
    for gas in (0., .5, 1., 1e8, np.nextafter(1e10, 0.), 1e10, np.nextafter(1e10, np.inf)):
        add(f"activation-cutoff-{gas:.17g}", scheme=2, gas=[1e13, gas, 0.])
    add("activation-vapor-limited", scheme=2, gas=[1e13, 1e12, 0.], timestep=1e8)
    add("kinetic-vapor-limited", scheme=3, gas=[1e13, 1e18, 0.], timestep=1e4)
    for temperature, humidity, sulfate in ((180., 0., 1e8), (190.15, .0001, 1e10),
                                          (300.15, 1., 1e15), (315., 1.2, 1e18)):
        add(f"binary-clamp-{temperature}", temperature=temperature, relative_humidity=humidity,
            gas=[1e13, sulfate, 0.], growth=1)
    add("heterogeneous-parameters", scheme=2, organic_enabled=True, sulfate_molar_mass=98.,
        organic_molar_mass=150., sulfate_density=1800., organic_density=1500.,
        mass_accommodation=np.linspace(.1, 1., 17), wet=dry*np.linspace(1., 2., 17))
    # Active binary nucleation at colder temperatures, including the binary
    # branch above the boundary layer for activation and kinetic configurations.
    # All three growth modes must see meaningful nonzero binary source terms.
    environments = ((220., 1e12, 1e4), (240., 1e13, 1e5), (260., 1e14, 1e6))
    for env_index, (temperature, sulfate, background) in enumerate(environments):
        for scheme in (1, 2, 3):
            for growth in (1, 2, 3):
                add(f"active-binary-env{env_index}-scheme{scheme}-growth{growth}",
                    scheme=scheme, growth=growth, boundary_layer=False,
                    organic_enabled=env_index != 0, number=background,
                    gas=[1e13, sulfate, 2e12], temperature=temperature,
                    pressure=40000. + 25000. * env_index,
                    mass_accommodation=np.linspace(.4, 1., 17),
                    sulfate_density=1841. if env_index < 2 else 1900.,
                    organic_density=2000. if env_index < 2 else 1800.)
    rng = np.random.default_rng(1949)
    for index in range(6):
        add(f"seeded-growth3-{index}", scheme=1 + index % 3, growth=3,
            boundary_layer=bool(index % 2), organic_enabled=bool(index % 2),
            number=10**rng.uniform(3, 5, 17), gas=[1e13, 10**rng.uniform(13, 14), 2e12],
            temperature=rng.uniform(220, 260), pressure=rng.uniform(25000, 80000),
            relative_humidity=rng.uniform(.2, .85), timestep=rng.uniform(.5, 30),
            wet=dry*rng.uniform(1.05, 2.2, 17), mass_accommodation=rng.uniform(.3, 1., 17))
    return {"labels": labels, **{key: np.stack([case[key] for case in records]) for key in records[0]}}


def run_oracle(executable, cases, repetitions=1, *, operation="coupled"):
    number = np.asarray(cases["number"], dtype=float)
    if number.ndim != 2 or number.shape[1] != 17 or not len(number):
        raise ValueError("number must have nonempty shape (cases, 17)")
    if not isinstance(repetitions, (int, np.integer)) or repetitions < 1:
        raise ValueError("repetitions must be positive integer")
    if operation not in ("coupled", "isolated"):
        raise ValueError("operation must be coupled or isolated")
    n = len(number)
    values = {"number": number}
    shapes = {"volume": (n, 17, 5), "gas": (n, 3), "wet_diameter": (n, 17), "mass_accommodation": (n, 17),
              **{name: (n,) for name in ("temperature", "pressure", "relative_humidity", "timestep", "scheme", "growth", "boundary_layer", "organic_enabled", *DEFAULT_PARAMETERS)}}
    for name, shape in shapes.items():
        value = np.asarray(cases.get(name, DEFAULT_PARAMETERS.get(name)), dtype=float)
        if name in DEFAULT_PARAMETERS and value.shape == ():
            value = np.broadcast_to(value, shape)
        if value.shape != shape:
            raise ValueError(f"{name} must have shape {shape}")
        positive = name not in ("volume", "gas", "relative_humidity", "boundary_layer", "organic_enabled")
        if not np.all(np.isfinite(value)) or np.any(value <= 0 if positive else value < 0):
            raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
        values[name] = value
    if not np.all(np.isfinite(number)) or np.any(number < 0):
        raise ValueError("number must be finite nonnegative")
    for name, allowed in (("scheme", [1, 2, 3]), ("growth", [1, 2, 3]), ("boundary_layer", [0, 1]), ("organic_enabled", [0, 1])):
        if not np.all(np.isin(values[name], allowed)):
            raise ValueError(f"Invalid {name}")
    lines = [f"{n} {repetitions} {int(operation == 'coupled')}"]
    for i in range(n):
        lines.append(" ".join(str(int(values[name][i])) for name in ("scheme", "growth", "organic_enabled", "boundary_layer")))
        for row in ([values[name][i] for name in ("temperature", "pressure", "relative_humidity", "timestep")],
                    [values[name][i] for name in DEFAULT_PARAMETERS], values["mass_accommodation"][i],
                    values["wet_diameter"][i], number[i], values["volume"][i].ravel(order="F"), values["gas"][i]):
            lines.append(" ".join(f"{x:.17e}" for x in row))
    result = subprocess.run([str(executable)], input="\n".join(lines) + "\n", capture_output=True, text=True, check=True)
    output = np.fromstring(result.stdout, sep=" ")
    if output.size != 6 + n * 113 or not np.all(np.isfinite(output)):
        raise ValueError("Unexpected or nonfinite native nucleation output")
    rows = output[6:].reshape(n, -1)
    state = {"critical_rate": rows[:, 0], "critical_molecules": rows[:, 1], "critical_organic_molecules": rows[:, 2],
             "critical_diameter": rows[:, 3], "molecular_rate": rows[:, 4:6], "fractions": rows[:, 6:8],
             "number": rows[:, 8:25], "volume": np.stack([row[25:110].reshape(17, 5, order="F") for row in rows]),
             "gas": rows[:, 110:113]}
    timing = {"kernel_wall_seconds": float(output[0]), "kernel_cpu_seconds": float(output[1]),
              "number_checksum": float(output[2]), "volume_checksum": float(output[3]), "gas_checksum": float(output[4]),
              "clock_ticks_per_second": int(output[5]), "repetitions": int(repetitions), "cases": n,
              "native_levels_per_case": 2, "operation": operation, "input_reset_included": True,
              "critical_capture_included": False}
    return state, timing

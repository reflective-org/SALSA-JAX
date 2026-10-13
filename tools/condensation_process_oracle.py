# Copyright 2026 Reflective
# SPDX-License-Identifier: Apache-2.0
"""Original SALSA condensation oracle, without nucleation or HAMMOZ/VBS.

Data-only module stubs provide the native species constants and supported grid.
The nucleation stub aborts if called. A separate mechanical copy assignment
captures actual collision rates outside timing; scientific arithmetic is verbatim.
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
SOURCES = ("parkind1.F90", "mo_kind.F90", "mo_math_constants.F90",
           "mo_physical_constants.F90", "mo_ham_salsa_dynamics.F90",
           "mo_ham_salsactl.F90", "mo_ham_species.F90", "mo_ham.F90")
DEFAULT_PARAMETERS = {"sulfate_molar_mass": 96.0631, "organic_molar_mass": 180.0,
                      "sulfate_density": 1841.0, "organic_density": 2000.0}
# Molar mass units are g/mol; density units are kg/m3, as in speclist.
STUBS = """module mo_species
  use mo_kind, only: dp
  implicit none
  type species_type
    real(dp) :: moleweight, density
  end type
  type(species_type) :: speclist(4)
end module
module mo_ham_species
  implicit none
  integer, parameter :: id_so4=1, id_oc=2, id_ss=3, id_wat=4
end module
module mo_ham_salsactl
  use mo_kind, only: dp
  implicit none
  integer, parameter :: in1a=1, in2a=4, fn2b=17, nbin(2)=[3,7]
  integer :: nsnucl=0
  real(dp), parameter :: d_sa=5.539376964394570e-10_dp, epsoc=0.15_dp, n3=158.79_dp
  real(dp) :: massacc(17), dpmid(17)
end module
module mo_ham
  implicit none
  integer :: subm_ngasspec=3, subm_naerospec_nowat=5, nsoa=0
end module
module mo_ham_subm_species
  implicit none
  integer :: isubm_so4g=2, isubm_ocnv=3
end module
module mo_time_control
  use mo_kind, only: dp
  implicit none
  real(dp) :: time_step_len=60._dp
end module
module mo_ham_salsa_nucleation
  use mo_kind, only: dp
  use mo_ham_salsactl, only: fn2b
  use mo_ham, only: subm_ngasspec
  implicit none
contains
  subroutine nucleation(kproma,kbdim,klev,krow,number,diameter,temp,rh,pres,dt,gas,rate,sa,oc,pbl)
    integer :: kproma,kbdim,klev,krow,pbl(kbdim)
    real(dp) :: number(kbdim,klev,fn2b),diameter(kbdim,klev,fn2b)
    real(dp) :: temp(kbdim,klev),rh(kbdim,klev),pres(kbdim,klev),dt
    real(dp) :: gas(kbdim,klev,subm_ngasspec),rate(kbdim,klev,2),sa(kbdim,klev),oc(kbdim,klev)
    error stop 'Nucleation must remain disabled in the condensation oracle'
  end subroutine
end module
"""
DRIVER = """program reference_condensation
  use iso_fortran_env, only: int64
  use mo_kind, only: dp
  use mo_species, only: speclist, species_type
  use mo_ham, only: subm_ngasspec
  use mo_ham_salsactl, only: massacc, dpmid, d_sa
  use mo_physical_constants, only: avo, argas, ak, p0sl_bg
  use mo_math_constants, only: pi
  use mo_ham_subm_species, only: isubm_ocnv
  use extracted_condensation, only: condensation, capture_rates, last_rates
  implicit none
  integer :: n, repetitions, i, j
  character(len=32) :: command
  integer, allocatable :: organic_enabled(:)
  integer :: pbl(1)
  integer(int64) :: wall_started, wall_finished, clock_rate
  real(dp), allocatable :: initial_number(:,:,:,:), initial_volume(:,:,:,:,:), initial_gas(:,:,:,:)
  real(dp), allocatable :: number(:,:,:,:), volume(:,:,:,:,:), gas(:,:,:,:), rates(:,:)
  real(dp), allocatable :: wet(:,:,:,:), accommodation(:,:), temperature(:,:,:), pressure(:,:,:), timestep(:), parameters(:,:)
  real(dp) :: rh(1,1), started, finished
  call get_command_argument(1,command)
  if (trim(command)=="--constants") then
    write(*,'(ES26.17E3)') avo,argas,ak,p0sl_bg,pi,d_sa
    stop
  end if
  read(*,*) n, repetitions
  allocate(initial_number(1,1,17,n), initial_volume(1,1,17,5,n),initial_gas(1,1,3,n))
  allocate(number(1,1,17,n), volume(1,1,17,5,n), gas(1,1,3,n), rates(17,n))
  allocate(wet(1,1,17,n), accommodation(17,n), temperature(1,1,n), pressure(1,1,n), &
           timestep(n),parameters(4,n),organic_enabled(n))
  pbl=1
  rh=0.5_dp
  do i=1,n
    read(*,*) temperature(:,:,i), pressure(:,:,i), timestep(i), organic_enabled(i)
    read(*,*) parameters(:,i)
    read(*,*) accommodation(:,i)
    read(*,*) wet(:,:,:,i)
    read(*,*) initial_number(:,:,:,i)
    read(*,*) initial_volume(:,:,:,:,i)
    read(*,*) initial_gas(:,:,:,i)
  end do
  capture_rates=.true.
  do i=1,n
    call run_case(i)
    rates(:,i)=last_rates
  end do
  capture_rates=.false.
  call cpu_time(started)
  call system_clock(wall_started, clock_rate)
  do j=1,repetitions
    do i=1,n
      call run_case(i)
    end do
  end do
  call system_clock(wall_finished)
  call cpu_time(finished)
  write(*,'(6(ES26.17E3,1X))') real(wall_finished-wall_started,dp)/real(clock_rate,dp), &
       finished-started, sum(number), sum(volume), sum(gas), real(clock_rate,dp)
  do i=1,n
    write(*,'(ES26.17E3)') number(:,:,:,i)
    write(*,'(ES26.17E3)') volume(:,:,:,:,i)
    write(*,'(ES26.17E3)') gas(:,:,:,i)
    write(*,'(ES26.17E3)') rates(:,i)
  end do
contains
  subroutine run_case(index)
    integer, intent(in) :: index
    speclist(1)=species_type(parameters(1,index),parameters(3,index))
    speclist(2)=species_type(parameters(2,index),parameters(4,index))
    speclist(3)=species_type(58.443_dp,2165._dp)
    speclist(4)=species_type(18._dp,1000._dp)
    massacc=accommodation(:,index)
    dpmid=wet(1,1,:,index)
    subm_ngasspec=2+organic_enabled(index)
    isubm_ocnv=3*organic_enabled(index)
    number(:,:,:,index)=initial_number(:,:,:,index)
    volume(:,:,:,:,index)=initial_volume(:,:,:,:,index)
    gas(:,:,:,index)=initial_gas(:,:,:,index)
    call condensation(1,1,1,1,number(:,:,:,index),volume(:,:,:,:,index), &
                      wet(:,:,:,index),gas(:,:,:,index),rh,temperature(:,:,index),pressure(:,:,index),timestep(index),pbl)
  end subroutine
end program
"""


def build_oracle(build_directory, compiler="gfortran", *, checked=True):
    """Compile the original repaired process, with optional runtime checks."""
    flags = FLAGS + CHECK_FLAGS if checked else FLAGS
    directory = Path(build_directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    process = _extract((ROOT / "mo_ham_salsa_dynamics.F90").read_text(), "SUBROUTINE", "condensation")
    (directory / "condensation_verbatim.inc").write_text(process + "\n")
    anchor = "          zcs_tot = sum(zcolrate)"
    if process.count(anchor) != 1:
        raise ValueError("Missing or ambiguous condensation rate capture anchor")
    diagnostic = "          IF (capture_rates) last_rates=zcolrate\n\n"
    instrumented = process.replace(anchor, diagnostic + anchor)
    wrappers = {
        "stubs.f90": STUBS,
        "condensation.F90": "module extracted_condensation\n  use mo_kind, only: dp\n"
            "  logical :: capture_rates=.false.\n  real(dp) :: last_rates(17)\ncontains\n" +
            instrumented + "\nend module\n",
        "driver.f90": DRIVER,
    }
    for name, text in wrappers.items():
        (directory / name).write_text(text)
    objects = []
    for path in [*[ROOT / name for name in SOURCES[:4]], *[directory / name for name in wrappers]]:
        obj = directory / (path.stem + ".o")
        subprocess.run([compiler, *flags, "-c", str(path), "-o", str(obj)], cwd=directory,
                       check=True, capture_output=True, text=True)
        objects.append(str(obj))
    executable = directory / "condensation_oracle"
    subprocess.run([compiler, *flags, *objects, "-o", str(executable)], cwd=directory,
                   check=True, capture_output=True, text=True)
    sha = lambda value: hashlib.sha256(value).hexdigest()
    compiled_constants = np.fromstring(subprocess.check_output([str(executable), "--constants"], text=True), sep=" ")
    if compiled_constants.shape != (6,):
        raise ValueError("Unexpected compiled constant probe output")
    provenance = {
        "compiled_constants": dict(zip(("avogadro", "gas_constant", "boltzmann", "reference_pressure", "pi", "molecule_diameter"), compiled_constants.tolist())),
        "species_parameters_supplied_per_case": True,
        "source_sha256": {name: sha((ROOT / name).read_bytes()) for name in SOURCES},
        "process_sha256": sha(process.encode()), "instrumented_process_sha256": sha(instrumented.encode()),
        "wrapper_sha256": {name: sha(text.encode()) for name, text in wrappers.items()},
        "diagnostic_overlay": diagnostic.strip(), "separate_compilation_without_lto": True,
        "configuration": {"nsoa": 0, "nsnucl": 0, "HAMMOZ": False, "bins": 17, "species": 5,
                          "gas_slots": ["SO2", "H2SO4", "OC"], "nucleation_stub_aborts": True,
                          "default_species_parameters": DEFAULT_PARAMETERS},
        "compiler": subprocess.check_output([compiler, "--version"], text=True).splitlines()[0],
        "flags": list(flags), "checked": checked,
    }
    return executable, provenance


def reference_cases():
    """Native grid plus gas/sink boundaries and heterogeneous supported states."""
    fixture = load_fixture(ROOT / "tests/fixtures/native-default")
    dry = np.array(fixture["entry"]["dpmid"]["values"])
    base_massacc = np.array(fixture["entry"]["massacc"]["values"])
    labels, records = [], []
    def add(label, *, number=None, volume=None, gas=None, wet=None, organic_enabled=True,
            temperature=280., pressure=101325., timestep=60., accommodation=None, **parameters):
        number = np.full(17, 1e8) if number is None else np.broadcast_to(number, (17,)).copy()
        if volume is None:
            volume = number[:, None] * (np.pi / 6 * dry**3)[:, None] * np.array([.5, .3, .1, 0., .1])
        labels.append(label)
        records.append({"number": number, "volume": volume,
                        "gas": np.array([1e13, 1e12, 2e12] if gas is None else gas),
                        "wet_diameter": dry * 1.3 if wet is None else np.array(wet),
                        "mass_accommodation": base_massacc.copy() if accommodation is None else np.array(accommodation),
                        "organic_enabled": organic_enabled, "temperature": temperature,
                        "pressure": pressure, "timestep": timestep, **DEFAULT_PARAMETERS, **parameters})
    add("mixed-positive-organic")
    add("zero-initial-condensate", volume=np.zeros((17, 5)))
    add("absent-organic", organic_enabled=False)
    add("zero-organic", gas=[1e13, 1e12, 0.])
    add("zero-vapors", gas=[1e13, 0., 0.])
    add("empty", number=0.)
    add("zero-timestep", timestep=0.)
    for label, amount in (("gas-below-threshold", np.nextafter(1e-10, 0.)),
                          ("gas-at-threshold", 1e-10),
                          ("gas-above-threshold", np.nextafter(1e-10, np.inf))):
        add(label, gas=[1e13, amount, amount], timestep=3600.)
    # Per-particle bin-1 rate measured from this native oracle at 280 K,
    # 101325 Pa, wet=1.3*dpmid, massacc=1. Bracket the sink threshold by 1%
    # rather than rely on a backend-specific last-bit equality.
    boundary_number = 1e-30 / 1.4327635482662702e-14
    for label, value in (("sink-below-threshold", .99 * boundary_number),
                         ("sink-above-threshold", 1.01 * boundary_number)):
        number = np.zeros(17)
        number[0] = value
        add(label, number=number, timestep=1e30)
    for index in (0, 1, 2, 3, 10, 16):
        number = np.zeros(17)
        number[index] = 1e10
        add(f"single-bin-{index + 1:02d}", number=number)
    add("equal-small-diameters", wet=np.full(17, 8e-9))
    add("unequal-accommodation", accommodation=np.linspace(.1, 1., 17))
    add("altered-species-parameters", sulfate_molar_mass=98., organic_molar_mass=150.,
        sulfate_density=1800., organic_density=1500.)
    rng = np.random.default_rng(8301)
    for index in range(8):
        add(f"random-{index}", number=10**rng.uniform(2, 11, 17),
            gas=10**rng.uniform(8, 15, 3), wet=dry*rng.uniform(1., 3.5, 17),
            temperature=rng.uniform(230, 310), pressure=rng.uniform(30000, 101325),
            timestep=10**rng.uniform(-2, 3), accommodation=rng.uniform(.1, 1., 17),
            organic_enabled=bool(index % 2))
    captured = load_condensation_fixture(ROOT / "tests/fixtures/condensation-default")["entry"]
    def record(name):
        item = captured[name]
        return np.asarray(item["values"]).reshape(item["shape"], order="F")
    if record("d_sa") != 5.539376964394570e-10 or not np.array_equal(record("gas_indices"), [2, 0]):
        raise ValueError("Default condensation fixture constants/mapping changed")
    labels[0] = "captured-default"
    native_gas = record("pc_gas")[0, 0]
    records[0] = {
        "number": record("pnaero")[0, 0], "volume": record("zvols")[0, 0],
        "gas": np.r_[native_gas, 0.], "wet_diameter": record("zdwet")[0, 0],
        "mass_accommodation": record("massacc"), "organic_enabled": False,
        "temperature": float(record("ptemp")[0, 0]), "pressure": float(record("ppres")[0, 0]),
        "timestep": float(record("ptstep")),
        "sulfate_molar_mass": record("species_molar_mass")[0],
        "organic_molar_mass": record("species_molar_mass")[1],
        "sulfate_density": record("species_density")[0], "organic_density": record("species_density")[1],
    }
    return {"labels": labels, **{name: np.stack([record[name] for record in records]) for name in records[0]}}


def run_oracle(executable, cases, repetitions=1):
    """Return state/rates and warmed compute timings including mutable-input reset."""
    number = np.asarray(cases["number"], dtype=float)
    if number.ndim != 2 or number.shape[1] != 17 or not len(number):
        raise ValueError("number must have nonempty shape (cases, 17)")
    if not isinstance(repetitions, (int, np.integer)) or repetitions < 1:
        raise ValueError("repetitions must be a positive integer")
    n = len(number)
    values = {"number": number}
    shapes = {"volume": (n, 17, 5), "gas": (n, 3), "wet_diameter": (n, 17),
              "mass_accommodation": (n, 17), "temperature": (n,), "pressure": (n,),
              "timestep": (n,), "organic_enabled": (n,), **{key: (n,) for key in DEFAULT_PARAMETERS}}
    for name, shape in shapes.items():
        value = np.asarray(cases.get(name, DEFAULT_PARAMETERS.get(name)), dtype=float)
        if name in DEFAULT_PARAMETERS and value.shape == ():
            value = np.broadcast_to(value, shape)
        if value.shape != shape:
            raise ValueError(f"{name} must have shape {shape}")
        values[name] = value
    for name, value in values.items():
        positive = name not in ("number", "volume", "gas", "timestep", "organic_enabled")
        if not np.all(np.isfinite(value)) or np.any(value <= 0 if positive else value < 0):
            raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    if not np.all(np.isin(values["organic_enabled"], [0, 1])):
        raise ValueError("organic_enabled must contain only booleans")
    lines = [f"{n} {repetitions}"]
    for i in range(n):
        lines.append(" ".join(f"{values[name][i]:.17e}" for name in ("temperature", "pressure", "timestep")) +
                     f" {int(values['organic_enabled'][i])}")
        for row in ([values[name][i] for name in DEFAULT_PARAMETERS], values["mass_accommodation"][i],
                    values["wet_diameter"][i], number[i], values["volume"][i].ravel(order="F"), values["gas"][i]):
            lines.append(" ".join(f"{x:.17e}" for x in row))
    result = subprocess.run([str(executable)], input="\n".join(lines) + "\n", capture_output=True,
                            text=True, check=True)
    output = np.fromstring(result.stdout, sep=" ")
    if output.size != 6 + n * 122 or not np.all(np.isfinite(output)):
        raise ValueError("Unexpected Fortran condensation oracle output")
    rows = output[6:].reshape(n, -1)
    state = {"number": rows[:, :17].copy(),
             "volume": np.stack([row[17:102].reshape(17, 5, order="F") for row in rows]),
             "gas": rows[:, 102:105].copy(), "rates": rows[:, 105:].copy()}
    timing = {"kernel_wall_seconds": float(output[0]), "kernel_cpu_seconds": float(output[1]),
              "number_checksum": float(output[2]), "volume_checksum": float(output[3]), "gas_checksum": float(output[4]),
              "clock_ticks_per_second": int(output[5]), "repetitions": int(repetitions), "cases": n,
              "input_reset_included": True, "rate_capture_included": False}
    return state, timing

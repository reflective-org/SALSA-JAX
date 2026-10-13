"""Checked standalone oracle for the original five-species SALSA coagulation.

Only dependencies are stubbed: the process and coagc text are extracted from
repository Fortran. A documented copy-only assignment exposes the actual zcc
matrix. Timing excludes diagnostics, startup, parsing, and output, but includes
resetting mutable inputs on every invocation.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import re
import subprocess

import numpy as np

from tools.coagulation_capture import load_fixture

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ("parkind1.F90", "mo_kind.F90", "mo_math_constants.F90",
           "mo_physical_constants.F90", "mo_ham_salsa_init.F90", "mo_ham_salsa_dynamics.F90")
FLAGS = ("-O2", "-fno-fast-math", "-fno-lto", "-ffree-line-length-none", "-cpp")
CHECK_FLAGS = ("-fcheck=all", "-ffpe-trap=invalid,zero,overflow", "-fbacktrace")
STUBS = """module mo_ham_salsactl
  use mo_kind, only: dp
  implicit none
  integer, parameter :: in1a=1, in2a=4, fn2a=10, in2b=11, fn2b=17
  real(dp) :: dpmid(17)
end module
module mo_ham
  implicit none
  integer, parameter :: subm_naerospec_nowat=5, nsoa=0
  integer :: sizeclass(17)
end module
"""
DRIVER = """program reference_coagulation
  use iso_fortran_env, only: int64
  use mo_kind, only: dp
  use mo_math_constants, only: pi
  use mo_ham_salsactl, only: dpmid
  use extracted_coagulation, only: coagulation, capture_coefficients, last_coefficients
  implicit none
  integer :: n, repetitions, i, j
  integer(int64) :: wall_started, wall_finished, clock_rate
  real(dp), allocatable :: initial_number(:,:,:,:), initial_volume(:,:,:,:,:)
  real(dp), allocatable :: number(:,:,:,:), volume(:,:,:,:,:), coefficients(:,:,:)
  real(dp), allocatable :: wet(:,:,:,:), dry(:,:), temperature(:,:,:), pressure(:,:,:), timestep(:)
  real(dp) :: core(1,1,17), started, finished
  read(*,*) n, repetitions
  allocate(initial_number(1,1,17,n), initial_volume(1,1,17,5,n))
  allocate(number(1,1,17,n), volume(1,1,17,5,n), coefficients(17,17,n))
  allocate(wet(1,1,17,n), dry(17,n), temperature(1,1,n), pressure(1,1,n), timestep(n))
  do i=1,n
    read(*,*) temperature(:,:,i), pressure(:,:,i), timestep(i)
    read(*,*) dry(:,i)
    read(*,*) wet(:,:,:,i)
    read(*,*) initial_number(:,:,:,i)
    read(*,*) initial_volume(:,:,:,:,i)
  end do
  capture_coefficients=.true.
  do i=1,n
    call run_case(i)
    coefficients(:,:,i)=last_coefficients
  end do
  capture_coefficients=.false.
  call cpu_time(started)
  call system_clock(wall_started, clock_rate)
  do j=1,repetitions
    do i=1,n
      call run_case(i)
    end do
  end do
  call system_clock(wall_finished)
  call cpu_time(finished)
  write(*,'(5(ES26.17E3,1X))') real(wall_finished-wall_started,dp)/real(clock_rate,dp), &
       finished-started, sum(number), sum(volume), real(clock_rate,dp)
  do i=1,n
    write(*,'(ES26.17E3)') number(:,:,:,i)
    write(*,'(ES26.17E3)') volume(:,:,:,:,i)
    write(*,'(ES26.17E3)') coefficients(:,:,i)
  end do
contains
  subroutine run_case(index)
    integer, intent(in) :: index
    dpmid=dry(:,index)
    core(1,1,:)=pi/6._dp*dpmid**3
    number(:,:,:,index)=initial_number(:,:,:,index)
    volume(:,:,:,:,index)=initial_volume(:,:,:,:,index)
    call coagulation(1,1,1,number(:,:,:,index),volume(:,:,:,:,index), &
                     wet(:,:,:,index),core,timestep(index),temperature(:,:,index),pressure(:,:,index))
  end subroutine
end program
"""


def _extract(source, kind, name):
    match = re.search(rf"^  {kind} {name}\(.*?^  END {kind} {name}", source, re.M | re.S)
    if match is None:
        raise ValueError(f"Could not find original {name}")
    return match.group(0)


def build_oracle(build_directory, compiler="gfortran", *, checked=True):
    """Compile original arithmetic; runtime checks default on, off for benchmarking."""
    flags = FLAGS + CHECK_FLAGS if checked else FLAGS
    directory = Path(build_directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    coagc = _extract((ROOT / SOURCES[-2]).read_text(), "FUNCTION", "coagc")
    process = _extract((ROOT / SOURCES[-1]).read_text(), "SUBROUTINE", "coagulation")
    # Keep the unmodified extraction for inspection; instrumentation only copies
    # an already-computed value, after its loop and before state updates.
    (directory / "coagulation_verbatim.inc").write_text(process + "\n")
    anchor = "    ! Original version:"
    if process.count(anchor) != 1:
        raise ValueError("Missing or ambiguous coagulation diagnostic anchor")
    diagnostic = "    IF (capture_coefficients) last_coefficients=zcc\n\n"
    instrumented = process.replace(anchor, diagnostic + anchor)
    wrappers = {
        "stubs.f90": STUBS,
        "coagc.f90": "module mo_ham_salsa_init\ncontains\n" + coagc + "\nend module\n",
        "coagulation.F90": "module extracted_coagulation\n  use mo_kind, only: dp\n"
            "  logical :: capture_coefficients=.false.\n  real(dp) :: last_coefficients(17,17)\n"
            "contains\n" + instrumented + "\nend module\n",
        "driver.f90": DRIVER,
    }
    for name, text in wrappers.items():
        (directory / name).write_text(text)
    executable = directory / "coagulation_oracle"
    objects = []
    for path in [*[ROOT / name for name in SOURCES[:4]], *[directory / name for name in wrappers]]:
        obj = directory / (path.stem + ".o")
        subprocess.run([compiler, *flags, "-c", str(path), "-o", str(obj)], cwd=directory,
                       check=True, capture_output=True, text=True)
        objects.append(str(obj))
    subprocess.run([compiler, *flags, *objects, "-o", str(executable)], cwd=directory,
                   check=True, capture_output=True, text=True)
    sha = lambda value: hashlib.sha256(value).hexdigest()
    provenance = {
        "source_sha256": {name: sha((ROOT / name).read_bytes()) for name in SOURCES},
        "function_sha256": sha(coagc.encode()), "process_sha256": sha(process.encode()),
        "instrumented_process_sha256": sha(instrumented.encode()),
        "wrapper_sha256": {name: sha(text.encode()) for name, text in wrappers.items()},
        "diagnostic_overlay": diagnostic.strip(),
        "separate_compilation_without_lto": True,
        "configuration": {"nsoa": 0, "HAMMOZ": False, "bins": 17, "species": 5,
                          "in1a": 1, "in2a": 4, "fn2a": 10, "in2b": 11, "fn2b": 17},
        "compiler": subprocess.check_output([compiler, "--version"], text=True).splitlines()[0],
        "flags": list(flags), "checked": checked,
    }
    return executable, provenance


def reference_cases():
    """Native geometry plus empty, single-bin, stiff, hydrated and seeded cases."""
    fixture = load_fixture(ROOT / "tests/fixtures/coagulation-default")
    def record(stage, name):
        field = fixture[stage][name]
        return np.array(field["values"]).reshape(field["shape"], order="F")
    dry = record("entry", "dpmid")
    base_number = record("entry", "pnaero")[0, 0]
    base_volume = record("entry", "zvols")[0, 0]
    labels, numbers, volumes, hydrations, temperatures, pressures, steps = [], [], [], [], [], [], []
    def add(label, number, volume=None, hydration=1.0, temperature=280.0, pressure=101325.0, timestep=60.0):
        labels.append(label)
        numbers.append(np.array(number, dtype=float))
        if volume is None:
            # Regime 1 supports sulfate and OC; insoluble bins exclude salt.
            fractions = np.tile(np.array([0.4, 0.3, 0.1, 0.1, 0.1]), (17, 1))
            fractions[:3] = [0.7, 0.3, 0.0, 0.0, 0.0]
            fractions[10:] = [0.45, 0.35, 0.1, 0.0, 0.1]
            volume = (number * np.pi / 6 * dry**3)[:, None] * fractions
        volumes.append(volume)
        hydrations.append(np.broadcast_to(hydration, (17,)))
        temperatures.append(temperature)
        pressures.append(pressure)
        steps.append(timestep)
    add("captured-default", base_number, base_volume, record("entry", "zdwet")[0, 0] / dry,
        float(record("entry", "ptemp")[0, 0]), float(record("entry", "ppres")[0, 0]),
        float(record("entry", "ptstep")))
    add("empty", np.zeros(17))
    for bin_index in range(17):
        number = np.zeros(17)
        number[bin_index] = 1e10
        add(f"single-bin-{bin_index + 1:02d}", number)
    add("mixed-zero-timestep", np.full(17, 1e10), timestep=0.0)
    add("mixed-dense-long-step", np.full(17, 1e12), timestep=3600.0)
    pair_number = np.zeros(17)
    pair_number[[3, 10]] = 1e12
    pair_volume = np.zeros((17, 5))
    pair_volume[:, 0] = pair_number * np.pi / 6 * dry**3
    add("cross-regime-budget-stress", pair_number, pair_volume, timestep=3600.0)
    add("unequal-hydration", np.geomspace(1e10, 1e3, 17), hydration=np.linspace(1.05, 3.0, 17))
    rng = np.random.default_rng(2381)
    for index in range(8):
        add(f"random-{index}", 10 ** rng.uniform(2, 11, 17),
            hydration=rng.uniform(1.0, 3.5, 17), temperature=rng.uniform(230, 310),
            pressure=rng.uniform(30000, 101325), timestep=10 ** rng.uniform(-2, 3))
    count = len(labels)
    wet = np.stack(hydrations) * dry
    wet[0] = record("entry", "zdwet")[0, 0]
    return {"labels": labels, "number": np.stack(numbers), "volume": np.stack(volumes),
            "dry_diameter": np.broadcast_to(dry, (count, 17)).copy(),
            "wet_diameter": wet,
            "temperature": np.array(temperatures), "pressure": np.array(pressures),
            "timestep": np.array(steps)}


def run_oracle(executable, cases, repetitions=1):
    """Return number, volume, actual collision matrices and warmed wall/CPU time."""
    number = np.asarray(cases["number"], dtype=np.float64)
    if number.ndim != 2 or number.shape[1] != 17 or not len(number):
        raise ValueError("number must have nonempty shape (cases, 17)")
    if not isinstance(repetitions, (int, np.integer)) or repetitions < 1:
        raise ValueError("repetitions must be a positive integer")
    n = len(number)
    values = {"number": number}
    for name, shape in (("volume", (n, 17, 5)), ("wet_diameter", (n, 17)),
                        ("dry_diameter", (n, 17)), ("temperature", (n,)),
                        ("pressure", (n,)), ("timestep", (n,))):
        value = np.asarray(cases[name], dtype=np.float64)
        if name == "dry_diameter" and value.shape == (17,):
            value = np.broadcast_to(value, shape)
        if value.shape != shape:
            raise ValueError(f"{name} must have shape {shape}")
        values[name] = value
    for name, value in values.items():
        positive = name in ("wet_diameter", "dry_diameter", "temperature", "pressure")
        if not np.all(np.isfinite(value)) or np.any(value <= 0 if positive else value < 0):
            raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    lines = [f"{n} {repetitions}"]
    for i in range(n):
        for row in ([values[key][i] for key in ("temperature", "pressure", "timestep")],
                    values["dry_diameter"][i], values["wet_diameter"][i], number[i],
                    values["volume"][i].ravel(order="F")):
            lines.append(" ".join(f"{x:.17e}" for x in row))
    result = subprocess.run([str(executable)], input="\n".join(lines) + "\n", capture_output=True,
                            text=True, check=True)
    output = np.fromstring(result.stdout, sep=" ")
    if output.size != 5 + n * (17 + 85 + 289) or not np.all(np.isfinite(output)):
        raise ValueError("Unexpected Fortran oracle output")
    rows = output[5:].reshape(n, -1)
    state = {"number": rows[:, :17].copy(),
             "volume": np.stack([row[17:102].reshape(17, 5, order="F") for row in rows]),
             "coefficients": np.stack([row[102:].reshape(17, 17, order="F") for row in rows])}
    timing = {"kernel_wall_seconds": float(output[0]), "kernel_cpu_seconds": float(output[1]),
              "number_checksum": float(output[2]), "volume_checksum": float(output[3]),
              "clock_ticks_per_second": int(output[4]), "repetitions": int(repetitions),
              "cases": n, "input_reset_included": True, "coefficient_capture_included": False}
    return state, timing

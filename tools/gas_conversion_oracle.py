"""Checked verbatim gas2mmr oracle using native constants and unit-code declarations."""
from pathlib import Path
import hashlib
import re
import subprocess

import numpy as np

from tools.coagulation_process_oracle import _extract, FLAGS, CHECK_FLAGS

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ("parkind1.F90", "mo_kind.F90", "mo_math_constants.F90", "mo_physical_constants.F90")
DRIVER = """program gas_reference
 use mo_kind, only: dp
 use mo_ham, only: subm_ngasspec,subm_gasspec,subm_gasunitconv
 use mo_species, only: speclist
 use extracted_gas_conversion, only: gas2mmr
 implicit none
 integer :: n,nt,ng,i,j
 real(dp),allocatable :: tracers(:,:,:),gas(:,:,:),masses(:)
 integer,allocatable :: indices(:),codes(:)
 real(dp) :: rho(2,1),pressure(2,1),temperature(2,1)
 read(*,*) n,nt,ng
 subm_ngasspec=ng
 allocate(subm_gasspec(ng),subm_gasunitconv(ng),speclist(ng))
 allocate(tracers(2,1,nt),gas(2,1,ng),masses(ng),indices(ng),codes(ng))
 do i=1,n
   read(*,*) rho(1,1),pressure(1,1),temperature(1,1)
   read(*,*) indices
   read(*,*) codes
   read(*,*) masses
   tracers=-4321.0_dp
   rho(2,1)=1.0_dp
   pressure(2,1)=1.0_dp
   temperature(2,1)=1.0_dp
   gas=0.0_dp
   read(*,*) tracers(1,1,:)
   read(*,*) gas(1,1,:)
   do j=1,ng
      subm_gasspec(j)=j
      subm_gasunitconv(j)=codes(j)
      speclist(j)%idt=indices(j)+1
      speclist(j)%moleweight=masses(j)
   end do
   call gas2mmr(1,2,1,nt,tracers,gas,rho,pressure,temperature)
   write(*,'(ES26.17E3)') tracers(1,1,:),tracers(2,1,:)
 end do
end program
"""


def gas_routine():
    return _extract((ROOT / "mo_convert_concentrations.F90").read_text(), "SUBROUTINE", "gas2mmr")


def unit_declarations():
    source = (ROOT / "mo_ham.F90").read_text()
    lines = re.findall(r"^\s*INTEGER,\s*PARAMETER\s*::\s*(?:immr2ug|immr2molec|ivmr2molec)\s*=.*$", source, re.MULTILINE)
    if len(lines) != 3:
        raise ValueError("Expected three native gas conversion-code declarations")
    return "\n".join(lines)


def build_oracle(build_directory, compiler="gfortran"):
    directory = Path(build_directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    routine, declarations = gas_routine(), unit_declarations()
    (directory / "gas2mmr_verbatim.inc").write_text(routine + "\n")
    stubs = """module mo_species
 use mo_kind, only: dp
 implicit none
 type species_type
   integer :: idt
   real(dp) :: moleweight
 end type
 type(species_type),allocatable :: speclist(:)
end module
module mo_ham
 implicit none
 integer :: subm_ngasspec
 integer,allocatable :: subm_gasspec(:),subm_gasunitconv(:)
""" + declarations + "\nend module\n"
    files = {name: (ROOT / name).read_text() for name in SOURCES}
    files.update({"stubs.f90": stubs,
                  "gas_conversion.f90": "module extracted_gas_conversion\ncontains\n" + routine + "\nend module\n",
                  "driver.f90": DRIVER})
    flags = [*FLAGS, *CHECK_FLAGS]
    objects = []
    for name, text in files.items():
        path = directory / name
        path.write_text(text)
        obj = path.with_suffix(".o")
        subprocess.run([compiler, *flags, "-c", str(path), "-o", str(obj)], cwd=directory,
                       check=True, capture_output=True, text=True)
        objects.append(str(obj))
    executable = directory / "gas_conversion_oracle"
    subprocess.run([compiler, *flags, *objects, "-o", str(executable)], cwd=directory,
                   check=True, capture_output=True, text=True)
    digest = lambda text: hashlib.sha256(text.encode()).hexdigest()
    provenance = {
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                          for name in (*SOURCES, "mo_ham.F90", "mo_convert_concentrations.F90")},
        "routine_sha256": digest(routine), "unit_declarations_sha256": digest(declarations),
        "wrapper_sha256": {name: digest(text) for name, text in files.items()},
        "flags": flags, "checked": True, "scope": "gas2mmr leaf only; no driver scaling or HAM wrapper",
        "compiler": subprocess.check_output([compiler, "--version"], text=True),
    }
    return executable, provenance


def run_oracle(executable, cases):
    tracers = np.asarray(cases["tracers"], dtype=float)
    gas = np.asarray(cases["gas"], dtype=float)
    if (tracers.ndim != 2 or gas.ndim != 2 or tracers.shape[0] != gas.shape[0] or
            min(*tracers.shape, gas.shape[1]) < 1):
        raise ValueError("Expected nonempty matching (cases,tracers) and (cases,gases)")
    n, nt = tracers.shape
    ng = gas.shape[1]
    arrays = {"tracers": tracers, "gas": gas}
    for name, shape in (("air_density", (n,)), ("pressure", (n,)), ("temperature", (n,)),
                        ("tracer_indices", (n, ng)), ("molar_mass", (n, ng)), ("unit_codes", (n, ng))):
        values = np.asarray(cases[name])
        if values.shape != shape:
            raise ValueError(f"Invalid shape for {name}: expected {shape}")
        arrays[name] = values
    for name, values in arrays.items():
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Nonfinite {name}")
    for name in ("air_density", "pressure", "temperature", "molar_mass"):
        if np.any(arrays[name] <= 0):
            raise ValueError(f"Nonpositive {name}")
    for name in ("tracer_indices", "unit_codes"):
        if not np.issubdtype(arrays[name].dtype, np.integer):
            raise ValueError(f"{name} must contain integers")
    if np.any(arrays["tracer_indices"] < 0) or np.any(arrays["tracer_indices"] >= nt):
        raise ValueError("Out-of-range tracer_indices")
    lines = [f"{n} {nt} {ng}"]
    for i in range(n):
        lines.append(" ".join(f'{arrays[name][i]:.17e}' for name in ("air_density", "pressure", "temperature")))
        for name in ("tracer_indices", "unit_codes"):
            lines.append(" ".join(map(str, arrays[name][i])))
        for name in ("molar_mass", "tracers", "gas"):
            lines.append(" ".join(f"{value:.17e}" for value in arrays[name][i]))
    result = subprocess.run([str(executable)], input="\n".join(lines) + "\n", capture_output=True,
                            text=True, check=True, timeout=30)
    output = np.fromstring(result.stdout, sep=" ")
    if output.size != n * nt * 2 or not np.all(np.isfinite(output)):
        raise ValueError("Unexpected gas conversion output")
    output = output.reshape(n, 2, nt)
    return {"tracers": output[:, 0, :], "inactive_padding": output[:, 1, :]}

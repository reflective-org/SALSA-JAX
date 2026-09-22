"""Checked oracle for the verbatim initial SALSA tiny-particle cleanup block."""
from pathlib import Path
import hashlib
import re
import subprocess

import numpy as np

from tools.coagulation_process_oracle import FLAGS, CHECK_FLAGS
from tools.hydration_oracle import SOURCES, STUBS

ROOT = Path(__file__).resolve().parents[1]
FLOAT_FIELDS = ("number", "volume", "gas")
HEADER = """module extracted_cleanup
 use mo_kind, only: dp
 use mo_species, only: speclist
 use mo_ham_species, only: id_so4,id_oc
 use mo_ham_salsactl, only: in1a,fn1a,in2a,fn2a,in2b,fn2b,nlim
 use mo_math_constants, only: pi
 use mo_physical_constants, only: avo
 implicit none
 integer,parameter :: isubm_so4g=2
 integer :: isubm_ocnv=0
contains
 subroutine cleanup(kproma,kbdim,klev,pnaero,zvols,pc_gas)
 integer,intent(in) :: kproma,kbdim,klev
 real(dp),intent(inout) :: pnaero(kbdim,klev,17),zvols(kbdim,klev,17,5),pc_gas(kbdim,klev,3)
 real(dp) :: zddry
 integer :: ii,jj,kk
"""
DRIVER = """program cleanup_reference
 use mo_kind, only: dp
 use mo_species, only: speclist,species_type
 use mo_ham_salsactl, only: nlim
 use extracted_cleanup, only: cleanup,isubm_ocnv
 implicit none
 integer :: n,i,k,organic
 real(dp) :: masses(5),densities(5)
 real(dp) :: number(1,1,17),volume(1,1,17,5),gas(1,1,3)
 read(*,*) n
 do i=1,n
   read(*,*) nlim,organic
   read(*,*) masses
   read(*,*) densities
   read(*,*) number
   read(*,*) volume
   read(*,*) gas
   do k=1,5
     speclist(k)=species_type(masses(k),densities(k))
   end do
   isubm_ocnv=merge(3,0,organic==1)
   call cleanup(1,1,1,number,volume,gas)
   write(*,'(ES26.17E3)') number,volume,gas
 end do
end program
"""


def cleanup_block():
    """Extract initial cleanup before the first actual hydration call."""
    text = (ROOT / "mo_ham_salsa.F90").read_text()
    start = "    DO kk = in1a,fn1a      ! size bin\n"
    end = "    CALL equilibration(kproma, kbdim, klev,        &\n"
    if text.count("\n" + start) != 1 or text.count("\n" + end) != 1:
        raise ValueError("Ambiguous initial cleanup anchors")
    first, last = text.index(start), text.index(end)
    if first >= last:
        raise ValueError("Invalid initial cleanup anchor order")
    return text[first:last].rstrip()


def normalized_block(block):
    return re.sub(r"\s+", "", "\n".join(line.split("!")[0] for line in block.splitlines())).lower()


def build_oracle(build_directory, compiler="gfortran"):
    directory = Path(build_directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    block = cleanup_block()
    (directory / "cleanup_verbatim.inc").write_text(block + "\n")
    texts = {name: (ROOT / name).read_text() for name in SOURCES[:4]}
    texts.update({"stubs.f90": STUBS, "cleanup.f90": HEADER + block + "\n end subroutine\nend module\n",
                  "driver.f90": DRIVER})
    flags = [*FLAGS, *CHECK_FLAGS]
    objects = []
    for name, text in texts.items():
        path = directory / name
        path.write_text(text)
        obj = path.with_suffix(".o")
        subprocess.run([compiler, *flags, "-c", str(path), "-o", str(obj)], cwd=directory,
                       check=True, capture_output=True, text=True)
        objects.append(str(obj))
    executable = directory / "cleanup_oracle"
    subprocess.run([compiler, *flags, *objects, "-o", str(executable)], cwd=directory,
                   check=True, capture_output=True, text=True)
    digest = lambda text: hashlib.sha256(text.encode()).hexdigest()
    provenance = {
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                          for name in (*SOURCES[:4], "mo_ham_salsa.F90")},
        "cleanup_block_sha256": digest(block), "wrapper_sha256": {name: digest(text) for name, text in texts.items()},
        "flags": flags, "checked": True, "scope": "initial tiny cleanup only; no mass conversion or hydration",
        "gas_layout": ["SO2", "H2SO4", "OC"], "organic_absent_index": 0,
        "compiler": subprocess.check_output([compiler, "--version"], text=True),
    }
    return executable, provenance


def run_oracle(executable, cases):
    number = np.asarray(cases["number"], dtype=float)
    if number.ndim != 2 or number.shape[1] != 17 or not len(number):
        raise ValueError("number must have nonempty shape (cases,17)")
    n = len(number)
    shapes = {"number": (n, 17), "volume": (n, 17, 5), "gas": (n, 3),
              "species_molar_mass": (n, 5), "species_density": (n, 5),
              "number_threshold": (n,), "organic_enabled": (n,)}
    values = {}
    for key, shape in shapes.items():
        value = np.asarray(cases[key], dtype=float)
        if value.shape != shape or not np.all(np.isfinite(value)):
            raise ValueError(f"Invalid finite shape for {key}: expected {shape}")
        positive = key in ("species_molar_mass", "species_density", "number_threshold")
        if np.any(value <= 0 if positive else value < 0):
            raise ValueError(f"Invalid {key}")
        values[key] = value
    if not np.all(np.isin(values["organic_enabled"], (0, 1))):
        raise ValueError("organic_enabled must contain booleans")
    lines = [str(n)]
    for i in range(n):
        lines.append(f'{values["number_threshold"][i]:.17e} {int(values["organic_enabled"][i])}')
        for field in ("species_molar_mass", "species_density", "number", "volume", "gas"):
            lines.append(" ".join(f"{v:.17e}" for v in values[field][i].ravel(order="F")))
    result = subprocess.run([str(executable)], input="\n".join(lines) + "\n", capture_output=True,
                            text=True, check=True, timeout=30)
    output = np.fromstring(result.stdout, sep=" ")
    if output.size != n * 105 or not np.all(np.isfinite(output)):
        raise ValueError("Unexpected native cleanup output")
    rows = output.reshape(n, 105)
    return {"number": rows[:, :17], "volume": np.stack([row[17:102].reshape(17, 5, order="F") for row in rows]),
            "gas": rows[:, 102:105]}

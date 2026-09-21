"""Build/run an isolated copy of the current Fortran tree, retaining provenance.

This is a compatibility audit, not yet a scientific validation or timing harness.
Run directories must be new, so failures cannot reuse stale executables/results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time
import tomllib

if __package__:
    from .runtime_metadata import FILENAME, instrument_driver, parse_metadata
    from .reference_scope import FILENAME as SCOPE_FILENAME, STATE_FILENAME, instrument_scope, parse_completion, parse_scope_state
    from .state_capture import FILES as STATE_FILES, instrument_salsa, load_capture
    from .coagulation_capture import FILES as COAGULATION_FILES, instrument_salsa_coagulation, load_capture as load_coagulation_capture
    from .condensation_capture import FILES as CONDENSATION_FILES, instrument_salsa_condensation, load_capture as load_condensation_capture
else:
    from runtime_metadata import FILENAME, instrument_driver, parse_metadata
    from reference_scope import FILENAME as SCOPE_FILENAME, STATE_FILENAME, instrument_scope, parse_completion, parse_scope_state
    from state_capture import FILES as STATE_FILES, instrument_salsa, load_capture
    from coagulation_capture import FILES as COAGULATION_FILES, instrument_salsa_coagulation, load_capture as load_coagulation_capture
    from condensation_capture import FILES as CONDENSATION_FILES, instrument_salsa_condensation, load_capture as load_condensation_capture

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def output(*command: str) -> str:
    return subprocess.check_output(command, cwd=ROOT, text=True).strip()


def logged_run(command: list[str], directory: Path, log: str, timeout: int) -> dict:
    start = time.perf_counter()
    with (directory / log).open("w") as stream:
        try:
            result = subprocess.run(
                command, cwd=directory, stdout=stream, stderr=subprocess.STDOUT,
                timeout=timeout, check=False,
            )
            status, returncode = "finished", result.returncode
        except subprocess.TimeoutExpired:
            status, returncode = "timeout", None
    return {"command": command, "status": status, "returncode": returncode,
            "wall_seconds_including_io": time.perf_counter() - start, "log": log}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new build/run directory")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/fortran-reference.toml")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--capture-runtime", action="store_true",
                        help="instrument only the copied driver to emit effective runtime metadata")
    parser.add_argument("--scope", choices=("full", "first-microphysics"), default="full",
                        help="full trajectory or a finite-state check after the first microphysics call")
    parser.add_argument("--capture-salsa-state", action="store_true",
                        help="capture native entry/exit arrays and mappings in the first-microphysics scope")
    parser.add_argument("--capture-coagulation-state", action="store_true",
                        help="capture actual coagulation inputs/outputs in the first-microphysics scope")
    parser.add_argument("--capture-condensation-state", action="store_true",
                        help="capture condensation with nucleation disabled in the first-microphysics scope")
    args = parser.parse_args()
    if args.capture_salsa_state and args.scope != "first-microphysics":
        parser.error("--capture-salsa-state requires --scope first-microphysics")
    if args.capture_coagulation_state and args.scope != "first-microphysics":
        parser.error("--capture-coagulation-state requires --scope first-microphysics")
    if args.capture_condensation_state and args.scope != "first-microphysics":
        parser.error("--capture-condensation-state requires --scope first-microphysics")
    if args.capture_salsa_state or args.capture_coagulation_state or args.capture_condensation_state:
        args.capture_runtime = True
    config = tomllib.loads(args.config.read_text())
    target = args.output.resolve()
    target.mkdir(parents=True, exist_ok=False)

    sources = sorted([*ROOT.glob("*.F90"), ROOT / "Makefile", ROOT / "Makefile.in",
                      ROOT / ".depend", ROOT / "configure", *ROOT.glob("include/*.inc")])
    inputs = sorted([ROOT / "namelist.echam", ROOT / "lut_kappa.nc", *ROOT.glob("input/*.nc")])
    manifest = {
        "schema_version": 1, "purpose": "compatibility audit; not performance measurement",
        "platform": platform.platform(), "machine": platform.machine(),
        "config": config,
        "scope": args.scope,
        "environment": {key: os.environ.get(key) for key in
                        ["FC", "FFLAGS", "LDFLAGS", "OMP_NUM_THREADS"]},
        "status": "preparing",
    }
    manifest_path = target / "manifest.json"
    try:
        manifest["source_revision"] = output("git", "rev-parse", "HEAD")
        manifest["working_tree_status"] = output("git", "status", "--short")
        manifest["config_sha256"] = sha256(args.config)
        manifest["harness_sha256"] = sha256(Path(__file__))
        manifest["source_sha256"] = {str(p.relative_to(ROOT)): sha256(p) for p in sources}
        manifest["input_sha256"] = {str(p.relative_to(ROOT)): sha256(p) for p in inputs}
        for path in sources + inputs:
            destination = target / path.relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
        (target / "data").mkdir()
        if args.capture_runtime:
            manifest["runtime_capture"] = instrument_driver(target / "driver.F90")
            manifest["runtime_capture"]["status"] = "instrumented_not_run"
        if args.scope == "first-microphysics":
            manifest["scope_capture"] = instrument_scope(target / "driver.F90")
            manifest["scope_capture"]["status"] = "instrumented_not_run"
        if args.capture_salsa_state:
            manifest["salsa_capture"] = instrument_salsa(target / "mo_ham_salsa.F90")
            manifest["salsa_capture"]["status"] = "instrumented_not_run"
        if args.capture_coagulation_state:
            manifest["coagulation_capture"] = instrument_salsa_coagulation(target / "mo_ham_salsa.F90")
            manifest["coagulation_capture"]["status"] = "instrumented_not_run"
        if args.capture_condensation_state:
            manifest["condensation_capture"] = instrument_salsa_condensation(target / "mo_ham_salsa.F90")
            manifest["condensation_capture"]["status"] = "instrumented_not_run"
        manifest["compiled_driver_sha256"] = sha256(target / "driver.F90")
        compiler = config["build"]["compiler"]
        manifest["compiler"] = output(compiler, "--version")
        manifest["netcdf_fortran"] = output("nf-config", "--version")
        manifest["netcdf_c"] = output("nc-config", "--version")
        # Override inherited linker flags explicitly: some local shells supply
        # colon-delimited -L entries that are not valid compiler arguments.
        flags = " ".join(config["build"]["flags"]) + " " + output("nf-config", "--fflags")
        libs = output("nf-config", "--flibs") + " " + output("nc-config", "--libs")
        command = ["make", f"-j{config['build']['jobs']}", "F90=" + compiler,
                   "F90FLAGS=" + flags, "LDFLAGS=", "LIBS=" + libs]
        manifest["build"] = logged_run(command, target, "build.log", config["build"]["timeout_seconds"])
        if manifest["build"]["returncode"] != 0:
            manifest["status"] = "build_failed"
            return 1
        manifest["executable_sha256"] = sha256(target / "ham_box")
        manifest["status"] = "built"
        if args.run:
            manifest["run"] = logged_run(["./ham_box"], target, "run.log", config["run"]["timeout_seconds"])
            manifest["status"] = "run_completed_unvalidated" if manifest["run"]["returncode"] == 0 else "run_failed"
            manifest["diagnostic_sha256"] = {
                str(p.relative_to(target)): sha256(p)
                for p in sorted([*target.glob("*.dat"), *target.glob("data/*.dat")])
            }
            if args.capture_runtime:
                capture = manifest["runtime_capture"]
                try:
                    runtime_path = target / FILENAME
                    capture["runtime_file"] = FILENAME
                    capture["runtime_sha256"] = sha256(runtime_path)
                    capture["metadata"] = parse_metadata(runtime_path.read_text())
                    capture["status"] = "captured"
                except (OSError, ValueError) as error:
                    capture["status"] = "capture_failed"
                    capture["error"] = str(error)
                    if manifest["run"]["returncode"] == 0:
                        manifest["status"] = "runtime_capture_failed"
                    return 1
            if args.scope == "first-microphysics":
                capture = manifest["scope_capture"]
                try:
                    scope_path = target / SCOPE_FILENAME
                    capture["completion_file"] = SCOPE_FILENAME
                    capture["completion_sha256"] = sha256(scope_path)
                    capture["completion"] = parse_completion(scope_path.read_text())
                    state_path = target / STATE_FILENAME
                    capture["state_file"] = STATE_FILENAME
                    capture["state_sha256"] = sha256(state_path)
                    capture["state"] = parse_scope_state(state_path.read_text())
                    capture["status"] = "captured"
                except (OSError, ValueError) as error:
                    capture["status"] = "capture_failed"
                    capture["error"] = str(error)
                    if manifest["run"]["returncode"] == 0:
                        manifest["status"] = "scope_capture_failed"
                    return 1
                if manifest["run"]["returncode"] == 0:
                    manifest["status"] = "first_microphysics_completed_unvalidated"
            if args.capture_salsa_state:
                capture = manifest["salsa_capture"]
                try:
                    capture["file_sha256"] = {name: sha256(target / name) for name in STATE_FILES.values()}
                    capture["state"] = load_capture(target)
                    capture["status"] = "captured"
                except (OSError, ValueError) as error:
                    capture["status"] = "capture_failed"
                    capture["error"] = str(error)
                    if manifest["run"]["returncode"] == 0:
                        manifest["status"] = "salsa_capture_failed"
                    return 1
            if args.capture_coagulation_state:
                capture = manifest["coagulation_capture"]
                try:
                    capture["file_sha256"] = {name: sha256(target / name) for name in COAGULATION_FILES.values()}
                    capture["state"] = load_coagulation_capture(target)
                    capture["status"] = "captured"
                except (OSError, ValueError) as error:
                    capture["status"] = "capture_failed"
                    capture["error"] = str(error)
                    if manifest["run"]["returncode"] == 0:
                        manifest["status"] = "coagulation_capture_failed"
                    return 1
            if args.capture_condensation_state:
                capture = manifest["condensation_capture"]
                try:
                    capture["file_sha256"] = {name: sha256(target / name) for name in CONDENSATION_FILES.values()}
                    capture["state"] = load_condensation_capture(target)
                    capture["status"] = "captured"
                except (OSError, ValueError) as error:
                    capture["status"] = "capture_failed"
                    capture["error"] = str(error)
                    if manifest["run"]["returncode"] == 0:
                        manifest["status"] = "condensation_capture_failed"
                    return 1
            return 0 if manifest["run"]["returncode"] == 0 else 1
        return 0
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        manifest["status"] = "setup_failed"
        manifest["error"] = str(error)
        return 1
    finally:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"{manifest['status']}: {manifest_path}")


if __name__ == "__main__":
    raise SystemExit(main())

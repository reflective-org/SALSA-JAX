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
    args = parser.parse_args()
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
            return 0 if manifest["run"]["returncode"] == 0 else 1
        return 0
    except (OSError, subprocess.CalledProcessError) as error:
        manifest["status"] = "setup_failed"
        manifest["error"] = str(error)
        return 1
    finally:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"{manifest['status']}: {manifest_path}")


if __name__ == "__main__":
    raise SystemExit(main())

# Lessons from local JAX ports

Inspection date: 2026-09-21. These are implementation precedents, not evidence
that SALSA already has a Python port or passes parity tests. Paths below are
relative to the SALSA repository root; neighboring repositories are not dependencies.

## Sources and provenance

| Repository | Inspected HEAD | Working-tree caveat |
|---|---|---|
| `../GLOMAP-JAX` | `b516108d496eba72d769714ef179ad4d475615f0` | Fortran box driver/config/state, tests and inputs have local changes; the cited harness/state/notes files were not listed modified. |
| `../mam4-app/mam4-jax` | `2f9aa3665a93428ad49e337ce66ab42b688c7b4e` | Clean at inspection. |
| `../acdc-jax` | `afe53ad48782b6ba78b5724d15c3e910eea86117` | Clean at inspection. |
| `../tomas-jax-condensation` | `d65e45e59865ce41a6eafd7f40bd23f4d5bdb1e9` | The cited Fortran comparison tests/build wrappers are untracked local work, not guaranteed by this revision. |

The inspection read code and documentation; it did not rerun these repositories'
tests or independently reproduce their reported accuracy and performance.

## Adopt in the SALSA plan

### A reference harness before physics

ACDC's `docs/plan/ULTRAPLAN.md` links phase plans with one verification step per
commit-sized task and keeps progress separately. Its sequence establishes the
Fortran bridge and fixtures before implementing physics, then adds compilation
and batching after eager correctness. Apply that sequence to each SALSA process.

MAM4's `docs/plans/011-coag-orchestration-port.md` separates coagulation helpers
from orchestration, captures before/after state with process switches, and
documents driver terms outside the comparison boundary. SALSA needs equivalent
contracts for every kernel and for their composition; condensation calls
nucleation internally, so switch isolation alone is insufficient.

### Explicit state and source provenance

GLOMAP's `src/glomap_jax/core/state.py` annotates immutable state fields with
units, shapes and index spaces and separates prognostic from derived quantities.
Use similarly explicit SALSA state, static configuration and index conversion
tables. Keep scientific constants traceable to their Fortran declarations.
Do not inherit GLOMAP's modal dimensions or padding policy for a sectional model.

MAM4's `scripts/build_reference.sh` copies sources into a build directory before
applying instrumentation patches. Adopt source hashes and isolated build/run
directories; move SALSA sources only after establishing a reproducible baseline.

### Trustworthy oracle capture

GLOMAP's `validation/leaf_common.py` documents persistent allocation and mutated
`SAVE` tables contaminating later captures. It launches fresh processes, reads
active configuration back from Fortran, checks intercepted error paths and rejects
unexpectedly identical configuration outputs. Test SALSA initialization/reset
semantics explicitly and default to independent capture processes.

GLOMAP's `validation/goldens_manifest.py` records array names, dtypes, shapes,
content hashes and provenance, detecting drift, missing files and unlisted files.
Use this pattern for SALSA, with reviewed regeneration instead of automatic
replacement of expected answers after a failure.

### Measured numerical acceptance

ACDC's `docs/validation.md` distinguishes exact decisions, algebraic evaluation,
stress states and solver convergence. Its fresh-process steady-state capture
avoids reference warm-start dependence. Set SALSA tolerances from SALSA evidence,
including per-variable absolute floors; another model's tolerance is not a spec.

GLOMAP's `docs/porting-notes.md` shows trajectory fixtures can miss important
branches and records rounding, expression-order and loop-carried-state hazards.
Construct threshold fixtures deliberately. Preserve observed scientific behavior
first and record suspected defects separately from any proposed corrections.

### Equivalent performance comparisons

MAM4's `scripts/benchmark_1000_sims.py` synchronizes JAX, excludes warmup from hot
timings and reports repeated measurements. Its Fortran timing includes subprocess
startup and NetCDF output while its hot JAX timing runs in process. SALSA will
report kernel and end-to-end timings separately with equal work on each side;
that example's headline speedup is not a transferable performance expectation.

TOMAS's local `tests/test_fortran_vs_jax.py` and
`tests/fortran_benchmark/build_fortran.py` demonstrate narrow f2py wrappers,
Fortran-contiguous arrays and explicit one-based/zero-based species indices.
Its module-wide skip when the extension is absent is unsuitable for a required
SALSA parity gate: that gate must fail if the oracle did not execute.

## Defer or avoid

- Defer new ODE solvers, GPU optimization and gradient claims until the original
  process order, clipping, substeps and budget behavior pass CPU comparisons.
- Avoid generic fidelity switches for hypothetical defects. Document an observed
  discrepancy and its impact before adding a named alternative behavior.
- Avoid copying dependency versions, tolerance values, model dimensions or
  scientific constants from neighboring models without SALSA-specific evidence.
- Keep Python packaging verification in the reorganization plan. MAM4's
  `pyproject.toml` records a missing-table failure after a package move; test an
  installed wheel when SALSA starts shipping lookup tables or other assets.

The concrete acceptance protocol is in
[Validation and benchmarks](validation-and-benchmarks.md).

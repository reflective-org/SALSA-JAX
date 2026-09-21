# Repository Guidelines

This is SALSA-JAX, the Python/JAX development repository. The preserved Fortran baseline is tagged `fortran-baseline-d87dc07`; `origin` belongs to `reflective-org/SALSA-JAX`. Use this checkout’s own `.venv`.

## Project Structure & Module Organization

Fortran sources remain in the root: `driver.F90` runs the box model and `mo_ham_salsa*.F90` implements sectional microphysics. `include/` holds namelists; `input/` holds NetCDF inputs; `data/` contains samples/diagnostics. JAX kernels live in `src/salsa_jax/`, tests in `tests/`, reference tools in `tools/`, and the commit-sized roadmap in `docs/plans/ULTRAPLAN.md`. Source relocation awaits reference validation.

## Build, Test, and Development Commands

Install GNU Fortran, Make, and NetCDF C/Fortran libraries; dependency regeneration requires `makedepf90`.

- `make`: builds `ham_box`.
- `sh configure`: overwrites `Makefile` from `Makefile.in` and refreshes `.depend`; use after dependency changes. Keep persistent build edits in `Makefile.in`.
- `./ham_box`: runs from the repository root, writing diagnostics.
- `make clean`: removes objects/modules but leaves `ham_box`.
- `.venv/bin/python -m pytest -q`: runs tests, including mandatory Fortran leaf comparisons.

Adjust compiler/library paths locally; preserve `SALSA` and `NOMPI` definitions. For Python tooling, use a virtual environment (`python3 -m venv .venv`) and document dependencies. Exclude environments, binaries, `.o`, `.mod`, and generated diagnostics from commits.

## Coding Style & Naming Conventions

Match local indentation using spaces, uppercase Fortran keywords, lowercase identifiers, and `&` continuations. Use `mo_*.F90`, explicit `INTENT`, and `REAL(dp)`. No formatter/linter is configured. Document units, dimensions, assumptions, and changed workflows. Put tunable settings in namelists/configuration; centralize scientific constants in existing constants modules rather than duplicating literals.

## Agent Workflow

- **Think first:** state assumptions and tradeoffs. Surface competing interpretations and simpler approaches; ask before implementing when ambiguity affects correctness. Use judgment for trivial tasks.
- **Keep it simple:** implement only the requested behavior. Avoid speculative features, single-use abstractions, unnecessary configurability, and impossible-scenario handling.
- **Edit surgically:** match existing style; avoid unrelated cleanup. Remove only code made unused by your changes; report pre-existing dead code separately.
- **Verify goals:** give multi-step tasks a brief plan pairing each step with a check. Iterate through implementation, verification, and correction until success criteria pass. Use agents for independent subtasks and worktrees for isolated changes when useful; review their combined results.

## Testing Guidelines

Use pytest (`test_*.py`); no coverage threshold exists. Reproduce bugs before fixing them. Compare JAX with actual Fortran, then check budgets, nonfinite/unphysical values and trajectories. Record compiler/configuration/results. The checked full model currently exposes an out-of-bounds organic-gas access: completing an unchecked run does not establish scientific validity. See `docs/development/quickstart.md` for isolated runs.

## Commit & Pull Request Guidelines

Use focused commits with descriptive subjects, following history (e.g., “Added VBS modules”). PRs should explain changes, link relevant issues, and report affected configurations, validation commands, and diagnostic comparisons.

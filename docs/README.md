# SALSA-JAX development documentation

- [Master plan and commit-sized backlog](plans/ULTRAPLAN.md)
- [Execution status and next tasks](plans/STATUS.md)
- [Development quickstart](development/quickstart.md)
- [Fortran architecture, units and known defects](development/architecture.md)
- [Independent Fortran repairs and remaining failures](development/reference-repairs.md)
- [Effective runtime metadata and capture scope](development/runtime-metadata.md)
- [Isolated nucleation contract and regression evidence](development/nucleation-reference.md)
- [Checked first-call microphysics scope](development/scoped-reference.md)
- [Standalone radiation initialization and data audit](development/radiation-reference-audit.md)
- [Lessons from local JAX ports](development/local-port-lessons.md)
- [Numerical validation and benchmark protocol](development/validation-and-benchmarks.md)
- [First coagulation-coefficient port](development/coagulation-coefficient.md)
- [2026-09-21 baseline and benchmark results](development/results-2026-09-21.md)

The current Python package implements one collision-coefficient leaf. Complete
coagulation, other aerosol processes and the full box model remain planned.

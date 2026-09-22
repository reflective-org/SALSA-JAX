# SALSA-JAX development documentation

- [Runnable sulfuric-acid-only experiment and trajectories](development/sulfuric-acid.md)
- [Seven-mode number initialization](development/initialization.md)
- [Aerosol material and HAM tracer initialization](development/aerosol-initialization.md)
- [HAM gas conversion and remaining wrapper audit](development/ham-conversion-audit.md)
- [Complete native SALSA API, validation and benchmarks](development/salsa-composition.md)
- [Master plan and commit-sized backlog](plans/ULTRAPLAN.md)
- [Execution status and next tasks](plans/STATUS.md)
- [Development quickstart](development/quickstart.md)
- [Fortran architecture, units and known defects](development/architecture.md)
- [Independent Fortran repairs and remaining failures](development/reference-repairs.md)
- [Effective runtime metadata and capture scope](development/runtime-metadata.md)
- [Isolated nucleation contract and regression evidence](development/nucleation-reference.md)
- [Native SALSA capture and checked fixture](development/native-state-capture.md)
- [State, units and component mapping contract](development/salsa-state-contract.md)
- [Checked first-call microphysics scope](development/scoped-reference.md)
- [Standalone radiation initialization and data audit](development/radiation-reference-audit.md)
- [Lessons from local JAX ports](development/local-port-lessons.md)
- [Numerical validation and benchmark protocol](development/validation-and-benchmarks.md)
- [Active native nucleation capture and acceptance](development/nucleation-native-capture.md)
- [Native grid, hydration and CPU benchmarks](development/grid-and-hydration.md)
- [Native remapping and post-remap correction](development/remapping.md)
- [Supported nucleation and condensation coupling](development/nucleation-process.md)
- [Nucleation rate leaves](development/nucleation-rates.md)
- [Standalone condensation port and benchmarks](development/condensation-process.md)
- [Complete standalone coagulation port and benchmarks](development/coagulation-process.md)
- [First coagulation-coefficient port](development/coagulation-coefficient.md)
- [2026-09-21 baseline and benchmark results](development/results-2026-09-21.md)

The Python package composes the supported native SALSA call for 17 bins and
five materials, with independently tested process APIs. Complete-call checks
cover process switches, native captures, recalculation and short repeated calls.
Number/material/tracer initialization and the first HAM gas conversion also pass
reference and actual driver-boundary checks. Diagnostic conversion, full HAM
coupling, broader trajectories and the box driver remain planned. A separate
pure-sulfate experiment now checks selected 1000-step trajectories and sulfur
budgets against original Fortran. See the [current validation status](plans/STATUS.md)
for evidence and inherited scientific limitations.

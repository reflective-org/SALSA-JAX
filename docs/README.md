# SALSA-JAX development documentation

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

The Python package implements grid generation, hydration, coagulation,
condensation, remapping and post-remap correction for 17 bins and five materials.
The default pre-remap chain and the following two-pass remap/correction chain
separately match native boundaries. Condensation has a separate coupled API for validated nucleation
schemes and growth options 1–3. Complete SALSA composition, seven-mode
initialization and box-model trajectories remain planned. All 541 tests pass in
the repository’s `.venv`; see the [current validation status](plans/STATUS.md)
and process guides for evidence and inherited scientific limitations.

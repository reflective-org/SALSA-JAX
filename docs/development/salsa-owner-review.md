# SALSA owner review: scientific assumptions and box configuration

These six items come from the comparison study's `docs/model_notes/salsa.md`
and were checked against this checkout. They remain owner-review questions,
separate from JAX/Fortran compatibility acceptance. D009 is a reproduced budget
defect; the other items require decisions about intended formulation or build
behavior. None is silently changed in the port.

| Item | Source evidence | Decision requested |
|---|---|---|
| Density assumptions: 1500 / 1841 / 1830 kg/m³ | Coagulation uses fixed [1500](../../mo_ham_salsa_dynamics.F90#L161) for particle masses; the sulfate registry uses [1841](../../mo_ham_species.F90#L301). Nucleation comments document [1830 and 98.08 g/mol](../../mo_ham_salsa_nucleation.F90#L465) in the growth derivation, beside a fixed numerical coefficient. | Confirm which densities and molecular masses should govern collision speeds, aerosol volumes and cluster growth; document or reconcile the approximations. **1830 is a documented derivation assumption, not a separately read runtime density.** Other growth expressions also use registry density. |
| Dry-midpoint kernel with wet-diameter rescaling | [Coagulation](../../mo_ham_salsa_dynamics.F90#L161) evaluates `coagc` at dry bin midpoints and fixed-density masses, then multiplies by `dpmid(m)*pdwet(n)/(dpmid(n)*pdwet(m)`. | Confirm intended validity of this approximation versus recomputing diffusion, particle mass and the complete transition kernel at evolving sizes. |
| Pair-mean Fuchs transition distance | [coagc](../../mo_ham_salsa_init.F90#L349) uses `mdiam=(diam1+diam2)/2` in both particles' `tva` expressions before combining them into `fmdist`. | Confirm pair-mean versus particle-specific diameter treatment; compare unequal-size pairs and transition-regime limits before selecting a change. |
| RH floor and no dry switch | [Hydration](../../mo_ham_salsa_properties.F90#L306) clamps RH to `[0.05,0.95]` for populated bins; soluble bins use equilibrium water uptake. | Decide whether an explicit dry-aerosol mode is needed. Setting input RH to zero does **not** request dry particles. The insoluble-bin hydration flag does not disable soluble-bin hydration. |
| D009 cross-regime volume creation | The [reproducer and transfer audit](reference-repairs.md#d009--cross-regime-coagulation-creates-volume-in-stiff-cases) show mismatched sequential 2a gains and 2b losses; a stiff two-bin case creates about 240% extra volume. | Select and validate a conservative update in a separate scientific change. Empty 2b bins avoid this case in the focused sulfuric-acid experiment; they do not repair it. |
| Box-build nucleation configuration | `nsnucl` [defaults to zero](../../mo_ham_salsactl.F90#L75); the [namelist read and broadcasts](../../mo_ham_salsactl.F90#L234) are inside `#ifdef HAMMOZ`, excluded by the standalone box build. | Expose and report effective standalone switches independently of HAMMOZ. The nucleation routines themselves are available; ordinary namelist edits do not activate them in this build. |

## Effect on the current sulfuric-acid experiment

The [JAX experiment](sulfuric-acid.md) exposes nucleation scheme/growth explicitly
and the native reference wrapper assigns those switches directly. Its active
nucleation checks therefore do not depend on the box namelist reader. Both
implementations retain the density choices, midpoint rescaling, pair-mean
transition distance and RH clamp. The default is hydrated at RH 0.5, not dry.
Fortran parity and sulfur closure do not resolve these formulation questions.

The sparse receiving-bin roundoff limitation documented in the experiment is
separate from D009: it occurs with empty 2b and does not imply cross-regime
volume creation. Keep their reproducers and proposed remedies distinct.

For each owner decision: record the intended formulation, add an independent
reference/budget case, change Fortran in a focused commit, then update JAX and
repeat process and trajectory comparisons. Preserve existing compatibility
results as historical evidence. Comparison-study percentage offsets are not
revalidated by this source audit and are not adopted as new measurements here.

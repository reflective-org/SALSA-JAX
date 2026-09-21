# Native SALSA state contract

This audit describes `mo_ham_salsa.F90:salsa` and its caller,
`mo_ham_subm.F90:ham_subm_interface`, for the current standalone
`SALSA`/`NOMPI` build without `HAMMOZ`. It specifies what a first-call
capture can safely read; it does not validate every process/configuration.
Use captured runtime counts, mappings and flags rather than assuming that
the namelist was read.

## Boundary, dimensions and valid entry fields

Array axes are Fortran `(horizontal, level, ...)`; only
`1:kproma` of the leading `kbdim` dimension is active. Preserve explicit
shape, axis order and one-based indices in the artifact. Do not read padding
cells merely because `kbdim` is larger.

| Native argument | Shape | Effective role and units |
|---|---|---|
| `ppres` | `(kbdim,klev)` | Input, Pa |
| `prh` | same | Input, relative humidity as a fraction |
| `ptemp` | same | Input, K |
| `ptstep` | scalar | Input, s |
| `ppbl` | `(kbdim)` | Input despite INOUT declaration; real-valued boundary-layer top level index, converted locally using `INT` |
| `pc_gas` | `(kbdim,klev,subm_ngasspec)` | Entry/exit gas concentrations; default registered gases are molecules/m³ air |
| `paerml` | `(kbdim,klev,naerocomp)` | Entry/exit flattened aerosol components; sulfate molecules/cm³ air, OC/BC/SS/dust µg/m³ air |
| `pnaero` | `(kbdim,klev,fn2b)` | Entry/exit particles/m³ air |
| `pm6rp` | same bin shape | **Exit only**, ambient particle radius in cm |
| `pm6dry` | same bin shape | **Exit only**, dry particle radius in cm |
| `prhop` | same bin shape | **Exit only**, particle density in g/cm³ |
| `pww` | same bin shape | **Exit only**, aerosol water kg/m³ air |

The last four arrays are declared `INTENT(INOUT)`, but SALSA assigns them
before reading them. Its caller supplies automatic locals `zm6rp`,
`zm6dry`, `zrhop`, `zww` with no preceding initialization on this path.
An entry serializer must **omit their contents**, not inspect them for
finiteness or invent zero inputs. At exit, all active bin cells are assigned.

The caller's dry-radius temporary is dimensioned with `nsol`, whereas the
native dummy uses `fn2b`. The current SALSA setup has
`nsol=nclass=fn2b=17`; capture/assert that compatibility rather than
generalizing to arbitrary configurations.

## Component, species, bin and tracer maps

`paerml` is not a dense bin-by-species tensor. `ham_define_bins` in
`mo_ham_init.F90` builds a flattened `aerocomp` list. The base standalone
construction has 69 active components. This is the authoritative HAM mapping,
not the conflicting region-major mapping produced by the D008 driver bug
described below:

| Material | Valid bins | Bin→component map | Base component slots |
|---|---|---|---|
| Sulfate | 1–17 | `iso4b` | 1–17 |
| Organic carbon | 1–17 | `iocb` | 18–34 |
| Black carbon | 4–17 | `ibcb` | 35–48 |
| Dust | 4–17 | `idub` | 49–62 |
| Sea salt | 4–10 | `issb` | 63–69 |

Unsupported material/bin entries in these maps are zero, not valid Fortran
indices. Bins 1–3 are 1a, 4–10 are 2a, and 11–17 are 2b. Sea salt is absent
from the 2b component list even when 2b hydration is enabled. Water is held
separately in `aerowater`; it is not one of the 69 components.
Additional `HAMMOZ` VBS/aqueous components are outside this base contract.

Capture each active component's `iclass`, `spid` and `idt`:
`aerocomp(component)%idt` selects the corresponding HAM tracer. Capture
`sizeclass(bin)%idt_no` for particle number and
`aerowater(bin)%idt` for water. The `idt_ms4/idt_moc/idt_mbc/idt_mss/idt_mdu`
arrays in `mo_ham_salsa_trac` contain **tracer indices**, whereas
`iso4b/iocb/ibcb/issb/idub` contain **component indices**.

Metadata needed for complete HAM wrapper reconstruction:

- The active `subm_gasspec` and `subm_gasunitconv` entries, plus
  `isubm_so4g/isubm_ocnv`; zero means absent optional organic gas.
- `subm_aerospec`, `subm_aerospec_nowat`, `subm_aero_idx`,
  `subm_aerounitconv` and their active lengths.
- Referenced species identities/names, molar masses [g/mol], densities [kg/m³],
  and `id_so4/id_oc/id_bc/id_ss/id_du/id_wat`. Preserve numeric IDs without
  assuming they equal chemical-axis positions.
- Bin names, `lsoluble` and `lsoainclass`; component and tracer maps are
  separate relationships and should be checked for consistency.

The current native fixture includes gas species/conversion/tracer maps, full
species/tracer names, material/component/number-tracer maps, hydration species
order, molar masses and densities. Direct water-tracer maps, aerosol conversion
tables and bin names/flags remain to be captured before reconstructing the full
HAM wrapper. F11b therefore remains partial.

Record only initialized fields. For example, `ham_define_bins` leaves
`aerowater%aero_idx` undefined. Its `%idt` is assigned later by tracer
definition and is valid at this capture boundary.

## D008: driver initialization overwrites the authoritative mapping

The first native capture in `build/native-capture-first/salsa-entry.txt`
exposed inconsistent material and component metadata. The mutation is inside
`mo_convert_concentrations.F90:conc2mmr`, not a separate `init_aero`
routine. Driver initialization calls it after `start_ham`,
`ham_define_tracer` and `ham_initialize` have established the
species-major component/tracer identities.

Before repair, its `CASE(HAM_SALSA)` reset a local counter and rewrote the maps while
initializing composition by region:

| Map | Region-major overwrite observed before D008 repair |
|---|---|
| `iso4b` | bins1–3→1–3; bins4–10→7–13; bins11–17→42–48 |
| `iocb` | bins1–3→4–6; bins4–10→14–20; bins11–17→49–55 |
| `ibcb` | bins4–10→21–27; bins11–17→56–62 |
| `issb` | bins4–10→28–34 |
| `idub` | bins4–10→35–41; bins11–17→63–69 |

For example, `iocb(1)=4` claims component4 is OC in bin1, but
`aerocomp(4)` still identifies sulfate in bin4. The same routine then
converts components into `pxtm1` using the unchanged `aerocomp` species
and tracer metadata. Thus the defect changes both bin assignment and units
during initialization; it is not merely a serialization-order discrepancy.
Previously cached `idt_m*` maps also retain the original tracer identities.

Commit `0f843dd` removes the counter-based **map assignments** from this
SALSA initialization branch and populates `paerml` through the maps already
defined by `ham_define_bins`. It preserves existing composition recipes:

- 1a: sulfate/OC volume fractions 0.2/0.8.
- 2a: SO4/OC/BC/SS/DU fractions 0.5/0.05/0.05/0.1/0.3.
- 2b: SO4/OC/BC/SS/DU fractions 0.01/0.01/0.01/0/0.97.

Do not reorder registered components or tracers to accommodate the overwrite.
Reference acceptance requires unchanged maps across `conc2mmr`, agreement
between every active bin/material map and `aerocomp%iclass/%spid`, correct
initial per-bin material volumes and tracer destinations. Use unequal number,
core volume and density values in regression fixtures. A conversion roundtrip
alone can reproduce the same wrong permutation and miss this defect.
Reject the pre-repair capture rather than relaxing the parser.

## Internal volume representation and exact conversions

The local `zvols(horizontal,level,bin,material)` contains
m³ of material per m³ of air. Its first five material slots are:

`1=SO4, 2=OC, 3=BC, 4=SS, 5=DU`.

This order differs from the flattened component-construction order, where
dust precedes sea salt. Never reshape `paerml` directly into `zvols`.
The active `subm_aerospec_nowat` mapping used by hydration must agree with
these fixed first-five assignments.

Let `rho_s` denote the captured species density and
`v_so4=(MW_so4/1000)/avo/rho_so4`, in m³/molecule. Entry conversion is:

- Sulfate: `zvols[...,bin,1] = paerml[...,iso4b(bin)] * 1e6 * v_so4`.
- Other present materials:
  `zvols[...,bin,s] = paerml[...,map_s(bin)] * 1e-9 / rho_s`.

The exit assignments invert these exactly:
sulfate is `zvols/(1e6*v_so4)`; other components are
`zvols*rho_s*1e9`. Absent material slots begin at zero.
The local `zvols` itself is not a defined native entry argument: serialize it
only after initialization/conversion, with a distinct internal-stage label.

Coagulation explicitly transfers sea-salt volume entering insoluble 2b bins
into the sulfate volume slot and zeros the sea-salt slot. This existing
approximation does not conserve sulfate and salt separately by species mass.
The native budget test checks their combined material volume, including sulfate
vapor molecular volume; OC, BC and dust retain individual budgets.

`zcore` is dry volume **per particle** [m³/particle], not a concentration.
`zdwet` is wet **diameter** [m], despite occasional radius comments.
The final assignments in `salsa` are:

```text
pm6dry = (zcore/(pi/6))**(1/3) / 2 * 100       # cm
pm6rp  = zdwet / 2 * 100                      # cm
pww    = ((pi/6)*zdwet**3-zcore)*rho_water*pnaero
```

Density is dry-material mass concentration plus `pww`, divided by total
material/water volume concentration, then divided by 1000 for g/cm³.
The low-volume fallback is water density in 1a/2a and BC density in 2b.
For empty/low-number bins hydration supplies nominal `dpmid` geometry;
nonzero nominal radius does not imply populated particles.

## Geometry and configuration dependencies

`set_sizebins` initializes `dpmid` [m], `vlolim/vhilim`
[m³/particle], and `vratiolo/vratiohi` [dimensionless]. Its 2b grid duplicates
2a. Regime limits are 3 nm, 50 nm, 700 nm and 10 µm; the configured bin
counts are 3, then 4+3, with the larger region duplicated for 2b.

These arrays are read by the active SALSA processes and are not reassigned
during this native call. Remapping changes number and material distributions,
not the stored reference grid. Capture initialized geometry and immutable
bin bounds/counts once; entry/exit equality can verify this contract.
Do **not** include `epsv` as presumed valid state: it is declared/exported
but this checkout has no initialization or active consumption of that array.

Capture `lscoag`, `lscond`, `recalc`, `nsnucl`, `nj3`,
`act_coeff`, `lsol2b`, `lhydr2b`, `locgas`, `nsoa`,
`time_step_len`, `nlim`, `massacc`, `d_sa`, `n3`, `epsoc`,
`slim`, `surfw0` and source-identified physical constants.
Hydration follows initialized `sizeclass%lsoluble`; `lhydr2b` affects that
initialization. Preserve the actual order: hydration, coagulation, optional
hydration, condensation with nested nucleation, and two remap/hydration passes.

## Distinguish native exit from HAM exit

Before calling SALSA, HAM applies `xt_borrow`, includes existing tendencies
in `pxtm1+pxtte*dt`, converts mass/number, and clamps aerosol mass/number
nonnegative. It converts number and molecular gas concentrations from cm⁻³
to m⁻³ immediately before the native call. Capture native entry **after**
those conversions, and native exit **before** their inverse.

After SALSA, HAM converts native radii /100 to m and density ×1000 to kg/m³,
reconstructs tendencies, stores water mixing ratios in `pxtm1`, and calls
`xt_borrow` again. Thus native exit is not identical to final HAM state.
Retain `pxtm1`, `pxtte`, timestep and air density at the wrapper boundary
if cross-boundary closure is claimed. Old optional-argument comments in
`ham_subm_interface` still describe cm/g·cm⁻³ even though its returned
assignments are SI.

## Remaining defects and limits

- **D007:** six tiny-diameter cleanup blocks in `salsa` still index
  `pc_gas(...,isubm_ocnv)` without guarding the absent-gas sentinel.
  Condensation/nucleation guards do not cover these blocks. Current reference
  input avoids this branch; arbitrary tiny-volume fixtures may expose it.
- HAM's aerosol-sulfate conversion uses `6.022e20/mw_so4`, while native
  molecular volume uses the separate `avo` constant. Preserve both legacy
  values; do not silently harmonize them when testing closure.
- The `HAMMOZ` diagnostic tail assigns `densaer_class=pm6dry` and
  `aerwat_class=prhop`, apparently mismatched quantities. It is excluded
  from the supported standalone capture, not repaired here.
- Extending to arbitrary gas registries needs a unit audit: HAM multiplies
  every gas slot by 1e6 at the SALSA boundary, while its conversion enum also
  permits mass-based gas slots. The default two registered gases use the
  molecular conversion code; do not generalize that unit label unchecked.
- Capturing finite arrays does not validate conservation, empty-bin thresholds,
  remapping convergence, unsupported SOA paths or a complete driver trajectory.

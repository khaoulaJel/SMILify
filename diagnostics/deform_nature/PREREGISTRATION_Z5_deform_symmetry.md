# Pre-registration — Z5: does constraining the free-form field to be bilateral improve transfer?

**Written and committed BEFORE the runs.** Date 2026-08-31. Folder `diagnostics/deform_nature/`.
Corpus `bench50_clean` — **50 real workers** (50/50 a subset of the 757-worker corpus, 0/50
overlap with ALL_ANTS_CLEAN). Workers are the only corpus of interest.

## 1. What Z4 established

On workers, the whole remaining gap is the free-form channel: `betas` alone transfer at
gen@20/spread **0.2936**, `deform_verts` alone at **0.9885**, the composite at **0.6147**.

Splitting `deform_verts` through the template's exact mirror involution (max match distance
0.00e+00, exact on 100.0% of vertices):

| | value | null |
|---|---:|---:|
| symmetric share of field energy | **0.6006** | **0.500 exactly, by construction** |
| gen@20/spread, symmetric part S | 0.9791 | — |
| gen@20/spread, antisymmetric part A | 0.9792 | — |
| gen@20/spread, `betas + S` (A projected out) | **0.5292** | — |
| gen@20/spread, `betas + deform` as shipped | 0.6147 | — |

The subspaces have equal dimension, so an isotropic random field splits exactly evenly. At 0.60
the field is **barely above the noise null**, and *neither half transfers* — so `deform_verts` is
predominantly per-specimen noise, not anatomy the 25-D space is missing. Nothing imposes this:
`symmetry_penalty` acts on joint scale/translation only, `midline_penalty` on declared midsagittal
vertices; `deform_verts` has never been constrained.

> **H: the antisymmetric half of the free-form field is noise, and refitting without it yields
> registrations that transfer better, without a material loss of fit.**

The 0.5292 above is a **post-hoc projection of finished fits**, i.e. an upper bound on what a
refit could realise. Z5 asks whether the constraint survives being applied during optimisation, or
whether that freedom was load-bearing for the data term.

## 2. Arms

`D1_PROD.yaml`, seed 0, identical but for `w_deform_sym`:

- **A** — as shipped, `w_deform_sym: 0` (= Z3 arm A's configuration; re-run so all arms share one
  job and environment).
- **B** — `w_deform_sym: 2.0`.
- **C** — `w_deform_sym: 10.0`.

Two weights because the term is new and has no prior calibration; §5 states in advance what is and
is not licensed by that. This is a soft penalty, not a hard projection, deliberately: a hard
projection cannot trade against the data term and would make a fit loss uninterpretable.

## 3. VOIDING checks

1. **The involution is exact.** Built from `v_template`, asserted `max match < 1e-6` and exact on
   > 99% of vertices at trainer construction, which raises rather than warns. A sloppy map would
   penalise real asymmetry as if it were error.
2. **The term binds.** Symmetric share of the field must rise above arm A's 0.6006 in B and C.
3. **Shape space open in every arm** — betas mean |z| > 0.5.
4. **Fit not collapsed** — final `chamfer` within **10%** of A, read from the fitter's own log.
   *(Z3's version of this check passed vacuously on `nan` because `metrics.csv` is only written
   when `optimise_moonshot` is asked to evaluate. The scorer here reads the log directly and
   treats a missing value as a failure.)*

## 4. Endpoint

**PRIMARY: `gen@20/spread` on rest-space shaped geometry, workers, vs arm A's ~0.6147.**

- **PASS** — the best non-void arm reaches **≤ 0.56** (i.e. realises at least half the 0.6147 →
  0.5292 headroom) with fit within 10%.
- **PARTIAL** — improves by ≥ 0.02 but does not reach 0.56.
- **FAIL** — no improvement ≥ 0.02, or every improving arm is voided by fit collapse.

**Reported alongside, not decisive:** symmetric share achieved, §6.10 head ratio, `deform_mag`,
per-channel gen@20. Reference is the instrument floor — exact correspondence, 0.00 at k=20.
ALL_ANTS_CLEAN is **not** a target: different corpus, and R6 showed it is in-sample for this shape
space (7.27% of its free-form field inside `span(shapedirs)` vs 0.90% for workers).

## 5. What each outcome licenses

- **PASS** → `w_deform_sym` becomes a D1_PROD candidate, and the promotion requires the full
  757-worker corpus, not this n=50 — the same standard `scale_cap` was held to (EXECUTION_PLAN §4).
  Weight calibration is licensed **only** at that point, as a deployment parameter.
- **PARTIAL** → report; do not ship; do not sweep further weights hunting for the bar.
- **FAIL** → the antisymmetric half is load-bearing for the data term, the 0.5292 projection is
  not reachable by fitting, and the free-form channel is closed as a route. Combined with Z4's
  finding that neither half transfers, that would mean the residual worker gap is **irreducible
  per-specimen scan and correspondence error**, and the honest next step is to re-derive what
  `gen@20/spread` can attain on real scans rather than to keep chasing 0.00.

## 6. Out of scope

Correspondence descriptors, scan degradation, pose, leg axes, anterior rotation limits (Z2),
joint blendshapes (Z3), and any change to the shape space itself.

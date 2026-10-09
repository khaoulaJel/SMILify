# Morphometric Validation (MV) — when can a SMILify measurement be trusted?

**This is Track B.** It is the home of everything answering *"which SMILify measurements should I
trust, and why?"* — as opposed to Track A (`diagnostics/groundtruth/`), which asks *why the fitter
gets anatomy wrong in the first place.*

Created 2026-09-09 to stop this work being scattered across `atta_reference/`, `groundtruth/` and
`morphometrics/`.

---

## The principle

> A number extracted from an articulated fitted model is **not automatically a morphometric.**

Four validity axes, which **fail independently** — this is why one is not enough:

| axis | question | what it needs | coverage |
|---|---|---|---|
| **B1 anatomical** | do the endpoints sit on the structure the trait is *named after*? | expert landmarks | 12 specimens |
| **B2 pose invariance** | does it hold constant when only articulation changes? | nothing but a fit | every specimen |
| **B3 biological** | does it reproduce a known scaling relationship? | reference measurements | 20 Atta |
| **B4 internal consistency** | bilateral asymmetry \|R−L\|/mean (ants are near-symmetric, so L/R disagreement is error) | **no ground truth** | all 757 |

A measurement can be perfectly **pose-stable and still anatomically wrong** — measuring the wrong
structure, consistently. That is exactly R3/R4's finding (34.7% → 0.2% anatomical error the moment
supervision names the correct part, while surface error reads 0.2% either way). Any single axis
certifies fits that are wrong.

## Results (20 Atta, ARM_A fit)

| trait | B2 pose Δ | B3 exp~BL | B1 | verdict |
|---|--:|--:|---|---|
| **HW** | **0.115%** | **1.205** ✓allometric | not landmark-based (Type III extent) | **strongest** — pose-invariant, reproduces published allometry |
| HL | 0.321% | 1.122 ✓ | adjudicated (moved 9.3% diag) | good |
| FL | 0.528% | 1.190 ✓ | **rig joint, not surface** | pose-stable but measured between rotation pivots |
| TBL | 0.678% | 1.014 ✗ | derived | no independent signal beyond body length |
| ML | 1.653% | 1.130 ✓ | adjudicated (moved 12.6%) | usable with caveat |
| WL | 1.926% | 1.010 ✗ | adjudicated (moved 7.8%) | the size normaliser itself; no independent signal |
| PetL | 2.266% | 1.040 ✗ | **unadjudicated** | unvalidatable |
| GL | 2.325% | 0.887 ✓ | **unadjudicated** | unvalidatable |
| **SL** | **6.081%** | 1.225 ✓ | adjudicated (moved **14.7%**) | **pose-fragile** — worst on B2 *and* B4 |

**B3 positive control passes exactly**: reference HW vs reference BL → 1.2352 (published 1.2352,
R²=0.9868). HW's model-derived 1.205 [1.154, 1.257] overlaps it, via a *different* head-width
definition than the Atta replication used — convergent validation.

## Two findings that change how the numbers are read

**1. The null exponent for B3 is 1.0, not 0.** `fitter_3d/utils.py::load_meshes` normalises every
specimen independently, so model-unit traits carry no absolute size (real specimens span 3.2×;
model-unit body length spans only 1.55×). Physical scale is restored per specimen via
`BL_mm / BL_model`; because that factor contains body length, a trait with no independent size
information lands at **exponent ≈ 1.0**. Deviation from 1.0 is the signal — R² is inflated by the
shared factor and is *not* evidence.

*This corrects an earlier claim of mine* that leg measurements show "essentially no size
relationship" (R²≈0.02). That regression used the compressed, noisy model-unit body length as its
x-variable. Against the clean physical reference, `FL` scales at **1.190 [1.088, 1.292]**, excluding
isometry. What survives about legs is their higher **pose** sensitivity, measured independently.

**2. B4 is a trait-level screen, NOT a per-specimen confidence score.** Across traits it ranks
correctly (SL worst on both B2 and B4). But *within* a trait, per specimen, asymmetry does not
predict pose contamination — all four correlations non-significant (p ≥ 0.10, n=20). Reporting it
as a per-specimen trust score would be the same proxy-without-mechanism error this project has
already hit seven times.


## B3b — measurement agreement (added 2026-09-09)

`mv_agreement.py` → `out/mv_agreement.json`. Bias, limits of agreement, proportional bias and
Lin's concordance against Fabian's physical reference. Deliberately **not** led by correlation:
R²=0.99 coexists with large systematic bias, and morphometric methodology treats biological signal
and measurement error as separate things to establish separately.

**Circularity determines which rows mean anything.** Scale is restored per specimen via
`BL_mm/BL_model`, so BL agreement is *exactly* circular (arithmetic check only) and HW-in-mm is
*partially* circular. The dimensionless **HW/BL ratio is the only fully independent test.**

| measurement | bias | 95% LoA | mean abs rel | CCC | prop. bias |
|---|--:|---|--:|--:|---|
| BL (mm) — *exactly circular* | +0.0000 | [0, 0] | 0.00% | 1.000 | no |
| HW_ext (mm) — partially circular | +0.0535 (+2.22%) | [−0.131, +0.238] | 4.61% | 0.995 | no |
| HW_reporter (mm) — partially circular | −0.0031 (−0.13%) | [−0.203, +0.197] | 4.11% | 0.996 | no |
| HW_ext / BL — **scale-free** | +0.0117 (+2.56%) | [−0.030, +0.053] | 4.61% | 0.859 | no |
| HW_reporter / BL — **scale-free** | +0.0022 (+0.47%) | [−0.042, +0.047] | 4.11% | 0.862 | **YES p=0.044** |

**Three things this surfaced that R² could not:**

1. **The two "head width" definitions are not interchangeable.** The reporter-bone distance is
   essentially unbiased (−0.13%); the Type III extent carries a systematic **+2.22%** bias. Which
   definition you use changes the answer.
2. **CCC collapses 0.996 → 0.862 once size is removed.** The near-perfect mm agreement is largely
   carried by the shared scale factor — the circularity concern, made quantitative.
3. **Significant proportional bias in the scale-free ratio (p=0.044)**: error grows with specimen
   size. This is the *same* defect as the recovered exponent landing at 1.205 against the published
   1.2352 — the model slightly compresses the allometric relationship. Two independent methods,
   one real finding. For a polymorphic species this matters: it is precisely the large/small
   extremes where the ratio is least trustworthy.


## M2 — synthetic morphometric recovery (added 2026-09-09)

`m2_synthetic_recovery.py` → `out/m2_results.json`, writeup in
`RESULTS_M2_synthetic_recovery.md`. Uses `synth_power48` (48 specimens, **noise = 0.0**, generated
by the model itself, so an exact solution provably exists) with existing `P48_*` fits — no new
fitting. Because synthetic specimens share the template topology, TRUE per-vertex correspondence
error is available, not a nearest-neighbour proxy.

**Headline: no diagnostic predicts morphometric accuracy.** Chamfer's correlation with trait
recovery error is |r| ≤ 0.34 across nine traits, mostly non-significant, three of them negative.
And the **oracle** diagnostics fail too — true correspondence error, true pose error, true beta
error (none obtainable on a real scan) reach max |r| ≈ 0.41. Geometric fit quality, even measured
perfectly against ground truth, is close to uninformative about whether a measurement is right.

**Recovery error on data where exact recovery is possible** (scale-corrected shape error, median):
WL 3.84%, HW 4.79%, SL 5.76%, HL 6.02%, TBL 7.17%, FL 11.73%, GL 12.31%, ML 20.74%, PetL 25.47%.
This is **search failure, not representational failure** — exactly what R9 predicts.

**Two failure modes, separated.** Pose sensitivity (B2) and recovery error (M2) are different: SL is
worst on pose (6.08%) but mid-pack on recovery (5.76%); FL is pose-stable (0.53%) but poorly
recovered (11.73%). This is the concrete evidence that there is no single scalar quality score.

## Files

| file | role |
|---|---|
| `mv_framework.py` | computes all four axes → `out/mv_results.json` |
| `m2_synthetic_recovery.py` | synthetic recovery + does-any-diagnostic-predict-error → `out/m2_results.json` |
| `mv_agreement.py` | Bland–Altman agreement vs physical reference → `out/mv_agreement.json` |
| `mv_report.py` | renders `out/morphometric_validation_report.html` (self-contained, offline) |
| `out/mv_results.json` | full per-trait, per-specimen numbers |
| `out/morphometric_validation_report.html` | the dashboard |

Reuses, does not redefine: `diagnostics/groundtruth/trait_extract.py` (GLAD trait protocol),
`diagnostics/morphometrics/measure.py` (model loading, body frame).

## Run

```bash
conda activate pytorch3d
python diagnostics/morphometric_validation/mv_framework.py   # ~20 s, CPU only
python diagnostics/morphometric_validation/mv_report.py
```

## Where the neighbouring work sits

| track | location | question |
|---|---|---|
| **A — why anatomy is wrong** | `diagnostics/groundtruth/` (R3, R4, R9, G-series) | does the objective even prefer correct anatomy? (R9: **no**) |
| **B — is a measurement valid** | **here** | which traits can be trusted, on what evidence |
| biological anchor | `diagnostics/atta_reference/` | 20 Atta, physical reference, head-width replication |
| corpus-scale traits | `diagnostics/morphometrics/` | the 757-specimen trait extraction this framework governs |

## Open

- B1 is currently a *provenance* check (is the landmark adjudicated, how far did it move) rather
  than a per-specimen measurement of endpoint-to-expert-anatomy distance. The 12 gt_expert
  specimens support the stronger version; not yet built.
- B2/B4 are computed on Atta only. Both need nothing but a fit, so both can run on all 757 —
  the natural next extension.
- No expert annotation exists for `petiole_*` or `gaster_apex_mid`, so PetL/GL/TBL stay
  unvalidatable until someone annotates them.

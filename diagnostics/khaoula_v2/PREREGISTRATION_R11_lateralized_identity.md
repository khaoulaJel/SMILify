# Pre-registration — R11: can semantic features recover LATERALIZED anatomical identity?

**Written and committed BEFORE the run.** Bars fixed here are not revised after seeing a number.

Date: 2026-09-09. Folder: `diagnostics/khaoula_v2/` (with T1.0/T1.1, whose machinery this reuses).

---

## 1. Why this is not a re-run of T1.0/T1.1

Both already ran (2026-08-19, on JUWELS) and their results define this question rather than
answer it:

- **T1.0 — PASSED decisively.** Raw DINOv2 on 8 rendered views separates the 5 coarse anatomical
  groups: purity 0.797, ARI 0.428 (chance 0.463 / 0.000). Foundation features carry real anatomical
  signal on this exact template.
- **T1.1 — FAILED its gate.** Within-part correspondence: DINO 10.34% median error vs
  coverage-matched HKS 13.51%. A real ~23% gain, but short of the pre-registered ≤6.76%, and >2×
  worse than the fitted pipeline's own 4.41%. Geodesic-refinement (Uzolas et al. 2025) gave 10.12% —
  did not rescue it.

**The reformulation.** T1.1 asked for dense *within-part* correspondence. R3/R4 show the fitter does
not need that: its CORRECT arm constrained each landmark to its correct **part** (`b_h`, `ma_r`,
`an_1_r`) — part-level identity — and that alone moved anatomical error 34.7% → 0.2%. So the
established negative is for a strictly harder problem than the one proven to carry the whole effect,
while the established positive (T1.0) is at exactly the granularity R3/R4 requires.

**The gap.** T1.0's `group_of()` is NOT lateralized — all legs collapse to "leg", all antennae to
"antenna", mandibles fold into "head". R3 needs `ma_r` vs `ma_l`. And laterality is precisely where
geometry is known to fail here (R6: instance recovery blocked by fused left/right structures; R7/R8:
no reliable bilateral reference frame, body rotationally degenerate) and where the literature
independently flags foundation features as weak (*Telling Left from Right*, CVPR 2024).

> **H: pretrained semantic features assign correct LEFT/RIGHT anatomical part identity, on a
> specimen they were not fitted to.**

## 2. Design — T1.0's machinery, changed only where the question changed

Reused unmodified from `t10_diff3f_sanity.py`: template loading, 8-view rendering
(`render_3d.make_renderer/render`), per-vertex backprojection of view features.
Replaced: `dino_features` now loads DINOv2 via `torch.hub` (`facebookresearch/dinov2`,
`dinov2_vitb14`) rather than `transformers`, because this cluster's env pins torch 2.3.1 and
transformers ≥5 requires ≥2.5. Same architecture and weights, different loader — verified by
embed_dim 768 and patch-token shape.

**Labels (Level 2, lateralized):** `head`, `thorax`, `gaster`, and the bilateral pairs
`antenna_L/R`, `mandible_L/R`, `foreleg_L/R`, `midleg_L/R`, `hindleg_L/R`. Level 3 (per-segment
laterality, `L_coxa` …) is explicitly NOT attempted — T1.1 already showed fine within-part
localisation is the harder regime, and starting there would repeat that mistake.

**Generalisation, not self-classification.** A probe trained and tested on the same template's
vertices is meaningless (neighbouring vertices are near-duplicates). Train on the template, test on
**held-out synthetic specimens** from `synth_power48`, whose labels are known because topology is
shared. Reported per specimen.

## 3. Endpoint and pre-registered bar

**Primary endpoint: LATERAL ACCURACY.** Restricted to vertices whose true label is one of the
bilateral parts, the fraction assigned the correct **side** (L vs R), ignoring which segment.
Chance = 50% by construction. This isolates the one thing at issue; overall part accuracy is
reported alongside but does not gate, because T1.0 already established coarse parts are separable
and a high overall score could be carried entirely by head/thorax/gaster.

- **PASS** — lateral accuracy ≥ **70%** on held-out specimens, **and** significantly above both
  50% chance (binomial, p<0.05) **and** the geometric baseline below.
- **PARTIAL** — significantly above chance and baseline, but below 70%.
- **FAIL** — not significantly above chance, or not above the geometric baseline.

70% is set from the downstream requirement, not from a pilot: R3/R4's effect comes from getting the
named part right, so a side-assignment that is wrong ~1 vertex in 3 cannot deliver it. The number is
committed here before any lateralized result is computed.

**Geometric baseline (must be beaten):** nearest-template-vertex by rigid proximity after alignment
— the baseline W1 established as surprisingly strong and which handcrafted and learned descriptors
have repeatedly lost to. Reported on the identical vertex population.

**Key diagnostic, reported regardless of outcome:** the **mirror-confusion rate** — among
misclassified bilateral vertices, the fraction whose predicted label is exactly the mirror
counterpart of the truth (`antenna_L` → `antenna_R`). A high value is the specific signature that
semantics are recovered but laterality is not, which is the literature's stated failure mode and
would point directly at geometry-aware correspondence as the next step rather than a bigger model.

## 4. Mechanism checks (any failure voids)

1. **Label correctness** — assert every bilateral class is non-empty and L/R vertex counts are
   near-equal on the template (the model is mirror-symmetric to 1e-3; grossly unequal counts mean
   the labelling is wrong, not the features).
2. **No train/test leakage** — the probe never sees a test specimen's vertices; assert specimen
   disjointness explicitly.
3. **Feature coverage** — backprojection leaves occluded vertices unlabelled. Report coverage and
   score the geometric baseline on the *same* covered subset, exactly as T1.1 was corrected to do.
4. **Chance control** — a label-shuffled probe must score ~50% lateral accuracy. If it does not,
   the evaluation is broken.

## 5. What each outcome licenses — fixed now

| outcome | reading |
|---|---|
| PASS | semantic features supply the identity R3/R4 needs, including laterality → proceed to R12 (predicted vs oracle identity → SMILify → morphometric error) |
| PARTIAL | signal exists but is insufficient alone → geometry-aware semantic correspondence (*Telling Left from Right* first) is the motivated next method, not a bigger backbone |
| FAIL, with **high mirror-confusion** | the specific, literature-predicted failure: semantics recovered, laterality not. Strongest possible motivation for geometry-aware processing |
| FAIL, with **low mirror-confusion** | features do not carry ant part identity at this granularity at all; T1.0's coarse result does not extend, and the foundation-feature route closes |

## 6. Not claimed

Synthetic specimens with shared topology and exact labels. Nothing about real scans, whose
fragmentation and fused left/right structures (R6) can only make this harder. A PASS here is a
necessary, not sufficient, condition for the real-scan case.

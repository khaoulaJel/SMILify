# The V-series: what anatomical supervision does to SMILify, and the measurement bug that hid it

**2026-09-07.** Nine experiments (V1–V10), one bug in our own instrument, three claims withdrawn —
two of them mine. Written so the reasoning survives without the person who ran it.

---

## 1. Where this started

The G-series had established a specific, actionable position:

- **G6** — the production objective contains **no term constraining joint position**. Chamfer,
  edge, normal, Laplacian, offset, symmetry, midline, scale are all surface or regularisation
  terms; `limit` bounds joint *rotations*, not *positions*. The skeleton is free to be wrong
  provided the surface matches. **The objective is under-determined w.r.t. the skeleton, not
  mis-weighted** — which is why every Z-series reweighting returned null.
- **G7** — adding a joint-position term cut joint error 38.9% → 27.5% for 19% more chamfer.
- **G8** — the benefit does **not** propagate to unsupervised joints below ~75% coverage.
- **T1** — 9.1% of the 757 production fits emit biologically impossible traits, mechanism unknown
  (T1's own proposal was refuted by T2).

The V-series set out to find an objective where **both** anatomical accuracy and surface morphology
beat production. It found something else first.

---

## 2. The arc, in order

### V1 — deformation is not the escape route (n=14)

Tested whether free per-vertex deformation is *why* the objective ignores the skeleton. Freezing it
should then let the joint term reach further.

**The opposite.** Freezing deform made joints **worse** (9.12% → 17.52%); removing it alone did
nothing for anatomy (36.65% vs production 38.86%). Deformation *helps* the joint term.
**"Freeze deform during the anatomical stage" is closed.**

An apparent finding: impossible-trait count tracked deform magnitude across all nine arms
(Spearman 0.932), and both zero-deform arms had zero violations.

### V2 — that finding was wrong, and I withdrew it (n=757)

`trainer.py:352` applies deformation last (`verts = verts + deform_verts`), so `verts − deform` is
an exact ablation. Stripping it from every fit in the corpus moved the impossible rate **9.1% →
8.2%**. Ten rescued, three newly created.

**I had generalised from the wrong verb.** V1 showed the joint term *creates* violations; it never
showed deformation *causes* the ones already present. V1's zero-deform arms were **re-optimised**,
so the whole solution moved. And V1's 14 specimens contained exactly **one** production violation —
a subset with no baseline signal cannot support a claim about 69 of them.

*Control: reproduced `traits_Z8.npz` to 0.000e+00 and T1's 69/757 exactly.*

### V3 — not an identifiability degeneracy either (no ground truth needed)

If anterior pose and shape spanned the same surface subspace, no optimiser could separate them.
Principal angles at each specimen's own production parameters: **anterior 12.14°, gaster control
18.40°**, and **no specimen in either region had any angle below 10°**. Flagged vs unflagged:
12.10° vs 12.14°, **p = 0.476**.

**Refuted.** β also moves anterior joints only **0.104×** as much as θ does, killing the
"β compensates for bad head pose" mechanism from a second direction.

**But the severity distribution was the useful part:** 68% of flagged specimens sit within 25% of
the bound, none beyond 2× the bound width. **A continuous drift, not a bimodal catastrophe** — the
regime where a weak regulariser suffices and multi-start/continuation are unnecessary.

### V4 — the hinge baseline (n=189)

`L_production + w·hinge`, hinge zero inside the biological bounds. **9.1% → 0.1% at 1.01× chamfer**,
saturating immediately at w=1. Unflagged variation retained 0.98×.

**Stated in the file at the time:** the violation count is not an independent endpoint, because the
hinge optimises the very quantity being scored. A mandible pushed just inside ML/HL = 2.00 is not
thereby a correct mandible.

### V5 — joint supervision on real scans, corrected annotations (n=12)

First run scored against `gt_expert` — the corrected annotations. **Every prior G-series experiment
had used the superseded `gt_batch1`**, whose mandibles and coxae are ~20% of a mesosoma off.

Joint error **32.74% → 15.20%** at 1.54× chamfer. The anterior starts far worse than the posterior
(37.31% vs 24.38%) and closes most of the gap by λ=0.1.

**And violations climbed 1 → 2 → 5 → 6 → 8 as joint error collapsed.** Conclusion at the time:
anatomy and surface-derived validity are *in conflict*.

### V6 — the measurement bug (n=12 → 757)

The first **surface** accuracy measurement in the project. Distance from the fitted surface to a
human landmark:

| | median, % of Weber's length |
|---|---:|
| the template vertex we *call* that landmark | **48.8%** |
| the **nearest point on the same surface** | **7.7%** |

**6.4×.** Reported raw, the first number says the fit is catastrophically wrong. It is not — the
landmark definitions are. `landmark_template_indices.json` was rule-computed on 2 Sept, tagged half
its entries `source="verify"`, and was **never verified** because no ground truth existed.

Recalibrating from the 12 human-annotated specimens moved indices **4–21% of a body length**. On all
757 fits: **ML/HL violations 44 → 0**, corpus median 0.95 → **0.40** against a human median of
**0.35**. The old mandible landmark inflated measured mandible length ~2.4×.

**This retroactively explains four consecutive nulls.** T1 proposed a mechanism, T2 refuted it; V1
proposed deformation, V2 refuted it; V3 proposed a degeneracy, null at p = 0.476. **Nobody could
find a fitting cause for the mandible violations because there was none — the failure was in the
ruler.**

### V5b — and so V5's headline was also wrong

Same experiment, only the ruler changed. The violation column is **flat at 0–1 across the entire
sweep**, with no relationship to λ.

**V5's "conflict" is withdrawn.** What survives: at λ=0.03, joint error **32.74% → 15.20%** with
**zero** impossible traits for 1.54× chamfer. A trade-off remains, but it is **anatomy vs chamfer**,
not anatomy vs biological validity. The two were conflated because the same broken landmarks
produced both numbers.

### V7 — the recalibration generalises, and is still not enough

Leave-one-specimen-out: calibrate on 11, evaluate on the 12th.

| | median, % of WL |
|---|---:|
| old indices | 53.6% |
| recalibrated, held out | **44.2%** |
| **floor** (nearest surface) | **10.1%** |

The correction is real (8 of 10 landmarks improve) but sits at **4.4× the floor**: **77% of the
remaining error is the landmark index, not the fit.** The 12 specimens' votes disagree by 6–17% of a
body length — **a single template vertex cannot represent an anatomical point across these
species.**

**This forced a design decision:** a landmark supervision term must be **point-to-surface**, never
"template vertex v belongs at human point p", because the latter asserts a correspondence the data
says is unstable. Not visible without the held-out split.

### V8 — sparse surface observations reach unseen anatomy, nearly for free (n=12)

`L = mean_p min_v ‖p−v‖²`, assignment recomputed every step. Four landmarks **never supervised**
decide the result.

**Held-out landmark error 12.9% → 6.9% — a 47% cut — for 1.07× chamfer.** Supervised points collapse
to 0.1%, which is reachability and is *not* the claim.

**And joint error is unchanged: 33.22% → 33.00%.** Surface supervision improves the surface and
leaves the rig where it was.

### V9 — the channels are orthogonal, and the combined objective is the one to ship (n=12)

Four arms, all judged on the same four never-supervised landmarks.

| arm | held-out surface | joint err | chamfer | deform |
|---|---:|---:|---:|---:|
| A  D1 | 13.4% | 32.80% | 1.02× | 5.68 |
| B  +landmarks | **7.6%** | 32.94% | **1.07×** | 5.80 |
| C  +joints | 12.9% | **15.91%** | 1.60× | 7.72 |
| D  +both | **7.1%** | **15.33%** | **1.59×** | 7.83 |

**The missing arrow is absent.** A 51% correction to the skeleton moves surface anatomy by 4% —
noise. With V8's surface → joints null, **the channels are independent in both directions.**

**Complementarity is additive, not synergistic — and that suffices.** D matches B on the surface
and C on the joints with no interaction term, so **D halves both errors at 1.59× chamfer, the price
of the joint term alone.** The landmark term is free once the joint term is paid for; **D dominates
C outright.**

**The compensation hypothesis is dead.** `deform_verts` RMS goes 3.91 → 7.72 (C) → 7.83 (D): the
joint term inflates deformation, reproducing V1, and landmark supervision does not suppress it.

### V10 — and V8's headline number was split-dependent

Pre-registered bar (every split improves >20%): **FAILED**.

| split | production | supervised | change |
|---|---:|---:|---:|
| S1 left+rear *(V8's)* | 12.9% | 7.4% | 42% |
| S2 right+front | 10.9% | 8.0% | 26% |
| S3 head midline | **7.2%** | 7.0% | **3%** |
| S4 mixed | 10.4% | 6.7% | 36% |

**V8's 47% was measured on the split with the worst production baseline and should not be quoted.**

But the *endpoints* barely move: **7.4 / 8.0 / 7.0 / 6.7** across four disjoint subsets. S3 does not
fail because supervision stopped working — its landmarks were already at 7.2% in production. Every
split converges to the same place; only the distance travelled differs.

**Post-hoc, and flagged as such:** the honest claim is not "supervision improves held-out landmarks
by 47%" but **"supervision drives held-out surface-landmark error to ≈7% of Weber's length
regardless of which landmarks are supervised."**

### V11 — human landmark repeatability is 2.4%, so the floor is the model (n=3 specimens, 36 pairs)

Three specimens re-annotated independently. Median disagreement **2.4% of Weber's length**, and it
does not depend on specimen size — the smallest, worst-fitting specimen is as repeatable as the
largest.

**The cross-check settles it.** Every V10 split's endpoint sits **2–3× above its own split's human
noise**, and S2 — built entirely from the four *most* repeatable landmarks — floors *highest*.
**The ~7% is the model, not the annotator.**

**V11b:** one landmark, `antennal_insertion_l`, read 37.5%. A third round showed it is **bimodal,
not noisy** — two rounds agree to ~2% and a third sits ~38% away, with the outlier changing between
specimens. The socket is a wide annulus and any rim point is defensible. **My occlusion explanation
was refuted** (the smallest, hardest-to-see specimen was the *most* consistent); the annotator's was
correct. Pinning the definition to "where the tube emerges" fixes it. This is V7's *a landmark is
not a point* reappearing inside the annotation protocol.

### V12 — CAPACITY IS NOT THE BOTTLENECK (n=12)

| arm | HELD-OUT | all 12 | chamfer |
|---|---:|---:|---:|
| production | 12.9% | 10.1% | 1.00× |
| S8 supervise 8 | 7.3% | — | 1.08× |
| **S12 supervise ALL 12** | 0.2% | **0.2%** | **1.09×** |
| S8_ORACLE vertex→point | 7.7% | — | 1.05× |

**All twelve landmarks held simultaneously at 0.2%, inside the model's own prior, for 1.09×
chamfer.** This is the honest capacity test G3's withdrawn 3.7% was not — no oracle leaves the
prior. **The representation is capable; do not rebuild it.**

**Correspondence identity is not the lever.** Handing the optimiser the per-specimen vertex
genuinely nearest each human landmark gives **7.7% vs 7.3%** — no effect.

### V13 — NO RESOLVABLE DENSITY CURVE, and the generalisation claim collapses (n=12)

**Read this before quoting any V8 or V9 number.**

My first output scored every held-out subset against the all-12 production median. That was wrong:
production error varies **six-fold** across landmarks (head_width_l 3.9% → scape_apex_l 23.6%), so
subset baselines ranged 5.8–22.2%. Corrected, scoring each draw against its own baseline:

| k supervised | median change on UNSEEN landmarks | range across 3 draws |
|---:|---:|---|
| 2 | −7% | −8 to −5 |
| 4 | +9% | −19 to +14 |
| 6 | +25% | +11 to +31 |
| **8** | **+3%** | **−10 to +28** |
| 10 | +42% | +3 to +60 |

**There is no curve.** Ranges overlap so heavily the ordering is not established.

**And this collapses V8/V9's generalisation claim.** V9's arm B reported held-out error 13.4% →
7.6% — a **43%** improvement — on one split. Three further draws at the same density give a
**median of +3%**, one of them **−10%**, worse than production.

> **Sparse surface supervision corrects the points it touches (to 0.1%) and does substantially less
> for the rest than this report previously claimed.** V8's 47% and V9's 43% are both withdrawn.

---

## 3. What the evidence now supports

**Two anatomical channels, independent in both directions.**

| supervision | improves | does **not** improve | chamfer |
|---|---|---|---:|
| interior joints (λ≈0.03) | joints 32.8% → **15.9%** | surface (13.4% → 12.9%, null) | 1.60× |
| surface landmarks (λ≈1) | **the points it supervises**, to 0.1% | joints (32.8% → 32.9%, null) | **1.07×** |
| both | joints → 15.3% AND supervised surface → 0.1% | — | 1.59× |

Where the rotation pivots sit and where the skin sits are **separate problems in this model**.
Neither correction propagates to the other.

**The representation is capable.** V12's S12 places all 12 landmarks at 0.2% inside the prior.
Nothing in this investigation supports rebuilding the shape, pose or deformation model.

**Anatomical constraints are LOCAL.** Below full coverage, generalisation to unsupervised anatomy is
not reliably demonstrated — on either channel. G8 found joint supervision needs ~75% coverage; V13
found surface supervision generalises erratically at every density below full. **This is the
strongest surviving conclusion, and it is a requirement, not a solution:**

> **Anatomical information must be DENSE. V13 establishes how much information the fitter needs, not
> how that information should be obtained.**

It could come from dense semantic correspondence, automatically detected features, a learned
descriptor, anatomical segmentation, a dense registration field, skeleton-plus-surface semantics, or
a different fitting formulation. **No mechanism is committed to on this evidence.**

**The ~7% floor is the model, not the instrument.** Human repeatability is 2.4% (V11), and every
V10 split's endpoint sits 2–3× above its own split's noise. But V13 reframes what that floor *is*:
unsupervised landmarks largely stay near where production put them, so ~7% is closer to "supervision
did not reach here" than to "supervision converged and stopped".

---

## 4. What is NOT established

- **The human noise floor is unmeasured, and V10 has made this the central open question.** Every
  configuration lands on ≈7%, below V7's 10.1% floor. If a second annotation of the same specimen
  disagrees with the first by ~7% of Weber's length, every anatomical number above has reached the
  instrument's ceiling and further tuning measures noise. If it disagrees by ~1–2%, the floor is
  something else and real headroom remains. **One specimen annotated twice settles it — ~10 minutes,
  and the highest-leverage outstanding action in the project.**
- **V8's 47% is withdrawn as a headline** (V10): it was the best of four splits, range 3–42%.
- **Everything anatomical is n = 12**, one run per arm, no seed replication. This fitter shows
  ~0.01 run-to-run variation (Z4/Z5); differences under ~10% relative are not readable.
- **Joint error is measured on the joints used as targets** — reachability, not generalisation.
  G8's ~75%-coverage requirement is untouched by anything here.
- **The trait layer remains correspondence-bound.** V7's 44.2% held-out landmark error is the
  ceiling on every trait number, and T1/T3/T4/V4 all inherit it. Point-to-surface fixes
  *supervision*, not *measurement*: a distance between two landmarks still needs two identified
  points.
- **T4's 2.8× surface-over-joint advantage** survives recalibration in direction but shrinks
  (1.97× → 1.57× on an internally consistent statistic). It should be recomputed properly before
  being quoted.
- **HW/HL got worse under recalibration** (33 → 48 violations), but the comparison is confounded:
  the old HW is a maximum *extent*, the new one a two-vertex distance. Only ML/HL and SL/HL are
  like-for-like.

---

## 5. What I would do next, in order

1. **Repeat annotation of one specimen** → the noise floor. Gates the interpretation of everything
   above. ~10 minutes of expert time.
2. ~~Read V9 and V10~~ — **done.** Channels are orthogonal and additive; V8's percentage was
   split-dependent but its endpoint is not.
3. **Corpus-scale validation of arm D**, but only after (1) — if ≈7% is the annotation floor, a
   corpus run would be measuring noise at considerable compute cost.
4. **Then** choose λ from the anatomy/chamfer Pareto curve — *not* from the violation criterion,
   which V6 and V5b showed was measuring the broken ruler.
5. **Deferred deliberately:** the 15 stratified annotations, hinge tuning, a learned pose prior, a
   joint predictor, new deformation priors, learned correspondence. V6 saved that annotation
   effort — running the original 2×2 on fixed template vertices could have "proved" surface
   landmarks don't work when the fault was the landmark representation.

---

## 6. Process notes worth keeping

- **Test an indexed-correspondence error against the nearest-surface distance** before concluding
  anything about fit quality. That single check turned "the fit is catastrophically wrong" into
  "the ruler is wrong" (V6).
- **A provisional input stays provisional.** `landmark_template_indices.json` said in its own header
  that every index "must be visually verified before it is used". T1, T4 and V4 used it anyway.
  The failure was escalation of confidence, not code.
- **Absence is data.** Cephalotes has no antennae; Eciton holds its mandibles crossed so the right
  apex sits on the left. Both were flagged as errors by checkers that assumed otherwise, and both
  checkers were wrong.
- **The annotator's anchors never landed on the animal** — the joint file and the `.obj` are in
  different frames (~100× scale, permuted axes). Solved exactly through the fitting pipeline rather
  than by brute force, because brute force left the *reflection* ambiguous and a wrong sign would
  have silently swapped left and right.

---

## 7. Artifacts

Results: `RESULTS_V1..V8_20260907.md`, `RESULTS_V5b_20260907.md`.
Code: `v1_real_scan_validation.py`, `v2_corpus_deform.py`, `v3_identifiability.py`,
`v4_hinge_baseline.py`, `v5_pilot.py`, `v6_surface_accuracy.py`, `v7_landmark_heldout.py`,
`v8_point_to_surface.py`, `v9_four_arm.py`, `v10_split_robustness.py`.
Data: `annotation/landmarks/*_traits.json` (12 specimens × 12 landmarks),
`annotation/landmark_indices_recalibrated.json`, `annotation/joint_to_obj_transform.json`,
`annotation/qc_landmarks.py`, `annotation/qc_annotations.py`.

# SMILify anatomical accuracy: what we asked, what we found, and what it means for the pipeline

**2026-09-08.** Written to be read on its own. Every number traces to a committed experiment; the
file names are given so any claim can be checked.

---

## The question

SMILify fits a parametric ant model to scans by minimising a surface objective — chamfer, edge,
normal, Laplacian, offset, symmetry, midline, scale. The morphometrics we want come from the fitted
model. So the question is not *"does the mesh match the scan?"* but:

> **Do measurements taken from the fitted model correspond to the animal's actual anatomy?**

To answer it we needed ground truth. Twelve specimens were annotated with expert joint positions,
and the same twelve with **twelve anatomical surface landmarks** each — the first surface ground
truth this project has had.

---

## Finding 1 — the objective does not constrain where the skeleton is

**Established earlier (G6, G7) and confirmed throughout.** Every term in the production objective
is a surface or regularisation term. `limit` bounds joint *rotations*, never joint *positions*. The
skeleton is free to be wrong provided the surface matches, and many skeletons match one surface.

Adding an explicit joint-position term cuts joint error against expert annotations from
**32.7% to 15.2%** of mesosoma length, for 1.54× chamfer *(V5, `RESULTS_V5b_20260907.md`)*.

**This is not a mis-weighting.** No reweighting of existing terms can constrain a quantity the
objective never mentions — which is why the earlier Z-series reweighting experiments all returned
null.

---

## Finding 2 — skeletal and surface anatomy are independent channels

*(V9, `RESULTS_V9_20260907.md`)*

| supervision | corrects | leaves unchanged | chamfer |
|---|---|---|---:|
| interior joints | joints 32.8% → **15.9%** | surface anatomy (13.4% → 12.9%) | 1.60× |
| surface landmarks | the supervised surface points | joints (32.8% → 32.9%) | **1.07×** |
| both | joints → 15.3% **and** surface points | — | 1.59× |

Correcting where the rotation pivots sit does **not** move the skin toward the anatomy, and vice
versa. The combined objective collects both benefits **for the price of the joint term alone**.

---

## Finding 3 — the representation is capable

*(V12, `RESULTS_V12_20260908.md`)*

Supervising all twelve landmarks places **every one at 0.2%** of Weber's length simultaneously, for
**1.09× chamfer**, using pose, shape and deformation exactly as production uses them — nothing
leaves the model's own prior.

> **The 25-PC shape space can express the anatomy. There is no evidence for rebuilding the model.**

---

## Finding 4 — anatomical constraints are local

*(G8, and V13 `RESULTS_V13_20260908.md`)*

Supervising a subset does **not** reliably fix the rest. On the skeletal channel, benefit needs
~75% coverage. On the surface channel, supervising 8 of 12 landmarks changes the other four by a
median of **+3%**, with draws ranging **−10% to +28%** — no reliable propagation at any density
tested.

> **Anatomical information must be dense. This is a requirement, not a prescription of mechanism.**

---

## Finding 5 — the headline result: the fitted model's named parts are not on the anatomy

*(R2 and R3, `RESULTS_R2_20260908.md`, `RESULTS_R3_20260908.md`)*

On production fits, the distance from each expert landmark to the model's **correspondingly named
part**:

| landmark | named part | distance to that part | distance to *any* surface |
|---|---|---:|---:|
| mandibular_apex_r | `ma_r` | **50.4%** | 12.1% |
| mandibular_apex_l | `ma_l` | **30.0%** | 14.9% |
| scape_apex_r | `an_1_r` | **73.6%** | 20.8% |
| scape_apex_l | `an_1_l` | **80.7%** | 23.6% |
| antennal_insertion_r | `an_1_r` | **42.7%** | 4.2% |
| antennal_insertion_l | `an_1_l` | **27.2%** | 4.3% |

*(% of Weber's length. The parts exist and are populated — `ma_r` has 195 vertices.)*

> **The fitted model's mandible is not where the real mandible is. Its antenna is not where the real
> antenna is. By 27–81% of a body length.**

**And it is fixable — but only by supervision that names the structure.** Three arms, identical
targets, differing only in which vertices may satisfy them:

| arm | err to the **named** part | err to any surface | chamfer |
|---|---:|---:|---:|
| production | 35.5% | 10.1% | 1.00× |
| **named-part supervision** | **0.2%** | 0.2% | **1.25×** |
| free point-to-surface | **34.7%** | 0.2% | 1.10× |

All arms reach the landmarks equally (0.2% to *some* surface). Only naming the part relocates the
anatomy. **Free supervision leaves the mandible exactly as misplaced as production did.**

**The effect is uniform, not driven by a subset** *(R4, `RESULTS_R4_20260908.md`)*. Named-part
supervision reaches **12 of 12 landmarks** (0.1–0.3%), and free supervision leaves the named parts
more than 10% away on **11 of 12 specimens** — from 5.4% on the best-fitting (*Leptogenys*) to
**63.4%** on *Eciton*. Per landmark under free supervision: `scape_apex_l` 63.0%,
`mandibular_apex_r` 44.7%, `scape_apex_r` 43.0%, `wl_anterior_r` 40.8% — while `head_width_l`
(0.3%) and `clypeal_ant_mid` (6.3%) are already close, because those landmarks sit on the head,
which is the one part the fit does place correctly.

---

## Consequence for the pipeline — a measurement convention

**Point-to-surface distance cannot distinguish a fit whose anatomy is correct from one that is 35%
of a body length wrong. Both read 0.2%.**

Every surface-accuracy number in this investigation, and chamfer itself, measures **surface
proximity, not anatomical correctness**. They are correct as stated and measure less than the words
suggest.

> **Any evaluation of an anatomical claim must report distance to the NAMED part.**

This also resolves the long-standing anterior puzzle (§6.10, X1, Z2): traits computed from the
fitted mandibles and antennae were never measuring the structures they name.

---

## Two things about the measurement layer that change what is trustworthy

**1. The GLAD trait landmarks were mis-specified.** `landmark_template_indices.json` was
rule-computed and half its entries were tagged `source="verify"`; verification was impossible
without surface ground truth. Against the expert landmarks the indices sat **4–21% of a body length**
from where they belong, and the mandible index inflated measured mandible length **~2.4×**
*(V6, `RESULTS_V6_20260907.md`)*. This is why 44 of 87 corpus trait violations were a measurement
artefact rather than a fitting failure.

**2. Weber's length is a Type III landmark pair, and it normalises every trait we report.**
Independent repeat annotation of three specimens gives median human repeatability of **2.4%** of
Weber's length, and the per-landmark ordering reproduces standard geometric-morphometric theory
exactly: **Type I/II anatomical points median 1.8%, Type III constructed extremes median 5.0%,
every I/II below every III (p = 0.002)** *(V11, `RESULTS_V11_20260908.md`)*. The annotation is of
textbook quality; the Type III normaliser is a protocol limitation, not a fitting defect.

---

## Where this leaves SMILify

The formulation the investigation converged on:

> **SMILify has sufficient model capacity to represent the anatomically correct solution, and
> anatomical constraints are demonstrably powerful, but the unsupervised fitting objective cannot
> reliably recover anatomical identity from scan geometry alone.**

Each clause is measured. *Capacity*: all 12 landmarks at 0.2% inside the model's own prior, 1.09×
chamfer (V12). *Constraints are powerful*: naming the part moves the named structures from 34.7% of
a body length off to 0.2% (R3, uniform across 12/12 landmarks and 11/12 specimens, R4). *Cannot
recover identity*: four annotation-free procedures for recovering laterality all fail (R6–R8), the
last of them because the body is rotationally degenerate about its long axis while the features that
would break the degeneracy are the posture-contaminated appendages.

**The narrow claim, stated deliberately:** for these imperfect pinned scans, the geometric
information available to the tested annotation-free procedures is insufficient — not "geometry
cannot determine anatomy".

**What this justifies:** supplying the anatomical identity by a learned semantic correspondence,
whose job is narrow — recover the information the current objective cannot infer — rather than
replacing SMILify's model or optimiser, neither of which the evidence indicts. And the value of that
information is already bracketed: **34.7% → 0.2%**. The open question is how much of it a learned
system recovers automatically, not whether it would be worth having.

## What we recommend

1. **Report surface-derived traits, not rig-derived ones** — surface traits carry ~1.6× the genus
   signal on identical fits, after landmark correction *(T4, re-verified)*.
2. **Report distance to the named part** in any anatomical evaluation. This is the single most
   important convention change.
3. **Do not rebuild the shape model.** Finding 3 shows it is capable.
4. **The missing ingredient is semantic, not geometric**: the objective needs to know *which
   structure* should reach a location. That is one bit per landmark, and it is worth the difference
   between 34.7% and 0.2%.

---

## Artifacts

Experiments `V1`–`V13`, `R1`–`R4` in `diagnostics/groundtruth/` (`RESULTS_*.md` plus the script and
JSON for each). Ground truth in `annotation/`: `gt_expert/` (joints, 12 specimens), `landmarks/`
(12 surface landmarks × 12 specimens), `landmarks_repeat/` and `landmarks_round3/` (repeatability),
`qc_landmarks.py`, `qc_annotations.py`, `qc_repeatability.py`.
A literature check of these findings against current work is in
`LITERATURE_VALIDATION_20260908.md`.

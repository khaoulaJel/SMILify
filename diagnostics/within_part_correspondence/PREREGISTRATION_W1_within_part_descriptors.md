# Pre-registration — W1: is anatomy-conditioning the missing information *within* a part?

**Written and committed BEFORE the run.** Bars below are fixed. Nothing in this file is revised
after seeing a number; if a bar turns out to be badly chosen, that is recorded next to the verdict
rather than edited away (standing practice, cf. F2's "threshold kept deliberately").

Date: 2026-08-30. Folder: `diagnostics/within_part_correspondence/` (self-contained; touches no
live module).

---

## 1. The question, and why this one

Five candidate locations for the correspondence gap have been tested and excluded — backbone
architecture (F1), target encoding (F6), selection (F2), post-hoc assignment (F4), embedding
capacity (F7). Two more attractive lines are closed on their own evidence: pose initialisation
(C14p/E1 — ground-truth pose is the ceiling and buys ~0.009) and bilateral symmetry
(`geom_leg_init.py:790` — true L/R leg-pose distance averages 33.7°, so ants are not symmetric
enough to average).

What remains un-probed is the thing an earlier result pointed at directly: **~83% of the residual
correspondence error is *within* anatomical parts, not across part boundaries.** Coarse anatomical
identity is substantially solved. The open question is the one inside it:

> Given that a surface region is **already known** to be the correct part (say `l2_r_fe`), does an
> **anatomy-conditioned** descriptor identify *which point inside it* corresponds, better than the
> nearest-surface geometry the optimizer currently uses?

This is deliberately the smallest experiment that can answer it. No network is trained. No fitter is
run. If the answer is no, the "inject semantic identity" line is constrained cheaply and early,
before any GPU is spent on a descriptor network.

## 2. Setup

- **Corpus**: `diagnostics/moonshot/synth_b2_train_corrb06.npz` — 4000 synthetic specimens sharing
  the template topology (10,235 verts). **Ground-truth correspondence is vertex-index identity**,
  verified from the corpus construction: `verts` is `(4000, 10235, 3)` on one fixed topology, so
  vertex *i* of specimen A and vertex *i* of specimen B are the same anatomical point by
  construction. This is what makes the task exactly scoreable with no annotation.
- **Specimens**: 12 held out from the tail of the corpus (indices fixed by seed 0), 20 ordered
  pairs (A → B) drawn from them.
- **Parts**: the 36 leg segment classes from `build_segment_taxonomy` — the *same taxonomy
  `leg_acc`/`seg_acc` score against*, not a new one. Segments with < 30 template vertices are
  **excluded and named**; `pt` (7–8 verts) and `ta` (38–39) are expected to fall out, consistent
  with the CSE-feasibility finding that they are unsupported.
- **Oracle part label on both sides.** Retrieval for a query point of class *s* on A is restricted
  to points of class *s* on B. This *is* the premise under test ("already-correct anatomical part"),
  not a leak: the experiment asks what remains hard once part identity is granted.

## 3. Metric

For query vertex *i* of class *s* on A, an arm returns a retrieved vertex *j* on B. The true
partner of *i* is *i*.

> **error(i) = ‖ B_verts[j] − B_verts[i] ‖ / seglen_B(s)**

3D error on the target mesh, normalised by that segment's own PCA extent, so short `tr` and long
`fe` are comparable. **3D error, not top-1 exact-vertex accuracy** — F7 established that top-1 is
the wrong yardstick on a dense mesh (19% of vertices have a >0.99-similar neighbour, and the
crowding is entirely local). Scoring on top-1 here would repeat F6's mistake of fixing in the
method something that only needed fixing in the framing.

The ceiling is exactly **0** (the true partner is always in the candidate set, since we retrieve
over full vertex sets rather than a sample), so there is no finite-sampling floor to subtract.
`gap_closed = 1 − err_arm / err_random`.

## 4. Arms

| arm | what it encodes | why it is here |
|---|---|---|
| `RANDOM` | uniform among same-part points on B | chance floor. Must be worst. |
| `XYZ_RIGID` | nearest neighbour in raw 3D after a part-local **Kabsch** rigid alignment of *s* from A to B | **the baseline that matters** — "nearest-surface geometry", i.e. what the optimizer effectively has once the part is roughly placed |
| `LOCAL_GEO` | Level 1: handcrafted local-geometry only — multi-scale local-PCA eigenvalue features (linearity / planarity / scattering) + normal orientation. No anatomy. | tests whether *generic* geometric descriptors suffice |
| `PART_FRAME` | Level 2, **the anatomy-conditioned arm**: coordinates in the segment's own frame — axial *t*∈[0,1] along the part axis, radial *r*/*R*, circumferential angle *θ* | encodes "distal portion of this segment", the semantic content the proposal argues is missing |
| `PART_FRAME_AXIAL` | *t* alone | isolates whether the axial coordinate carries all of it |

## 5. Pre-registered bar

Primary comparison is **`PART_FRAME` vs `XYZ_RIGID`**, per segment, **individually** — no pooled
averages (F2's lesson: pooled means hid that fixed and broken cases cancelled). Bar is set on the
four segments with adequate support and standing interest: **`co`, `tr`, `fe`, `ti`**, aggregated
across legs and sides within each.

- **PASS** — median normalised 3D error is lower for `PART_FRAME` than `XYZ_RIGID` on **all four**
  segments individually, each with paired sign test **p < 0.05** *and* **≥ 15% relative** reduction.
- **PARTIAL** — the direction holds on some but not all four, or holds on all four but under 15%.
- **FAIL** — no segment improves. Anatomy-conditioning is not the missing information at
  within-part resolution, and this line closes as cheaply as it opened.

Reported alongside, gating nothing: `LOCAL_GEO` (does generic geometry alone do it?) and
`PART_FRAME_AXIAL` (is the axial coordinate the whole story?).

## 6. Mechanism checks — mandatory, reported with the endpoint

Standing rule from the six decoupling instances: **an endpoint alone is not evidence.** Four checks
are fixed here; the first two can void the run.

1. **Self-retrieval no-op (voids the run if it fails).** With A = B, every geometric arm must
   return exactly 0.000 error for every point. Anything else means the retrieval harness or the
   index bookkeeping is wrong, and no number in the run may be read.
2. **Frame-consistency control (voids `PART_FRAME`/`PART_FRAME_AXIAL` if it fails).** The part
   frame is built by PCA on the part's own vertices, whose axis **sign and circumferential
   reference are ambiguous**. Sign is resolved to point away from the proximal joint and θ is
   referenced to a fixed anatomical direction. The **axis-flip rate against a reference specimen
   must be ≈ 0**. If frames flip between specimens the descriptor is meaningless and its numbers
   are not a result — this is the arm's most likely silent failure, so it is checked explicitly
   rather than assumed.
3. **Axial / circumferential decomposition of the `PART_FRAME` residual.** Prediction, from F1
   (circumferential angle linearly decodable only at 52.9° vs 90° chance) and SurfEmb: the residual
   is **circumferential-dominated**, axial ≪ circumferential. A gain arriving with axial ≈
   circumferential would mean the frame is not doing what the arm claims and is reported as such.
4. **Chance control.** `RANDOM` must be clearly worst. Any arm landing at `RANDOM` indicates a
   broken descriptor rather than a finding.

## 7. What each outcome licenses — fixed now, so the reading is not chosen after the fact

- **PASS** → anatomy-conditioned descriptors carry within-part information that raw geometry does
  not. This licenses **exactly one** next step: feeding this correspondence into the fitter and
  scoring on the *fitter's* metric. It does **not** license training a descriptor network — note
  that retrieval gains have failed to reach the fitter three times (C11, C12, F2), so a retrieval
  win here is a hypothesis about the fitter, not a result about it.
- **PARTIAL** → report which segments and stop. No network.
- **FAIL** → the within-part gap is not addressable by anatomy-conditioning, the last cheap
  hypothesis is closed, and the remaining bottleneck is the one the record already names: absence of
  ground truth on real specimens.

**Not claimed under any outcome:** anything about real scans. This is synthetic, with exact
correspondence by construction and no scan noise, partiality, or topology defects. Transfer to real
ants is an assumption, and the bench_10 annotation benchmark is what would test it.

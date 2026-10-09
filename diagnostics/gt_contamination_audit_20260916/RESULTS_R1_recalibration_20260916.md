# R1 — Landmark recalibration in the corrected frame

**2026-09-16.** `recalibrate_landmarks.py`, output `landmark_indices_recalibrated_v2.json`.
Rebuild of `annotation/landmark_indices_recalibrated.json`, which was fitted in the Blender Z-up
frame. n = 11 expert specimens (Dolichoderus excluded — annotated on the Discothyrea mesh).
Fits: JAB production `A_prod_s0`, cached `verts`, no re-optimisation.

---

## 1. The frame correction is validated, decisively

Distance from each annotated landmark to the nearest point of its own scan surface, as % of the
mesh diagonal — a discriminating test, because a landmark in the wrong frame cannot sit on the scan:

| | median over 11 specimens |
|---|---:|
| corrected `obj = (x, z, −y)` | **0.122%** |
| raw `original`, as R3/R9/A1/slide8 read it | **2.272%** |

A **19× improvement**, and the assert `med_corrected < 0.5%` passed on all 11 specimens
individually. This independently confirms the 2026-09-14 frame audit by a route it did not use.

## 2. The old recalibrated file is confirmed badly wrong

Displacement between the old vertex and the new consensus vertex, median over specimens, in % of
Weber's length:

| landmark | old v | new v | displacement % WL |
|---|---:|---:|---:|
| mandibular_apex_l | 6067 | 6855 | **79.1** |
| scape_apex_r | 1498 | 1510 | **77.7** |
| mandibular_apex_r | 6125 | 6855 | **74.1** |
| clypeal_ant_mid | 5676 | 1603 | 62.2 |
| head_width_l | 250 | 6305 | 50.5 |
| cephalic_post_mid | 917 | 640 | 47.2 |
| scape_apex_l | 170 | 5423 | 47.1 |
| head_width_r | 6263 | 934 | 40.6 |
| antennal_insertion_r | 5685 | 1243 | 40.4 |
| wl_anterior_r | 1368 | 782 | 33.8 |
| wl_posterior_r | 3194 | 2497 | 29.2 |
| antennal_insertion_l | 5502 | 5847 | 22.0 |

Every trait computed with the old file used vertices 22–79% of a body length away from where the
corrected frame puts them.

## 3. **The rebuild does not produce a usable instrument — and that is the finding**

The consensus vertex is chosen to minimise mean distance over all 10,235 template vertices. Even at
that optimum, agreement is poor and leave-one-specimen-out stability is worse:

| landmark | n | vertex | mean % WL | max % WL | LOO stable |
|---|---:|---:|---:|---:|---:|
| wl_posterior_r | 11 | 2497 | 16.6 | 64.1 | 3/11 |
| mandibular_apex_r | 11 | 6855 | 23.8 | 57.5 | 10/11 |
| wl_anterior_r | 11 | 782 | 23.9 | 48.7 | 10/11 |
| mandibular_apex_l | 11 | **6855** | 24.1 | 68.9 | 9/11 |
| clypeal_ant_mid | 11 | 1603 | 24.2 | 43.0 | 2/11 |
| cephalic_post_mid | 11 | 640 | 28.0 | 64.1 | 9/11 |
| head_width_l | 11 | 6305 | 31.6 | 67.5 | 2/11 |
| head_width_r | 11 | 934 | 32.0 | 51.4 | **1/11** |
| antennal_insertion_r | 10 | 1243 | 33.5 | 67.9 | **0/10** |
| antennal_insertion_l | 10 | 5847 | 37.5 | 92.1 | 4/10 |
| scape_apex_r | 10 | 1510 | 50.5 | **115.1** | 7/10 |
| scape_apex_l | 10 | 5423 | 57.5 | **116.6** | 4/10 |

Three things stand out:

1. **No landmark achieves good agreement.** The best is 16.6% WL; most sit at 24–38%; the scape
   apices exceed 50% with maxima above a full Weber's length.
2. **Leave-one-out instability.** For 5 of 12 landmarks the chosen vertex changes when a single
   specimen is removed (`antennal_insertion_r` 0/10, `head_width_r` 1/11). A consensus that depends
   on which 11 animals you happened to annotate is not a definition.
3. **`mandibular_apex_l` and `mandibular_apex_r` collapse to the SAME vertex, 6855.** The
   optimisation could not distinguish the sides.

   **CORRECTION, added 2026-09-16 after checking the targets themselves.** The original wording
   ("the left and right mandible tips cannot be one point") overstated this. Measured separation
   between the two annotated apices, as % of Weber's length:

   | specimen | sep % WL | | specimen | sep % WL |
   |---|---:|---|---|---:|
   | Eciton burchellii | **32.6** | | Formica cf. fusca | 3.2 |
   | Aenictus ceylonicus | **13.4** | | Aphaenogaster gracillima | 2.9 |
   | Cephalotes atratus | 10.2 | | Discothyrea patrizii | 2.2 |
   | Cataglyphis iberica | 8.1 | | Cyphomyrmex major | 1.9 |
   | Gigantiops destructor | 6.7 | | Odontomachus bauri | **1.4** |
   | Leptogenys peuqueti | 3.3 | | | |

   For most specimens the two apices are only 2–8% WL apart, so one vertex serving both is not by
   itself surprising. The defensible claim is narrower: **a single consensus vertex minimised mean
   distance to both sides even though some specimens separate them by 13–33% WL** — the search had
   no purchase on laterality. Do not present this as "left and right are the same point".

### The confound, stated explicitly

Agreement here is measured on **fitted** meshes, and the production fit's own joint error is
25.1% WL (JAB). So the 24–38% WL figures **conflate** (a) registration error with (b) genuine
absence of a stable vertex-level correspondence. This design cannot separate them, and no claim
should be made that does.

But two observations are **not** explained by isotropic fit error:
- the **L/R collapse** at the mandible — an isotropic error would not merge two laterally separated
  points onto one vertex;
- **scape apex maxima above 100% WL** — larger than the fit error by a factor of four.

## 4. Consequences

- **`HL` and `WL` are not rescuable by recalibration.** Both are landmark-pair traits, and no
  stable pair exists. So the M4 scale-free traits cannot be fixed by swapping in v2 — the
  denominator problem is not a bad-file problem, it is that a fixed vertex pair does not track this
  anatomy across specimens.
- **`HW` remains the survivor**, and for a structural reason worth stating: `trait_extract.py:67`
  defines it as an *extent of a body part*, not a pair of named vertices. Extent-based definitions
  do not require vertex-level correspondence; landmark-pair definitions do.
- **This is the clean evidence for the talk's slide 21** ("a correct label is not yet a correct
  location"), replacing the void V6 numbers — and it is a stronger statement, because it holds
  *after* the frame was fixed.
- **v2 must not be quietly substituted into `mv_framework.py`.** It is published as a diagnostic,
  not as a replacement instrument. Doing so would repeat the original error: shipping a consensus
  whose dispersion was never reported.

## 5. What would actually settle it

The confound is separable with a design this one cannot reach:
- repeat the consensus on the **oracle-supervised** JAB fits (O1, 17.4% WL) — if agreement improves
  proportionally, the residual is fit error; if it does not, it is genuine correspondence absence;
- or measure landmark agreement **specimen-to-specimen on the raw scans**, with no model in the
  loop, which removes fit error entirely.

The second is the cleaner experiment and is cheap. Recommended before any further trait work.

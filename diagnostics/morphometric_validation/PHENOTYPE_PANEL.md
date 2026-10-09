# SMILify Validated Phenotype Panel — frozen definitions and evidence-based tiers

**Purpose**: fix which measurements may be used for biology, at what granularity of claim, and on
what evidence — *before* any large-corpus analysis. Written 2026-09-09.

**The principle this enforces** (V6): *a template vertex is not automatically a biological
landmark.* V6 found the named template vertex sat ~48.8% of Weber's length from the human landmark
while the nearest fitted surface point was 7.7% away — a ~6.4× discrepancy. Definitions are frozen
here so the fitter is never tuned until old landmark definitions happen to work.

---

## The selection rule — a CONJUNCTION, not a tally

A phenotype is admissible only if it clears **all three** independent filters. "Passed enough axes"
is not the rule, because the filters answer different questions and a failure in any one of them
invalidates a different part of the claim:

```
        measurement accuracy   x   reproducibility   x   biological validity
        (M2, vs exact GT)          (M1 + M3)             (agreement + allometry
                                                          + landmark adjudication)
```

| filter | question | source | status |
|---|---|---|---|
| **accuracy** | how close to truth when truth is known? | M2, 48 synthetic, exact GT | done |
| **reproducibility** | same specimen, same answer? | M1 pose invariance + M3 seeds (3 seeds, 48 specimens) | **done 2026-09-09** |
| **biological validity** | right structure, real scaling law? | B1 adjudication + B3 allometry + Bland-Altman vs physical reference | done |

**This panel is frozen BEFORE the large-corpus analysis (M4).** Traits are selected on
independently established measurement validity, never because they produce an interesting
biological pattern. That ordering is the difference between a validated phenotyping pipeline and
post-hoc trait shopping.

## The four evidence axes, and why they must disagree

| axis | question | source |
|---|---|---|
| **B1 anatomical** | do the endpoints sit on the named structure? | expert landmarks, n=12 |
| **B2 pose invariance** | constant when only articulation changes? | 20 Atta, observed vs canonical |
| **B3 biological** | reproduces a known scaling law? | 20 Atta vs physical reference |
| **M2 recovery** | how close to truth when truth is known? | 48 synthetic, exact GT |

**These axes disagree, and that is the point.** A trait can be pose-stable and poorly recovered
(ML), pose-fragile and well recovered (SL), or well recovered and biologically uninformative (WL).
No single number substitutes for the panel.

## Evidence table

| trait | B2 pose Δ | B3 exponent | allometric? | **M2 recovery** | B1 status |
|---|--:|--:|:--:|--:|---|
| **HW** | **0.115%** | 1.205 [1.154,1.257] | yes | **4.79%** | Type III extent (not landmark) |
| **HL** | 0.321% | 1.122 | yes | 6.02% | adjudicated (moved 9.3% diag) |
| FL | 0.528% | 1.190 | yes | 11.73% | **rig joint, not surface** |
| TBL | 0.678% | 1.014 | no | 7.17% | derived |
| ML | 1.653% | 1.130 | yes | **20.74%** | adjudicated (moved 12.6%) |
| WL | 1.926% | 1.010 | no | **3.84%** | adjudicated (moved 7.8%) |
| PetL | 2.266% | 1.040 | no | **25.47%** | **unadjudicated** |
| GL | 2.325% | 0.887 | yes | 12.31% | **unadjudicated** |
| SL | **6.081%** | 1.225 | yes | 5.76% | adjudicated (moved 14.7%) |

B3 null is **1.0, not 0** — per-specimen normalisation means a trait with no independent size
information lands at ~1.0. Deviation from 1.0 is the signal; R² is inflated by the shared scale
factor and is not evidence.

---

## ROLE ASSIGNMENT (M4 uses this, and only this)

**FINAL — conjunction complete 2026-09-09 (M2 accuracy x M3 reproducibility x biological validity).**

| role | traits | accuracy (M2) | reproducibility (M3 seed CV mean/max) | validity |
|---|---|---|---|---|
| **PRIMARY phenotype** | `HW` | 4.31% ✓ | 0.88% / 6.6% ✓ | ✓ bias −0.13%, exp 1.205 |
| **PRIMARY phenotype** | `HL` | 5.51% ✓ | 1.08% / 8.5% ✓ | ✓ exp 1.122 (landmark moved 9.3%) |
| **SECONDARY** | **none — slot eliminated** | — | `FL` was the only candidate and **failed** at 40.9% max CV | — |
| **COVARIATE, not phenotype** | `WL`, `TBL` | ✓ | ✓ (0.59% / 0.94%) | isometric — no independent signal |
| **CONDITIONAL** | `SL` | 5.33% ✓ | 2.42% / 19.5% ~ | pose-fragile 6.08% ✗ — matched pose only |
| **EXCLUDED** | `FL` | 9.93% ~ | **40.9% max ✗** | rig joint, not surface ✗ |
| **EXCLUDED** | `ML` | **21.73% ✗** | **44.3% max ✗** | ✓ |
| **EXCLUDED** | `GL` | 14.81% ✗ | 29.2% max ✗ | **unadjudicated ✗** |
| **EXCLUDED** | `PetL` | **34.41% ✗** | **93.1% max ✗** | **unadjudicated ✗** |

**M4 uses `HW` and `HL` only, with `WL`/`TBL` as size covariates.**

No trait may be promoted into PRIMARY after M4's results are seen.

---

## PRIMARY — validated, may headline biological claims

### `HW` — head width
- **Definition (frozen)**: Type III maximum mediolateral extent of the `b_h` part, after rigid
  alignment to the template part frame (`trait_extract.py::_ext(b_h, axis 0)`).
- **Alternative definition, also validated**: `b_h_l`–`b_h_r` reporter-bone distance. **These are
  not interchangeable** — against the physical reference the reporter version is essentially
  unbiased (−0.13%) while the Type III extent carries a systematic **+2.22%** bias. State which one
  is used; do not mix them within an analysis.
- Evidence: best on every axis. Pose 0.115%; exponent 1.205 overlapping the independently
  reproduced published 1.2352; agreement bias −0.13% (reporter); synthetic recovery 4.79%.

### `HL` — head length
- **Definition (frozen)**: `clypeal_margin_ant_mid` → `cephalic_margin_post_mid`, **using the
  expert-recalibrated indices** (`annotation/landmark_indices_recalibrated.json`), NOT
  `landmark_template_indices.json`.
- Evidence: pose 0.321%, exponent 1.122 (allometric), recovery 6.02%.
- **Caveat**: recalibration moved its endpoints 9.3% of the body diagonal. Any value computed with
  the un-adjudicated template indices carries that offset.

## SECONDARY / QUALIFIED

### `FL` — hind femur  **[DEMOTED TO EXCLUDED by M3, 2026-09-09]**
- **Definition (frozen)**: `l_3_fe_r` → `l_3_ti_r`, **joint-to-joint**.
- **M3 verdict**: 4.14% mean / **40.9% max** seed CV — fails reproducibility outright, independent of
  its definitional problem. Nearly half its total error (CV/err 0.42) is optimizer noise alone.
- **The qualification is definitional, not statistical.** FL is pose-stable (0.528%) and allometric
  (1.190), but it is measured between **rotation pivots, not surface anatomy** — the class G2 found
  unreliable. Pose-stability here does not imply biological validity.
- Recovery 11.73%, i.e. mid-pack. Usable for relative comparison; not a substitute for a
  surface-defined femur length.

## CONDITIONAL, COVARIATE, and EXCLUDED

### `SL` — scape length
- Definition: `antennal_insertion_r` → `scape_apex_r` (recalibrated indices).
- **Fails on ONE axis, not all**: worst pose sensitivity in the panel (6.081%) and largest landmark
  displacement (14.7%), but recovers *well* (5.76%). Therefore: **admissible only where pose is
  controlled or matched**; inadmissible for comparisons across specimens in differing postures.

### `ML` — mandible length
- Definition: `mandibular_apex_r` → `clypeal_margin_ant_mid` (recalibrated). Convention O2 (apex to
  clypeus) remains unresolved upstream.
- **Demoted from any primary role by M2**: recovery error **20.74%**, second worst of nine, despite
  good pose stability (1.653%) and allometry (1.130). Looking good on the Atta axes did not survive
  the synthetic ground-truth test. This is the clearest single case for why the panel needs M2.

### `WL`, `TBL` — Weber's length, total body length
- WL is the **normaliser** for everything else; TBL is a derived sum.
- Both land at exponent ~1.0 → **no independent size signal beyond body length**. WL is the
  best-recovered trait in the panel (3.84%) and simultaneously the least informative as a
  phenotype. Use as size covariates, not as phenotypes.

## EXCLUDED — not usable for biology at present

### `PetL`, `GL` — petiole length, gaster length
- **No expert annotation exists** for `petiole_ant_mid`, `petiole_post_mid`, `gaster_apex_mid`, so
  B1 cannot be evaluated *even in principle*. PetL additionally has the worst recovery in the panel
  (25.47%).
- Unblocking these requires annotation work, not fitter work.

---

## Rules of use

1. **Cite the landmark file.** Any trait value must record whether it used the recalibrated or the
   template indices. `trait_extract.py` currently defaults to the **un-adjudicated** template file.
2. **Do not average across roles.** A composite index mixing HW with PetL inherits PetL's validity.
2b. **The conjunction is not negotiable.** A trait strong on two filters and weak on the third is
   excluded, not averaged in — ML is the worked example (pose ✓, allometry ✓, recovery ✗).
3. **Atta validation does not transfer automatically.** The 20-Atta agreement establishes validity
   *for Atta*, on those traits. For the broader corpus, separate: (a) directly validated,
   (b) mechanistically supported by pose/recovery evidence, (c) exploratory.
4. **Never gate on fit quality.** M2: chamfer's correlation with morphometric error is |r| ≤ 0.34,
   and even oracle diagnostics reach only ~0.41. Selecting "good" specimens by chamfer is
   indefensible.
5. **Report the failure mode, not a score.** "HW is reliable to ~0.1% under pose change and ~4.8%
   against known truth" is a usable statement; "quality = 0.82" is not.

## Provenance

Axes computed by `mv_framework.py` (B1/B2/B3/B4), `mv_agreement.py` (Bland–Altman vs physical
reference), `m2_synthetic_recovery.py` (recovery vs exact GT). Underlying evidence:
`RESULTS_M2_synthetic_recovery.md`, `RESULTS_R3/R4_20260908.md` (why B1 exists), V6 (landmark
recalibration).

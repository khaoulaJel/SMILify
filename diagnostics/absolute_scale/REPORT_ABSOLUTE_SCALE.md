> **Update**: `REPORT_VALIDATION_PASS.md` (same directory) runs four follow-up checks on this
> report's results, including a leak-free lot-residualisation test. Its main correction: §4/§6's
> characterisation of the combined PCA's PC1 as "not simply size... more complex" is too cautious —
> direct measurement shows it is, quantitatively, almost exactly a size axis (r=-1.00 vs. log-size).
> The genus-signal and guild-lift findings below are largely confirmed, with the guild lift shown
> to be more confound-sensitive than the genus result. Read that report alongside this one.

# Task 1 — Recover absolute physical scale: findings

Two phases, kept explicitly separate as instructed: **calibration** (is mm recovery technically
possible, and for whom) and **evaluation** (does it add information). The calibration phase is
unchanged from the prior checkpoint — this report documents the evaluation phase only, run on the
276-specimen calibrated subset, never on the full 757/616-specimen baseline.

## 0. What corpus this runs on, and why it's smaller than the full baseline

| Corpus | n | Role |
|---|---|---|
| Full worker corpus (existing baseline, `morphometrics.csv` on `feature/registration_moonshot`) | 757 | NOT used here — different genus/lot composition, not comparable arm-to-arm |
| `ALL_ANTS_CLEAN` | 81 | NOT used here — no calibration path exists at all (no JSON, no specimen identity, pre-normalised with unrecorded original units) |
| **Calibrated subset (`calibration_status == "ok"`)** | **276** | **This evaluation, all three arms** |
| ...after the existing protocol's `min_n >= 3` genus filter (genera with too few members can't support a leave-one-out test) | **149** | The number actually going into the classification tests below |

The `min_n>=3` filter is not new — it's `analyse.py`'s own default (`--min_n 3`), applied here
unchanged. 149 specimens, 29 genera, 27 accession lots.

**Coverage accounting, no silent removals:**
- 757 worker specimens total → 279 have a raw AntScan source scan with `voxel_size` in the mirror
  used → 276 pass the calibration plausibility gate (3 excluded: `Leptogenys_cf.binghamii_CASENT0877639`,
  `Polyrhachis_wolfi_CASENT0878096`, `Pseudoneoponera_rufipes_CASENT0877628` — confirmed mesh
  artifacts, raw-voxel bbox implies 50–93mm "ants," not real biology; see prior checkpoint).
- 478 worker specimens have no raw scan in this mirror at all (confirmed by direct accession-code
  search, not a naming mismatch) → permanently excluded from any absolute-scale analysis until more
  source data is located.
- 81 ALL_ANTS_CLEAN specimens → permanently proportions-only, no calibration path exists.
- Of the 276, 127 further genera fail `min_n>=3` (singletons/pairs) → excluded from the
  classification tests only (kept in the distribution/plausibility checks in §3).

## 1. Matched three-arm evaluation

Lot-blind leave-one-lot-out 1-NN, exactly `analyse.py`'s protocol (`pcs`, `loo_1nn`, `perm_test`
reused verbatim, not reimplemented). 95% CI is new — a lot-level cluster bootstrap (1000 resamples,
percentile interval), since the existing framework didn't have one; not present in the original
code. n=149, 29 genera, 27 lots, identical across all three arms by construction.

| Arm | Accuracy | Null | p | Lift | 95% CI |
|---|---|---|---|---|---|
| **A. Proportions only** (existing Mosimann log-shape-ratio + shape-dev) | 14.1% | 4.5±2.0% | <0.0001 | **3.14×** | 4.4–19.7% |
| **B. Absolute size only** (log-mm measurements, not size-normalised) | 10.7% | 4.6±2.0% | 0.0075 | **2.33×** | 3.5–18.9% |
| **C. Proportions + absolute** (block-balanced combination) | 14.8% | 4.4±2.0% | <0.0001 | **3.35×** | 5.8–21.3% |

**Combined (14.8%) > proportions-only (14.1%)** — true, but the gap is small (0.7 points) and the
two arms' 95% CIs overlap almost completely (4.4–19.7% vs 5.8–21.3%). This is a real but modest
lift, not a large or clearly decisive one at this sample size.

**Size alone carries real genus signal**: 2.33× lift, p=0.0075. Weaker than shape alone, but not
noise — this confirms body size is not just a scale-free-proportions phenomenon; the phylogenetic
size differences among ant genera are large enough to be genus-informative on their own, even
without shape.

## 2. Lot confound check for absolute size

This is the guard the task asked for explicitly. **Absolute size does show a lot-associated
signal**: predicting collection lot from size-only features gives 8.1% accuracy vs. a 4.2±1.9% null
(p=0.0375, lift 1.93×). This is weaker than the genus signal above (which is the real question), but
it's real and above the 0.05 threshold — **flagged, not swept aside**.

Interpretation: this is plausible without invalidating the genus result, because genus and lot are
correlated by construction — a lot mostly holds one species (95.7% of prior analysis's species-level
signal came from accession number alone, per `analyse.py`'s docstring), so ANY genus-tracking signal
will show *some* lot association just from genus-lot correlation, independent of any pipeline
artifact. The lot-blind protocol in §1 already controls for this at the genus level (that's the whole
point of leave-one-LOT-out rather than leave-one-specimen-out). But this result means the absolute-size
arm's genus signal should be read as "real but partially entangled with batch/fixation/scan-session
effects that happen to track lot," not as unambiguously pure morphology. A stronger test (residualise
size on lot, then re-test genus signal) is the natural follow-up if this pipeline moves toward
production use — not done here, flagged as a limitation instead of quietly fixed.

## 3. Distribution and plausibility of the recovered absolute measurements

The size proxy used in §§3–5 is `exp(mean(log(all 34 mm measurements)))` — a composite geometric
mean across every measured structure (head, mesosoma, gaster, mandible, antenna, leg segments).
**This deliberately understates true body length** — it mixes large structures (mesosoma length)
with tiny ones (individual antennal segments, mandible width), so its absolute value should be read
as an isometric-size index, not literally "body length in mm."

- Range: 0.13–5.01mm, median 0.45mm, across the 276 calibrated specimens.
- **Smallest 5**: *Mayriella* (0.13mm), *Probolomyrmex* (0.17mm), *Discothyrea* ×2 (0.18mm), *Oxyepoecus*
  (0.19mm). Checked against literature, not just internal consistency: Discothyrea (minute
  egg-predators, some species <1.5mm total length) and Probolomyrmex (slender subterranean ants) are
  both genuinely among the smallest ant genera — the ranking is biologically sound even though the raw
  number understates true body length.
- **Largest 5**: *Acromyrmex balzani* (5.01mm), *Formica sanguinea* ×2 (3.85–4.05mm), *Bothroponera*
  ×2 (3.64–3.88mm) — consistent in relative ordering with these genera's known larger body sizes.
- No remaining implausible taxon-size combinations found in this spot check beyond the 3 already
  excluded at the calibration stage.
- **No genus or lot dominates**: most-represented genus is *Strumigenys* at 5.4% of the subset (15/276),
  most-represented lot 8.3% (23/276) — neither is close to a majority-driving concentration.
- 127 distinct genera and 32 distinct lots represented in the full 276 (before the min_n filter).

See `out_evaluation/fig_size_distribution.png`.

## 4. Does absolute size change the shape space?

- **PC1 vs log(size)**: r=-0.062, p=0.45 — no significant linear relationship.
- **PC2 vs log(size)**: r=0.129, p=0.12 — not significant either, though the weakest of the three.
- **PC3 vs log(size)**: r=-0.008, p=0.92 — no relationship.
- None of the leading three scale-free shape axes individually track this size index. The current
  Mosimann-ratio PCA is, by this test, doing what it was designed to do: removing isometric size.

- **But the combined PCA's PC1 is a genuinely different axis**: correlation between shape-only PC1
  and combined-PCA PC1 is only r=0.081 — almost orthogonal, not just a rescaled/rotated version of
  the same direction. Given absolute-length measurements are mutually strongly correlated with each
  other (nearly tautologically, as different rulers on the same growing body), the block-balanced
  combined PCA's dominant axis is pulled toward that shared covariance structure, producing a leading
  axis that isn't well described as either "shape PC1" or simply "log(size)" alone — it's a genuinely
  new direction the current representation doesn't capture. **No causal claim is made here** — this is
  a description of what varies, not why.

See `out_evaluation/fig_pc_vs_size.png`.

## 5. Ecological / functional guild signal

Only 53/149 min_n-filtered specimens carry a guild label (arboreal 9, trap-jaw 20, army ant 20,
leafcutter/fungus-grower 4) — above the 15-specimen/3-guild threshold set in advance for a formal
test, so this is reported as a real (not merely exploratory) result, though n remains small and
results should not be over-generalised.

| Arm | Accuracy | Null | p | Lift |
|---|---|---|---|---|
| A. Proportions only | 54.7% | 31.2% | <0.0001 | 1.75× |
| B. Absolute size only | 54.7% | 32.0% | 0.0025 | 1.71× |
| C. Proportions + absolute | **60.4%** | 31.8% | <0.0001 | **1.90×** |

- **Combining does help here**, by more than in the genus test: +5.7 points over proportions alone.
- **Size alone matches proportions alone** (54.7% both) for guild — despite carrying no significant
  individual correlation with PC1/PC2/PC3 in §4, absolute size alone is just as informative for
  ecological guild as the full shape representation is, which is a genuinely interesting result: guild
  membership (trap-jaw/army-ant/arboreal/leafcutter) tracks body size about as strongly as it tracks
  shape.
- **PC1 guild separation strengthens with size added**: eta² 0.217 (shape-only) → 0.258 (combined) —
  guilds are somewhat better separated on the combined PC1 than the shape-only one.

See `out_evaluation/fig_guild_pca.png`.

## 6. Final interpretation

**1. Does absolute physical size improve genus discrimination over proportions alone?**
Marginally yes (14.8% vs. 14.1%, lift 3.35× vs. 3.14×), but the effect is small and the confidence
intervals overlap substantially. At this sample size (149 specimens after the genus filter), this is
a real but not decisive improvement.

**2. Does size alone contain meaningful genus signal?**
Yes — 2.33× lift, p=0.0075. Weaker than shape, but statistically real, not noise.

**3. Does combining size + proportions outperform proportions alone?**
Yes for genus (small margin, overlapping CIs) and yes more clearly for ecological guild (+5.7 points,
60.4% vs 54.7%). The combined representation is never worse than proportions-only in this evaluation.

**4. Is any size signal vulnerable to collection-lot confounding?**
Yes, and this is flagged rather than hidden: absolute size predicts lot at 1.93× lift (p=0.0375).
This doesn't invalidate the lot-blind genus result (which already controls for lot at the group
level), but it means the size signal is not cleanly separable from batch/fixation/scan-session
effects, and that entanglement should be disclosed wherever this result is used.

**5. Does adding size reveal a biologically meaningful axis absent from the current scale-free
representation?**
Yes, structurally: the combined PCA's PC1 is nearly orthogonal to the current shape-only PC1
(r=0.081), meaning it captures variation the current Mosimann-ratio representation removes by
construction. But no single leading shape PC (1, 2, or 3) individually correlates with this size
index, so this new axis is not simply "add a size dimension" — it reflects the block's covariance
structure more than a single scalar. Correlational only; no causal claim.

**6. Does it change the ecological/functional interpretation of the shape space?**
Somewhat. Guild separation on PC1 strengthens with size added (eta² 0.217 → 0.258), and absolute size
alone matches full shape's guild-classification accuracy. This is consistent with body size being a
real, independent part of what distinguishes ecological guilds (a trap-jaw ant is not just
"differently shaped" but reliably differently sized), not an artifact of the shape representation.

### Bottom line for SMILify

Absolute scale is technically recoverable (prior checkpoint) and, on the data currently available,
**adds real but modest information beyond the existing scale-free proportions** — clearest for
ecological/functional guild classification, present but small for genus discrimination, and
genuinely revealing a shape-space axis the current pipeline removes by design. It comes with two
honest caveats that any production use must carry forward: (a) only 276/757 (36%) of the existing
worker corpus can be calibrated with currently available source data, and (b) the size signal is not
fully separable from a lot/batch confound. Worth incorporating as an additional, clearly-labelled
representation alongside (not instead of) the existing proportions — not worth replacing proportions
with, and not worth overselling as a large effect at the current sample size.

## Artifacts

- `custom_processing/calibrate_scale.py` — calibration utility (unmodified this phase)
- `diagnostics/absolute_scale/recompute_absolute_measurements.py` — mm back-transform (unmodified this phase)
- `diagnostics/absolute_scale/morphometrics_absolute_mm.csv` — calibrated measurement table (unmodified this phase)
- `diagnostics/absolute_scale/evaluate_absolute_scale.py` — this evaluation (new)
- `diagnostics/absolute_scale/out_evaluation/results.json` — full numeric results
- `diagnostics/absolute_scale/out_evaluation/fig_size_distribution.png`
- `diagnostics/absolute_scale/out_evaluation/fig_pc_vs_size.png`
- `diagnostics/absolute_scale/out_evaluation/fig_guild_pca.png`

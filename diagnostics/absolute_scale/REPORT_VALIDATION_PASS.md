# Task 1 — validation pass addendum

Follow-up to `REPORT_ABSOLUTE_SCALE.md`, same 276-specimen calibrated subset (149 after
`min_n>=3`, 29 genera, 27 lots — unchanged). Does not revisit or modify the calibration.
Full numeric output: `out_validation/results.json`, figure: `out_validation/fig_retained_vs_excluded.png`.

## A. Does the size→genus signal survive lot residualisation? (the main question)

**Method**: leave-one-out within-lot mean subtraction — each specimen's absolute-size features
have its own lot's *other* members' mean removed (self excluded). This uses only the specimen's
own already-known lot label and its lot-mates' measurements; it never touches a genus label, and
it's computed once, globally, before any classification — safe under the existing leave-one-LOT-out
protocol because that protocol already structurally forbids a specimen's own lot from being used as
its nearest-neighbour candidate, regardless of how the feature was preprocessed. 3 of 27 lots are
singletons and fall back to grand-mean centring (documented, not silently handled).

| | Accuracy | Null | p | Lift | 95% CI |
|---|---|---|---|---|---|
| Before residualisation | 10.7% | 4.6±2.0% | 0.0075 | 2.33× | 3.5–18.9% |
| **After residualisation** | **10.1%** | 4.5±2.0% | **0.0100** | **2.24×** | 3.5–17.8% |

**96.1% of the original lift survives.** The size→genus signal is **not** mostly collection
structure — it holds up almost unchanged once the lot-level mean shift is explicitly removed. This
directly answers the concern raised by the lot-confound flag in the prior report: that flag was real
(size does predict lot, 1.93× lift) but it does not explain away the size→genus result.

## B. Do the guild results survive the same correction?

Same five representations tested lot-blind on the 53-specimen, 4-guild subset:

| Representation | Accuracy | Null | p | Lift |
|---|---|---|---|---|
| Shape only | 54.7% | 31.2% | <0.0001 | 1.75× |
| Size only, before residual. | 54.7% | 32.0% | 0.0025 | 1.71× |
| Size only, after residual. | 60.4% | 30.7% | <0.0001 | 1.96× |
| Combined, before residual. | 60.4% | 31.8% | <0.0001 | 1.90× |
| Combined, after residual. | 56.6% | 30.2% | <0.0001 | 1.88× |

The **+5.7pp** combined-over-shape gap reported previously **shrinks to +1.9pp** after
residualisation — about a third of its original size survives. By the letter it "survives" (stays
positive, both still highly significant vs. their own nulls), but the honest reading is that **this
result is noticeably more sensitive to the lot confound than the genus result in §A was** (96% of
lift retained there vs. ~33% of the gap retained here). At n=53 across 4 guilds this is also a small
sample for a two-decimal-place comparison — take the direction (weak positive) as more reliable than
the exact magnitude. Interestingly, size-alone accuracy *increased* after residualisation (54.7% →
60.4%), which cuts against a simple "the confound was propping up size's contribution" story — most
plausibly sampling noise at this n, not a phenomenon to build further claims on without more data.

## C. Is the 276-specimen calibratable subset biased?

Compared against the 481 excluded worker specimens (757 total workers = 276 + 481; both figures
computable for every specimen since they don't require calibration — `log_size` is a model-unit
isometric-size proxy that exists for all 757 rows in the pre-calibration measurement set, and
shape PC1/PC2 come from a single PCA fit over all 757 workers together).

- **Isometric size** (log_size, model units): retained mean -1.259 vs. excluded -1.268 — Welch t
  p=0.43, KS p=0.56. No detectable difference.
- **Shape PC1**: p=0.56 (t), p=0.56 (KS). No detectable difference.
- **Shape PC2**: p=0.057 (t), p=0.25 (KS) — borderline on the t-test only, not on the
  distribution-shape-sensitive KS test; not treated as a real effect on one borderline test alone.
- **Fit asymmetry** (asym_median, a mesh-quality proxy): p=0.30 (t), p=0.28 (KS). No difference.
- **Genus coverage**: 127 genera in the retained set, 45 genera exist *only* in the excluded 481
  (never calibratable with current source data) — e.g. `daceton`, `cardiocondyla`, `brachyponera`,
  `dorymyrmex`, `anoplolepis` among others (full list in `results.json`). These are a real gap, not
  a random sample loss — some ecologically distinct genera (e.g. `daceton`, a trap-jaw genus) are
  entirely absent from anything Task 1 can currently say about absolute size.
- **Subfamily composition**: proportionally similar between retained/excluded (Myrmicinae dominates
  both at roughly the same ~34–37% share; no subfamily is conspicuously over- or under-represented
  in the retained set — see `results.json` for the full table).

**Verdict**: on every distributional test performed, the calibratable subset is not statistically
distinguishable from the excluded specimens in size, shape, or fit quality. The real limitation is
narrower and more specific: **complete absence of 45 genera**, not a systematic quality or size bias
in the specimens that did make it in. Conclusions from §§1–5 of the main report generalise
reasonably to "ant genera with source-scan coverage," not obviously to the missing 45.

## D. Inspecting the "size axis" — checked twice, and revised again

**Correction to this same section, one round earlier.** The r=-1.000 below was first reported as
an empirical discovery. That needed a second check: absolute measurements were built as
`exp(Z + log_size) * mm_per_unit`, and `log_size`/`log(mm_per_unit)` are added identically across
all 34 columns of a given specimen — so a PCA finding a component that IS that shared additive term
is close to mathematically guaranteed, not 34 independently-agreeing measurements. Two checks
(`check_pc1_tautology.py`, full log in `out_validation/pc1_tautology_check.log`):

- **PCA on only the 3 already-scale-free shape indices** (cephalic/mandible/scape ratios, which by
  construction carry no size term at all): PC1 vs. log(size) → **r=-0.018, p=0.83.** No leak
  elsewhere in the pipeline — reassuring, but expected.
- **Row-demean the absolute block first** (subtract each specimen's own mean across its 34 mm
  measurements, which should algebraically cancel the shared size term) **then** recompute PC1:
  correlation with log(size) **collapses from r=-1.000 to r=0.111 (p=0.18, not significant)**, and
  the resulting axis correlates **r=-0.841** with the pre-existing proportions-only PC1 — i.e. once
  the shared size term is removed, what's left is essentially the same shape information the
  proportions block already had, not something new.

One genuine complication surfaced doing this carefully: the analytical prediction was that
row-demeaning should cancel the size term *exactly* (max discrepancy ~1e-6). It didn't — the actual
discrepancy was 0.21–0.45 log-units. Checked directly: these 34 columns exactly match the source
pipeline's `CORE_BLOCKS` (12+8+5+3+3+3=34 via `block_of()`), so this isn't a column-selection bug in
this validation. The likely explanation is that `log_size` in the upstream `morphometrics.csv`
(`feature/registration_moonshot`) was computed from a *larger* column set (e.g. all ~40 measured
features) than the 34 "core" columns actually written to that CSV — a genuine provenance
inconsistency in the source data, flagged here rather than silently reconciled. It does **not**
affect the mm back-transform in `recompute_absolute_measurements.py` (`exp(Z + log_size)` recovers
the original raw value exactly regardless of what log_size was averaged over, since Z was defined
relative to that same log_size) — it only means Check 2's demeaning is an approximation, not an
exact cancellation, so its result is genuinely empirical rather than analytically forced to land at
zero. That it still collapsed the correlation this dramatically is the stronger evidence, not
weaker.

**Verdict: PC1 restates size by construction of the feature set — this is not an empirical
discovery about ant morphology, and should not be reported as one.** The classification results in
§§A and B above, and in the main report, are unaffected: they use the raw log(mm) features directly
for 1-NN distance, and never depended on PC1 being a "discovery" one way or the other. Only this
specific descriptive framing needed correcting.

The following numbers (loadings, guild means, genus extremes) are kept for the record, since they
remain factually accurate descriptions of what the *raw absolute features* look like — they should
just not be read as "PCA discovered that these correlate with size," given the above.

Original (now-corrected) description of what those numbers show:

- **98.5%** of PC1's squared loading mass sits in the absolute-measurement block (vs. 1.5% in the
  proportions block) — every one of the top-15 loadings by magnitude is an absolute-mm feature.
- **PC1 vs. log(geometric-mean absolute size): r = -1.000, p<0.0001.** Effectively a restatement of
  the size index, not merely correlated with it.
- **PC1 vs. log(mesosoma_len_mm)** — used here as the nearest available **proxy** for Weber's length;
  this pipeline does not compute the standardised AntWeb landmark measurement (anterior pronotal
  margin to posterior propodeal lobe), only a generic mesosoma-extent bounding dimension, so this is
  explicitly a proxy, not the real thing: **r = -0.999, p<0.0001.**
- **PC1 vs. classic scale-free shape indices** (cephalic index head_wid/head_len, mandible index
  mandible_len/head_len, scape index seg_an_1/head_len — all computed as differences of log-shape-
  ratios, so genuinely scale-free by construction): **all non-significant, |r|<0.09, p>0.3.** PC1
  carries no detectable shape-index information, consistent with it being a size axis and not a
  shape axis in disguise.
- **Guild means** (sign convention: more negative PC1 = larger absolute size): leafcutter/fungus-
  grower -1.70 (by far the largest-bodied guild in this subset), army ant -0.10, arboreal +0.26,
  trap-jaw +0.34. This matches established myrmecology — leafcutters (*Atta*/*Acromyrmex*) are
  notably large-bodied — without being tuned to produce that match.
- **Genus extremes**: largest end includes *Camponotus*, *Formica*, *Odontomachus*, *Eciton*,
  *Acromyrmex* — genuinely large-bodied genera; smallest end includes *Strumigenys*, *Carebara*,
  *Temnothorax*, *Solenopsis* — genuinely small-bodied genera. Face-valid.

**These numbers are real, but describe a near-tautology, not a discovery** — see the correction at
the top of this section. The loading mass and correlation are large because absolute-mm features
share an additive size term by construction, not because 34 independently-informative measurements
happened to agree. The guild-mean and genus-extreme face validity is still worth noting (it shows
the recovered *raw* absolute measurements behave sensibly, i.e. the calibration itself is sound),
but it is evidence about the calibration, not evidence that PCA discovered a "size axis."

## Summary of what changed from the main report

| Claim in `REPORT_ABSOLUTE_SCALE.md` | Status after validation |
|---|---|
| Size alone carries genus signal (2.33× lift) | **Confirmed robust** — 96% of lift survives lot residualisation |
| Combined beats proportions for guild (+5.7pp) | **Partially confirmed** — direction holds, magnitude shrinks to ~+1.9pp under residualisation; more confound-sensitive than the genus result |
| Combined PC1 is "not simply size... more complex" | **Corrected, then corrected again**: it does restate size (r=-1.00, 98.5% loading mass), but this is a near-tautology of how absolute features were constructed (shared additive size term), not an empirical discovery — confirmed by two follow-up checks (pure shape-index PCA: no size correlation; demeaned absolute block: correlation collapses to r=0.11 n.s.) |
| Calibrated 276 might be a biased subsample | **Not supported** — no detectable size/shape/quality bias; the real gap is 45 entirely-uncalibratable genera, not a skew in the ones that are |

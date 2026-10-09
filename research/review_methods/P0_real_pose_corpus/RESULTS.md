# P0 results: real-scan fitted-pose library (657 worker scans, 161 genera, no JAB genus)

Rules in PREREGISTRATION.md; wording per DEVIATIONS D7. Output `out/P0_articulation.json`,
`out/analyse.log`. All 657 fits reproduced (JAB loader guard).

| measurement | result | pre-registered reading |
|---|---|---|
| validity gate: fitted vs expert articulation, JAB (n = 11) | Spearman **−0.10** (p 0.77); median fitted − expert +3.6 deg | **FAIL**: fitted articulation is not a per-specimen articulation measure |
| stability, 48 specimens x 3 seeds | within SD 3.96 deg vs between 8.54 deg, ratio 0.46 | usable (<= 0.5) |
| fitted articulation distribution | median 34.8 deg (p5 20.6, p25 28.0, p75 41.6, p95 51.6) | reported as *fitted* articulation |
| pose_train vs pose_eval | 35.1 vs 34.3 deg, KS p 0.81 | split balanced |
| JAB within P0 | fitted-articulation percentiles 16-73 | JAB is typical of the corpus |

## What this shows

1. **Population level, at n = 657, independent of JAB:** fitted real articulation sits far beyond the
   synthetic training range (training per-specimen maximum 19 deg; P0 5th percentile 20.6 deg), at the
   same level as the expert skeletons (+3.6 deg). The distribution-level articulation shift found on
   11 specimens (Q3a A7) holds on 657 specimens from 161 genera, with the caveat that these are fitted
   values.
2. **Per specimen, fitted poses are not trustworthy** (no rank agreement with experts). The fitter
   reproduces the *amount* of articulation in the population, not which specimen is how bent. This
   is consistent with JAB's 25% WL joint error and is itself a result about the fitter.
3. Consequences, fixed before using P0: S2-confirm tests *realistic fitted articulation as a
   population*; any training on P0 poses uses them as a distribution, and the broadened-synthetic arm
   (C), which does not depend on the fitter, is mandatory alongside it.

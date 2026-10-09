# M4 — scientific results, FROZEN 2026-09-10

**Experimental phase closed.** M4-A, M4-B, M4-C and M4-E are complete on all 757 specimens. No
further experiments will be run on this question. M4-D is **optional**, not unfinished (§4).

Hard stop, recorded so it is not relitigated: **no M4-F, no new correspondence method, no new loss,
no new QC metric, no trait shopping, no post-hoc promotion of HL, no invented size proxy to rescue
M4-D.** The next phase is analysis consolidation and writing.

---

## The chain

Every link was pre-registered before its result was seen, and each defines the admissible sample of
the next.

| | question | result |
|---|---|---|
| **M4-A** | Can we trust the measurements? | HW **99.5%** admissible (753/757). HL 93.9%, failing through a specific, **genus-structured** AP-inversion (perm. p=0.025) |
| **M4-B** | Is there biological structure? | HW/WL differs substantially between genera, rank η² = **0.331**, perm. p < 5×10⁻⁵ — as *shape*, not claimed size-independent |
| **M4-C** | Where is that structure? | HW/WL's genus component **survives** removal of singleton species (22.5% vs 19.8% species) — not a species-composition artefact. HL's **reverses** |
| **M4-E** | Is the scalar part of the 3D phenotype? | Yes. Cross-validated held-out **η² = 0.409 ± 0.044** vs **0.236 ± 0.049** for random dense directions, p = 1.8×10⁻¹⁵ |

Complete: **3D reconstruction → measurement validity → biological structure → connection back to
the 3D representation.**

## The headline numbers, as they should be reported

- M4-E: **held-out η² = 0.409 ± 0.044 vs 0.236 ± 0.049 random, p = 1.8×10⁻¹⁵.** The in-sample 0.350
  is **not** the headline — it is partly circular, since the direction is fitted to predict a
  scalar that is itself genus-structured.
- M4-A: permutation p = 0.025 for genus-structured HL failure. The asymptotic χ² (0.015) is invalid
  at these counts and is not reported.
- M4-C: the decisive figure is the **singleton robustness check**, not the raw variance components.

## Why HW carries the study

HW and HL separate on **three independent axes** and agree on one:

| axis | HW | HL |
|---|---|---|
| M4-A admissibility | 99.5% | 93.9%, genus-structured exclusion |
| M4-C singleton robustness | genus > species holds | ordering **reverses** |
| M4-B / M4-C role | primary | secondary |
| M4-E bridge | 0.409 | 0.412 (agrees) |

One axis of agreement does not overturn three of disagreement. **HL is not promoted.**

## 4. M4-D — optional, not unfinished

The core biological endpoint is established **without** absolute scale. M4-D was to be the
absolute-scale analysis; absolute scale is unidentifiable on `worker_ALT` (7× implied spread,
`voxel_size` refuted as the cause, no rescaling in preprocessing, no external reference).

- **If the scale question is resolved**: run the predefined analysis. It adds a result.
- **If it is not**: the study stands as complete, with the limitation stated plainly —

> *Absolute-scale allometry could not be identified from the available scan metadata, so the
> biological analysis was restricted to scale-free morphology and within-corpus comparisons for
> which the estimand was identifiable.*

That is a limitation, not a failure of the study. **No alternative size proxy will be invented to
rescue it.**

## 5. Statistical discipline — the pattern to preserve in writing

Three defects were caught **before** becoming claims, and each is documented in its results file:

1. **Asymptotic χ² rejected** for permutation tests wherever many small groups made expected counts
   ≪ 5 (M4-A, M4-B). The genus p moved 0.015 → 0.025; the larger, honest number is reported.
2. **`GENUS_TO_SUBFAMILY` keys are lowercase** — the unfixed mismatch silently mapped all 757
   specimens to UNKNOWN, which would have produced a spurious "no subfamily structure" result.
3. **M4-E's in-sample η² replaced by a cross-validated one**, in response to a circularity concern
   in its own design.

## 6. The conclusion

> SMILify can recover a small set of anatomically validated morphometrics from imperfect
> articulated insect scans, but **geometric fit quality alone does not establish morphometric
> validity**. Reliability is trait-specific, real-corpus failure modes can be anatomically
> structured, and validated morphometrics can recover biologically structured components of the
> dense 3D phenotype.

## 7. Remaining action

One, and it is not an experiment: **send Fabian the scale question**
(`diagnostics/atta_reference/MESSAGE_TO_FABIAN_20260908.md`, drafted, unsent). A defensible scale
source unlocks the predefined M4-D. Its absence does not block the study.

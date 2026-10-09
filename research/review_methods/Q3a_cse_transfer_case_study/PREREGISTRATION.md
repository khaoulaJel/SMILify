# Q3a: why CSE helps some real specimens and harms others (mechanism case study)

Written 2026-10-06. **Exploratory case study, not a confirmatory test.** With n = 11 and two or three
failures, no "property X predicts failure" claim will be made. The output is a set of named
mechanisms, each of which Q3b then reproduces (or fails to reproduce) as a controlled shift on
synthetic data.

## Disclosure: what had been seen before this was written

- JAB per-specimen final-stage `med_fk` for `A_prod` and `I_cse` (seen 2026-09-14 and again today).
- Today, before writing this: the per-stage seed-mean `med_fk` trajectory of both arms
  (H0, H1, H2, Stage_2, Stage_3) and the final `side_confusion`. Copied into `out/A1_stage_table.txt`
  by the analysis script so it is on disk rather than in a chat log.

What that showed, stated now so it cannot be rediscovered later as a "finding":
1. Cephalotes is worse under CSE from H0 onward (65.5 vs 57.5% WL), i.e. before any fitting stage
   could compensate.
2. On several specimens CSE is better at H2 and the advantage disappears or reverses in the
   moonshot surface stages (Discothyrea 30.9 vs 47.1 at H2, 49.8 vs 45.2 final; Aphaenogaster,
   Gigantiops, Leptogenys similar), while A_prod improves from H2 to Stage_3.

Pattern 2 is a hypothesis generated on the test set. It is labelled **H-stage** below and is
tested only on synthetic data (Q3b).

## Analyses (all CPU, no new fitting, all on existing JAB artefacts)

| id | question | measurement |
|---|---|---|
| A1 | Where in the pipeline does CSE help or hurt? | per-specimen, per-stage paired difference I_cse − A_prod in `med_fk`, by JAB region |
| A2 | Is the pre-fit correspondence itself anatomically wrong, and how (wrong leg, wrong side, wrong segment, wrong region)? | **expert-skeleton proxy labels** (below) vs the part of each CSE-retrieved template vertex |
| A3 | Which template regions receive too many or too few correspondences? | per-part template coverage of `cse_all.npz`'s mask; concentration (points per retrieved vertex) |
| A4 | Does the GT-free cycle-consistency signal see the failure? | per-specimen and per-region cycle distance with the C3 checkpoint, loaded strictly |
| A5 | How does the fitter respond: does it absorb wrong correspondence through pose, deformation or offsets? | H2 -> Stage_3 change in `med_fk`, `deform_rms`, `chamfer`, per region, both arms |
| A6 | What is unusual about each specimen as an input? (descriptive, for hypothesis generation only) | scan point count, connected components, bbox aspect, A_prod fitted betas Mahalanobis norm (morphology distance from the shape-space mean), fitted pose magnitude, log_beta_scales spread |

### A2 proxy, and its validation gate

Real scans have no dense correspondence ground truth. The proxy assigns each scan point to the
nearest **expert skeleton bone** (segment between an expert joint and its parent; laterality from
the joint name) and maps the bone to a part label at the resolution {body, head, mandible_l/r,
antenna_l/r, leg i side s segment}.

**Validation gate (run first, on synthetic P48, where true labels exist):** build the same proxy
from the ground-truth FK joints and compare it with the true per-point part labels.
- Usable at a resolution only if proxy accuracy there is >= 0.85 (leg identity: which leg and
  side) and >= 0.75 (leg segment). Below that, A2 is reported at the coarser resolution only, or
  not at all.
- The proxy's own confusion pattern is reported, so that CSE errors which coincide with proxy
  errors (e.g. at segment boundaries) are not counted as CSE failures.

## Output

`RESULTS.md`: per-specimen mechanism cards (Cephalotes, Aphaenogaster, Formica, Eciton,
Discothyrea, plus one neutral specimen as a reference), then a short list of candidate mechanisms,
each with the Q3b synthetic experiment that would reproduce it. Probes and figures stay in `out/`.

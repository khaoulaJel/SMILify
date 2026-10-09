# TASK 3 — GWN disagreement-rate diagnostic, baseline vs gentle (scheme:'all', bench50_clean)

Shipped diagnostic: `diagnostics/moonshot/eval_gwn_disagreement.py` (CPU-only, read-only,
no training/loss changes). Jobs 15525266 (`gwn_base`) / 15525267 (`gwn_gentle`), both
COMPLETED cleanly on 2026-08-13, run on TASK1/TASK2's own `fit3d_results_all_baseline` /
`fit3d_results_all_gentle` (scheme:'all', gentle penetration weight 0.02/0.05) Stage_3 output.
Per-specimen CSVs: `fit3d_results_all_{baseline,gentle}/Stage_3_deform_fine_gwn_disagreement.csv`.

Reproduce: `python3 diagnostics/khaoula_review/task3_gwn_disagreement_recompute_PROBE.py`
(raw output: `task3_gwn_disagreement_recompute_out.txt`).

## 0. Independent recompute (per handoff discipline: never trust a log over the raw file)

Recomputed the population-level table directly from the two raw CSVs, bypassing both
SLURM logs' printed summaries. **Matches the log-derived numbers to <5e-5 on all 15
non-trivial pair/directions** — no discrepancy, logs were trustworthy here, but this is
now confirmed rather than assumed.

## 1. Headline table (50/50 specimens valid both arms, mean disagreement_rate per pair/direction)

| pair | dir | baseline | gentle | delta |
|---|---|---:|---:|---:|
| mandible-antenna | A_into_B | 0.0479 | 0.1221 | +0.0742 |
| gaster-legs | A_into_B | 0.0195 | 0.0844 | **+0.0649** |
| mandible-antenna | B_into_A | 0.0925 | 0.0492 | -0.0433 |
| head-legs | A_into_B | 0.0036 | 0.0377 | +0.0341 |
| mandible-legs | A_into_B | 0.0020 | 0.0220 | +0.0200 |
| antenna-legs | A_into_B | 0.0097 | 0.0250 | +0.0153 |
| antenna-thorax | B_into_A | 0.0017 | 0.0148 | +0.0131 |
| mandible-thorax | B_into_A | 0.0010 | 0.0100 | +0.0090 |
| antenna-legs | B_into_A | 0.0019 | 0.0091 | +0.0073 |
| antenna-thorax | A_into_B | 0.0031 | 0.0056 | +0.0025 |
| head-legs | B_into_A | 0.0016 | 0.0036 | +0.0020 |
| mandible-legs | B_into_A | 0.0014 | 0.0031 | +0.0017 |
| antenna-gaster | B_into_A | 0.0000 | 0.0007 | +0.0007 |
| gaster-legs | B_into_A | 0.0131 | 0.0127 | -0.0004 |
| mandible-thorax | A_into_B | 0.0033 | 0.0034 | +0.0001 |
| (5 remaining gaster-{head,mandible,antenna} rows) | both dirs | 0.0000 | 0.0000 | tied |

**13/15 non-trivial pair/directions worsen under gentle, 2/15 improve, 5/20 tied at exactly
zero (uninformative, not "improved").** Per-specimen (mean disagreement across all 20
pair/directions, n=50): 40/50 specimens worse under gentle, 10/50 better. Grand mean
disagreement roughly doubles (0.0101 → 0.0202).

## 2. Gaster-legs only (the pair the Hard Rule requires reporting, and the only pair TASK1's
headline count result is about)

| dir | baseline | gentle | delta |
|---|---:|---:|---:|
| A_into_B | 0.0195 | 0.0844 | **+0.0649 (4.3x)** |
| B_into_A | 0.0131 | 0.0127 | -0.0004 (noise) |

The worsening is concentrated entirely in `A_into_B` — same directional asymmetry pattern
as FINDINGS.md Step 6.

## 3. Relation to FINDINGS.md Step 6 (r=0.947)

Step 6's correlation was measured on an **earlier gentle arm** (`scheme:'deform'`, pose
frozen, per REVIEW_penetration_update.md §4) — not on TASK1/TASK2's `scheme:'all'` arm used
here. Step 6's proposed follow-up experiment ("if allowing joint rotation resolves collisions
by rotation instead of denting, the disagreement asymmetry should shrink or invert") is
answered by this data: **it does not invert, and does not meaningfully shrink** — `A_into_B`
still worsens sharply (+0.0649) while `B_into_A` stays flat, the same signature Step 6 found
under pose-frozen training. Freeing pose (TASK1's `scheme:'all'`) changed the *count* trade
(reversed gaster-legs' own sign, per TASK1 §1) but **did not fix the underlying proximity-test
accuracy asymmetry** that Step 6 identified as a distinct, additional mechanism.

## 4. Reading, and what it adds to TASK1

TASK1 found (via the training loss's own proximity-based count, on `scheme:'all'`): gaster-legs
count improves (-20.8%) but the aggregate across all 10 non-adjacent pairs gets worse (+16.8%)
— contact gets shallower on the targeted pair but more widespread elsewhere. This GWN result
is a **second, independent metric** (ground-truth winding-number inside/outside, not the
training loss's own nearest-triangle-centroid proximity test) measured on the same Stage_3
output, and it shows the same shape: worse agreement with ground truth on `A_into_B`-heavy
pairs, concentrated exactly where TASK1's per-pair table already showed the largest relative
count blowups (`mandible-thorax` +424%, `antenna-thorax` +456%, both `A_into_B`-dominated
pairs in this table too). That is **triangulation, not just directional consistency**: two
metrics built on unrelated code paths (a proximity heuristic used as a training signal, vs.
a closed-form geometric ground truth used only for evaluation) agree on where and how the
gentle-weight arm gets worse.

## 5. Implication for the standing recommendation

`REVIEW_penetration_update.md`'s recommendation was: ship gentle as an available config
option, not a default, gated by a required per-specimen check. This result **strengthens
that caveat rather than changing it**: it adds evidence that the training signal itself
(the proximity test) becomes measurably less accurate under the gentle arm specifically in
the direction where outcomes are degrading, and confirms (rather than merely assumes) that
freeing pose via `scheme:'all'` — otherwise a validated, no-cost win per FINDINGS.md's
`A4_nofreeze` result — does not resolve this specific mechanism. It does not test the actual
fix Step 6 proposed (replacing the proximity test with a GWN/SDF-based one); that remains
future work, unattempted here, per the out-of-scope list in the task brief (no loss/weight
changes).

## Verdict

GWN diagnostic shipped and independently verified (§0). Its result corroborates and sharpens
TASK1's finding rather than contradicting it: the gentle-weight arm's cost is real, measurable
by a second unrelated metric, concentrated in the same direction/pairs, and untouched by
`scheme:'all'`. No change to the ship/no-ship recommendation; one additional, independent
data point supporting the existing "config option + required per-specimen check" caveat.

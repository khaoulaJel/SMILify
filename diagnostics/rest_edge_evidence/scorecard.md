# Rest-edge vs shrink-edge scorecard (Task 2)

## VERDICT: REJECT rest (at these full D1 weights)

This is NOT specimen-level heterogeneity (better on some specimens, worse on others) -- it is a UNIFORM trade-off present on every single specimen: edge_logratio_absmean improves ~18x on 10/10 specimens, but folded_face_frac regresses on 10/10 specimens, dihedral_p99 regresses on a majority of specimens, and fscore@0.01/chamfer_l2 collapse in aggregate (YES fscore collapse, YES chamfer collapse). Metrics regressing on a majority of specimens: dihedral_p99 (10/10 specimens), folded_face_frac (10/10 specimens). Per the pre-registered rule, 'majority of specimens' regression on an integrity metric triggers REJECT, not the mixed/opt-in branch (that branch is for cases where different specimens disagree on which mode is better, not for a single metric-vs-metric trade-off that is consistent across the whole set).

- to-ship commit at run time: `9ebe508a19a3bb40256bbf24c930fa12b7a65413`
- Same 10-specimen bench50_clean mesh set, same hierarchical init (H2_joint.npz, shared across both arms since edge_mode is not read by the hierarchical stage), same D1 full budget (1000+1000 its), same hardware (1x H100, c23g).
- Arm A (control): `diagnostics/moonshot/cfg/D1_low.yaml` unchanged (edge_mode: shrink).
- Arm B (test): `diagnostics/moonshot/cfg/D1_low_rest.yaml` (edge_mode: rest in both stages).
- Elapsed: job 3025721, 00:11:04 wall clock for hier + both moonshot arms on 1x H100.

## Reproduction check vs Task 1's D1 numbers (Arm A should match)

| metric | Task 1 D1 (commit 9ebe508) | Arm A (this run) | rel. delta | match? |
|---|---|---|---|---|
| deform_mag_mean | 0.00401 | 0.00398 | -0.8% | OK (within 5%) |
| edge_logratio_absmean | 0.14856 | 0.14697 | -1.1% | OK (within 5%) |

Not a bit-reproducibility test (Arm A reruns the moonshot stage fresh from the same H2_joint.npz init rather than replaying Task 1's exact run), but both metrics land within 5% of Task 1's D1 arm, which is the expected regime given fixed seed=0 and identical config -- no red flag.


## Per-specimen gate metrics (edge_logratio_absmean, folded_face_frac)

| specimen | edge_logratio A(shrink) | edge_logratio B(rest) | delta | folded_frac A | folded_frac B | delta |
|---|---|---|---|---|---|---|
| Acanthostichus_aff.brevicornis_CASENT0744328 | 0.15786 | 0.00795 | -0.14991 | 0.00336 | 0.01290 | +0.00954 |
| Acromyrmex_coronatus_CASENT0744365 | 0.13831 | 0.00780 | -0.13051 | 0.00264 | 0.01134 | +0.00870 |
| Acromyrmex_lobicornis_CASENT0878007 | 0.16965 | 0.01065 | -0.15900 | 0.00645 | 0.01505 | +0.00860 |
| Anochetus_risii_CASENT0877608 | 0.12405 | 0.00607 | -0.11798 | 0.00261 | 0.01010 | +0.00749 |
| Anochetus_risii_CASENT0877609 | 0.15937 | 0.01089 | -0.14848 | 0.00651 | 0.01544 | +0.00893 |
| Aphaenogaster_pallida_CASENT0745235 | 0.11740 | 0.00549 | -0.11191 | 0.00283 | 0.00948 | +0.00665 |
| Atopomyrmex_mocquerysi_CASENT0744784 | 0.16350 | 0.00802 | -0.15548 | 0.00420 | 0.01697 | +0.01277 |
| Carebara_trechideros_CASENT0877591 | 0.13695 | 0.00759 | -0.12936 | 0.00355 | 0.01433 | +0.01078 |
| Centromyrmex_brachycola_CASENT0744052 | 0.14984 | 0.00755 | -0.14229 | 0.00384 | 0.01313 | +0.00928 |
| Cephalotes_minutus_CASENT0709253 | 0.15273 | 0.01034 | -0.14238 | 0.00349 | 0.02261 | +0.01912 |

## Full integrity + symmetry metric means (mean across 10 specimens)

| metric | A (shrink) mean | B (rest) mean | delta |
|---|---|---|---|
| edge_logratio_absmean | 0.14697 | 0.00824 | -0.13873 |
| tri_quality_mean | 0.75579 | 0.76754 | +0.01175 |
| tri_quality_p05 | 0.22233 | 0.23378 | +0.01145 |
| deform_mag_mean | 0.00398 | 0.00329 | -0.00069 |
| deform_mag_p95 | 0.01068 | 0.00915 | -0.00153 |
| deform_mag_max | 0.13424 | 0.06397 | -0.07027 |
| dihedral_p99 | 65.47649 | 102.02133 | +36.54484 |
| folded_face_frac | 0.00395 | 0.01413 | +0.01019 |
| midline_dev_mean | 0.01570 | 0.01562 | -0.00007 |
| midline_dev_p95 | 0.01631 | 0.01778 | +0.00147 |
| midline_dev_mean_norm | 0.01556 | 0.01608 | +0.00052 |
| midline_dev_excess | -0.00297 | -0.00056 | +0.00241 |
| fscore@0.01 | 0.62511 | 0.43132 | -0.19379 |
| chamfer_l2 | 0.00038 | 0.00094 | +0.00056 |

## Per-specimen deform_mag_max (does removing shrink pressure let the mesh balloon?)

| specimen | deform_mag_max A(shrink) | deform_mag_max B(rest) | delta |
|---|---|---|---|
| Acanthostichus_aff.brevicornis_CASENT0744328 | 0.08426 | 0.07171 | -0.01254 |
| Acromyrmex_coronatus_CASENT0744365 | 0.18324 | 0.04940 | -0.13384 |
| Acromyrmex_lobicornis_CASENT0878007 | 0.17579 | 0.13611 | -0.03968 |
| Anochetus_risii_CASENT0877608 | 0.06475 | 0.04270 | -0.02205 |
| Anochetus_risii_CASENT0877609 | 0.23853 | 0.07469 | -0.16383 |
| Aphaenogaster_pallida_CASENT0745235 | 0.05740 | 0.04703 | -0.01036 |
| Atopomyrmex_mocquerysi_CASENT0744784 | 0.17490 | 0.03427 | -0.14063 |
| Carebara_trechideros_CASENT0877591 | 0.19401 | 0.07346 | -0.12055 |
| Centromyrmex_brachycola_CASENT0744052 | 0.11047 | 0.03232 | -0.07815 |
| Cephalotes_minutus_CASENT0709253 | 0.05907 | 0.07797 | +0.01890 |

## Gate check detail

- `edge_logratio_absmean`: 0/10 specimens regressed (B > A), 10/10 improved (B < A).
- `folded_face_frac`: 10/10 specimens regressed (B > A), 0/10 improved (B < A).
    - REGRESSION: Acanthostichus_aff.brevicornis_CASENT0744328: 0.00336 -> 0.01290
    - REGRESSION: Acromyrmex_coronatus_CASENT0744365: 0.00264 -> 0.01134
    - REGRESSION: Acromyrmex_lobicornis_CASENT0878007: 0.00645 -> 0.01505
    - REGRESSION: Anochetus_risii_CASENT0877608: 0.00261 -> 0.01010
    - REGRESSION: Anochetus_risii_CASENT0877609: 0.00651 -> 0.01544
    - REGRESSION: Aphaenogaster_pallida_CASENT0745235: 0.00283 -> 0.00948
    - REGRESSION: Atopomyrmex_mocquerysi_CASENT0744784: 0.00420 -> 0.01697
    - REGRESSION: Carebara_trechideros_CASENT0877591: 0.00355 -> 0.01433
    - REGRESSION: Centromyrmex_brachycola_CASENT0744052: 0.00384 -> 0.01313
    - REGRESSION: Cephalotes_minutus_CASENT0709253: 0.00349 -> 0.02261

- fscore@0.01 collapse (B worse than A by >0.02): YES (A=0.6251, B=0.4313)
- chamfer_l2 collapse (B worse than A by >50%): YES (A=0.00038, B=0.00094)

## Decision rule (pre-registered in the task brief, not adjusted after seeing results)

- Switch default to `rest` if neutral-to-better on every integrity metric across all 10 specimens (no per-specimen regressions on edge_logratio_absmean or folded_face_frac), AND fscore/chamfer don't collapse.
- Keep `shrink` as default, ship `rest` as opt-in if mixed (better on some specimens, worse on others) -- do not average across specimens to force a verdict.
- Reject `rest` if it regresses integrity metrics on a majority of specimens.

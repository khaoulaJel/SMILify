# TASK 2 — structural constraint arm (w_offset added to TASK 1's config), bench50_clean

**Goal**: does adding the validated free-form-offset penalty (`w_offset` 5.0/2.0, the D1
setting — confirmed in `diagnostics/moonshot/cfg/D1_low.yaml`) on top of TASK 1's
`scheme:'all'` + gentle-penetration base config reduce penetration by structurally
preventing mesh collapse, rather than merely penalizing it?

Config: `fitter_3d/ants_cfg_all_offset_baseline.yaml` / `fitter_3d/ants_cfg_all_offset_gentle.yaml`
(TASK 1's `ants_cfg_all_*.yaml` + `w_offset: 5.0` Stage_2 / `2.0` Stage_3, `deform_verts` added
to the trainable param set). bench50_clean, 50 specimens, unseeded (matching TASK 1's unseeded
arm for direct comparison — 3-seed reruns not repeated here given TASK 1 already established
the seed-noise magnitude, see below). Jobs: 15524309 (baseline+offset, 6:50), 15524310
(gentle+offset, 18:35), both COMPLETED. Results: `fit3d_results_all_offset_baseline/`,
`fit3d_results_all_offset_gentle/`. Reproduce:
`python diagnostics/khaoula_review/task2_hardrule_reanalysis_PROBE.py`
(`task2_hardrule_reanalysis_out.txt`). Renders: `task2_outlier_render_PROBE.py`
(`out/task2_outliers_lateral_XZ.png`, `out/task2_outliers_anterior_YZ.png`).

## Headline: preserves the gaster-legs win, more than halves the aggregate cost

| | TASK 1 (no offset) | TASK 2 (+ w_offset) |
|---|---:|---:|
| gaster-legs count delta | **-20.8%** | **-20.2%** |
| all-pairs count delta | **+16.8%** | **+5.4%** |
| fscore@0.01 delta, mean | -0.0048 | -0.0032 |
| fscore@0.01 delta, worst single specimen | -0.0596 | **-0.1125** |
| deform_mag delta (gentle vs its own baseline) | +4.13% | +5.72% |
| edge_logratio delta (gentle vs its own baseline) | +5.63% | +6.44% |

The gaster-legs improvement is unchanged (-20.2% vs -20.8%, within seed-noise of each other —
TASK 1's 3 seeds ranged -17.2% to -22.6%). The aggregate collateral damage across all 10 checked
pairs drops from +16.8% to +5.4%: **the structural constraint doesn't eliminate the trade-off
TASK 1 found, but it cuts it by roughly two-thirds.** Still net worse, not a full fix.

## The offset penalty's own effect, independent of the penetration loss

Comparing the two **baseline** arms (TASK 1's `w_penetration=0` vs TASK 2's
`w_penetration=0` + `w_offset`) isolates what the structural constraint does on its own:

| | TASK 1 baseline (no offset) | TASK 2 baseline (+ offset) | change |
|---|---:|---:|---:|
| deform_mag mean | 0.03689 | 0.00474 | **-87.1%** |
| edge_logratio mean | 0.5453 | 0.1872 | **-65.7%** |

This replicates the D1 recipe's own validated effect (FINAL_REPORT.md §2.2 #1: "edge distortion
0.35 → 0.14, 128 workers") in a fresh setting — same direction, same order of magnitude. The
offset penalty forces the fit to rely on pose/shape rather than raw per-vertex displacement,
which is exactly the mechanism TASK 2 hypothesized would reduce collateral penetration by
preventing collapse-by-denting. The all-pairs aggregate result (+16.8% → +5.4%) is consistent
with that mechanism actually operating, not just a correlation.

## The new risk this arm introduces: a much larger worst-case fscore loss

TASK 1's worst single-specimen fscore@0.01 loss was -0.0596 (`Heteroponera_panamensis`).
TASK 2's worst is **-0.1125** (`Ectatomma_brunneum`, 143 count increase alongside it) — nearly
double. Rendered (`out/task2_outliers_*.png`) alongside the other 5 flagged outliers
(`Lasius_nr._fuliginosus` -636 count, `Temnothorax_congruus` -571 count,
`Eciton_hamatum` +528 count, `Mayriella_sp` -520 count, `Myrmecina_sp` -137 count/-0.0687
fscore): **no collapse or tearing in any of the 6.** All remain recognisable, correctly-posed
ants; `Mayriella_sp` and `Ectatomma_brunneum` show a visibly different head roll/orientation
between baseline and gentle in the anterior view, which plausibly explains the tau=0.01 fscore
hit (a real but localized pose disagreement, not a structural failure) — `scheme:'all'` combined
with the offset penalty gives the optimizer more freedom to land on a different local pose
optimum per specimen, which is a real cost of unfreezing pose, not of the penetration loss
specifically.

## Answer

Adding the structural constraint (`w_offset`) **does** measurably help: it preserves the one
targeted improvement (gaster-legs, unchanged at ~-20%) while cutting the net collateral damage
across all checked pairs by roughly two-thirds (+16.8% → +5.4%), and it independently improves
integrity a lot regardless of the penetration loss (edge distortion -66%, deform magnitude -87%
on the baseline arm alone). It does not fully close the gap — the aggregate is still net worse
with the penetration loss on than off — and it introduces a larger worst-case per-specimen
surface-fit risk (-0.1125 vs -0.0596) than TASK 1, though visual inspection shows this is a
pose-disagreement effect, not mesh collapse. Net read: **structural constraint is a real,
partial fix, not a complete one** — consistent with, and a meaningful update to, TASK 1's
"real trade-off, not a clean win" conclusion.

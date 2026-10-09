# Anatomical Initialization Ceiling Test — RESULTS (2026-08-20)

## Decision (falsification outcome, stated first per protocol §6)

**FAILURE — do not build a learned initializer yet.** The cheap anatomical init (Arm B)
*regresses* leg-level correspondence accuracy relative to zero-init (0.732 vs 0.857, a
12.5-point drop) on 11 of the 12 specimens (`failure_case_gallery.txt`), and *regresses* every
morphometric block except leg_distal (antenna R 0.598 vs 0.763, all-feature R 0.518 vs 0.745,
leg_prox R 0.261 vs 0.768). The protocol's success criterion requires recovering GT-init's gain
on **both** leg-level correspondence accuracy **and** distal-leg/antenna R, consistently across
specimens. Arm B clears neither: it moves leg_distal R substantially toward the GT-init ceiling
(0.144→0.538, vs. GT-init's 0.604) but at the cost of a broad, specimen-consistent regression
everywhere else, including leg-level correspondence accuracy itself — the opposite of the
"substantial fraction of the GT-init gain, consistent across most specimens" bar. Per protocol
§6, the correct read is not "coarse pose is solved and something else is now the bottleneck" —
it is that **this specific cheap initializer's own errors are worse than useful**, and a
different diagnosis (of the initializer, not of the fitting pipeline) is needed before this
question can be asked again.

## Key numbers (n=12, `synth_clean`, D1 recipe — `summary_table.csv`)

| arm | leg_rot_err_deg (pre-opt) | leg_acc | within_leg_seg_acc | antenna_side_acc | antenna_seg_acc | leg_distal_R | antenna_R | all_feature_R |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| zero_init | 23.1 | 0.857 | 0.828 | 0.985 | 0.770 | 0.144 | 0.763 | 0.745 |
| cheap_anatomical | 28.5 | **0.732** | **0.686** | 1.000 | 0.760 | **0.538** | **0.598** | **0.518** |
| gt_init | 0.0 | 0.938 | 0.866 | 1.000 | 0.828 | 0.604 | 0.677 | 0.779 |

**A→B (zero→cheap):** leg_acc −0.125, within_leg_seg_acc −0.143, leg_distal_R **+0.394**,
antenna_R −0.165, all_feature_R −0.227. **B→C (cheap→gt):** leg_acc +0.206,
within_leg_seg_acc +0.181, leg_distal_R +0.066, antenna_R +0.079, all_feature_R +0.261 — gt_init
recovers cleanly from the same corpus, so the regression above is specific to the cheap
initializer's own errors, not an artifact of the D1 pipeline or the corpus.

## Causal trajectory: why the divergence (leg_distal R up, everything else down)

Section A (pre-optimization, no fitting): the cheap initializer's raw leg joint-rotation error
(28.5° mean) is already *worse* than zero-init's (23.1°) — see
`section_A_init_quality.json`/`fig_joint_error_boxplot_preopt.png`. This was found and reported
honestly during implementation, not after the fact (see `cheap_anatomical_init.py`/
`simple_leg_heuristic.py` docstrings for the two prior heuristics — `fitter_3d/geom_leg_init.py`
per-segment chain-IK, and this module's own predecessor design — that were tried and also
failed this same pre-optimization check before the current single-rigid-rod heuristic was
settled on as the least-bad option). The D1 pipeline does not recover from the worse start: it
converges to a leg_distal-favoring but globally worse basin, most visibly on leg-level
correspondence (`fig_confusion_leg.png`) and on blocks that should be UNAFFECTED by a leg-only
init (head R 0.956→0.553, gaster R 0.589→0.243) — collateral damage that is broad, not
leg-specific, echoing the same pattern already seen in this project's GNC track (per
[[project-correspondence-accuracy-metric]]: "GNC's antenna-cost framing undersold the real
collateral damage, which is broad, not antenna-specific"). The mechanism appears to be that a
badly-placed leg pose destabilizes the WHOLE fit's convergence basin (shared symmetry loss,
partition churn during H1/H2, global chamfer competition), not just the legs it directly
perturbs — worth flagging as a candidate general finding, not yet independently confirmed here.

## What this does and doesn't say about initialization as a lever

GT-init's own numbers (leg_acc 0.938 vs 0.857 baseline, independently bootstrap-confirmed per
[[project-correspondence-accuracy-metric]]) still show initialization CAN be a powerful lever
when the init itself is accurate. This experiment does not contradict that — it shows that THIS
project's current best cheap, correspondence-free, no-fitting-required heuristic for producing
that accurate an init does not yet exist. Two heuristics were tried and both failed the
pre-optimization sanity check before any GPU time was spent on the full pipeline for the first
one; only the second (less bad) was carried through to a full D1 run, and it failed the
post-fit falsification criterion too. The honest conclusion is narrower than "abandon the
learned-initializer direction" — it is "coarse geometric heuristics for leg pose, as attempted
here (per-segment chain IK, single-rigid-rod alignment), are not yet good enough to be worth
learning FROM; a learned initializer would need a better training signal than either of these,
or a fundamentally different approach to the raw-geometry-to-pose problem (e.g. a learned
regressor trained directly against GT poses, rather than an analytic heuristic distilled into a
learned model)."

## Deviations from the pre-registered protocol (disclosed)

1. **`fitter_3d/geom_leg_init.py` reused, then abandoned** (two-step user decision,
   2026-08-20): the protocol said not to reuse prior geometry-only initializer code; the first
   round of user sign-off approved reusing it anyway (efficiency); validating it end-to-end for
   the first time (its own docstring implied prior validation that never actually happened —
   no output ever existed on disk) showed it makes every leg joint substantially WORSE than
   zero-init (leg_distal position error 0.436 vs 0.163, ~3x), so a second round of user sign-off
   approved writing a fresh heuristic instead (`simple_leg_heuristic.py`). Both findings are
   preserved (`probe_B0_geom_init_validation.py` outputs, `scratch_probes/`) rather than
   discarded.
2. **Arms A (zero_init) and C (gt_init) reuse pre-existing runs** (`SYN_clean_w5`,
   `SYN_clean_pose25_gtinit_w5`, both on `synth_clean`, same D1 recipe) instead of being
   re-fit, since they already exist on disk and are the basis of the already-validated
   correspondence-accuracy comparison in [[project-correspondence-accuracy-metric]]. Only Arm B
   was fit fresh.
3. **Global orientation (protocol step 1)** is computed via PCA and reported as a Section-A
   diagnostic only (mean estimation error 13.7°, `section_A_init_quality.json`), but is NOT fed
   into the leg-pose computation or the fitter: `optimise_hierarchical.py --init_joint_rot_from`
   has no global-orientation override (always zero-inits `global_rot`), and `synth_clean` is
   canonically aligned by construction (true global orientation is always identity), so identity
   is both the numerically correct choice and the one the fitter actually uses regardless.
4. **"Per-specimen R" plots**: R (Pearson correlation across all 12 specimens) is a
   corpus-level statistic with no single-specimen value. `fig_per_specimen_trajectory.png`'s
   leg_acc panel is genuinely per-specimen; its R panels show one bar per arm (corpus-level),
   not per specimen.
5. **Failure-case gallery**: `failure_case_gallery.txt` ranks specimens by leg_acc delta
   (worst/best 3), as specified. Overlaid template+scan mesh renders for visual inspection were
   NOT produced (would need a full per-specimen 3D render pass, out of scope for this pass) —
   the ranked list identifies which specimens to inspect manually if that is wanted next.
6. **Section D (surface/integrity metrics)**: not computed. `calib_features.py`/`run_audit.py`
   do not emit chamfer/f-score/edge-distortion/deformation-magnitude directly; those live in a
   separate morphometrics pipeline (`morphometrics.csv`/`genus_table.json` machinery) not wired
   into this experiment's driver. Flagged as not done rather than silently omitted.

## Deliverables

- `summary_table.csv`, this file
- `section_A_init_quality.json` (pre-optimization rotation/global-orientation error)
- `../../correspondence_accuracy/out/correspondence_confusion.json` (full confusion matrices,
  all arms including `ACI_cheap`)
- `../../morphometrics/out/feature_reliability_all_runs.json` (full per-feature R/snr/bias, all
  3 arms)
- `fig_per_specimen_trajectory.png`, `fig_confusion_leg.png`, `fig_confusion_within_leg_seg.png`,
  `fig_joint_error_boxplot_preopt.png`, `fig_R_paired_diff.png`, `failure_case_gallery.txt`
- Cheap-initializer source: `../cheap_anatomical_init.py`, `../simple_leg_heuristic.py`
  (committed; see also the abandoned-but-preserved `probe_B0_geom_init_validation.py` finding)
- Exact commands: `../submit_ACI_cheap.sbatch` (Arm B fit, SLURM job 3089878, seed 0 default);
  Arms A/C are `SYN_clean_w5` / `SYN_clean_pose25_gtinit_w5`, produced in an earlier session —
  their exact original command line was not preserved (see `cheap_anatomical_init.py`'s
  docstring and [[project-correspondence-accuracy-metric]]), only the recipe
  (`--midline 2.0 --beta_prior 0.0 --limit 0.273 --offset 30.0`, `D1_low.yaml`/`D1_SYN.yaml`,
  the latter byte-identical in substance) and their fitted output.

# Evidence for the registration recipe

Tables behind `../README.md`. One row per variant; all fits use the code of branch `feature/investigation`.

| File | Content |
|---|---|
| `resolved_stage_weights.csv` | Every stage of the default recipe as the two entry points resolve it: iterations, learning rates, data term and all loss weights. |
| `variants_inside_recipe.csv` | Variants of the default recipe: `edge_mode: rest`, `w_trans`, `w_beta_prior`, and `w_scale` in the skeleton stages. Joint error on 11 expert-annotated AntScan scans (seeds 0–2) against a default fitted in the same batch, and surface fit on 64 AntScan workers (seed 0). |
| `geodesic_coherence.csv` | `w_geocoh` 1.0 in the surface stages, same joint-error protocol; surface fit on the 64 AntScan workers (seed 0) against a default fitted in a separate batch. |
| `dense_correspondence.csv` | Correspondence targets supplied to the fitter: learned surface embedding, GBCPD and NICP; same joint-error protocol. |
| `penetration_loss.csv` | Proximity penetration loss in the earlier single-trainer pipeline (`fitter_3d.optimise`, model `SMIL_OmniAnt.pkl`), 50 AntScan scans × 3 seeds: three measures of gaster–leg self-intersection, surface fit and free-form offset, each variant against its reference. |
| `penetration_loss_preset_bars.csv` | The same variants against the bars fixed before the runs, per seed. |
| `penetration_loss_intersection_curve.csv` | Gaster–legs loss against its reference, 50 scans, mean of seeds 0–2: length of the curve along which the two surfaces cross, triangle pairs, contained leg vertices and their depth. |
| `fit_quality_by_corpus.csv` | The default recipe on the 81 cleaned scans (seeds 0–2; *Atta* and other genera separately) and on the 757 AntScan workers (two runs), and the earlier recipe (`fitter_3d.optimise`) on both sets: medians and shares per fit set. |
| `candidates_surface_and_plausibility.csv` | Candidates 1 and 2 against the default, paired by specimen, on the 757 AntScan workers and the 81 cleaned scans. |
| `candidates_joint_error.csv` | Joint error of every candidate, same protocol as the variants. |
| `candidates_genus_accuracy.csv` | Collection-lot-blind genus accuracy on 616 workers of 82 genera for the default, candidates 1 and 2 and the earlier recipe, with the exact McNemar test against the default. |
| `skinning_weight_defect.csv` | Left against right mandible in fits with the model as it is and with normalised skinning weights. |
| `penetration_cause_synthetic.csv` | Synthetic scans with known part identity (33 per set, seeds 0–2): gaster–legs crossings and contained leg vertices of fits with no, learned, true-part and true dense correspondence and of the ground-truth mesh; share of leg vertices paired with gaster surface. |
| `variant_configs/` | The surface-stage configuration of each variant (a copy of `diagnostics/moonshot/cfg/D1_PROD.yaml` with the one change named in its first line). |

All code and configuration paths below are on branch `feature/investigation`.

## Columns

- `shift_pct_WL`: Hodges–Lehmann estimate of the paired difference in per-scan median joint error (variant minus
  default), in percent of Weber's length; `ci95_*`, `ci90_*`: exact Wilcoxon-inverted intervals. A variant counts as
  equivalent when the 90% interval lies within ±2.5.
- Joint error: distance between fitted and expert-annotated joints after forward kinematics, median over the joints
  of a scan, mean over seeds.
- `chamfer_ratio`: median over scans of the variant's chamfer distance divided by the default's.
- `F001_abs_*`: F-score at an absolute threshold of 0.01 in the normalised frame (scan centred, maximum absolute
  coordinate 1).
- Intersecting triangle pairs: pairs of triangles, one from each of two anatomically non-adjacent body parts
  (`fitter_3d/part_groups.py`), that intersect; exact triangle–triangle tests, no distance threshold.
- Contained leg vertices: leg vertices inside the gaster by the generalised winding number.
- Proximity count: the count used by `fitter_3d/penetration_loss.py` (vertices within a distance threshold of a
  non-adjacent part).
- Intersection curve length: the summed length of the segments shared by intersecting triangle pairs.
- `surface_distance`: exact point-to-surface distance between scan and fit, mean of squared distances, summed over
  both directions; normalised frame.
- `scale_ratio`: the largest per-joint scale factor of a fit or its reciprocal, whichever is larger (1 = no scaling); `freeform_offset`: mean magnitude of the
  per-vertex offset.
- `mahalanobis2`: squared Mahalanobis distance of the 25 shape coefficients from the prior mean; a fit is inside the
  95% region of the shape prior when it is below the 95% quantile of the chi-squared distribution with 25 degrees of
  freedom (37.65).
- `pct_joint_limit_over_5deg`: share of fits with a joint more than 5° beyond its limit.
- `mandible_asymmetry`: RMS residual between the left mandible and the mirrored right mandible after a best
  similarity fit, as a share of mandible size. Mandible length: extent of the mandible's vertices along their first
  principal axis (2nd to 98th percentile).
- Genus accuracy: leave-one-out nearest neighbour on principal components of the fitted measurements, with every
  specimen of the query's collection lot removed from the candidates; `times_permutation_null` is the accuracy divided by its mean under
  permuted genus labels.

## Running a variant

Skeleton stages: the default command with the flag named in `change_to_default_command` (`--trans_cap`,
`--beta_prior`, `--scale_cap`). Surface stages: `fitter_3d.optimise_moonshot` with the file from `variant_configs/`
as `--yaml_src`. `w_geocoh` uses `diagnostics/khaoula_v2/optimise_geocoherence.py` in place of
`fitter_3d.optimise_moonshot`. Correspondence targets are passed to both entry points with
`--cse_correspondence_from`. The penetration variants are the configurations `fitter_3d/ants_cfg_all_*.yaml` run with
`python -m fitter_3d.optimise --yaml_src <file> --seed <seed>`. Candidate 2 is the default command with
`variant_configs/candidate2_plausibility_terms.yaml` as `--yaml_src` of the surface stages. The earlier recipe is
`python -m fitter_3d.optimise --yaml_src diagnostics/atta_reference/cfg_arm_a_master.yaml` (file on
`feature/investigation`) on the code of `master`.

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

## Running a variant

Skeleton stages: the default command with the flag named in `change_to_default_command` (`--trans_cap`,
`--beta_prior`, `--scale_cap`). Surface stages: `fitter_3d.optimise_moonshot` with the file from `variant_configs/`
as `--yaml_src`. `w_geocoh` uses `diagnostics/khaoula_v2/optimise_geocoherence.py` in place of
`fitter_3d.optimise_moonshot`. Correspondence targets are passed to both entry points with
`--cse_correspondence_from`. The penetration variants are the configurations `fitter_3d/ants_cfg_all_*.yaml` run with
`python -m fitter_3d.optimise --yaml_src <file> --seed <seed>`.

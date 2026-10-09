# Registration stages on one cleaned Atta worker

`fig_registration_stages.png` (300 dpi, 180 mm wide; `.pdf` has editable text) shows the default mesh
registration recipe (D1_PROD) on one *Atta vollenweideri* worker (WOLO scAnt scan 13, 12.5 mg), one
panel per stage: **a** rest template (init), **b** after skeleton placement (pose, shape and per-joint
scale; no free-form deformation), **c** after the coarse free-form stage (Stage 2), **d** after the fine
free-form stage (Stage 3). The fitted template (blue) is drawn over the scan (grey): grey that stays
visible is scan surface the fit has not reached. Same camera and lighting in all panels.

Under each panel: chamfer distance, F-score at tau = 0.01, and penetration count.

| stage | chamfer | F@0.01 | penetration |
|---|---|---|---|
| init | 1.35e-2 | 0.149 | 18 |
| pose | 8.88e-4 | 0.508 | 46 |
| Stage 2 | 8.30e-5 | 0.929 | 3 |
| Stage 3 | 7.02e-5 | 0.947 | 6 |

## Files
- `stages.csv` per-stage metrics for scan 13 (`panel=True` rows are the four panels shown); `stages_all20.csv` the same for all 20 Atta scans.
- `score_stages.py` computes the metrics, `blender_render.py` renders the layers (Blender Cycles), `compose_figure.py` assembles the figure.

## What was run
- **Recipe (D1_PROD), seed 0:** `fitter_3d.optimise_hierarchical --midline 2.0 --beta_prior 0.0 --limit 0.273 --offset 30.0 --skip_h3`, then `fitter_3d.optimise_moonshot --yaml_src diagnostics/moonshot/cfg/D1_PROD.yaml --init_from <H2_joint.npz> --eval`, with model `3D_model_prep/OmniAnt_25PCs_joint_limited.pkl` (md5 `08b69daf119a6eda0cd699b8894ee7f8`). The recipe and its weights are described in `diagnostics/SHIPPED_RECIPE.md`.
- **Scan 13** was fixed before fitting by an input-only rule: a single watertight component, standard mesh density, and body mass closest to the median of the 20 scans.
- **Pose panel** is the output of the skeleton stages (`H2_joint`); **init** is the model's default state (zero pose, mean shape) before any fitting.
- **Metrics**, in the fitter's frame (scan centred, max |coordinate| = 1):
  - chamfer: bidirectional squared nearest-neighbour distance, 30k area-weighted samples per surface (mean of 5 sampling seeds);
  - F@0.01: F-score at tau = 0.01 (1% of the half-extent);
  - penetration: template vertices lying inside a non-adjacent body part (`fitter_3d/penetration_loss.py`, proximity tau 0.03, max depth 0.08).

## Reproduce
1. Fit the Atta scans with the two commands above.
2. `python score_stages.py --hier_dir <results>/D1_s0_hier --moon_dir <results>/D1_s0 --mesh_dir <atta_scans>` writes the two CSVs and `probes/focus_meshes.npz`.
3. `blender_render.py --npz probes/focus_meshes.npz --out layers` (needs Blender 5 as a Python module), then `python compose_figure.py`.

## Reading the figure
A clean scan in a near-rest posture, so this shows the recipe on an easy case. The grey rims along the legs at Stage 3 show the fitted legs are thinner than the scanned ones. Surface agreement alone does not establish that the anatomy is correct.

# Task 2 pre-registration: registration stage figure on one cleaned Atta worker

Frozen 2026-10-09, before any fit was run. Approved by Khaoula 2026-10-09. The sha256 of this file
is recorded in `PREREGISTRATION.FREEZE` before the job is submitted; any later change is logged in
`DEVIATIONS.md`, never edited in here.

Task source: `holotype_paper/WORK_PACKAGE.md`, Task 2 (due Mon 12 Oct): "Run D1_PROD on one of the
80 cleaned Atta scans and render the template at each stage (init, pose, Stage_2, Stage_3) with
chamfer, F@0.01 and penetration count per stage underneath. Same camera, same lighting, four panels
in a row." Deliverable: 300-dpi PNG + per-stage CSV in this folder.

## 1. Specimen

Only 20 Atta scans exist on disk (`/hpcwork/nao48500/atta20/01..20.obj`, the WOLO
*A. vollenweideri* workers, 1.1-47.1 mg). Fabian's "80 cleaned" is read as 60 artist-cleaned + 20
Atta; to be confirmed with him.

Selection rule, fixed on input properties only:
1. clean input: one connected component and watertight, evaluated on position indices (as
   pytorch3d `load_obj` reads them; trimesh's default load splits at UV seams and must not be used);
2. standard mesh density: exclude 02, 03, 07 (11-19k vertices vs 5-8k);
3. closest to the geometric-median body mass of all 20 (7.4 mg) in log units.

Eligible after 1-2: 05, 06, 08, 13, 14, 16, 17, 18, 19. Result: **13.obj** (12.5 mg, 5.52 mm,
|log distance| 0.52). Backup if 13's fit fails technically: **08.obj** (0.57).

All 20 are fitted in one batch (as the production driver runs); the figure shows 13, and
`stages_all20.csv` gives the per-stage distribution and 13's rank.

## 2. Recipe (D1_PROD), code state, seed

- Model `3D_model_prep/OmniAnt_25PCs_joint_limited.pkl`, md5 `08b69daf119a6eda0cd699b8894ee7f8`
  (identical on disk and on `master`; matches `diagnostics/SHIPPED_RECIPE.md`).
- Moonshot config `diagnostics/joint_alignment_benchmark/cfg/A_prod.yaml`, md5
  `1efa7a1481119aaf05e32a89375f4218`; weights identical to `diagnostics/moonshot/cfg/D1_PROD.yaml`
  (md5 `d1af1fe7a038f29a300de7493fe4ba9e`; only comments and inert `results_dir` differ).
- Code: HEAD `302d1439` + uncommitted patch to `fitter_3d/` and `smal_model/`, line-identical to
  `diagnostics/joint_alignment_benchmark/data/code_state_uncommitted.patch` (JAB A_prod's code state).
- No learned checkpoint is used (no learned init, no CSE).
- Seed 0. `SMILIFY_COUPLE_JOINT_BLENDSHAPES=0`. `--eval` added vs JAB: it runs after `manager.run()` and only writes `metrics.csv` (needed for gate 2); it cannot change the fit.

```bash
export SMILIFY_SMAL_FILE=3D_model_prep/OmniAnt_25PCs_joint_limited.pkl
export SMILIFY_COUPLE_JOINT_BLENDSHAPES=0
M=/hpcwork/nao48500/atta20   R=/hpcwork/nao48500/fig_registration_stages
python -u -m fitter_3d.optimise_hierarchical --mesh_dir $M \
  --midline 2.0 --beta_prior 0.0 --limit 0.273 --offset 30.0 --skip_h3 \
  --seed 0 --results_dir $R/D1_s0_hier
python -u -m fitter_3d.optimise_moonshot --mesh_dir $M \
  --yaml_src diagnostics/joint_alignment_benchmark/cfg/A_prod.yaml \
  --init_from $R/D1_s0_hier/H2_joint.npz --seed 0 --results_dir $R/D1_s0 --eval
```

## 3. Panels

| panel | source |
|---|---|
| init | `SMAL3DFitter` default state, no optimisation (zero pose, mean betas, identity rotation, zero translation, zero deform) = state at the start of H0 |
| pose | `H2_joint.npz` (skeleton placement: pose + shape + joint scales, no free-form deform) |
| Stage_2 | `Stage_2_deform_coarse.npz` |
| Stage_3 | `Stage_3_deform_fine.npz` |

pytorch3d rasteriser via `diagnostics/moonshot/render_3d.py::make_renderer` (orthographic, one
point light). One camera framed once from the normalised scan and reused for all panels;
dorsolateral elev 22, azim 140. Extras: dorsal and lateral views, the scan from the same camera,
text-free transparent single panels. Main figure 300 dpi, 180 mm wide, numbers under each panel.

## 4. Metrics (fitter frame: scan centred, max |coord| = 1)

- chamfer: `diagnostics/moonshot/metrics.py::surface_metrics` `chamfer_l2`, 30k area-weighted
  samples per surface, mean and spread over 5 sampling seeds.
- F@0.01: same function, tau = 0.01 absolute (1% of half-extent). Secondary, labelled column:
  `fitter_3d/eval_metrics.f_score` at 1% of bbox diagonal (definition used by trainer.py).
- penetration: `fitter_3d/penetration_loss.penetration_loss_batched` with trainer.py's eval
  arguments (`PART_GROUPS_COARSE`, non-adjacent pairs, proximity_tau_fraction 0.03,
  max_depth_fraction 0.08, unramped). Report hard `num_penetrating` (vertex x direction x pair
  triples), `soft_num_penetrating`, and the gaster-legs pair separately.

## 5. Gates (a number is not reported until it passes)

1. Saved-parameter reproduction: verts rebuilt from each npz's parameters match saved `verts`
   to < 1e-4; init reconstruction has zero deform and equals the template.
2. Stage_3 chamfer / F@0.01 for 13 match D1_PROD's own `metrics.csv` within sampling noise.
3. Independent recompute (numpy + scipy cKDTree) of chamfer / F@0.01 for one stage.
4. Fit-on-scan overlays at every stage in `probes/`, inspected and described before any number is
   written (leg-in-gaster, head direction, leg configuration).
5. Penetrating vertices rendered on the Stage_3 mesh; GWN gaster-legs as a secondary cross-check.
6. Orientation: H0 overlay must show the head on the template's head side; if not, stop and report.
7. 13's metrics placed against the 20-specimen distribution.

## 6. Known ways the figure could mislead (caption must handle)

1. Clean Atta is a positive control inside a failure paper.
2. The 20 Atta fits in the shape space used master `fitter_3d.optimise` (`init_rot_lock`, no
   hierarchy, no w_offset / w_limit / w_scale), not D1_PROD; stage names coincide, recipes differ.
3. Surface metrics do not certify anatomy (C2); no expert joints exist for Atta.
4. Init numbers are dominated by scale and placement mismatch.
5. Single seed; fitter run-to-run noise ~0.01 F; hard penetration count CV ~40%.
6. Two F@0.01 definitions in the repo; must match whatever the Task 4 table uses.
7. A fixed view can hide penetration (extra views provided).
8. The watertight criterion selects one of the cleaner Atta inputs.

# Slide 8, "THE SCORE CAN LIE" — single-specimen visual reproduction of R9

Real pipeline run, real ant scan, real optimisation — not a mockup. Extends
`diagnostics/groundtruth/r9_objective_audit.py` / `r3_correct_parts.py`'s exact machinery
(same production checkpoints, same `D1_PROD.yaml` `Stage_3_deform_fine` objective, same
anatomically-CORRECT landmark supervision) to a **single** specimen so a concrete mesh+params
pair could be saved to disk and rendered — R9/R3 only ever reported aggregate 12-specimen-batch
numbers; no per-specimen mesh was ever written to disk before this.

## Probe facts (stated before running, per CLAUDE.md)

- The 12 specimens in `annotation/landmarks/*_traits.json` (used by R3/R9) correspond 1:1 to the
  12 specimens in `annotation/gt_expert/` (the authoritative annotation set; `gt_batch1` is
  superseded and not used here). **All 12 gt_expert specimens were already used by R9** — there
  is no 13th annotated specimen available to be "fresh" with. Per the task's own fallback clause,
  one of the 12 was reused.
- Production parameters live in `diagnostics/moonshot/runs/Z8_W*/Stage_3_deform_fine.npz`
  (`labels`, `betas`, `global_rot`, `joint_rot`, `trans`, `log_beta_scales`, `betas_trans`,
  `deform_verts`), keyed by specimen id — these ARE saved to disk, one entry per specimen, so the
  production fit did not need to be re-optimised, only re-evaluated.
- Target meshes: `/hpcwork/nao48500/worker_alt_data/{sid}_processed.obj`.
- Model: `3D_model_prep/OmniAnt_25PCs_joint_limited.pkl` (`SMILIFY_SMAL_FILE` env var, matching
  R3/R9), `SMILIFY_COUPLE_JOINT_BLENDSHAPES=0`.
- `SMAL3DFitter.forward(..., return_joints=True)` returns `(verts, joints)`; joint parent indices
  are `smal.smal_model.parents` (`dd["kintree_table"][0]`) — used for the skeleton-edge overlay.
- Cluster: no GPU on the login node (`nvidia-smi`: command not found); submitted via SLURM,
  account `rwth2151`, partition `c25g` (available, `mix-` state), following the exact
  conda-activation pattern in `diagnostics/groundtruth/submit_R9_c25g.sbatch`
  (`source /home/nao48500/miniforge3/etc/profile.d/conda.sh && conda activate pytorch3d`).

## Specimen chosen

**`Odontomachus_bauri_CASENT0878072`** — a trap-jaw ant with unusually elongated, articulated
mandibles. Chosen (not randomly) because R9's own writeup flags long mandibles / unusual
antennae as the morphology where the objective's anatomical mismatch is largest and most visible
("named parts drift furthest on the morphologies — long mandibles, unusual antennae"), which
makes the effect legible in a single still frame, not just in aggregate numbers.

## What was run

`diagnostics/slide8_score_can_lie/slide8_score_can_lie.py`, submitted as SLURM job **4149368**
(`c25g`, 1 GPU, ~40s wall time; log `sbatch_logs/slide8_4149368.log`):

1. **Production arm**: production params loaded as-is from the `Z8_W*` checkpoint, evaluated
   once through `MoonshotStage.forward` (no optimisation — this is production's actual shipped
   fit).
2. **Corrected arm**: identical to R3/R9's `CORRECT` arm — starting from the same production
   params, `global_rot, joint_rot, betas, trans, deform_verts` optimised for 800 Adam iterations
   (lr=0.005) against `D1_PROD.yaml`'s full production objective **plus** a landmark term pinning
   each of the 12 expert-annotated landmarks to its anatomically-correct named part (`b_h`,
   `ma_r`/`ma_l`, `an_1_r`/`an_1_l`, `an_2_r`/`an_2_l`, `b_t`), `LAM=1.0`, exactly as in
   `r3_correct_parts.py`/`r9_objective_audit.py`.

Both arms' meshes, free parameters, and target mesh were saved to disk (new — R9/R3 never did
this): `mesh_production.obj`, `mesh_corrected.obj`, `mesh_target.obj`,
`fit_production.npz`, `fit_corrected.npz`.

## Verified numbers (this specimen only; from `slide8_numeric_results.json`)

| | production | anatomically-corrected | ratio (corrected/production) |
|---|---:|---:|---:|
| **Total `D1_PROD` objective** | 0.0004525 | 0.0008298 | **1.834x** (83% higher) |
| **Chamfer term alone** | 0.0004419 | 0.0004974 | **1.126x** (13% higher) |
| **Joint error, median % of Weber's length** | **23.34%** | **0.14%** | production is **167x worse** |

Per-landmark joint error (% WL) is in the json; the worst production landmarks are
`scape_apex_l` (95.6%), `wl_posterior_r` (49.3%), `antennal_insertion_l` (42.1%), and
`mandibular_apex_r` (35.8%) — exactly the anterior/appendage structures R9 predicted.

**This replicates the R9 pattern, and more sharply than the 12-specimen aggregate**: R9's
aggregate total-objective delta was +67%; this single specimen shows +83%. The chamfer
(pure-surface) term is only 13% worse for the anatomically correct fit — i.e. the two fits are
close to indistinguishable at the surface level — while the joint/anatomy error differs by two
orders of magnitude. No retry against other specimens was needed; the first specimen tried
already gave a clean, even stronger, replication.

## Rendering / animation (what was actually done, and the one deviation from the plan)

Renderer: **matplotlib `plot_trisurf`** (not the PyTorch3D/EGL renderer) — chosen because it
works headlessly on this cluster with zero setup risk (`matplotlib.use("Agg")`, already the
pattern in `fitter_3d/utils.py`), and because the deliverable needed skeleton-edge overlays
(line segments from `smal.smal_model.parents`) which are easiest to add directly in matplotlib
rather than via a photorealistic mesh renderer.

Script: `diagnostics/slide8_score_can_lie/render_slide8.py`.

1. `still_production_vs_target.png`, `still_corrected_vs_target.png` — each fitted mesh
   (translucent blue) over the target scan point cloud (gray), demonstrating the surfaces are
   visually near-identical to the raw scan in both arms.
2. `frame_000.png` … `frame_030.png` (31 frames) — **all five free parameters
   (`global_rot, joint_rot, betas, trans, deform_verts`) linearly interpolated** between the
   production and corrected solutions, mesh + skeleton (red) re-rendered at each step.
   **Deviation from the brief, documented as instructed**: the brief asked to interpolate pose
   while "holding shape fixed" if the surface deviates too much. Here shape parameters
   (`betas`, `deform_verts`) were interpolated too, not held fixed, because the CORRECT arm
   optimised shape+pose+deform jointly (same `FREE_P` as R3/R9) — freezing shape to only the
   production value would not reproduce the corrected end-state at t=1, defeating the point.
   The visible cost is a mild surface bulge/wobble mid-sequence; this is a real interpolation
   artifact of linear parameter-space blending, not hidden or smoothed away.
3. `slide8_score_can_lie.gif` — the 31 frames assembled into a looping (ping-pong) GIF via
   `imageio` (`duration=0.07s/frame`, `loop=0`).
4. `still_skeleton_comparison.png` — final comparison frame: the corrected-arm surface with
   **both** skeletons overlaid in different colours (red = production, gold = anatomically
   corrected) over the target scan — the head/mandible/antenna region visibly diverges between
   the two skeletons while the leg skeleton is nearly identical, exactly matching the "43% offset
   + 21% chamfer, but overwhelmingly an anterior effect" story from R9.

## Artifacts (all under `diagnostics/slide8_score_can_lie/`, none deleted)

- `slide8_score_can_lie.py` — the single-specimen fit script (production eval + CORRECT-arm
  optimisation), run via `submit_slide8.sbatch` (SLURM job 4149368).
- `render_slide8.py` — the rendering/animation script (run directly, CPU, ~2 min).
- `slide8_numeric_results.json` — full numeric results (per-landmark errors, both arms).
- `fit_production.npz`, `fit_corrected.npz` — saved free parameters for both arms.
- `mesh_production.obj`, `mesh_corrected.obj`, `mesh_target.obj` — saved meshes.
- `still_production_vs_target.png`, `still_corrected_vs_target.png`,
  `still_skeleton_comparison.png` — key still frames.
- `frame_000.png` … `frame_030.png` — interpolation frame sequence.
- `slide8_score_can_lie.gif` — final looping animation.
- `sbatch_logs/slide8_4149368.log` — full SLURM job log.
- `submit_slide8.sbatch` — the SLURM submission script.

## Limits (stated, not hidden)

- n=1 specimen — this is a demonstration reproduction for the slide visual, not a new
  statistical claim; the statistical claim (n=12, all terms move the same direction) is R9's,
  unchanged and un-superseded by this.
- The "corrected" arm is reachable only under full landmark supervision (R3's finding), not
  something production's actual pipeline currently discovers unsupervised.
- Shape parameters were interpolated jointly with pose for animation continuity (see above); a
  strict pose-only interpolation was not attempted since it would not connect the two saved
  end-states.

## Addendum, same day: rendering pass v2/v3 (presentation quality)

The v1 renders above (`still_*.png`, `frame_*.png`, `slide8_score_can_lie.gif`) are correct but
not presentation-legible: the object filled only a small fraction of the frame (matplotlib 3D's
default camera distance), and in the skeleton-comparison still, gold was drawn fully on top of
red, hiding the production (wrong) skeleton anywhere the two anatomies were close.

Fixes, in `render_slide8_v3.py` (data and numbers unchanged, presentation only), output under
`fig/`:
- **Zoom**: `ax.set_box_aspect(ranges, zoom=2.0)` (matplotlib >=3.3) plus zero-margin axes
  (`fig.add_axes([0.02,0.02,0.96,0.96], projection="3d")`) so the mesh fills the frame instead of
  ~20% of it.
- **Dual-stroke skeleton** (`plot_skel_dual`): production's skeleton (red) is drawn first at
  linewidth 7, the corrected skeleton (gold) drawn on top at linewidth 3.2. Where the two
  anatomies agree, you see a gold line with a visible red fringe -- proof two skeletons are
  overlapping there, not one. Where they diverge, red and gold separate into two fully distinct
  paths. This directly fixes "can't see the wrong solution" without changing the layout/palette.
- **Probed, not assumed, that a localized "zoom into the head" panel would work**: computed
  per-joint Euclidean distance between the two fits' joint sets (`fig/` script output, top-8
  divergent joint indices `[28, 31, 29, 30, 54, 53, 27, 15]`, distances 0.016-0.045 in the same
  normalized units as Weber's length ~0.447) and found their bounding box spans nearly the full
  body, not a local region. This is expected, not a bug: re-optimising proximal joint rotations
  propagates through the kinematic chain to every distal joint (leg tips, antenna tips), so a
  tight local crop would hide most of the effect rather than isolate it. Plan A (a magnified
  head-only inset with a full-body locator) was tried and discarded for this reason plus a
  layout bug; the full-body dual-stroke view is the accurate representation of where the two
  fits actually differ.
- **Deliverable format**: the v1 GIF was 25MB (unusable in a slide deck). Re-encoded as
  `fig/slide8_score_can_lie.mp4` (H.264, 18fps, ~4MB) as the primary deliverable, with
  `fig/slide8_score_can_lie_v3_compressed.gif` (adaptive palette, 0.55x scale, ~12.8MB) kept as a
  GIF fallback for contexts that can't embed video.

Final artifact set for the slide (all under `diagnostics/slide8_score_can_lie/fig/`):
- `slide8_score_can_lie.mp4` -- primary animation.
- `slide8_score_can_lie_v3_compressed.gif` -- GIF fallback.
- `hero_skeleton_comparison_v3.png` -- static dual-stroke comparison still (the single best frame
  for a non-animated slide layout).
- `still_production_vs_target_v3.png`, `still_corrected_vs_target_v3.png` -- surface-match
  stills.
- `render_slide8_v3.py` -- the script that produced all of the above (re-run reproducibly from
  `fit_production.npz` / `fit_corrected.npz` / `mesh_target.obj`, no re-optimisation needed).

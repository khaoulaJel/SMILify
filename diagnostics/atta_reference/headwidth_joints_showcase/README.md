# Head-width joints: showcase of SMILify as a measurement tool (*Atta vollenweideri*, n = 20)

Everything Fabian asked for after the head-width replication, in one place:

> *"Produce a clean log-log plot (head width over body mass), pick sensible axis limits, report the
> slope from the linear fit. Make some screenshots, maybe a little diagram of how you added the
> additional joints to the model, and then an overview of where they end up regressed on the
> registered shapes to demo you are measuring at the correct, widest point on the head."*

---

## Headline results

| | value |
|---|---|
| **Head width ~ body mass slope** | **0.379** (95% CI 0.368–0.390), **R² = 0.997**, n = 20 |
| 2025 reference slope | 0.395 (0.377–0.413), R² = 0.992 |
| Head width ~ body length slope | 1.187 (1.137–1.237); reference 1.235 (1.164–1.306) |
| **Joints capture of the true maximum head width, measured on the real scans** | **98.8%** median (96.8–100%) |
| Joint → real scan surface | 0.41% of body diagonal (median) |
| Joint height within the head | 52% of head height (mid-height, full-face widest level) |
| Same anatomical locus across specimens (SD, % head extent) | lateral 0.56 · antero-posterior 3.25 · dorso-ventral 1.64 |
| Python re-computation vs Blender export | 0.05% mean, 0.15% max |
| Pose sensitivity (pose removed) | head width 0.06% vs body length 1.44% |
| Left/right symmetry about the head midplane | 0.92% mean, 3.1% max |
| Scatter about the scaling line (log₁₀ residual SD) | **replication 0.012** · submitted export 0.017 · 2025 reference 0.019 |

**Reading:** the two added joints land on the lateral surface of the head at its widest point, on
every specimen from 1.1 to 47.1 mg. The head width they give reproduces the expected positive
allometry, and it scatters *less* around the scaling line than the 2025 reference measurements do.

---

## Suggested order to show Fabian

1. **F01a**: the log-log plot he asked for.
2. **F02**: how the joints were added (diagram).
3. **F03**: the template with the joints (screenshots).
4. **F05**: where the joints end up on all 20 registered specimens.
5. **F06**: the same joints on the real scans.
6. **F07**: quantitative evidence that this is the widest point.

Everything else is supporting material.

---

## Figure catalogue

All figures are in `figures/`, as `.png` (250–300 dpi) and `.pdf`.

### Measurement demonstration
- **F01a `headwidth_vs_mass_loglog_clean`**: head width (mm) against body mass (mg), both log
  axes, limits 0.8–60 mg and 0.9–5.2 mm. OLS fit in log-log space with 95% confidence band, and
  a dashed isometry line (slope 1/3) through the data centroid, following the convention of the
  *A. vollenweideri* 3D head-capsule literature. *Slope 0.379, R² 0.997, n 20.*
- **F01b `…_with_reference`**: the same plot with the 2025 reference points as open circles
  (reference slope 0.395).
- **F10 `headwidth_vs_bodylength_loglog`**: head width against body length (b_t–b_a_5).
  Positive allometry: slope 1.187, which excludes isometry at 1.0; reference 1.235.
- **F15 `coherence_with_scaling_law`**: each dataset's deviations from its own fitted scaling
  line. The replication scatters least (SD 2.8% vs 4.5% for the reference). The slope stays the
  same across all three sources.

### How the joints were added
- **F02 `how_the_joints_were_added`**: (A) where `b_h_l`/`b_h_r` sit in the skeleton, as leaf
  children of `b_h` and siblings of the mandible and antenna roots; they are inert reporters with
  no skinning. (B) The five-step Blender procedure. (C) What the exporter computes: each bone is
  tied to its 10 nearest Basis vertices by 1/d weights, the same weights applied to every
  registered specimen, followed by the conversion to mm.
- **F03 `template_with_headwidth_joints`**: the OmniAnt template in dorsal, lateral and anterior
  views, plus head close-ups showing `b_h_l`, `b_h_r`, `b_h`, `ma_l` and `ma_r`.
- **F04 `regressor_weights`**: zoom on the 10 vertices behind each joint, labelled with their
  weights. **Note:** `b_h_r` was placed almost on a vertex, so that one vertex carries 80% of its
  weight, whereas `b_h_l` spreads over its ten (maximum 30%). Both joints still lie on the
  surface; `b_h_r` simply follows one vertex more closely.
- **F14 `measurement_pipeline`**: from μCT mesh to head width in mm, with the validation checks
  listed.

### Where they end up on the registered shapes
- **F05 `all20_registered_heads_fullface`** (and **F05b** dorsal): all 20 registered heads in
  full-face view, each in its own head-aligned frame, with the regressed joints and the
  head-width line. Ordered from smallest to largest specimen.
- **F06 `overlay_on_real_scans`**: the smallest (01), a mid-sized (10) and the largest (20)
  specimen. The real scan surface (head crop, tan) is shown next to the registered SMIL head
  (grey), with the same joint positions drawn on both, in anterior and dorsal views.
- **F13 `placement_consistency`**: joint positions of all 20 specimens in normalised head
  coordinates, with 95% ellipses. The lateral spread is 0.56% of head width. The larger
  antero-posterior spread (3.25%) does not matter for the measurement, because head width is
  nearly constant along that stretch of the head (see F07A).

### Evidence it is the widest point, and that the measurement is sound
- **F07 `widest_point_evidence`**: (A) the head-width profile along the head for each real scan,
  with a dot at the joints' level. The joints sit on the plateau where the head is widest.
  (B) Joint separation as a fraction of the maximum head width: 98.8% median against the scan,
  98.2% against the registered mesh. (C) Distance from each joint to the real scan surface.
- **F08 `python_vs_blender_export`**: the joints drawn in every figure are exactly the joints the
  addon exported (0.05% mean difference).
- **F09 `agreement_with_2025_reference`**: identity plot (Lin's CCC 0.996) and a Bland–Altman plot
  (bias +0.8%; limits of agreement −9.0% to +10.7%; no significant proportional bias, p = 0.09).
- **F11 `pose_invariance`**: removing all articulation changes head width by 0.06%, against 1.4%
  for body length.
- **F12 `bilateral_symmetry`**: the left and right joints sit equidistant from the head midplane
  (0.9% mean asymmetry).

---

## An important reconciliation: which fit these figures use

Two Blender exports exist for the same 20 specimens, run with the same stock-fitter config:

| export | fit it came from | median \|error\| vs 2025 reference | slope vs mass |
|---|---|--:|--:|
| `blender_export/` (**the one first submitted**, 2026-09-06) | a session state that **cannot be reproduced**: no npz on disk matches it | 1.77% | 0.380 |
| `blender_export_rematch_20260907/` (**used here**) | `ATTA20_ARM_A_rematch.npz`; reproduced in Python to 0.05% | 4.12% | 0.379 |

- **Bone placement is not the cause.** The Base-pose `b_h`→`b_h_l` distances in the two exports
  agree to 0.1% (0.19827 vs 0.19848; `b_h_r` 0.19367 vs 0.19369).
- **Ordinary fitter variation is not the cause either.** Five repeat runs vary by about 0.2% CV
  (`RESULTS_X2_step0_noisefloor_20260907.md`).
- **The first export matches no fit on disk.** `RESULTS_X2_step0_reporter_validation_rematch_20260908.md`
  had already shown this.

These figures therefore use the **reproducible** fit, because every joint drawn has to be traceable
to a file. **The 1.77% agreement figure in the original submission should not be quoted.** The
reproducible number is 4.1%.

This doesn't change the conclusion. The slope is identical (0.379 vs 0.380), and the reproducible
replication follows the scaling law *more* tightly than the 2025 reference (F15). The larger
per-specimen differences from the reference are consistent with the reference's own scatter
around that line, as Fabian noted about the outliers.

---

## Method

1. **Fit.** 20 *A. vollenweideri* μCT meshes (`/hpcwork/nao48500/atta20/01–20.obj`, from
   `ALL_ANTS_CLEAN`). Stock SMILify fitter at `origin/master b5bf9565`, model
   `OmniAnt_25PCs_joint_limited.pkl`, config `cfg_arm_a_master_rerun_20260907.yaml`.
2. **Import.** SMIL Model Importer, with PCA, clean_mesh, symmetrise and regress_joints all OFF,
   giving one shape key per specimen.
3. **Joints.** `b_h_l` and `b_h_r` added as children of `b_h`, placed by hand on the **Basis**
   mesh at the widest point of the head capsule in full-face view. All shape-key values were 0
   during placement, because the exporter reads Basis coordinates and shape keys are additive.
4. **Regression.** The exporter gives each joint a J_regressor row over its 10 nearest Basis
   vertices, with weights `w_i = (1/d_i) / Σ(1/d_k)`. On specimen *s*, `j_s = Σ w_i v_{i,s}`.
   Rest positions were trilaterated from the export (`fitter_3d/joint_limits.py`), and the
   regressor was rebuilt in Python (`build_head_width_reporter_matrix`) and checked against the
   export (F08).
5. **Millimetres.** `HW_s = ||j_l − j_r|| × BL_mm,s / BL_model,s`, where BL is the b_t–b_a_5
   distance: taken from the reference CSV in mm, and measured on the same fit in model units.
6. **Widest-point test.** Each head was rigidly aligned (Kabsch) to the template head, with
   lateral, antero-posterior and dorso-ventral axes taken from the template. Each real scan was
   normalised exactly as `fitter_3d/utils.py::load_meshes` does (centre on the vertex mean,
   divide by max |coordinate|), and cropped to points within four edge lengths of the fitted head.
   Head width was measured as the 0.5–99.5 percentile lateral extent: over the whole head, and per
   antero-posterior slice (25 slices). The joint separation was then compared with the maximum.
7. **Statistics.** OLS on log₁₀ values with a t-based 95% CI; Lin's CCC; Bland–Altman on
   percentage differences; pose invariance by setting joint rotations to zero while keeping
   betas, per-joint scales, translations and free-form deformation.

### How the head-width definition compares with the literature

- **Standard taxonomic HW:** the maximum width of the head in full-face view (AntWiki,
  Morphological Measurements), with authors differing on whether the eyes are included. Wilson's
  *Atta* work measures across the widest point in full-face view.
- **Our joints** sit at 52% of head height and on the widest-width plateau, which matches that
  full-face widest-point definition.
- **Eyes:** the template has no separate eye geometry, so "including" versus "excluding" the eyes
  can't be distinguished. *A. vollenweideri* eyes are small and don't form the widest point in
  full-face view.
- **An earlier open question is now resolved:** `REPORT_ATTA_HEADWIDTH.md` noted that
  trilateration put the joints "toward the mandibular/genal region rather than mid-head". F03,
  F05 and F07 show they sit at mid-height on the lateral margin. They are closer to the mandible
  bases than to the `b_h` pivot only because `b_h` is an internal neck pivot, not a surface point.

### Literature this presentation follows
- Three-view mesh renders (dorsal, lateral, anterior), and log-log allometry against body mass
  with an OLS line, 95% CI and a dashed isometry reference: the μCT/LDDMM study of
  *A. vollenweideri* head capsules and mandibles (PMC11021802).
- Automated-versus-manual landmark validation reported as agreement with the manual reference,
  together with the placement shown on the template: ALPACA / SlicerMorph (Porto et al. 2021,
  *Methods Ecol. Evol.*).
- Method agreement reported with Bland–Altman plots and a concordance coefficient rather than
  correlation alone: standard in morphometric and landmark-digitisation reliability work.
- Head-width definition: AntWiki "Morphological Measurements"; *Atta* worker polymorphism and
  allometry (Wilson).

---

## Limitations
- **n = 20, one species, one bone placement**, with no repeated placement by a second person. The
  placement is audited here (F07, F13) but not replicated.
- **mm conversion uses the reference body length**, so SMILify supplies head width *relative to*
  body length. Absolute size comes from the reference measurement.
- **`b_h_r` is dominated by a single vertex** (80% of the weight; F04). It works, but re-placing it
  slightly away from a vertex would make the regression smoother.
- **Raw scans have spiky neck-cut edges** (visible in F06). The width percentiles (0.5–99.5) keep
  these from inflating the maximum width.
- **Blender screenshots.** All renders here come from a Python renderer. `blender/blender_screenshots.py`
  produces real Blender images from `bone_placement.blend`, but it has **not been run**, because
  Blender isn't installed on the cluster.

---

## Reproduce

```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate pytorch3d
cd diagnostics/atta_reference/headwidth_joints_showcase/scripts
python check_widest_point.py      # data/widest_point_check.json
python figs_data.py               # F01, F08–F12, F15 + data/*.csv/json
python figs_render.py             # F03–F07, F13
python figs_diagrams.py           # F02, F14
# real Blender screenshots (on a machine with Blender):
blender -b ../../blender_export/bone_placement.blend -P ../blender/blender_screenshots.py -- ./blender_screens
```

## Folder layout
```
figures/            F01–F15 (png + pdf); _probes/ = orientation check render
data/               head_width_per_specimen.csv, summary_statistics.json,
                    widest_point_check.json, placement_consistency.json
data/source/        reference CSVs + head-width rows extracted from both Blender exports
scripts/            hw_core.py (loading, regression, frames), hw_render.py (renderer),
                    check_widest_point.py, figs_data.py, figs_render.py, figs_diagrams.py
blender/            blender_screenshots.py (untested bpy script for true Blender screenshots)
```
Large source files are referenced rather than copied: `../blender_export/bone_placement.blend`,
`../blender_bundle_rematch_20260907/ATTA20_ARM_A_rematch.npz`, and the scans at `/hpcwork/nao48500/atta20/`.

# WOLO: head-width measurement validation

This folder documents a replication of the WOLO head-width measurement with the current SMILify
tool and model. It covers 20 *Atta vollenweideri* workers, 1.1–47.1 mg, the same specimens as the
2025 measurements.

It shows three things:
- head width recovered from the fitted model follows the expected scaling with body mass;
- how the two head-width joints were added to the model;
- that those joints land at the widest point of the head on the registered specimens.

---

## 01 · Head width vs body mass

**`headwidth_vs_bodymass_loglog.png`**

Head width against body mass across the 20 specimens, both axes logarithmic, with the fitted line
and its 95% confidence band. The dashed line marks isometric scaling (slope 1/3).

Linear fit in log–log space:
- n = 20
- **slope = 0.379**
- 95% CI = [0.368, 0.390]
- R² = 0.997

For comparison, the 2025 reference data give a slope of 0.395 [0.377, 0.413].

**`head_width_per_specimen.csv`**: the measurements behind the plot, one row per specimen. Columns:
mass, body length, 2025 reference head width, replicated head width, and the per-specimen check
values used in folder 05.

## 02 · Addition of the head-width joints

**`how_the_joints_were_added.png`**

1. **Where they sit.** Two new bones, `b_h_l` and `b_h_r`, are children of the head joint `b_h`. They
   are measurement-only: they don't deform the mesh or change the fit.
2. **Placement in Blender.** Import the fitted model, show the template shape, place one bone at the
   widest point of the head on each side, and export the joint distances.
3. **How they follow each specimen.** Each bone is tied to its 10 nearest template vertices, with
   weights proportional to 1/distance. The same vertices and weights are used on every registered
   specimen, so the joint positions move with the fitted head.

Head width is the distance between the two joints, converted to mm using each specimen's measured
body length (`b_t`–`b_a_5`).

## 03 · Blender screenshots

**`03_blender_screenshots/`**: renders and interface screenshots from Blender:
- **Template head with both joints:** full-face view, head only.
- **See-through head:** shows the head-width bar between the two joints.
- **Full model:** for orientation.
- **The two bones in the skeleton hierarchy:** under `b_h`.

## 04 · Joint placement after registration

**`template_with_joints.png`**: the model template from dorsal, lateral and front views, with head
close-ups showing the two added joints next to the head and mandible joints.

**`joints_on_all_20_registered_heads.png`**: where the two joints end up on each of the 20 fitted
specimens, full-face view, ordered from smallest to largest.

**`joints_on_real_scans_vs_model.png`**: the smallest (01), a mid-sized (10) and the largest (20)
specimen. Each is shown as the real scan next to the registered model, with the same joint
positions drawn on both, in front and dorsal views.

## 05 · Widest-point validation

**`widest_point_evidence.png`**: evidence that the joints measure the widest point of the head
capsule, taken from the real scans:
- **A:** head width along the length of the head for each specimen. The joints sit on the plateau
  where the head is widest.
- **B:** the distance between the joints covers **98.8%** of the maximum head width (median; range
  96.8–100%).
- **C:** the joints lie on the scan surface (median distance 0.41% of body diagonal).

The joints sit at mid head height and at the same anatomical position on all 20 specimens.

**`coherence_with_scaling_law.png`**: how far each specimen lies from its dataset's fitted scaling
line. The replicated head widths scatter less (SD 2.8%) than the 2025 reference measurements do
(SD 4.5%), and the slope is essentially the same.

---

## Note on earlier numbers

In my first submission I reported a median agreement of 1.77% with the 2025 head widths. That
number came from a Blender session I can't reproduce, because no fit on disk matches that export.
Everything here uses a reproducible fit. Its joint positions were checked against the Blender export
(0.05% difference), and it agrees with the 2025 values at 4.1% median. The slope is unchanged
(0.379 vs 0.380), and the scatter comparison in folder 05 shows the replicated values follow the
scaling relationship more tightly.

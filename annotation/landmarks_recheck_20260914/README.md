# Landmark re-annotations uploaded 2026-09-14

NOT installed anywhere. Nothing reads this folder automatically.

| file | what it is | status |
|---|---|---|
| `Dolichoderus_cf.bidens_CASENT0744033_traits.json` | new 12 surface landmarks, placed on the correct Dolichoderus scan | verified: scene mesh == worker_alt obj (proper similarity, resid 1e-15); landmarks on the scan surface (≤ 7e-6 diag) |
| `Dolichoderus_cf.bidens_CASENT0744033_traits_scene.blend` | the Blender scene those landmarks were placed in (originally uploaded as `Dolichoderus.blend`) | exact frame record for the JSON above |
| `Discothyrea_patrizii_CASENT0744991_traits.json` | Discothyrea landmarks from the first upload | differs from `landmarks/`, `landmarks_repeat/` and `landmarks_round3/`, so it is an additional round; its scene file was overwritten by the Dolichoderus upload, and its frame is not independently verified |

**Joints were NOT re-annotated.** The joint markers in the Dolichoderus scene are unplaced
placeholders (3 unique positions, mostly [0,0,0]). Dolichoderus remains EXCLUDED from the joint
ground truth; `gt_expert/Dolichoderus_*_joints.json` is an annotation of the Discothyrea mesh.

**Frame warning:** `original` in these JSONs has the same defect as `annotation/landmarks/`: Blender
Z-up, so obj = (x, z, -y). Verified here: the Dolichoderus JSON matches the scene markers at 3e-8
after that conversion.

Possible later uses: a second annotation of Dolichoderus landmarks and a 4th Discothyrea round for
landmark repeatability, when the surface-landmark results are re-checked.

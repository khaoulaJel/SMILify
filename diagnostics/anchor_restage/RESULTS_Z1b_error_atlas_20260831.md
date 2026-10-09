# Z1b — whole-body atlas on open-shape fits: the anterior rotation gap, not a binding defect

Whole-body atlas on OPEN-shape-space fits, since Z1 showed closed-shape-era diagnoses are
unreliable. Same 50 workers (`bench50_clean`), three arms, zero new compute.

## The rotation representation gap

`OmniAnt_25PCs_joint_limited.pkl` constrains **99 of 165** rotation axes:

| group | axes | constrained | free | skin mass |
|---|---:|---:|---:|---:|
| thorax | 3 | 3 | 0 | 1380.1 |
| mandible | 6 | 6 | 0 | 374.0 |
| legs | 108 | 90 | 18 | 4697.8 |
| **head (`b_h`)** | **3** | **0** | **3** | **2180.7** |
| **petiole+postpetiole** | 6 | 0 | 6 | 227.2 |
| **gaster** | 9 | 0 | 9 | 1009.2 |
| **antennae** | 18 | 0 | 18 | 361.6 |
| wing (unbindable) | 12 | 0 | 12 | 0.0 |

**`b_h` is the highest-skin-mass joint in the entire model and all three of its axes are free.**
(X1 recorded "45 of 48 non-leg non-mandible axes unconstrained"; the figure is 36 of 39 once the
12 unbindable wing axes are set aside.)

## What the free axes do

Fitted `|joint_rot|` on `bench50_G1_learned`, degrees:

| joint | skin mass | median | p99 | **max** |
|---|---:|---:|---:|---:|
| `b_h` head | 2180.7 | 3.1 | 61.3 | **101.1** |
| `an_2_l` | 102.0 | 36.8 | 127.2 | **174.0** |
| `b_a_5` | 5.6 | 17.4 | 107.0 | **135.9** |
| `b_a_1` petiole | 77.4 | 2.9 | 72.1 | 78.0 |
| *`l_1_fe_r` (limit 70°)* | 155.1 | 19.8 | 70.0 | **70.1** |
| *`l_1_tr_r` (limit 50°)* | 276.7 | 19.8 | 50.0 | **50.0** |

Two things follow. **The hinge works** — where limits are authored, the fitted maximum lands on
the band to a tenth of a degree. And **the free joints run far past anatomy**: a 101° head or a
174° antennal funiculus is not an ant. Medians are 2–6° on the body axis, so this is purely a
**tail** — which is what a hinge is for, and what no surface metric can see.

## The head defect, replicated on open-shape fits

Free-form deformation per group, reproducible across all three arms:

| group | \|deform\|/thorax | var share ÷ vert share |
|---|---:|---:|
| **head** | **0.70** | **0.24** |
| antenna_r / _l | 2.46 / 2.31 | 4.04 / 3.03 |
| mandible | 1.96 | 1.79 |
| gaster | 1.42 | 1.13 |

The head is 21% of all vertices and absorbs the least free-form work of any group. §6.10's
0.72–0.76 and X1's 0.793 both replicate here at **0.70**. This symptom is real and current, and
it sits on the joint whose rotation is unconstrained.

## Scale: real, but cosmetic

`b_a_5` reaches p99 `|log s|` 3.16 (23.5×) and max 28.2×. Across all 51 bound joints, scale abuse
correlates negatively with skinning mass in **8 of 8 runs** (ρ −0.19 to −0.56; sign test on the
direction p = 0.008) — low-evidence joints drift, because the chamfer term barely restrains them.
But `b_a_5` owns 5.56 of 10,235 vertices' worth of skin. **A joint that owns almost nothing
scaling 28× changes almost nothing.** Recorded, and explicitly excluded from Z2's endpoint in
advance so it cannot be promoted post hoc.

## What this licensed

**Z2**, submitted as job 3352880: `D1_PROD.yaml` unchanged, two arms differing only in the model
file, symmetric bands authored on the 12 free non-leg joints. Mechanism endpoint is the §6.10 head
ratio; bar and voiding checks fixed in `../anterior_limits/PREREGISTRATION_Z2_anterior_limits.md`
before submission.

## Artifacts

`z1b_error_atlas.py`, `z1b_error_atlas_out.txt`, `z1_rescore_worker_anchor.py` + output + JSON.
`fitter_3d/part_groups.py` now names j24/25/44/45 `wing`, `waist` = [1, 2] (petiole b_a_1 +
postpetiole b_a_2, skin mass 77.4 / 149.9, both normally bound), and `gaster` = [3, 4, 5]. The
atlas output predates that rename, so its "waist" row is the wing row and its "gaster" row
includes the petiole and postpetiole.

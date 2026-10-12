# Mesh-registration recipe (D1_PROD)

A rigged template is registered to a scan mesh in five stages: three skeleton stages (body chain, legs, all joints)
and two surface stages that add a per-vertex free-form offset. Shape coefficients, global pose, joint rotations and
per-joint scale and translation are free throughout.

| | |
|---|---|
| Model | `3D_model_prep/OmniAnt_25PCs_joint_limited.pkl`, md5 `08b69daf119a6eda0cd699b8894ee7f8` (10,235 vertices, 25 shape components, 55 joints) |
| Code | `fitter_3d.optimise_hierarchical`, then `fitter_3d.optimise_moonshot`; branch `feature/investigation` |
| Scan normalisation | centred on the vertex mean; maximum absolute coordinate scaled to 1 |
| Optimiser | Adam; cosine-annealed learning rate in the skeleton stages, constant in the surface stages |
| Cost; repeatability | 30–45 s per specimen on one NVIDIA H100; seeded, not bit-reproducible on GPU: F@0.01 varies by 0.001 and joint positions by 2.2% of Weber's length (WL) between seeds |

## Table 1. Stages and loss weights

| | H0_body | H1_legs | H2_joint | H3 | Stage_2 (coarse) | Stage_3 (fine) |
|---|---|---|---|---|---|---|
| Iterations | 900 | 1200 | 900 | skipped | 1000 | 1000 |
| Joint rotations free | body chain (18 joints) | legs (36); body frozen | all (54) | — | all | all |
| Free-form offset | off | off | off | — | on | on |
| Data term (bidirectional chamfer) | whole body, 8000 area samples per surface, Geman–McClure kernel 0.25 | per body part, template vertices against 8000 scan samples | as H1 | — | whole body, 6000 area samples per surface | as Stage_2 |
| Learning rate: pose, shape / joint scale, translation / joint rotation / offset | 0.02 / 0.005 / 0.01 / — | 0.008 / 0.002 / 0.015 / — | 0.005 / 0.00125 / 0.005 / — | — | 0.002 / 0.002 / 0.002 / 0.002 | 0.0005 / 0.0005 / 0.002 / 0.0005 |
| `w_chamfer` | 1.0 | 1.0 | 1.0 | — | 1.0 | 0.5 |
| `w_edge` (mode) | 0.05 (rest length) | 0.05 (rest length) | 0.05 (rest length) | — | 0.8 (shrink) | 0.2 (shrink) |
| `w_laplacian` | 0.005 | 0.005 | 0.005 | — | 0.01 | 0.001 |
| `w_normal` | not computed | not computed | not computed | — | 0.005 | 0.002 |
| `w_offset` | — | — | — | — | 5.0 | 2.0 |
| `w_sym` / `w_midline` | 0.5 / 2.0 | 0.5 / 2.0 | 0.5 / 2.0 | — | 0.5 / 2.0 | 0.5 / 2.0 |
| `w_limit` (joint limits) | 0.273 | 0.273 | 0.273 | — | 0.006 | 0.006 |
| `w_scale` (per-joint scale cap) | 0 | 0 | 0 | — | 0.052 | 0.052 |
| `w_beta_prior`, `w_trans`, penetration, `w_geocoh`, dense correspondence | 0 | 0 | 0 | — | 0 | 0 |

These weights reproduce the logged total loss at all 900 logged iterations of the 757-worker AntScan run. `--offset
30.0` in the command sets the offset weight of the skipped stage H3 and has no effect; it is kept so that the command
is the one that produced the fits.

## Command

```bash
export SMILIFY_SMAL_FILE=3D_model_prep/OmniAnt_25PCs_joint_limited.pkl SMILIFY_COUPLE_JOINT_BLENDSHAPES=0
python -m fitter_3d.optimise_hierarchical --mesh_dir <scans> --midline 2.0 --beta_prior 0.0 --limit 0.273 \
    --offset 30.0 --skip_h3 --seed 0 --results_dir <out>_hier
python -m fitter_3d.optimise_moonshot --mesh_dir <scans> --yaml_src diagnostics/moonshot/cfg/D1_PROD.yaml \
    --init_from <out>_hier/H2_joint.npz --seed 0 --results_dir <out> --eval
```

## Table 2. Terms that are off, and why

Joint error is the median error against expert annotation on 11 AntScan scans (default: 25% WL), each variant
against a default fitted in the same batch; shifts in % WL with 95% intervals; within ±2.5% WL counts as equivalent.
F@0.01 changes are at an absolute threshold of 0.01 on 64 AntScan workers.

| Term | Tested | Result | Evidence |
|---|---|---|---|
| Penetration loss | earlier single-trainer pipeline (`fitter_3d.optimise`) and model (`SMIL_OmniAnt.pkl`), 50 AntScan scans × 3 seeds; not implemented in the recipe's trainers | Halves the leg vertices contained in the gaster (194 → 83) while the two surfaces cross along a boundary several times longer (102 → 691 intersecting triangle pairs) and F@0.01 falls by 0.03; no variant passes its pre-set bars in any seed | `evidence/penetration_loss.csv`, `evidence/penetration_loss_preset_bars.csv` |
| `edge_mode: rest` in Stage_2/3 | inside the recipe, 3 seeds | Worse: +7.0 (2.6 to 14.6), 10 of 11 scans; chamfer ×2.3 | `evidence/variants_inside_recipe.csv` |
| `w_geocoh` 1.0 | inside the recipe, 3 seeds | Holds the fit near the template: chamfer ×2.0; joints +2.4 (−0.9 to 5.9), 8 of 11 worse | `evidence/geodesic_coherence.csv` |
| `w_trans` 0.02 / 0.2 | inside the recipe, 3 seeds | Joints equivalent (−0.5, −0.1); F@0.01 −0.003, −0.014 | `evidence/variants_inside_recipe.csv` |
| `w_beta_prior` 2·10⁻⁶ / 2·10⁻⁵ | inside the recipe, 3 seeds | Joints equivalent (+0.1, +0.6); F@0.01 −0.007, −0.022 | `evidence/variants_inside_recipe.csv` |
| Dense correspondence (learned surface embedding as vertex targets) | inside the recipe, 3 seeds | No gain on real scans: +2.3 (−11.2 to 11.9), single scans from −30 to +46; chamfer ×3.0. Targets from standard non-rigid registration are no better (GBCPD +10.4, NICP +35.6) | `evidence/dense_correspondence.csv` |
| `w_scale` in H0–H2; H3 | inside the recipe, 3 seeds | Cap in the skeleton stages changes neither joints (+0.2) nor final scale ratios; H3's output is not used by the surface stages | `evidence/variants_inside_recipe.csv` |

## The cleaned scans

*To be written.*

## Candidates (tested inside the recipe; not in the defaults)

*To be written.*

*Provenance.* Configuration `diagnostics/moonshot/cfg/D1_PROD.yaml`; resolved per-stage settings in
`evidence/resolved_stage_weights.csv`. Sets: 757 AntScan workers (seed 0); 11 expert-annotated AntScan scans
(seeds 0–2); 64 AntScan workers (seed 0); 50 AntScan scans for the penetration loss (seeds 0–2). Intersection =
intersecting triangle pairs between anatomically non-adjacent body parts (exact test). The configuration of every
variant is in `evidence/variant_configs/` and described in `evidence/README.md`.

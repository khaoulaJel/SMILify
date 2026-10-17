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
| Cost; repeatability | 30–47 s per specimen on one NVIDIA H100; seeded, not bit-reproducible on GPU: between seeds F@0.01 varies by 0.001 and joint error by 2.2% of Weber's length (WL) (per-specimen SD, medians) |

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
| Penetration loss | earlier single-trainer pipeline (`fitter_3d.optimise`) and model (`SMIL_OmniAnt.pkl`), 50 AntScan scans × 3 seeds; not implemented in the recipe's trainers | Halves the leg vertices contained in the gaster (194 → 83) while the two surfaces cross along a boundary 14 times longer (curve length 0.30 → 4.02; 102 → 691 intersecting triangle pairs) and F@0.01 falls by 0.03; no variant passes its pre-set bars in any seed. The crossings lie where leg vertices are paired with gaster surface, and on synthetic scans the true dense correspondence removes the excess crossings without this loss (147 → 50 triangle pairs; ground truth 86) | `evidence/penetration_loss.csv`, `evidence/penetration_loss_preset_bars.csv`, `evidence/penetration_loss_intersection_curve.csv`, `evidence/penetration_cause_synthetic.csv` |
| `edge_mode: rest` in Stage_2/3 | inside the recipe, 3 seeds | Joints +7.0 (2.6 to 14.6), 10 of 11 scans worse (p = 0.014; 0.11 after correction for the eight variants tested together); chamfer ×2.3; F@0.01 −0.14, lower in 63 of 64 workers | `evidence/variants_inside_recipe.csv` |
| `w_geocoh` 1.0 | inside the recipe, 3 seeds | Holds the fit near the template: chamfer ×2.0; joints +2.4 (−0.9 to 5.9), 8 of 11 worse; F@0.01 −0.23, lower in 64 of 64 workers | `evidence/geodesic_coherence.csv` |
| `w_trans` 0.02 / 0.2 | inside the recipe, 3 seeds | Joints equivalent (−0.5, −0.1); F@0.01 −0.003, −0.014 | `evidence/variants_inside_recipe.csv` |
| `w_beta_prior` 2·10⁻⁶ / 2·10⁻⁵ | inside the recipe, 3 seeds | Joints equivalent (+0.1, +0.6); F@0.01 −0.007, −0.022 | `evidence/variants_inside_recipe.csv` |
| Dense correspondence (learned surface embedding as vertex targets) | inside the recipe, 3 seeds | No gain on real scans: +2.3 (−11.2 to 11.9), single scans from −30 to +46; chamfer ×3.0. Targets from standard non-rigid registration are no better (GBCPD +10.4, NICP +35.6) | `evidence/dense_correspondence.csv` |
| `w_scale` in H0–H2; H3 | inside the recipe, 3 seeds | Cap in the skeleton stages changes neither joints (+0.2) nor final scale ratios; H3's output is not used by the surface stages | `evidence/variants_inside_recipe.csv` |

## The cleaned scans

The 81 cleaned scans of the shape space (61 artist-cleaned, one of them an *Atta*; 20 *Atta vollenweideri* from
scAnt) run through this recipe unchanged: same model, stages, weights and command. The model file is an input of the
recipe, so these are refits of the scans the shape space was built from.

| Fit set (default recipe, seed 0) | n | F@0.01, 1% of diagonal | F@0.01, absolute | Gaster and legs intersect | Any intersecting parts | Inside 95% region of shape prior |
|---|---|---|---|---|---|---|
| Cleaned scans | 81 | 0.994 | 0.906 | 0 of 81 | 1 of 81 | 91% |
| Cleaned scans, *Atta* | 21 | 0.996 | 0.934 | 0 of 21 | — | 100% |
| Cleaned scans, other genera | 60 | 0.994 | 0.898 | 0 of 60 | — | 88% |
| Uncleaned AntScan workers | 757 | 0.823 | 0.460 | 50% | 89% | 47% |

Cleaned scans, seeds 1 and 2: F@0.01 (absolute) 0.910 and 0.909, no gaster–leg intersection. *Atta* is a quarter of
the shape space and fits slightly closer than the other genera, which still fit far above the AntScan level.

The earlier recipe (`fitter_3d.optimise` as on `master`: pose and shape for 100 and 300 iterations, then free-form
offsets alone for 2 × 1000 iterations; no hierarchy and no symmetry, midline, offset, joint-limit or scale terms)
reaches a higher F@0.01 by letting the per-joint scale and the free-form offset grow:

| Default against earlier recipe | F@0.01, absolute | Per-joint scale ratio | Free-form offset | Gaster and legs intersect |
|---|---|---|---|---|
| 81 cleaned scans | 0.906 / 0.947 | 1.6 / 8.7 | earlier 8.4 times larger | 0 / 0 of 81 |
| 757 AntScan workers | 0.464 / 0.636 | 2.0 / 55 | earlier 8.5 times larger | 51% / 84% |

On the cleaned scans each of these differences has the same sign in 81 of 81 specimens. The AntScan row uses the
default refitted in the same batch as the earlier recipe; two runs of the default on the 757 workers agree in every
measure (paired tests, p ≥ 0.07). Evidence: `evidence/fit_quality_by_corpus.csv`.

## Candidates (tested inside the recipe; not in the defaults)

Joint error as in Table 2. Surface distance is the symmetric exact point-to-surface distance between scan and fit. On the
757 workers each candidate is compared with the default refitted in the same batch. Genus accuracy is the
collection-lot-blind nearest-neighbour accuracy on 616 workers of 82 genera.

1. **Closer surface at unchanged skeleton.** Exact point-to-surface distance as the data term (`corr_mode: p2s`),
   `w_chamfer` 2.0 / 1.0 and, in Stage_2/3, `w_limit` 0.273, `w_trans` 0.02, `w_beta_prior` 2·10⁻⁶. Surface distance
   −25% on AntScan (lower in 738 of 757) and −37% on the cleaned scans (81 of 81); F@0.01 (absolute) 0.464 → 0.528 and
   0.906 → 0.942; every fit inside the shape prior (default: 47% and 91%); genus accuracy 10.6% against 10.9%
   (p = 0.89). Joints equivalent: −0.9 (−1.9 to 0.3), eight seeds. Costs: free-form offset 25% larger on AntScan;
   intersecting triangle pairs on AntScan 156 against 152 (median).
2. **Joint-limit, translation and shape-prior terms alone** (the three weights of candidate 1 with the default data
   term). AntScan: fits outside the shape prior 53% → 0%, joint-limit violations above 5° 7% → 0%, genus accuracy
   12.8% against 10.9% (p = 0.13). Joints equivalent (−0.2). Cost: surface distance +10% on AntScan and +12% on the
   cleaned scans.
3. **Model with normalised skinning weights.** In the model above, 134 vertices of the left mandible have skinning
   weights that sum to less than 1 (minimum 0.40). Every fit therefore has a distorted left mandible: the left–right
   mismatch is 0.43 of mandible size on the cleaned scans and 0.33 on AntScan, and the left mandible measures 31% and
   5% longer than the right. Dividing each vertex's weights by their sum removes the distortion (mismatch 0.03 and
   0.09; lengths within 2–3%) with joints equivalent (+0.2), alone and together with candidate 1. Until the model is
   repaired, mandible measurements should be taken on the right side.
4. **Area sampling of the template in H1/H2** (the symmetric chamfer sampling of the earlier pipeline) shows no
   benefit inside the recipe: joints −1.2 (equivalent); F@0.01 on 64 workers not different (higher in 38, lower in
   26). Its earlier F-score gain belongs to a pipeline whose data term sampled the template by vertex.

Neither the sixteen variants of the objective tested nor `w_geocoh` improves joint accuracy.

Availability. Candidate 2 runs on the branch as it is (`evidence/variant_configs/candidate2_plausibility_terms.yaml`).
Candidates 1 and 4 use options that are not yet on the branch (`corr_mode: p2s`; area sampling in the skeleton
stages), and the repaired model file of candidate 3 is not in the repository. Evidence:
`evidence/candidates_surface_and_plausibility.csv`, `evidence/candidates_joint_error.csv`,
`evidence/candidates_genus_accuracy.csv`, `evidence/skinning_weight_defect.csv`.

*Provenance.* Configuration `diagnostics/moonshot/cfg/D1_PROD.yaml`; resolved per-stage settings in
`evidence/resolved_stage_weights.csv`. Sets: 757 AntScan workers (seed 0); 81 cleaned scans (seeds 0–2);
11 expert-annotated AntScan scans (seeds 0–2; eight seeds for candidate 1); 64 AntScan workers (seed 0); 50 AntScan
scans for the penetration loss (seeds 0–2). Intersection =
intersecting triangle pairs between anatomically non-adjacent body parts (exact test). The configuration of every
variant is in `evidence/variant_configs/` and described in `evidence/README.md`.

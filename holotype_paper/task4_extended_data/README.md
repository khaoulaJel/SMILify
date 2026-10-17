# Extended Data: automatic registration of the AntScan corpus

All fits use the default recipe (D1_PROD; model `OmniAnt_25PCs_joint_limited.pkl`) unless a row names another.
Seed 0 unless seeds are given. WL = Weber's length. F@0.01 is the F-score at 1% of the bounding-box diagonal;
"abs." marks the absolute threshold of 0.01 in the normalised frame. "Intersecting parts" = at least one
intersecting triangle pair between anatomically non-adjacent body parts (exact triangle test). Bilateral measurements
are the mean of left and right, except the mandible, which is measured on the right side (section 6, model defect).

## 1. Corpus and registration quality

AntScan (Katzke et al. 2026): 2,193 specimens; its surface meshes are automatic full-body segmentations by a network
trained on 12 manually pre-segmented scans. Used here: 757 worker scans and the 81 cleaned scans of the shape
space (61 artist-cleaned, one of them an *Atta*; 20 *Atta vollenweideri* from scAnt); 838 specimens, 187 genera.

| Corpus | n | F@0.01 | F@0.01 abs. | Surface distance | Gaster and legs intersect | Any intersecting parts | Outside 95% of shape prior |
|---|---|---|---|---|---|---|---|
| Cleaned scans | 81 | 0.994 | 0.906 | 5.2·10⁻⁵ | 0 of 81 | 1 of 81 | 9% |
| AntScan workers | 757 | 0.823 | 0.460 | 4.8·10⁻⁴ | 50% | 89% | 53% |

Cleaned scans, seeds 1 and 2: F@0.01 abs. 0.910, 0.909; gaster–leg intersection in 0 of 81. Cleaned scans of other
genera (60) against *Atta* (21): F@0.01 abs. 0.898 against 0.934 (Mahalanobis² 14.4 against 8.3). Two independent
runs of the recipe on the 757 workers differ in no measure (paired; p ≥ 0.07); this table gives the first run, and comparisons between configurations below use the
default refitted in the same batch (F@0.01 abs. 0.464, gaster–leg intersection 51%).

Evidence: `../task3_registration_recipe/evidence/fit_quality_by_corpus.csv`, `evidence/mesh_properties.csv`.

## 2. Accuracy against expert-annotated joints (11 AntScan scans, seeds 0–2)

| Region | Coxae | Proximal leg | Body axis | Mandibles | Antennae | Distal leg | All joints |
|---|---|---|---|---|---|---|---|
| Median joint error (% WL) | 11.6 | 12.9 | 24.1 | 42.8 | 54.9 | 56.3 | 25.1 |

Per-specimen range 13.3–68.6% WL; seed-to-seed SD per specimen 2.2% WL (median).

Evidence: `evidence/joint_error_by_region.csv`.

## 3. Registration strategies tested (joint error against the default fitted in the same batch; 11 scans)

Shift = Hodges–Lehmann estimate in % WL, positive = worse; 95% interval; scans better / worse.

| Strategy | Shift | 95% CI | Better / worse | Surface |
|---|---|---|---|---|
| **Recipe components removed** | | | | |
| No hierarchy (surface stages only) | +8.2 | 3.1 to 13.8 | 1 / 10 | chamfer ×1.19 |
| No joint limits | +4.0 | 0.8 to 7.1 | 2 / 9 | ×0.96 |
| No offset and normal terms | +5.0 | −0.3 to 9.8 | 3 / 8 | ×0.45 |
| No part partition; no scale cap; allometric prior | +0.5; +0.5; 0.0 | all include 0 | — | ×0.97–1.00 |
| **Objective variants (16 tested)** | | | | |
| Rest-length edge term (of the other fifteen, twelve are equivalent within ±2.5% WL and three inconclusive) | +7.0 | 2.6 to 14.6 | 1 / 10 | ×2.3 |
| Data weight doubled | −0.3 | −1.7 to 0.9 | 6 / 5 | surface distance −26% (64 workers) |
| **Pose initialisation** | | | | |
| Learned initialisation | +0.7 | −1.2 to 15.2 | 4 / 7 | ×1.00 |
| **Correspondence supplied to the fitter** | | | | |
| Learned surface embedding | +2.3 | −11.2 to 11.9 | 3 / 8 | ×2.97 |
| GBCPD (geodesic coherent point drift) | +10.4 | −0.5 to 21.2 | 4 / 7 | ×1.6 |
| NICP (Amberg) | +35.6 | 19.3 to 45.9 | 1 / 10 | ×3.4 |
| **Registration without the model** | | | | |
| NICP alone, joints from the regressor | 71.3% WL against 24.2 | — | — | — |
| **Earlier recipe** (`fitter_3d.optimise`, master; one seed) | 38.5 against 25.1 | — | 3 / 8 | F@0.01 abs. 0.636 against 0.464 (757 workers) |
| **Geodesic motion coherence** (`w_geocoh` 1.0) | +2.4 | −0.9 to 5.9 | 3 / 8 | chamfer ×2.0; F@0.01 abs. −0.23, lower in 64 of 64 workers; largest loss on gaster and head (×2.0, ×1.9), smallest on legs (×1.3) |
| **Point-to-surface data term, doubled data weight, joint-limit, translation and shape-prior terms** (8 seeds) | −0.9 | −1.9 to 0.3 | 7 / 4 | surface distance −25% (738 of 757 workers) |

Learned pose initialisation on 50 AntScan scans (surface measures, one seed): unscaled prediction worse than zero
initialisation (chamfer worse in 33 of 50, Wilcoxon p = 0.014); scaled-down predictions not different (p = 0.46–0.96).
Earlier recipe on all 757 workers: per-joint scale ratio 55 against 2.0, free-form field 8.5 times larger,
joint-limit violation above 5° in 70% of specimens against 7%, gaster and legs intersecting in 84% against 51%,
lot-blind genus accuracy 8.3% against 10.9% (p = 0.11). Geometric pose estimators are not reported: the per-specimen
outputs of the August comparison on 12 synthetic specimens were not recovered, so it could not be re-derived.

Evidence: `evidence/strategies_joint_error.csv`, `evidence/pose_initialisation_50_scans.csv`, `../task3_registration_recipe/evidence/variants_inside_recipe.csv`, `../task3_registration_recipe/evidence/dense_correspondence.csv`, `../task3_registration_recipe/evidence/candidates_joint_error.csv`.

## 4. Self-intersection losses (earlier single-trainer pipeline, model `SMIL_OmniAnt.pkl`; 50 AntScan scans, seeds 0–2)

Each attempt against its reference in the measures and bars fixed in August (proximity count; F@0.01 at 1% of the
diagonal). Ranges are over the three seeds.

| Attempt | Gaster–legs count | All-pairs count | Mean F change | Worst F change | Seeds | Bars passed |
|---|---|---|---|---|---|---|
| *Bar* | ≤ −15% | ≤ 0% | ≥ −0.010 | ≥ −0.080 | | all four |
| T1 proximity loss, all part pairs | −1% to −10% | +11% to +16% | −0.004 to −0.005 | −0.028 to −0.036 | 3 | in no seed (both count bars fail) |
| T2 same, with offset penalty | −1% to −9% | +16% to +25% | 0.000 to +0.003 | −0.014 to −0.064 | 3 | in no seed (both count bars fail) |
| T4 loss on gaster–legs only | −7% to −23% | −5% to −18% | −0.008 to −0.012 | −0.045 to −0.060 | 3 | in no seed (gaster–legs bar fails in 2, mean F in 2) |
| T5 T4 with scale and translation barriers | −16% to −26% | −19% to −25% | −0.017 to −0.019 | −0.077 to −0.109 | 3 | in no seed (mean F fails in 3, worst F in 2) |
| T7 symmetric chamfer sampling (10 scans) | — | −7%, −13%, +3% | +0.051, +0.041, +0.035 | — | 3 | split: F up in every seed, count not consistently lower |
| T3 | diagnostic of the count, no loss (see below) | | | | | |
| T6 winding-number loss | three specimens, one run (see below) | | | | 1 | — |
| T8 BVH loss | not run: its third-party CUDA extension fails | | | | 0 | — |
| T9 post-hoc repair | completed in August with a loss of fit; not re-run, no number reported | | | | 0 | — |

The proximity count is not a measure of penetration: it changes sign between seeds in 24–30 of 50 specimens. The same
fits with an exact test, gaster–legs, per-specimen mean of three seeds, medians, reference → with loss:

| Attempt | Contained leg vertices (winding number) | Intersecting triangle pairs | F@0.01 abs. | Free-form field |
|---|---|---|---|---|
| T1 | 174 → 119 | 165 → 404 | −0.014 | +4% |
| T2 | 194 → 182 | 102 → 141 | −0.001 | +3% |
| T4 | 194 → 83 | 102 → 691 | −0.030 | +18% |
| T5 | 194 → 93 | 102 → 525 | −0.027 | +24% |

Containment falls in every seed for every attempt while the crossings rise. For T4 the measured length of the
gaster–leg intersection curve rises from 0.30 to 4.02 normalised units (×13.6; higher in 47 of 50 specimens, lower in
none) and the leg vertices that remain contained lie shallower (mean depth ×0.62): the loss moves the legs to the
gaster boundary. T3: the proximity test and the winding number disagree most where the count worsens (r = 0.947
between disagreement asymmetry and count change; count worse in 44 of 50; an earlier fit set). T6: intersecting triangle pairs
908, 721, 96 against 904, 648, 273 without a loss. T7, exact test: intersecting triangle pairs −10%, −11%, −12%.

Evidence: `evidence/penetration_bars_per_seed.csv`, `../task3_registration_recipe/evidence/penetration_loss.csv`, `../task3_registration_recipe/evidence/penetration_loss_intersection_curve.csv`, `evidence/penetration_T6_winding_number_loss.csv`, `evidence/penetration_T7_symmetric_sampling.csv`, `evidence/proximity_against_winding_number.csv`.

## 5. Why registration fails on AntScan

| Finding | Number | Data |
|---|---|---|
| The model can hold the expert skeleton | joints supervised: 6.2% WL with shape inside the prior (Mahalanobis² 0.36); chamfer +6% over the unsupervised fit it starts from (27.0% WL) and +11% over a control continued for the same iterations (30.0% WL) | 11 scans, seed 0 |
| A closer surface does not place the skeleton | chamfer ×0.45 with joint error +5.0% WL (8 of 11 worse); surface distance −26% with joint error −0.3% WL | 11 scans; 64 workers |
| Supervision does not transfer between regions | region error with all other regions supervised against unsupervised: antennae 51.9 / 54.5, distal leg 48.9 / 53.1, mandibles 36.0 / 38.9 (p ≥ 0.24); supervised directly: 28.9, 19.7, 32.8 | 11 scans, seed 0 |
| Dense correspondence is the largest single lever | share of the gap to a fully informed fit closed by one restored quantity: dense correspondence 60–62%, part membership 35–38%, joint positions 30–31%, shape 5–17%, true pose 1.5–1.9%; true dense correspondence also brings gaster–leg crossings to the level of the true mesh without a penetration term (147 → 50; true mesh 86) | synthetic, real-level articulation, 33 × 3 seeds |
| Correct correspondence removes most of the error; wrong-leg targets remove the benefit | joint error 2.43 → 0.89% of body length (48 of 48); with 20% wrong-leg targets 3.17 (26 better, 22 worse); with 40% 4.57 (7 / 41) | synthetic, 48 |
| Learned correspondence fails under real articulation and real per-joint scales, not under real shape coefficients | leg-identity accuracy: reference 0.908; real shape coefficients 0.920; real poses 0.745 (median leg bend 26°); real per-joint scales 0.675 (21°); all real 0.568 | synthetic, 136 |
| Real leg articulation exceeds the training range | median leg bend 18–38° on the expert scans against 8–19° in training | 11 scans |

Mesh properties: the cleaned scans are not topologically cleaner (median 2,985 connected components against 2,735);
they have fewer open edges (3.2% against 6.1%). Two scan-level posture measures do not track the expert-measured leg
bend (Spearman −0.13, +0.19; n = 11), and 14 of 757 workers fall inside the cleaned range on all measures: mesh
construction and posture are not separable in these data.

Evidence: `evidence/supervised_fit_capacity.csv`, `evidence/supervision_transfer_by_region.csv`, `evidence/oracle_budget.csv`, `evidence/synthetic_correspondence_target_quality.csv`, `evidence/learned_correspondence_under_shift.csv`, `evidence/leg_articulation.csv`, `evidence/mesh_properties.csv`, `evidence/posture_measures.csv`, `../task3_registration_recipe/evidence/penetration_cause_synthetic.csv`.

## 6. What the registrations carry

| Result | Number | Data |
|---|---|---|
| Genus signal, collection-lot-blind | accuracy 10.9–11.4% in three independent runs, 5.0–5.4 times the permutation null | 616 workers, 82 genera |
| Species equals collection lot | within-genus species accuracy from the accession number alone 88.6%, from shape 32.8% (null 24.7%); 74 of 105 species lie in one lot | 271 workers, 33 genera |
| Scale cap | genus accuracy 11.7% without, 11.4% with; largest anterior scale ratios 8.9 → 3.8 (head), 10.6 → 4.1 (mandible), 13.6 → 4.1 (antenna); medians unchanged (1.39, 1.63, 2.13 → 1.39, 1.62, 2.04) | 838 specimens |
| Joint-limit, translation and shape-prior terms | outside the shape prior 53% → 0%; limit violation above 5° 6.7% → 0%; genus accuracy 12.8% against 10.9% (p = 0.13); joints equivalent (−0.2% WL) | 757 workers; 11 scans |
| The same with a point-to-surface data term at doubled weight | surface distance −25% (lower in 738 of 757) and −37% on cleaned scans (81 of 81); all fits inside the shape prior; genus accuracy 10.6% against 10.9% (p = 0.89); joints equivalent (−0.9% WL, eight seeds) | 757 workers; 81 cleaned; 11 scans |
| Cross-corpus genus retrieval | 4 of 46 cleaned specimens assigned to the correct genus among 38 (chance 2.6%; permutation p = 0.010; binomial p = 0.033); same four specimens with three seeds of the cleaned fits | 757 workers, 46 cleaned |
| Cross-corpus agreement of genus means (Pearson R, 95% CI) | gaster slenderness 0.78 (0.64–0.90); cephalic index 0.59 (0.34–0.76); waist constriction 0.55 (0.20–0.79); mesosoma slenderness 0.55 (0.20–0.77); the other eleven indices, including every leg, antenna and mandible index (mandible slenderness 0.32, −0.15 to 0.61), not different from zero after correction | 38 genera |
| Trap-jaw genera | mandible slenderness on the right mandible: three trap-jaw genera among the four most slender of 82 (*Myrmoteras* 6.5, first; *Strumigenys* 4.2, third; *Odontomachus* 3.9, fourth; *Mayaponera* 4.6 between them); *Odontomachus* 3.9 against 3.5 in the cleaned scans, and the narrowest head of the 82 genera (cephalic index 0.75) | 757 workers; 81 cleaned |
| Head width | measurable in 752 of 757 workers by a landmark-free criterion | 757 workers |
| Model defect | 134 left-mandible vertices with skinning weights summing below 1; left–right mandible asymmetry of fits 0.43 → 0.03 (cleaned) and 0.33 → 0.09 (AntScan) with normalised weights; the largest mismatch lies at one of these vertices in 81 of 81 and 63 of 64 fits with the model as it is; left mandible length reads 31% (cleaned) and 5% (AntScan) above the right with the defect, 2% and 3% below without; joint error equivalent (+0.2% WL) | 81; 64; 11 scans |

Evidence: `evidence/genus_accuracy_default_runs.csv`, `../task3_registration_recipe/evidence/candidates_genus_accuracy.csv`, `evidence/species_against_collection_lot.csv`, `evidence/scale_cap.csv`, `evidence/cross_corpus_genus_retrieval.csv`, `evidence/cross_corpus_index_agreement.csv`, `evidence/mandible_slenderness_by_side.csv`, `evidence/head_width_gate.csv`, `../task3_registration_recipe/evidence/skinning_weight_defect.csv`.

## 7. Figures

`fig_crosscorpus.png`: genus means of fifteen indices, AntScan workers against cleaned scans, ordered by R (300 dpi).
`fig_genus_indices.png`: six indices for the 46 genera with at least five workers (300 dpi). Both use the right mandible
for the two mandible indices; *Myrmoteras* has three workers and is not in the second figure.

## 8. Diagnosis (Discussion)

The recipe that registers the cleaned scans fits the AntScan meshes worse. Gaster and leg surfaces intersect in 50%
of 757 worker fits and none of 81 cleaned scans. Against expert annotation (11 specimens), median joint error is
12–13% of Weber's length for coxae and proximal legs and 43–56% for distal legs, antennae and mandibles. Model
capacity does not explain this: given expert joints, the model reproduces the annotated skeleton (6% error) with
shape coefficients inside the prior. Nor does the data term: a quarter less surface distance left joint error
unchanged, and neither sixteen objective variants nor two non-rigid methods reduced it. The missing quantity is
anatomical correspondence, which also accounts for the intersections: legs enter the gaster where the data term pairs
them with gaster surface, and on synthetic scans true dense correspondence removes the excess crossings that a
penetration loss only displaces. Learned correspondence fails under real leg articulation and real per-joint scales,
not under real shape coefficients. On synthetic scans, true dense correspondence closed 60–62% of the gap to a fully
informed fit; the true pose, under 2%. Correspondence robust to both, at minimum correct leg identity, is necessary
and, by the same measurement, not sufficient.

## Provenance

Specimen sets: 757 AntScan workers; 81 cleaned scans; 11 expert-annotated AntScan scans; 50 AntScan scans for the
self-intersection losses; synthetic sets of the correspondence study. Recipe, model and command:
`../task3_registration_recipe/README.md`. Tables behind every number: `evidence/` (described in `evidence/README.md`).

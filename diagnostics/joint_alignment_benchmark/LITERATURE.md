# Literature grounding for the Joint-Alignment Benchmark

Scope: the **evaluation methodology**, i.e. how to decide which registration strategy aligns the
model skeleton best with hand-annotated joints. Each entry states the design decision it grounds.
Every source was located and checked in this session (2026-09-14); links are to the publisher,
PubMed or arXiv record. Method-side literature (SMAL, CSE, correspondence) is in
`diagnostics/groundtruth/LITERATURE_VALIDATION_20260908.md` and is not repeated here.

---

## 1. The reference frame must not be fitted to what is being evaluated

**Fitzpatrick, J.M. (2009). Fiducial registration error and target registration error are
uncorrelated.** *Proc. SPIE 7261.* [ADS](https://ui.adsabs.harvard.edu/abs/2009SPIE.7261E..02F/abstract)
· West & Fitzpatrick on the TRE distribution:
[ResearchGate](https://www.researchgate.net/publication/11764825_The_Distribution_of_Target_Registration_Error_in_Rigid-Body_Point-Based_Registration)

The residual on the points used to *compute* an alignment (FRE) says nothing about error at points
not used (TRE). **Grounds §3.1:** earlier project numbers aligned the ground truth to the fitted
skeleton and then measured residuals on the same joints. That is an FRE-type quantity. The benchmark
registers GT to the scan through the annotation scene mesh, so no evaluated joint enters the frame.
Measured consequence on production: 42.7% → 25.1% WL median error (`figures/F1_gt_audit`).

**Umeyama, S. (1991). Least-squares estimation of transformation parameters between two point
patterns.** *IEEE TPAMI 13(4).* The closed-form similarity with a reflection guard. **Grounds §3.2:**
the guard was deliberately lifted (`allow_reflection=True`), which is how the four mirrored
annotation scenes were detected (det −1, residual ≤ 7e-8).

## 2. Absolute error first; Procrustes-aligned error only as a secondary view

**von Cramon-Taubadel, N., Frazier, B.C., Mirazón Lahr, M. (2007). The problem of assessing landmark
error in geometric morphometrics: theory, methods, and modifications.** *Am J Phys Anthropol
134(1):24–35.* [PubMed 17503448](https://pubmed.ncbi.nlm.nih.gov/17503448/)
· *How exactly did the nose get that long? A critical rethinking of the Pinocchio effect* (Evol Biol
2020): [Springer](https://link.springer.com/article/10.1007/s11692-020-09520-y)

Generalised Procrustes superimposition smears localised error across all landmarks (the "Pinocchio
effect"), so per-landmark error read after GPA is misattributed. **Grounds §2:** the primary endpoint
applies no post-hoc alignment. Observed here: PA error *exceeded* absolute error on 9/11 D1_PROD
specimens (median 35.5 vs 25.1% WL), because a few grossly misplaced legs dominate the least-squares fit.

**Limitations of (Procrustes) alignment in assessing multi-person human pose and shape estimation
(2024).** (Author list not verified in this session; cite from the arXiv record.) [arXiv 2409.16861](https://arxiv.org/abs/2409.16861)

In body-model evaluation, PA-MPJPE masks systematic global errors; world-coordinate metrics
(W-MPJPE) should be preferred whenever absolute placement matters. **Grounds §7 S1:** PA error is
reported, labelled as articulation-only.

**Xu, J. et al. (2023). Animal3D: A comprehensive dataset of 3D animal pose and shape.** *ICCV 2023.*
[CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Xu_Animal3D_A_Comprehensive_Dataset_of_3D_Animal_Pose_and_Shape_ICCV_2023_paper.html)
· [arXiv 2308.11737](https://arxiv.org/abs/2308.11737)

The SMAL-family benchmark standard: PA-MPJPE plus PCK with a body-size-normalised threshold
(PCK@HTH = half head-to-tail). **Grounds §7 S2:** PCK at thresholds normalised by a body-size
standard. For ants that standard is Weber's length, not head-to-tail (below).

## 3. Size normalisation: Weber's length

**AntWiki, Morphological Measurements** ([link](https://www.antwiki.org/wiki/Morphological_Measurements))
· GLAD trait notes ([PDF](https://globalants.org/static/trait-descriptions.pdf))

Weber's length (diagonal mesosoma length, pronotum–cervical shield junction to the posterior basal
angle of the metapleuron) is the standardised ant body-size measure. It excludes spines and is
posture-independent (a single rigid tagma). **Grounds §2:** errors are % WL. Joint centroid size is
posture-dependent (splayed legs), so it is used only as a sensitivity analysis.

## 4. Accuracy = trueness + precision; bias analysed separately

**ISO 5725-1:2023. Accuracy (trueness and precision) of measurement methods and results — Part 1.**
[ISO](https://www.iso.org/standard/69418.html)

Accuracy decomposes into trueness (bias of the mean from the reference) and precision (spread).
**Grounds §7 S6:** a per-joint mean signed error in a specimen-local anatomical frame, with a
leave-one-specimen-out correction. It separates systematic joint-*definition* offsets from fitting
error.

**Keller, M. et al. (2023). From Skin to Skeleton: Towards biomechanically accurate 3D digital
humans (SKEL).** *ACM TOG (SIGGRAPH Asia).* [ACM](https://dl.acm.org/doi/10.1145/3618381)
· [arXiv 2509.06607](https://arxiv.org/abs/2509.06607)

Graphics body models (SMPL) place joints where skinning needs them, not at anatomical articulation
centres, and the offset is systematic. **Grounds §5.1 and S6:** the rig's FK pivot and the
annotator's anatomical articulation are *different definitions*. A consistent per-joint bias across
all arms and specimens is read as a definition mismatch, not a strategy failure. It is also why the
model-side definition (FK vs regressed vs skin) is pre-registered and varied in sensitivity.

## 5. Measurement error in manual and automated landmarks

**Fruciano, C. (2016). Measurement error in geometric morphometrics.** *Dev Genes Evol 226:139–158.*
[PubMed 27038025](https://pubmed.ncbi.nlm.nih.gov/27038025/)

Review of random vs systematic error sources, replicate-based estimation, and visual inspection of
per-landmark dispersion. **Grounds §8.3:** the smallest effect of interest is tied to measured
annotation repeatability (V11, 2.4% WL), not chosen arbitrarily. It also grounds the requirement for
per-specimen visual probes (`probes/gt_on_scan_PROBE.png`).

**Percival, C.J. et al. (2019). The effect of automated landmark identification on morphometric
analyses.** *J Anat 234(6):917–935.* [Wiley](https://onlinelibrary.wiley.com/doi/10.1111/joa.12973)
· **Porto, A. et al. (2021). ALPACA: a fast and accurate computer vision approach for automated
landmarking of three-dimensional biological structures.** *Methods Ecol Evol.*
[Wiley](https://besjournals.onlinelibrary.wiley.com/doi/10.1111/2041-210X.13689)

The accepted validation design for automated 3D landmarking: compare automated placements against
independent expert placements on the same specimens, and report per-landmark error, not only a
global mean. **Grounds the whole design:** automated joints vs expert joints on the same scans, with
per-joint and per-region reporting (§7 S3, F4, F8).

## 6. Comparing several methods on few test cases

**Demšar, J. (2006). Statistical comparisons of classifiers over multiple data sets.** *JMLR 7:1–30.*
[ACM DL](https://dl.acm.org/doi/10.5555/1248547.1248548)

Paired Wilcoxon signed-rank for two methods over data sets; Friedman plus post-hoc (Nemenyi CD
diagram) for many. **Grounds §8.2 and §8.5.** Here the "data sets" are specimens.

**Benavoli, A., Corani, G., Demšar, J., Zaffalon, M. (2017). Time for a change: a tutorial for
comparing multiple classifiers through Bayesian analysis.** *JMLR 18.*
[JMLR](http://www.jmlr.org/papers/v18/16-305.html)

Argues for reporting effect sizes with a region of practical equivalence instead of p-values alone.
**Grounds §8.4:** verdicts combine a Holm-controlled test with a practical-equivalence band (±SESOI),
and every contrast is reported as an effect with its CI.

**Hodges–Lehmann estimator with the exact Wilcoxon-inverted CI** (standard nonparametric
location-shift estimation; see e.g. [Geyer, U. Minnesota notes](https://www.stat.umn.edu/geyer/s06/5102/notes/rank.pdf)).
**Grounds §8.2.** The implementation was checked against brute-force test inversion
(`tools/analyze.py`; agreement to grid resolution on 5 random draws).

**Lakens, D. (2017). Equivalence tests: a practical primer for t tests, correlations, and
meta-analyses.** *Soc Psychol Personal Sci 8(4):355–362.*
[SAGE](https://journals.sagepub.com/doi/10.1177/1948550617697177)

Absence of a significant difference is not evidence of equivalence; test it with TOST against a
smallest effect size of interest. **Grounds §8.4 EQUIVALENT:** a strategy is called equivalent only
when its 90% CI lies inside ±2.5% WL.

## 7. Dependence structure and optimiser noise

**Saravanan, V., Berman, G.J., Sober, S.J. (2020). Application of the hierarchical bootstrap to
multi-level data in neuroscience.** *NBDT.* [PubMed 33644783](https://pubmed.ncbi.nlm.nih.gov/33644783/)

Treating nested observations as independent inflates false positives beyond 45% at a nominal 5%.
**Grounds §8.1:** the specimen, not the joint, is the unit of inference. Joint-level pooling appears
only inside the mixed model, which models the nesting.

**Baayen, Davidson, Bates (2008). Mixed-effects modeling with crossed random effects for subjects
and items.** *J Mem Lang 59(4).* [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0749596X07001398)

Crossed random effects for two non-nested grouping factors. **Grounds §7 S7:** specimens and joints
are crossed (every joint on every specimen), with runs nested within specimen × arm.

**Bouthillier, X. et al. (2021). Accounting for variance in machine learning benchmarks.** *MLSys.*
[arXiv 2103.03098](https://arxiv.org/abs/2103.03098)

Seed and initialisation variance changes benchmark conclusions; a single run per method is not
evidence. **Grounds §6 (3 seeds per arm) and §8.7** (an effect is interpreted only above the seed SD).

---

## What the literature does NOT settle, and how the benchmark handles it

1. **No published joint-annotation repeatability for ant rigs.** The SESOI borrows the measured
   *surface*-landmark repeatability (V11). Interior joints are probably less repeatable, so the
   band is if anything conservative about calling strategies equivalent. Stated as a limitation.
2. **No accepted definition of an insect "joint centre" for body models.** Handled by
   pre-registering three model-side definitions and the per-joint bias analysis (S6), rather than
   claiming the rig pivot is anatomical.
3. **n = 11 is small by ML-benchmark standards and typical for 3D expert-landmark validation.** Power
   is stated in advance (§8.6), and INCONCLUSIVE is a legitimate outcome.

# Which registration strategies align the SMILify skeleton best with expert-annotated joints?

**Joint-Alignment Benchmark (JAB), 2026-09-14.** 11 real ant scans · 12 strategies (the shipped
recipe + 11 single-factor variants) · 3 optimiser seeds each · 396 fits · pre-registered
(`PREREGISTRATION.md`, frozen with SHA-256 before any strategy was run; deviations in `DEVIATIONS.md`).
Every number below is produced by `tools/analyze.py` from `data/`; nothing was computed by hand.

---

## Answer

1. **The shipped recipe D1_PROD is the best of the implemented annotation-free strategies. No
   single-factor change improves joint alignment.** None of the 11 contrasts reaches IMPROVES, and
   D1_PROD has the best mean rank (4.45 of 12; Friedman χ² = 24.4, p = 0.011). The median joint
   error is **25.1% of Weber's length** (95% CI 17.5–44.8).
2. **Removing any of four D1_PROD components makes joints worse in direction; three of them are also
   significant in the joint-level mixed model:**
   - hierarchical skeleton placement (+8.2% WL, 10/11 specimens worse; joint-level ×1.23)
   - the free-form offset/normal penalty (×1.25)
   - keeping pose free during surface fitting (×1.18)
   - with weaker evidence, authored joint limits (+4.0% WL, 9/11 worse)

   Only the hierarchical stage comes close to the pre-registered multiplicity bar (Holm p = 0.054).
   The others are consistent in direction but underpowered at n = 11.
3. **The scale cap, the allometric prior and extra anterior limits are joint-neutral**
   (EQUIVALENT within ±2.5% WL). Keep or drop them on other grounds.
4. **Joint alignment is won in one place: the hierarchical leg stage** (H0 → H1: −8.4% WL, 11/11
   specimens, p = 0.001). The three later steps add small gains, none individually significant
   (paired estimates −0.8, −3.0, −0.3% WL).
5. **Surface fit and skeleton accuracy dissociate.** Removing the offset/normal penalty gives the
   *closest* surface of any strategy (0.45× chamfer) and one of the *worst* skeletons. Chamfer must
   never be used to choose a strategy for joint accuracy.
6. **The remaining error belongs to the specimen and the anatomy, not the strategy.**
   - The same specimens fail under nearly every strategy: Eciton 68%, Discothyrea 45%, Formica 45% WL
     (median over strategies). The exception is CSE correspondence on Formica and Eciton (§5).
   - Coxae and trochanter/femur are placed at 12–13% WL; tibia/tarsus, antennae and mandibles at
     43–56% WL.
   - Joint-level variance: joint 0.36 and specimen 0.17, against 0.02 for the optimiser run (log scale).
7. **Expert joint supervision shows what is reachable, but it does not transfer.** Supervising all
   joints cuts error 26.9 → 17.4% WL (10/11). Supervising all regions *except* one leaves that region
   essentially unchanged, with every held-out CI including 0. This confirms the V-series conclusion
   under a corrected ruler: anatomical constraints act locally, so a strategy that improves
   distal legs, antennae or mandibles must supply information *about those regions*.

**What the evidence supports doing:** keep D1_PROD as the production recipe, and use this benchmark,
unchanged, as the acceptance gate for any future strategy. For the paper, report joint positions
region by region: proximal leg joints are usable (≈12% WL), while distal leg, antennal and mandibular
joints are not yet (43–56% WL).

---

## 1. Why a new ruler was needed (ground-truth audit)

The benchmark's first finding concerns the measurement, not the fitter. `figures/F1_gt_audit`,
`data/gt_registration_audit.json`.

| issue | evidence | handling |
|---|---|---|
| Old joint errors were measured in a frame fitted **to the production skeleton** (V9, `joint_to_obj_transform.json`) | that frame absorbs global errors (Fitzpatrick's FRE ≠ TRE) | GT registered through the scan mesh stored in each annotation `.blend`; residual ≤ 7e-8 of the diagonal; no fit involved. Production error: **42.7% → 25.1% WL** |
| 4 annotation scenes were **mirror images** (Aenictus, Aphaenogaster, Leptogenys, Odontomachus) | similarity det −1; labels follow the mirrored anatomy (12/12 posture cue) | reflect back **and** swap `_r`/`_l`; the production fit independently prefers the swap on 5/5 |
| Cephalotes was annotated on an older mesh | exact to `bench_10` mesh | ICP to the fit target, 0.15% diag median residual |
| **Dolichoderus was annotated on the Discothyrea mesh** | scene mesh = Discothyrea (8e-9); markers 92% inside it | excluded → **n = 11** (re-annotation attempt produced landmarks only; N1) |
| Surface-landmark `original` coordinates are Blender Z-up, not `.obj` | corrected points lie on the scan at ~1e-6 diag; uncorrected, 26–80% WL away | Weber's length (a distance) unaffected; **all earlier surface-landmark results (V6–V13, R-series, `landmark_indices_recalibrated.json`) need re-checking** |

## 2. Design

- **Estimand.** Per joint, ‖model joint − expert joint‖ / Weber's length, in the scan frame, with
  **no post-hoc alignment**. Summarised per specimen by the median over joints, averaged over seeds.
  Procrustes-aligned error is secondary: for D1_PROD it was *larger* than absolute error on 9/11
  specimens (median 35.5 vs 25.1% WL), because least-squares alignment spreads a few grossly
  misplaced legs over every joint (Pinocchio effect).
- **Model joint.** Forward-kinematics rotation pivots (`J_transformed + trans`). Two alternatives
  (regressed on final vertices; regressed on skinned vertices) change no verdict (sensitivity §6).
- **Joints.** 50 per full specimen (535 placed in total). Wings are absent on workers; `b_h` is
  degenerate in the rig. Regions: body axis, coxae, trochanter/femur, tibia/tarsus, mandibles, antennae.
- **Strategies.** Each changes exactly one factor of D1_PROD, using settings already established in
  the project (none tuned on these specimens):

| arm | factor removed / changed |
|---|---|
| A | D1_PROD (reference) |
| B | no hierarchical skeleton placement |
| C | no anatomical part partition in the skeleton stage |
| D | no authored joint-rotation limits |
| E | + anterior joint limits (135/165 axes) |
| F | no per-joint scale cap |
| G | allometric scale prior instead of the cap |
| H | learned leg-pose initialisation |
| I | learned dense correspondence (CSE head), both stages |
| J | no free-form offset / normal penalties |
| K | pose frozen during surface stages |
| L | separate distal leg groups |

- **Statistics.** Specimen = unit of inference. Each arm vs A: Hodges–Lehmann shift with exact
  Wilcoxon-inverted CI, exact p, Holm across 11 contrasts. EQUIVALENT only if the 90% CI lies within
  ±2.5% WL, the measured human landmark repeatability. Omnibus Friedman + Nemenyi; mixed model on log
  error with crossed specimen and joint effects; 6 sensitivity analyses. Power stated in advance: at
  n = 11 a contrast passes Holm only with ≈10/11 consistent specimens.
- **Controls passed.** Every checkpoint reproduces its saved vertices (max 1.2e-6). The fresh
  reference agrees with the historical production fits (median 25.1 vs 25.0% WL, Spearman 0.93
  across specimens). The oracle voiding control (re-optimisation with λ = 0) does not move error
  (HL −0.2 [−0.9, +0.5]). The learned-init and CSE inputs were confirmed active on all 11 specimens.

## 3. Primary result (`figures/F2_forest_primary`, `F3_paired_specimens`, `F9_critical_difference`)

Median joint error per arm, and paired change vs A (positive = worse).

| arm | median % WL | Δ vs A (HL) [95% CI] | better / worse | p exact | p Holm | verdict | joint-level ratio [95% CI] | chamfer vs A |
|---|---:|---|:-:|---:|---:|---|---|---:|
| **A** D1_PROD | **25.1** | — | — | — | — | reference | — | 1.00× |
| B no hierarchical | 37.9 | +8.2 [+3.1, +13.8] | 1 / 10 | 0.005 | 0.054 | inconclusive | **1.23 [1.13, 1.35]** | 1.19× |
| C no partition | 26.0 | +0.5 [−2.6, +3.9] | 5 / 6 | 0.64 | 1.00 | inconclusive | 0.97 [0.89, 1.06] | 0.97× |
| D no limits | 29.9 | +4.0 [+0.8, +7.1] | 2 / 9 | 0.014 | 0.14 | inconclusive | 1.05 [0.96, 1.15] | 0.96× |
| E + anterior limits | 27.6 | +1.3 [+0.1, +2.5] | 3 / 8 | 0.024 | 0.22 | **equivalent** | 1.02 [0.93, 1.11] | 0.99× |
| F no scale cap | 26.0 | +0.5 [−0.3, +1.4] | 4 / 7 | 0.28 | 1.00 | **equivalent** | 1.01 [0.93, 1.10] | 1.00× |
| G allometric prior | 25.8 | +0.0 [−1.3, +1.8] | 6 / 5 | 1.00 | 1.00 | **equivalent** | 0.99 [0.90, 1.08] | 1.00× |
| H learned pose init | 38.2 | +0.7 [−1.2, +15.2] | 4 / 7 | 0.37 | 1.00 | inconclusive | 1.07 [0.98, 1.17] | 1.00× |
| I CSE correspondence | 25.3 | +2.3 [−11.2, +11.9] | 3 / 8 | 0.52 | 1.00 | inconclusive | 1.08 [0.99, 1.18] | 2.97× |
| J no offset/normal | 32.7 | +5.0 [−0.3, +9.8] | 3 / 8 | 0.083 | 0.58 | inconclusive | **1.25 [1.15, 1.37]** | 0.45× |
| K pose frozen | 29.3 | +3.8 [−0.7, +7.1] | 2 / 9 | 0.067 | 0.54 | inconclusive | **1.18 [1.08, 1.29]** | 2.18× |
| L split distal | 30.8 | +0.9 [−1.2, +4.8] | 5 / 6 | 0.37 | 1.00 | inconclusive | 1.06 [0.97, 1.15] | 1.01× |

Mean rank (lower is better): A 4.45 · G 4.64 · F 5.27 · C 5.45 · H 5.73 · L 5.91 · I 6.36 · E 6.45 ·
D 7.55 · K 8.00 · J 8.45 · B 9.73 (Nemenyi CD = 5.02).

**Reading "inconclusive" correctly.** For B, D, J and K every sign points the same way (worse), the
joint-level mixed model confirms B, J and K, and none improves on more than 3/11 specimens.
"Inconclusive" means the pre-registered specimen-level test at n = 11 could not rule out chance
after correcting for 11 comparisons. It does not mean "no effect". For H and I the wide intervals
are the result: both are **high-variance strategies**, not neutral ones (§5).

## 4. Where in the anatomy and where in the pipeline

**Regions** (`F4_region_effects`, `F8_joint_error_skeleton`; exploratory, no multiplicity claim).
A's median error by region: coxae 11.6 · trochanter/femur 12.9 · body axis 24.1 · mandibles 42.8 ·
antennae 54.9 · tibia/tarsus 56.3% WL. Every region contrast whose 95% CI excludes 0 is a
*worsening*, all at the proximal leg:
- B: coxae +5.2, trochanter/femur +12.9
- J: coxae +4.4, trochanter/femur +8.7
- K: coxae +4.7

No strategy improves any region with a CI excluding 0.

**Pipeline stages** (`F7_stage_progression`), within A, paired per step:

| step | Δ median % WL [95% CI] | specimens improved |
|---|---|:-:|
| H0 body → H1 legs | **−8.4 [−11.6, −5.4]** | **11/11** (p = 0.001) |
| H1 → H2 joints | −0.8 [−2.1, +0.8] | 6/11 |
| H2 → surface coarse | −3.0 [−6.4, +1.0] | 8/11 |
| coarse → fine | −0.3 [−1.2, +0.1] | 9/11 |

The leg stage is the decisive step, which is why removing the hierarchical stage (B) hurts proximal
legs most. Arm H shows the reverse: its learned initialisation reaches the same error as A after the
leg stage (34.5 vs 34.3% WL), then loses ground during surface fitting (H2 32.5 → coarse surface 39.4%
WL, median over specimens).

## 5. Surface cost and heterogeneous strategies

**Surface vs skeleton** (`F6_accuracy_vs_chamfer`).
- J halves the chamfer (0.45×) but inflates free-form deformation 9× and worsens the skeleton ×1.25:
  the surface is fitted by deforming the skin instead of moving the skeleton.
- K (2.18×) and I (2.97×) raise the chamfer.
- No strategy reduces both errors.

**CSE correspondence (I) is not neutral, it is bimodal** (`F13_qualitative_cse`). Its seed SD is the
lowest of all arms (0.6% WL), so the effect is systematic per specimen, not noise:
- large wins: Formica 44.8 → 14.5, Eciton 68.6 → 47.1, Cataglyphis 25.1 → 20.2
- large losses: Cephalotes 23.5 → 69.7, Aphaenogaster 13.3 → 25.3

**Learned pose initialisation (H)** is neutral on 9 specimens and catastrophic on 2 (Cyphomyrmex
22.6 → 55.6, Cataglyphis 25.1 → 38.2), consistent with the earlier bench50 finding that learned
init loses to zero init on real scans.

## 6. Robustness (`results/results.json` → `sensitivity`)

No sensitivity analysis turns any contrast into IMPROVES or WORSENS. Under every definition the
signs of B, D, J and K stay positive and G stays ≈ 0.

| analysis | what changes |
|---|---|
| regressed joints (REG) / skinned joints (SKIN) | no verdict changes; B and K Holm p ≈ 0.11 under both, J 0.11 (REG) / 0.43 (SKIN) |
| mean instead of median | effects shrink (outlier joints dilute); D becomes equivalent |
| centroid-size normalisation | B strongest (+11.7, Holm 0.08); E and G move to inconclusive |
| excluding the ICP-registered specimen | nothing |
| excluding the 4 mirrored specimens (n = 7) | power drops; no verdict flips to a new direction |

**Systematic bias (ISO 5725 trueness).** Subtracting each joint's leave-one-specimen-out mean
offset makes error worse (median 24.9 → 37.4% WL). The large offsets are specimen-specific, not a
constant disagreement between the rig's pivot and the annotator's joint definition. So residual
error cannot be removed by a per-joint correction table. Caveat: the local anatomical frame is least
reliable on Leptogenys and Eciton.

## 7. Oracle references (`F12_oracle_transfer`; use expert joints at fit time, never ranked)

| supervision | effect |
|---|---|
| O0 control, λ = 0 | 26.9 → 27.3% WL, HL −0.2 [−0.9, +0.5]: re-optimisation alone does nothing |
| O1 all joints, λ = 0.03 | 26.9 → **17.4%** WL, HL −11.9 [−17.9, −5.7], 10/11 |
| O2 all regions except r, scored on r | body axis +0.1 [−1.4, +5.6] · coxae +1.0 [−3.7, +2.3] · trochanter/femur +1.0 [−3.6, +5.5] · tibia/tarsus −2.6 [−7.7, +2.4] · mandibles −1.5 [−6.9, +2.8] · antennae +0.1 [−3.7, +4.1] |

Where supervision helps most (tibia/tarsus 54 → 20, antennae 53 → 29% WL), holding that region out
recovers 15% and 6% of the gain. The skeleton's worst regions are therefore not reachable from
correct information elsewhere in the body.

## 8. Pre-registered predictions vs outcome

| arm | predicted | observed | |
|---|---|---|---|
| B | worsens | worse 10/11, Holm 0.054; mixed model confirms | direction ✓, significance ✗ (marginal) |
| C | worsens | inconclusive, HL +0.5 | ✗ |
| D | equivalent | worse 9/11, HL +4.0, not significant | ✗ (worse than predicted) |
| E | equivalent | equivalent | ✓ |
| F | equivalent; anterior worse | equivalent; mandibles +0.9 (CI includes 0) | ✓ / not shown |
| G | equivalent | equivalent | ✓ |
| H | equivalent or worsens | inconclusive; catastrophic on 2/11 | partly ✓ |
| I | inconclusive; coxae improve | inconclusive; coxae +2.2 (not improved); bimodal | ✓ / ✗ |
| J | worsens | worse 8/11; mixed model confirms | direction ✓ |
| K | worsens | worse 9/11; mixed model confirms | direction ✓ |
| L | equivalent | inconclusive, HL +0.9 | ✗ (not shown equivalent) |

## 9. Limitations

- **n = 11 specimens, 5 subfamilies, no Dolichoderinae.** Small effects are undetectable by design (§2).
- **No joint-annotation repeatability exists.** The ±2.5% WL band is the *surface*-landmark
  repeatability; interior joints are probably less repeatable, which makes the equivalence verdicts
  conservative.
- **Laterality of 4 mirrored specimens is inferred** (strong support on 4/5; weakest on Leptogenys).
- **One-factor ablations do not test interactions.** Combinations such as "CSE where it helps" were
  deliberately not tuned on these 11 specimens; doing so would turn the test set into a training set.
- **The oracle uses a single λ from earlier work** (0.03, V5b/V9) and one seed.

## 10. Reproduce

```bash
python diagnostics/joint_alignment_benchmark/tools/gt_registration.py       # GT → fitter frame (needs .blend dumps)
sbatch diagnostics/joint_alignment_benchmark/submit_arms.sbatch              # 12 arms × 3 seeds
sbatch diagnostics/joint_alignment_benchmark/submit_oracle.sbatch            # O0/O1/O2
python diagnostics/joint_alignment_benchmark/tools/score.py --legacy
python diagnostics/joint_alignment_benchmark/tools/analyze.py
python diagnostics/joint_alignment_benchmark/tools/figures.py
```

Fits: `/hpcwork/nao48500/jab_runs/`. Code state: HEAD `302d1439` + `data/code_state_uncommitted.patch`.
Methods literature: `LITERATURE.md`.

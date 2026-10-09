# Pre-registration — Joint-Alignment Strategy Benchmark (JAB)

**Frozen:** 2026-09-14, before any strategy arm was fitted.
**What had been seen before freezing:** (i) the ground-truth audit (§3), (ii) the *production*
fit's joint error on the 11 specimens under the new and old rulers (`probes/instrument_audit_PROBE_out.txt`).
No arm other than the pre-existing production fit had been run or scored. Any deviation from this
document is recorded in `DEVIATIONS.md` with its date and reason, and marked in the report.

---

## 1. Question

Fabian (paraphrased from his message): *"the bigger picture is still some validation on which
strategies lead to the best model (specifically joint) alignment with the ground-truth
hand-annotated joints."*

**Operational question.** Among the registration strategies that SMILify already implements and
that need no annotation at inference time, which produce model joint positions closest to
independently hand-annotated anatomical joints on real ant scans? Which anatomical regions does
each strategy help or hurt, and at what cost in surface fit?

**Not in scope:** developing new methods, tuning hyperparameters against these 11 specimens (the
benchmark set is a test set, see §9), or morphometric trait validation (Track B).

## 2. Estimand

For specimen *i*, joint *j*, strategy *a* and optimiser seed *k*:

  e_ijak = ‖ x̂_ijak − x_ij ‖ / WL_i × 100          (% of Weber's length)

- x̂ is the model joint (§5.1) in the fitter's normalised scan frame.
- x is the expert joint, placed in that same frame **without consulting any fit** (§3).
- WL_i is Weber's length from the human trait landmarks (the ant-morphometric body-size
  standard; a distance, so unaffected by the landmark-frame defect in §3.4).

**Primary per-specimen score:** S_ia = mean over seeds of the median over evaluated joints.
**Primary arm summary:** the median over specimens of S_ia, with a 95% specimen-bootstrap CI.

No post-hoc alignment is applied for the primary endpoint. Procrustes-aligned error hides global
misplacement and spreads local failures across joints (Pinocchio effect). The instrument audit
confirmed this on the production fit: PA error exceeded absolute error on 6/11 specimens. PA error
is kept as a secondary, articulation-only view.

## 3. Ground truth — frozen after audit

Source: `annotation/gt_expert/*_joints.json` (authoritative, user-confirmed 2026-09-07).
Registration: `tools/gt_registration.py` → `data/gt_joints_fitframe.json`, audit in
`data/gt_registration_audit.json`.

3.1 **Fit-independent frame.** Each `.blend` stores the scan mesh in annotation coordinates.
The mesh→`.obj` similarity is solved with vertex correspondence (residual ≤ 7e-8 of the diagonal).
Then comes `load_meshes`' normalisation. Earlier work solved the frame against the production
skeleton (V9; `annotation/joint_to_obj_transform.json`), which biases comparisons toward that fit
and raised production's apparent error from 25.1% to 42.7% WL.

3.2 **Mirrored scenes (4).** Aenictus, Aphaenogaster, Leptogenys and Odontomachus were annotated
in a mirror-imaged scene (det −1). Points are reflected back and `_r`↔`_l` are swapped. Evidence:
a posture-robust chirality cue is positive for all 12 in the annotation frame. The pre-existing
production fit prefers the swap on 5/5 mirrored/chained specimens, by 11–35% vs 53–88% WL.

3.3 **Tiers.** EXACT (7), MIRRORED (4, exact reflection), CHAIN (Cephalotes: annotated on the
older `bench_10` mesh; exact to that mesh, then ICP to the fit target at 0.15% diag median).
**EXCLUDED: Dolichoderus_cf.bidens_CASENT0744033.** Its scene mesh is the Discothyrea scan and its
markers lie inside that mesh, so it is not an annotation of Dolichoderus. **n = 11 specimens.**

3.4 **Landmark frame defect (affects WL source only).** `annotation/landmarks/*_traits.json`
`original` is in Blender Z-up coordinates, not `.obj`: obj = (x, z, −y). This is corrected here,
and the WL endpoints then lie on the scan. WL itself is rotation-invariant. The defect invalidates
earlier surface-landmark *positions* (V6–V13, R-series); that is reported, not repaired, here.

3.5 **Annotation provenance.** The JSON is authoritative where it post-dates the `.blend`
(Aenictus `l_1_ta_r`; Eciton `b_h`, `ma_*`, committed 2026-09-07). Edited points were verified to
lie on or inside the scan.

## 4. Evaluated joint set (frozen before arms)

All placed GT joints, except:
- **wing joints** `w_{1,2}_{r,l}` — never placed (absent on workers);
- **`b_h`** — structurally degenerate in the model (`b_h` ≡ `b_t` at every fit), so its error
  would measure the rig definition, not the strategy. Reported descriptively only.

All 535 placed joints carry visibility `clear`, so no visibility stratification is possible.

**Regions** (by joint name): body_axis `b_t, b_a_1..5` · coxa `l_*_co_*` · leg_proximal
`l_*_tr_*, l_*_fe_*` · leg_distal `l_*_ti_*, l_*_ta_*, l_*_pt_*` · mandible `ma_*` ·
antenna `an_*`.

## 5. Model-side definitions

5.1 **Model joint.** PRIMARY = forward-kinematics pivots (`J_transformed + trans`), the rig's
actual rotation centres. SECONDARY = `J_regressor` on the final deformed vertices (G/V-series
definition). The two differ by a median 0.6% WL on production (audit), so the choice is not
expected to matter; both are reported.
5.2 **Loading guard.** Parameters are shape-checked. Vertices regenerated from the loaded
parameters must reproduce the saved `verts` (< 1e-4 unit box), or the fit is refused.

## 6. Strategy arms

Reference **A = D1_PROD**, the shipped recipe (`diagnostics/SHIPPED_RECIPE.md`): hierarchical
skeleton placement (`--midline 2.0 --beta_prior 0.0 --limit 0.273 --offset 30.0 --skip_h3`)
followed by `optimise_moonshot` with `cfg/D1_PROD.yaml` (includes `w_scale 0.052`).

Every other arm changes **one** factor relative to A (a one-factor-at-a-time ablation), so each
contrast attributes an effect to one design decision.

| arm | factor | exact change vs A | origin in project |
|---|---|---|---|
| B `nohier` | hierarchical skeleton placement | moonshot D1_PROD from zero init, no hierarchical stage | Z2/X2 protocol |
| C `nopartition` | anatomical part partition in skeleton stage | hierarchical `--no_partition` | M1_nopart |
| D `nolimits` | authored joint-rotation limits | `--limit 0`, `w_limit 0` both stages | LIM_0 |
| E `anteriorlimits` | extra anterior limits (135/165 axes) | model `OmniAnt_25PCs_anterior_limited.pkl` | Z2_B |
| F `noscalecap` | per-joint scale barrier | `w_scale 0` (pre-2026-08-13 production) | X1_A |
| G `allometric` | allometric scale prior instead of cap | `w_scale 0`, `w_allo 0.00051` | X2_C |
| H `learnedinit` | learned leg-pose initialisation | `--init_joint_rot_from` learned (lowpose05 ckpt) | bench50_G1d |
| I `cse` | learned dense correspondence (CSE head) | `--cse_correspondence_from` in both stages (C3 head, cycle filter, keep 0.5, all segments, `--normalise`) | C8 / H_A |
| J `nooffnorm` | free-form offset + normal penalties | `w_offset 0`, `w_normal 0` both moonshot stages | A1/A3 |
| K `posefrozen` | pose refinement during surface stages | `scheme: deform` in both moonshot stages | C0_control |
| L `splitdistal` | separate distal leg groups | hierarchical `--split_distal` | EXPERIMENTAL_DEFAULTS |

**Seeds:** 0, 1, 2 for every arm; the fitter is not deterministic (~0.01 run-to-run, Z4/Z5).
**Inputs:** the 11 fit-target meshes from `/hpcwork/nao48500/worker_alt_data`, batched together.
Code state: this commit, recorded per run.

### 6.1 Reference analyses (not candidates; exploratory)
- **Stage progression** within A: H0_body → H1_legs → H2_joint → Stage_2 → Stage_3.
  Where in the pipeline is joint alignment gained or lost?
- **Oracle ceiling O1** — joint supervision on all evaluated joints (λ = 0.03, 800 its, from A's
  seed-0 fit; V9 arm C protocol). Circular by construction: it shows *reachability*.
- **Oracle generalisation O2** — leave-one-region-out joint supervision: supervise all regions but
  *r*, evaluate on *r*. Shows whether a partial skeleton would transfer.
O1/O2 use the ground truth at fit time and are never ranked against deployable strategies.

## 7. Metrics

**Primary:** §2 (FK, absolute, % WL, per-specimen median, seed-averaged).
**Secondary** (all per arm and per region):
- S1 PA error (per-specimen similarity Procrustes on evaluated joints)
- S2 PCK@τ, τ ∈ {5, 10, 15, 20, 25}% WL, and AUC of PCK over τ ∈ [0, 50]
- S3 region medians; arm × region matrix of paired effects vs A
- S4 bone-length error: |ℓ̂ − ℓ| / ℓ for GT-placed parent→child segments on the kinematic tree
- S5 side confusion: share of bilateral joints closer to the *contralateral* GT joint
- S6 systematic bias (ISO 5725 trueness): per-joint mean signed error in a specimen-local
  anatomical frame (anterior, dorsal, right), pooled over specimens, leave-one-specimen-out
- S7 variance decomposition: linear mixed model on log error, crossed random intercepts for
  specimen and joint, arm as fixed effect, seed as residual level; seed SD vs arm effects
- S8 safety: bidirectional chamfer (normalised frame) and `deform_verts` RMS
**Sensitivity:** REG joint definition · excluding CHAIN specimen · excluding MIRRORED specimens ·
mean instead of median · normalising by GT body-axis centroid size instead of WL.

## 8. Statistics and decision rules

8.1 **Unit of inference: the specimen (n = 11).** Joints within a specimen are not independent
(Saravanan et al. 2020): pooling joints as observations inflates Type-I error.
8.2 **Primary contrasts: each arm B–L vs A (family = 11 contrasts).** Paired differences
d_i = S_i,arm − S_i,A. Report the Hodges–Lehmann estimate with the exact Wilcoxon-inverted 95% CI,
the exact two-sided signed-rank p, the win count k/11 (sign test), and Holm-adjusted p across the
11 contrasts (Demšar 2006).
8.3 **Smallest effect size of interest (SESOI): Δ = 2.5% WL.** This matches the measured human
surface-landmark repeatability (V11: 2.4% WL; distances within a frame are unaffected by the §3.4
defect). An improvement smaller than annotation precision is not practically meaningful.
8.4 **Classification of each contrast:**
- **IMPROVES** — Holm p < 0.05 and HL < 0.
- **WORSENS** — Holm p < 0.05 and HL > 0.
- **EQUIVALENT** — 90% CI within (−Δ, +Δ) (TOST; Lakens 2017).
- **INCONCLUSIVE** — otherwise.
8.5 **Omnibus:** Friedman test over all 12 arms on S_ia, with a critical-difference diagram
(Nemenyi) as a visual summary only.
8.6 **Power, stated up front.** With n = 11 the smallest exact two-sided Wilcoxon p is 0.00098. The
Holm threshold for the strongest contrast is 0.05/11 = 0.0045, reached only with ≥ 10/11
consistent signs of broadly consistent rank. Small effects will be INCONCLUSIVE; that is a property
of n = 11, not evidence of no effect, and is reported as such.
8.7 **Seed noise:** an arm effect is only interpreted if |HL| exceeds the within-arm seed SD of S.
8.8 **Regions:** region-level contrasts are exploratory (no multiplicity claim), shown with CIs.

## 9. Guards against bias

- No arm is tuned on these specimens; every setting is the project's existing one (§6 table).
- The GT frame is independent of every arm (§3.1).
- The same 11 meshes, batch, iterations and seeds for all arms.
- The report must show per-specimen paired lines, not only summaries.
- If an arm crashes or produces a non-reproducing checkpoint, it is reported as failed, never
  silently dropped.

## 10. Pre-registered predictions (from prior project evidence)

| arm | prediction | basis |
|---|---|---|
| B nohier | WORSENS (legs) | hierarchical stage exists to place the skeleton |
| C nopartition | WORSENS (legs) | M-series partition gains |
| D nolimits | EQUIVALENT overall | limits bound rotations, not positions (G6) |
| E anteriorlimits | EQUIVALENT | Z2 null |
| F noscalecap | EQUIVALENT overall; anterior worse | X1 |
| G allometric | EQUIVALENT | X2 mechanism fail |
| H learnedinit | EQUIVALENT or WORSENS | G1 lost to zero-init on real scans |
| I cse | INCONCLUSIVE overall; coxa improves | H_A coxa fix, W5 deployment failure |
| J nooffnorm | WORSENS | A3 guard-broken |
| K posefrozen | WORSENS | A4/EXPERIMENTAL_DEFAULTS |
| L splitdistal | EQUIVALENT | Task 7 null on synthetic correspondence |

## 11. Literature grounding (methods)

Target vs fiducial registration error (Fitzpatrick 2009) · Procrustes and Pinocchio-effect
limitations (von Cramon-Taubadel et al. 2007; arXiv 2409.16861 (2024) on PA-MPJPE) ·
measurement error in morphometrics (Fruciano 2016) · automated vs manual landmark validation
(Percival et al. 2019; Porto et al. 2021) · trueness/precision (ISO 5725-1) · PCK/PA-MPJPE for
animal body models (Animal3D, Xu et al. 2023) · classifier/method comparison across datasets
(Demšar 2006; Benavoli et al. 2017) · seed variance (Bouthillier et al. 2021) · hierarchical data
(Saravanan et al. 2020) · equivalence testing (Lakens 2017) · anatomical vs graphics joint
definitions (Keller et al. 2023, SKEL). Full references and URLs: `LITERATURE.md`.

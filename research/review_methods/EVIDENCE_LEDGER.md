# Evidence ledger

Which prior results may enter the paper's evidence chain. Built 2026-10-06 from the result
documents on disk. Numbers were copied from those documents; re-verify against the source before
quoting any of them in the paper.

## A. Admissible evidence

| id | source on disk | what it establishes | data | caveats that travel with it |
|---|---|---|---|---|
| **JAB** | `diagnostics/joint_alignment_benchmark/REPORT.md`, `data/scores_specimen.csv` | D1_PROD 25.1% WL median FK joint error (n = 11 real, expert GT, fit-independent frame). No single-factor change improves it. Surface and skeleton dissociate (arm J: 0.45x chamfer, x1.25 joint error). Region errors: coxa 12, tr/fe 13, body 24, mandible 43, antenna 55, ti/ta 56% WL. Supervising all joints 26.9 -> 17.4%, which does not transfer to a held-out region. | real | n = 11; Dolichoderus excluded; 3 optimiser seeds |
| **JAB I_cse** | same, arm `I_cse` | CSE correspondence on real scans is specimen-dependent (seed means, % WL): Formica 44.8 -> 14.5, Eciton 68.6 -> 47.1, Cataglyphis 25.1 -> 20.2; Cephalotes 23.5 -> 69.7, Aphaenogaster 13.3 -> 25.3; the other six within about +-8 | real | one CSE checkpoint (C3), cycle filter keep 0.5 |
| **M2** | `diagnostics/morphometric_validation/RESULTS_M2_synthetic_recovery.md` | On P48 (noise 0, exact solution exists), trait recovery error is 4-25% median (HW 4.79). No diagnostic predicts it: chamfer \|r\| <= 0.34, oracle correspondence/pose/beta error \|r\| <= 0.41. | synthetic | recipe D1_no_deform, not D1_PROD. **The memory note links M2 to R9; that interpretation is void. Only M2's own measurements are admissible.** |
| **C4 / cycle** | `diagnostics/anatomical_pose_init/RESULTS_C4_cse_fit_20260827.md` | Predicted CSE correspondence fed as direct vertex identity recovers ~84% (cycle filter: 92%) of the dense-GT-oracle seg_acc gain on synthetic; the centroid/IK conversion was the bottleneck. leg_acc unmoved. Cosine confidence is anti-correlated with correctness; cycle consistency works. | synthetic, n = 12 | arm-vs-arm comparisons n.s. at n = 12 |
| **C14p** | `RESULTS_C14p_gtinit_ceiling_20260828.md` | Ground-truth pose initialisation does not move the coxa (co 0.4204 zero-init vs 0.4114 GT-init); it does fix distal segments without CSE. Coxa floor on an exact mesh is 0.0908 (E0). | synthetic P48 | read co against the 0.09 floor |
| **HA2** | `RESULTS_HA2_deform_off_20260828.md` | With `deform_verts` exactly zero, the CSE coxa gain survives (co 0.4115 -> 0.2226, 43/48), and coxa placement improves 2.3x on 45/48. Free-form deformation was not the mechanism. | synthetic P48 | registered verdict STILL-DECOUPLED; placement measurement was post hoc |
| **F1** | `RESULTS_F1_circumferential_probe_20260828.md` | Circumferential information reaches the head: ridge probe error 52.89 deg at `fp1` vs 90 deg chance (shuffled control 89.1 deg). Its pre-registered reading: the failure is in loss/retrieval, so **neither an equivariant retrain nor an attention decoder is justified**. Directly relevant to M03. | synthetic | verified 2026-10-06: the reported run loaded strictly (207 tensors); the earlier random-weights run was caught and replaced before reporting |
| **F2** | `RESULTS_F2_trained_selector_20260828.md` | A learned per-point re-ranker buys 2.5-4.2% of an oracle@5 gap; per-point independent selection is a dead end | synthetic | |
| **F4** | `RESULTS_F4_F5_20260828.md` | Sinkhorn/OT on retrieval: 4.3% (tr) / 4.5% (fe) relative, 35/40 and 33/40, below the 5% bar. Global assignment is worth ~4.5%, not ~30%. | synthetic | bar had no PARTIAL band |
| **F5** | same | Conformal fit-quality calibration fails its bar | synthetic | |
| **F6** | `RESULTS_F6_soft_targets_20260828.md` | Geodesic-soft training targets: -3.0% median 3D error, smaller than epoch-to-epoch noise | synthetic | one run |
| **F7** | `RESULTS_F7_embedding_geometry_20260828.md` | Key table uses 14.8/16 dims; crowding is entirely local; 0.0001 of leg vertices have a nearest key on another leg. Capacity is not the bottleneck. | synthetic | |
| **W1-W5** | `diagnostics/within_part_correspondence/RESULTS_W*.md` | Within a correct part, handcrafted, anatomy-normalised and learned descriptors all lose to part-local rigid proximity; normalising part proportions destroys signal (W1). W3b beats it only on coxa (+25.2%), and fails the deployment gate (W5: 0/48, 1.8% template coverage). | synthetic | |
| **R11** | memory `project_r11_semantic_laterality_closes.md` | DINO lateralised identity 82.6% vs rigid proximity 86.1%; both collapse under rotation | real/synthetic | verified 2026-10-06: R11 used synth_power48 template part labels, no landmark coordinates, so the frame defect cannot apply. Fixed render cameras: realistic regime is 0-30 deg |
| **Z4/Z5** | `diagnostics/deform_nature/` | `deform_verts` is per-specimen noise; fitter is non-deterministic (~0.01 gen@20 run to run) | worker scans | |
| **Anatomical init ceiling** | `diagnostics/anatomical_pose_init/out_ceiling_20260820/RESULTS.md` | Two analytic leg-pose initialisers are worse than zero init; `geom_leg_init.py` chain-IK is broken | synthetic | |

## B. Excluded (invalidated; provenance only, never in the evidence chain)

| id | why excluded | audit |
|---|---|---|
| G1, G3, G6-G10 | used the superseded `gt_batch1` annotations | `diagnostics/gt_contamination_audit_20260916/AUDIT_20260916.md` |
| R1-R9 (incl. R3, R9), A1, A3 | supervision targets read in the Blender Z-up frame, median 50.3% WL off anatomy | same |
| V1, V6-V13 | same frame defect; `landmark_indices_recalibrated.json` 1.54x wrong | same |
| slide8 "score can lie" rebuild | same frame defect, and circular | same |
| M4 scale-free traits | WL denominator from the contaminated recalibration | same |

Conclusions that rested only on excluded items do not exist for this programme, including when they
point the same way as an admissible result.

## C. Claims to avoid (not established by admissible evidence)

- "Perfect correspondence does not eliminate the real-scan error." No dense-correspondence oracle
  exists on real scans. The admissible real-scan oracle is joint supervision (JAB: 26.9 -> 17.4% WL).
- "The objective prefers the wrong anatomy." This was R9/A1, both void; JAB arm J points the other
  way.
- Any capacity claim of the form "the model can hold all landmarks at 0.2%": that was V12, excluded.

## Q1: is the problem actually correspondence? (admissible evidence only)

1. **On synthetic, correspondence is largely solved for what the fitter needs.** C4/cycle recover
   92% of the dense oracle's gain, F7 finds the embedding structure correct, and C14p shows GT pose
   leaves distal error small. The remaining synthetic gap is the coxa, and part of it is a 0.09
   metric floor.
2. **On real scans, correspondence is neither solved nor reliably helpful.** JAB I_cse ranges from
   -30.2 to +46.1% WL per specimen. Distal legs, mandibles and antennae sit at 43-56% WL under every
   annotation-free strategy.
3. **Supplied anatomical information helps only locally** (JAB oracle does not transfer across
   regions). So any successful source of anatomy has to be dense over the regions that fail.
4. **What the admissible evidence cannot answer** is *why* real-scan transfer fails, and which
   information would close the real-scan gap. That is Q2 (oracle budget on degraded synthetic) and
   Q3 (the JAB case study, then controlled shifts).

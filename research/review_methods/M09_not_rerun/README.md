# Review ideas not re-run, and the evidence that closes each

A written accounting, not an experiment. Each idea from the external review was checked against
admissible evidence (`../EVIDENCE_LEDGER.md` §A) before deciding not to spend compute on it again.
Where the existing test differs from what the review proposed, the difference is stated, and the
untested part is routed to the folder that owns it.

| review idea | already tested as | what it showed | untested remainder | where it goes |
|---|---|---|---|---|
| Soft correspondence / optimal transport | **F4** (Sinkhorn on the C11 head's retrieval) | 4.3% (tr) / 4.5% (fe) relative gain, 35/40 and 33/40 specimens, below the pre-set 5% bar; assignment not degenerate (entropy 4.31, 1.19x vertex coverage) | uncertainty used **inside the fitting objective** (expected targets, confidence weights) | `M02_soft_uncertainty_fitter/` (gated on Q2/Q3) |
| Soft targets for training | **F6** (geodesic-softened InfoNCE targets) | −3.0% median 3D error, smaller than epoch-to-epoch noise of one run | none worth funding | closed |
| Reduce deformation freedom (review experiment E) | **HA2** (`deform_verts` exactly 0, synthetic); **JAB arm J** (offset/normal penalty removed, real); **Z4/Z5** (structure of deform) | synthetic: the CSE coxa gain survives without deform (co 0.4115 → 0.2226, 43/48); real: freer surface gives the best chamfer (0.45x) and a worse skeleton (x1.25); deform is per-specimen noise | the *absorption* reading from Q3a A5 (every free channel inflates under CSE on real scans) is new and is tested causally in Q3b S1 | `../Q3b_controlled_shift/` S1; dose curve only as cleanup in `M07_deformation_dose/` |
| Skeleton-first / pose initialisation (review idea 7) | **C14p** (ground-truth pose init, synthetic); **anatomical init ceiling test** (two analytic initialisers); **JAB arm H** (learned init, real) | GT pose init does not move the coxa (0.4204 → 0.4114) though it fixes distal segments; both analytic initialisers are worse than zero init; learned init catastrophic on 2/11 real specimens | skeleton used as **input to correspondence** rather than as an initialiser | `M01_skeleton_conditioned/` (gated) |
| Intrinsic / anatomical fingerprints (idea 9) | **W1** (handcrafted local geometry, part-normalised coordinates), **W2** (C11 descriptor), **W5** (deployment gate), **R11** (DINO) | all lose to part-local rigid proximity *within an already-correct part*; normalising part proportions destroys signal (W1) | none at within-part level; the review's version as a scan-side feature needs joints and becomes M01's anatomical-coordinate arm | `M01_skeleton_conditioned/` arm D |
| DiffusionNet / functional maps (idea 10) | Z1 retired the functional-map programme | real scans are fragmented (17-193 connected components per JAB specimen, Q3a A6), so intrinsic operators on the scan are ill-defined | none | closed; reason stated in the paper |
| Plain transformer replacing PointNet++ | **F1** | circumferential information already reaches the head (52.9° vs 90° chance); its pre-registered reading: neither an equivariant retrain nor an attention decoder is justified | a cross-attention model justified by a *specific* relational hypothesis that Q3 produces | `M03_cross_attention/` (not committed) |

## Ideas the review raised that this programme does run

- Oracle information budget (on degraded synthetic): `../Q2_oracle_budget/`.
- Leave-one-morphology-out: folded into `../Q3b_controlled_shift/`. Q3a found real shapes *inside*
  the training shape distribution, so a full retraining LOMO runs only if S2's SHP / REALSHAPE
  conditions show a morphology effect.
- CPD / NICP baselines: `../M08_classical_baselines/`.

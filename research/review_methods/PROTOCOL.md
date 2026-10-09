# Evaluation protocol: review methods programme

Written 2026-10-06, before any new method was run on any data. Frozen by `PROTOCOL.FREEZE`
(SHA-256 of this file and of every file listed in §6). Any later change goes in `DEVIATIONS.md`
with date and reason; this file is not edited after the freeze.

## 1. Four metric families, never merged

The project's central empirical fact is that these come apart (JAB arm J: best surface, worse
skeleton). Every experiment reports each family it can measure **separately**. None is a proxy for
another, and no method is selected on G.

| family | quantity | synthetic | real (JAB) |
|---|---|---|---|
| **C** dense correspondence | (a) pre-fit retrieval: per scan point, distance between the retrieved and the true template vertex, in canonical rest space, as a fraction of that segment's length, per segment. (b) post-fit: leg-level and per-segment misassignment rate (`diagnostics/correspondence_accuracy/confusion.py`, the C13 scorer). | yes | **not measurable** (no dense GT) |
| **S** skeleton | FK rotation pivots `J_transformed + trans` vs ground truth. | per-joint error / GT body length L (below), by JAB region | `med_fk`, % Weber's length, JAB `tools/score.py`, by region |
| **G** registration geometry | chamfer | `diagnostics/moonshot/eval_run.py` | JAB `tools/score.py` |
| **M** morphometric | head width HW (`diagnostics/groundtruth/trait_extract.py`, extent of part `b_h`; the only trait that survived the 2026-09-16 audit unaffected) | vs GT mesh | reported, not a decision endpoint |

Synthetic body length **L** = bounding-box diagonal of the ground-truth mesh. Chosen because it needs
no landmark indices: the recalibrated Weber's-length indices were found 1.54x wrong (audit
2026-09-16). Regions are JAB's `region()`: body_axis, coxa, leg_proximal (tr, fe), leg_distal
(ti, ta, pt), mandible, antenna. Excluded joints as in JAB (`w_*`, `b_h`).

## 2. Frozen baseline: D1_PROD

D1_PROD is exactly JAB arm `A_prod`: `optimise_hierarchical` with
`--midline 2.0 --beta_prior 0.0 --limit 0.273 --offset 30.0 --skip_h3`, then `optimise_moonshot`
with `diagnostics/joint_alignment_benchmark/cfg/A_prod.yaml`, model
`3D_model_prep/OmniAnt_25PCs_joint_limited.pkl`, `SMILIFY_COUPLE_JOINT_BLENDSHAPES=0`, seeds 0, 1, 2.
On JAB the existing `A_prod` and `I_cse` fits are reused as baselines, not re-run; they are loaded
through `model_joints.load_fit`, which refuses any fit whose regenerated vertices differ from the
saved ones by more than 1e-4. On synthetic P48, D1_PROD is re-run under the same code state as every
new arm.

The existing CSE head is `diagnostics/anatomical_pose_init/out_C3_cse_head_20260826/best_model.pt`
(the checkpoint JAB arm `I_cse` used), loaded strictly (`strict=True`, zero missing/unexpected keys,
one named tensor compared to the raw file).

## 3. Use of the real JAB specimens

1. JAB is **never** used to choose a hyperparameter, threshold, checkpoint, epoch, segment subset,
   loss weight or any other design choice. All such choices are made on synthetic data and written
   into the method's `PREREGISTRATION.md` before its JAB run.
2. Each Q4 method and each baseline (B) is run on JAB **once**, in its pre-registered final form,
   3 seeds (0, 1, 2), through the D1_PROD pipeline with only its own factor changed. Every run is
   reported. No reruns after seeing results except for a documented crash, logged in `DEVIATIONS.md`.
3. **Verdict** uses JAB's own pre-registered machinery unchanged (`tools/analyze.py`): per-specimen
   seed-mean `med_fk`; paired difference vs `A_prod`; Hodges-Lehmann shift with 90% CI; exact
   signed-rank p; win count k/11; Holm correction with the family size **fixed now at m = 6**
   (4 Q4 interventions + CPD + NICP), applied at m = 6 even if fewer arms are eventually run, so
   that dropping arms cannot make survivors look stronger; IMPROVES / WORSENS / EQUIVALENT (TOST, SESOI = 2.5% WL) / INCONCLUSIVE.
   **Power, stated in advance.** At n = 11 the exact two-sided signed-rank p is 0.00098 at best;
   the strongest contrast's Holm threshold is 0.05/6 = 0.0083, which needs W+ <= 4 (e.g. 11/11 wins,
   or 10/11 with the single loss among the 4 smallest |differences|). A real but heterogeneous
   effect (help on some specimens, harm on others, as CSE showed) will read INCONCLUSIVE by
   construction. That is why item 4 is mandatory, not decoration.
4. **Distribution, not just a centre.** Each JAB report gives the per-specimen table, median and
   mean, the specimen-bootstrap 95% CI of the median, the paired-difference table, and the
   region-wise breakdown. The question "where does it help and where does it hurt" is answered from
   the per-specimen table, never only from the median.
5. The 25.1% WL figure is a reference point, **not** a success criterion. Success is a paired
   improvement over D1_PROD as defined in 3.3.
6. **Declared contamination.** Q3a inspects the existing JAB `A_prod` and `I_cse` outputs, which were
   seen on 2026-09-14. Hypotheses that come out of Q3a are tested on synthetic data only. Any Q4
   method motivated by Q3a carries this disclosure into the paper. With n = 11 there is no held-out
   real test set; this limitation is stated rather than hidden.

## 4. Synthetic data roles

| set | content | role |
|---|---|---|
| training corpus | 4000 specimens, `synth_b2_train_corrb06` generator, regenerated with joints and parameters (`common/make_corpus_with_joints.py`, vertex-verified against the original) | training only |
| retrieval dev | last 5% (200) of the training corpus, the split C3/C11/F6 used | C(a), for development decisions |
| P48 | `diagnostics/moonshot/synth_power48`, 48 specimens with GT parameters | synthetic fitter evaluation (C(b), S, G, M) |
| degraded corpora | built by a declared generator with fixed seeds (Q2/Q3b) | oracle budget and controlled shifts |
| held-out morphology splits | defined in betas space, written down before any training | Q3b |

## 5. Statistics on synthetic data

Specimen is the unit. Paired comparisons report sign test, Wilcoxon and paired t together, plus a
bootstrap CI of the mean paired difference. The fitter is not deterministic (about 0.01 run to run on
gen@20, Z5), so every synthetic fitter arm runs **at least 2 seeds**, and no difference smaller than
the measured seed-to-seed spread of the same arm is interpreted.

## 6. Verification rules (from CLAUDE.md, applied to every experiment)

- Every checkpoint is loaded with `strict=True`; one named tensor is compared to the raw file.
- Every data-pipeline change ships with a round-trip or equivalence probe that is kept on disk.
- Ground-truth joints are rebuilt through the model and the vertex round-trip is asserted before use.
- Probes and intermediate outputs stay in each folder's `out/` until the user has reviewed them.

Files frozen with this protocol: listed in `PROTOCOL.FREEZE`.

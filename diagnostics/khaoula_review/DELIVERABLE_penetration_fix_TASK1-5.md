# Penetration/registration fix — TASK 1-5 deliverable

bench50_clean (50 specimens), gentle proximity penetration loss (0.02/0.05) + `scheme:'all'`
vs `w_penetration=0` baseline. Full detail in the five linked docs; this is the summary.

## TASK 1 — does gentle + scheme:'all' help? (`TASK1_gentle_penetration_scheme_all_bench50.md`)

Pure re-analysis of existing runs, re-scored under the Hard Rule. On gaster-legs (the pair
called out as anatomically dominant): **yes**, count -20.8%, consistent across 3 seeds, no
collapse. On all 10 non-adjacent pairs the loss actually trains on: **no** — aggregate
+16.8% worse, the gaster-legs win is outweighed by new shallow contact elsewhere (mandible/
antenna/head vs. legs/thorax). Both true at once — contact gets shallower on the targeted
pair, more widespread everywhere else.

## TASK 2 — does a structural constraint (w_offset) fix it? (`TASK2_structural_constraint_offset_penalty.md`)

Same config + w_offset (5.0/2.0). Preserves the gaster-legs win (-20.2%, within noise of
TASK1) and **more than halves** the aggregate cost (+16.8% → +5.4%), at the cost of a worse
single-specimen F-score outlier (-0.1125 vs -0.0596). Real, partial fix — not a full one.

## TASK 3 — GWN disagreement diagnostic, shipped + verified (`TASK3_gwn_disagreement_baseline_vs_gentle.md`)

Diagnostic shipped (`diagnostics/moonshot/eval_gwn_disagreement.py`), independently
recomputed from raw per-specimen CSVs (matches logs to <5e-5). An unrelated ground-truth
metric (GWN inside/outside) **corroborates TASK1's finding from a second code path**:
13/15 non-trivial pair/directions disagree with ground truth more under gentle, concentrated
in the same `A_into_B` direction and the same pairs (mandible/antenna/head-thorax) where
TASK1's own count table showed the largest blowups. Confirms this is a real degradation in
signal quality, not a training-loss-specific artifact — and confirms (FINDINGS.md Step 6's
open question) that freeing pose via `scheme:'all'` does **not** resolve the underlying
proximity-test accuracy asymmetry, even though it does change the count outcome.

## TASK 4 — pair-scoped loss: does restricting training to gaster-legs alone make TASK 2's
partial fix a clean one? (`TASK4_pairscoped_gentle_penetration.md`)

Pre-registered follow-up, not the original spec's TASK 4 (torch-mesh-isect, still untried).
Same config as TASK 2 plus a new `penetration_train_pairs` option (`fitter_3d/trainer.py`)
restricting the w_penetration loss to gaster-legs only; evaluation still scores all 10 pairs.
Mixed result, not the clean win the hypothesis predicted: the all-pairs aggregate does turn
genuinely positive for the first time (+5.4% → **-13.6%**, PASS), but 2 of 4 pre-registered
criteria fail — mean F-score cost roughly quintuples (-0.0032 → **-0.0170**) and integrity
cost roughly quadruples (deform_mag/edge_logratio ~+6% → **~+21%**) — and the gaster-legs win
itself gets *weaker*, not stronger, when trained alone (-20.2% → **-16.8%**). A same-arm seed
check shows the effect size isn't stable to better than ~±10pp at this sample size. No
collapse in outlier renders. Reads as a squeezed-balloon effect: removing gradient pressure
from the other 9 pairs does eliminate their collateral damage, but concentrates deformation
(and its cost) onto gaster-legs rather than genuinely resolving more of it.

## TASK 5 — add Fabian's scale/trans barriers on top of TASK 4 (`TASK5_pairscoped_offset_scalecap.md`)

One pre-registered bench50 arm: TASK4's pair-scoped config plus `w_scale=0.052` (the one
validated value in the branch, T0.4, previously shipped only on the moonshot/hierarchical
pipeline, never run on this one) and `w_trans=1.0` (the only value ever tried anywhere,
explicitly uncalibrated). Newly wired into `fitter_3d/trainer.py`, ported verbatim from
`trainer_moonshot.py`. Result: **2 of 4 pre-registered criteria still fail** (mean F-score
−0.0195, worst −0.0890, both worse than the −0.01/−0.08 bar) — same two criteria TASK4
failed. The barriers did help on two axes that weren't required to pass: all-pairs aggregate
strengthened further (−13.6% → **−16.9%**, the best net win yet) and edge-distortion
integrity improved sharply (+21.1% → **+4.9%**, a 4.3x reduction) — but did not fix
mean/worst F-score, which is the criterion every arm across TASK1-5 has failed. No collapse.

## Verdict

No arm produces an unconditional win. Gentle + scheme:'all' alone trades one pair's
improvement for broader, GWN-confirmed degradation elsewhere (TASK1, TASK3). Adding
w_offset (TASK2) is a real, partial fix: keeps the gaster-legs win, cuts the aggregate
cost by more than half, but does not eliminate it and worsens the worst-case single-specimen
F-score. Restricting training scope further (TASK4) turns the aggregate net-positive but at
a larger, not smaller, per-pair fit/integrity cost and a weaker own-pair win. Adding the
scale/trans barriers on top (TASK5) improves the aggregate and edge-distortion further but
still does not clear the F-score bar. **Standing recommendation from
`REVIEW_penetration_update.md` — ship as an available config option, not a default, gated by
a required per-specimen check — holds.** No arm across TASK 1-5 clears all Hard-Rule-style
criteria at once; the mean/worst F-score bar in particular has failed on every soft-loss arm
tried. Per the pre-registered decision tree for TASK 5: treat soft proximity as optional and
gated, not the solution; default stays D1/no-penetration; the product bar ("net aggregate ≤0
and mean F within −0.01, simultaneously") has not been met. Part-filtered conical/BVH loss
(the original spec's TASK 4, and TASK5's own step 3) remains untried and is explicitly gated
behind a product decision on whether lower residual contact is still required — not something
decided in this task.

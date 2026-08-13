# Penetration/registration fix — TASK 1-4 deliverable

bench50_clean (50 specimens), gentle proximity penetration loss (0.02/0.05) + `scheme:'all'`
vs `w_penetration=0` baseline. Full detail in the four linked docs; this is the summary.

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

## Verdict

No arm produces an unconditional win. Gentle + scheme:'all' alone trades one pair's
improvement for broader, GWN-confirmed degradation elsewhere (TASK1, TASK3). Adding
w_offset (TASK2) is a real, partial fix: keeps the gaster-legs win, cuts the aggregate
cost by more than half, but does not eliminate it and worsens the worst-case single-specimen
F-score. Restricting training scope further (TASK4) turns the aggregate net-positive but at
a larger, not smaller, per-pair fit/integrity cost and a weaker own-pair win — a different
trade-off, not a strictly better one. **Standing recommendation from
`REVIEW_penetration_update.md` — ship as an available config option, not a default, gated by
a required per-specimen check — holds.** No arm across TASK 1-4 clears all four Hard-Rule-style
criteria at once. TASK 4's spec original (torch-mesh-isect) remains a live, untried next step;
if pursued, a proper 3-seed rerun of TASK 4's pairscoped arm should come first, since the
single seed-check here already shows the effect size isn't well-pinned.

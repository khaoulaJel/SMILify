# Model-free correspondence — does R1's ~24% WL reflect fit error or genuine correspondence absence?

**2026-09-17.** `modelfree_pairwise.py`, 11 gt_expert specimens (Dolichoderus excluded), 55
ordered pairs → 2480 landmark-transfer rows. No parametric model anywhere in this script:
landmarks transferred specimen-to-specimen by registering the **raw scans** directly.

## The question this was built to answer

R1 (`RESULTS_R1_recalibration_20260916.md`) found that even the best possible single template
vertex sits a median ~24% of Weber's length from its annotated landmark. That number is measured
**through the production fit**, whose own joint error is 25.1% WL — so it conflates fit error
with genuine correspondence absence, and the write-up flagged this explicitly rather than resolve
it. This experiment supplies the missing control: an estimate of the same kind of error with
**zero model-fitting error**, obtained by registering scans directly to each other.

## Result: the model-free arm is much worse, not better

| method | median error, % WL | what it means |
|---|---:|---|
| ICP (similarity: rotation + uniform scale + translation) | **162.4%** | direct scan-to-scan transfer |
| Oracle (best possible similarity transform, informed by every *other* landmark, leave-one-out) | **31.7%** | the theoretical floor for ANY similarity transform |
| *for reference:* R1, through the production fit | ~24% | — |
| Registration quality achieved (bidirectional chamfer) | median 26.0% WL | the registrations themselves were reasonable, not failed alignments |

Even the **oracle** arm — which is told the correct answer for every landmark except the one it's
predicting, and finds the best possible rigid+scale transform from that — is worse than R1's
through-the-fit figure. The plain ICP arm is catastrophic (162% median; per-landmark medians run
177–253% WL for the antennae, mandibles and scapes).

## Verdict: R1's ~24% is NOT dominated by fit error

If R1's residual were mostly the production fit's own 25% WL joint error riding along, a method
with **zero fit error** should do noticeably better. It does the opposite — over 30% WL even in
its best-case oracle form, nearly 7x worse in its plain form. **The deformable parametric model,
despite its own imperfection, is doing something a similarity transform structurally cannot: it
captures genuine non-rigid shape difference between species.** A trap-jaw ant and a turtle ant do
not differ by a rotation, scale and translation, and no amount of oracle information rescues a
transform class that assumes they do.

**Consequence for the talk's claim.** "No stable vertex-level correspondence exists" (R1) is not
weakened by this control — it is corroborated from a different angle. The absence is not an
artifact of measuring through an imperfect fit; a method with no fit error at all does worse,
because it has no shape model at all. The honest generalisation is:

> **Cross-species anatomical correspondence requires a non-rigid shape model. A rigid or
> similarity transform, however well-informed, is the wrong class of solution — and the
> deformable model's own ~24% residual, imperfect as it is, is already outperforming the
> best rigid alternative by a wide margin.**

## Important limitation — do not overclaim

This experiment did **not** implement a non-rigid model-free arm (e.g. functional maps, thin-plate
splines, or coherent point drift between raw scans). It rules out *rigid and similarity*
registration as the explanation for R1's residual; it does not rule out the possibility that some
non-rigid, model-free method could close the gap. That is a real open question, not one this
result answers. Framed narrowly, what is shown is: **the parametric model beats the best rigid
alternative, not that it beats every conceivable alternative.**

## Artifacts

- `modelfree_pairwise.py` — the registration + transfer script, now checkpointing per-pair
  (`modelfree_pairwise_ckpt.json`) so an interruption does not lose progress; this run completed
  all 55 pairs in 48s once restarted clean.
- `modelfree_pairwise.json` — full output, 2480 rows, both `icp` and `oracle` methods, per-pair
  chamfer.
- `modelfree_local.py` / `modelfree_local.log` — the on-surface validation pass (median 0.121%
  corrected vs 2.272% raw, consistent with R1 and the audit).

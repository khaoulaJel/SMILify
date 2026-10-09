# Pre-registration — W3: a descriptor trained *for* within-part cross-specimen correspondence

**Written and committed BEFORE the run.** Bars fixed here are not revised after seeing a number.

Date: 2026-08-30. Folder: `diagnostics/within_part_correspondence/`.

---

## 1. Why this is not already answered by W2

W2 measured the existing C11 CSE head and it FAILED — worse than plain rigid proximity on
`tr`/`fe`/`ti`, a tie on `co`. W2's own conclusion was that a new descriptor is justified **only**
with a specific argument for why a different objective would beat 0.0778 / 0.0958 / 0.0968.

Here is that argument, stated concretely enough to be wrong:

C11 is already a *contextual learned embedding* — a PointNet++ MSG backbone with multi-scale
neighbourhoods, trained with InfoNCE on exact vertex identity. What it is **not** is trained for the
task W2 measured. C11's objective is `query point → argmax over a fixed per-template-vertex key
table` — full-vocabulary retrieval, with negatives spread across all 10,235 vertices, i.e.
overwhelmingly *other parts*. The task that actually matters here is `query on specimen A → match
among candidates on specimen B, restricted to the same part`. Its negatives are, by construction,
**within-part**: the ~100–300 vertices on the same femur.

So C11 was optimised almost entirely on easy negatives (is this the femur or the gaster?) and
evaluated by W2 on hard ones (which femur point?). W3 tests the one change that follows:
**train with the negatives the evaluation actually uses.**

If that does not clear the bar, the "learned dense descriptor" line is closed on a fair test, not
on a mismatched one.

## 2. What is trained

- **Backbone**: `SMILCSENet`'s PointNet++ MSG backbone, unchanged, warm-started from C11 (asserted
  copied, not silently random).
- **Objective**: cross-specimen query-to-query InfoNCE. For a pair (A, B) and a part *s*, the
  positive for vertex *i* is **vertex *i* on B**; the negatives are **all other vertices of part *s*
  on B**. This is the evaluation task, used as the loss.
- **Input representation** — following W1's lesson explicitly: raw posed vertex coordinates. The
  part is **not** canonicalised, its length and radius are **not** normalised away. W1 measured that
  doing so is actively harmful, so the network is given the morphological variation and left to
  learn which of it is meaningful.

## 3. Splits — and the one thing that would invalidate everything

- **Train**: corpus specimens **0–3799**.
- **Eval**: **3988–3999**, the same 12 held-out specimens W1 and W2 used.
- These are disjoint by construction and disjointness is **asserted at runtime**, not assumed.
  Note 3800–3999 is C11's own validation tail, so the warm-start never saw the eval specimens
  either.

## 4. The density trap, and how evaluation avoids it

W2's registered density control caught that feeding all 10,235 vertices to this backbone is ~5×
its training density and drives it to at-or-below chance — an out-of-distribution artefact, not a
descriptor result. W3 therefore fixes one density everywhere:

**4096 vertices, sampled by index, identically for both specimens of a pair, at train and eval.**

Because the candidate set per part is then smaller than W1's full-vertex sets, the retrieval task is
*easier* and W1's published baselines are **not** the fair comparison. So:

> **Every arm — `RANDOM`, `XYZ_RIGID`, `PART_FRAME`, `CSE_C11`, `W3` — is recomputed on exactly the
> same sampled subsets**, and the bar is set against the **subset-matched `XYZ_RIGID`**, not against
> W1's numbers. W1/W2 figures are reported alongside as context only.

Aggregated over 8 independent subsets to avoid a single lucky sample.

## 5. Pre-registered bar

Primary: **`W3` vs subset-matched `XYZ_RIGID`**, per segment, individually — no pooled averages.

- **PASS** — `W3` median normalised 3D error is lower on **all four** of `co`/`tr`/`fe`/`ti`, each
  with paired sign test **p < 0.05** *and* **≥ 15% relative** reduction.
- **PARTIAL** — direction holds on some but not all four, or on all four but under 15%.
- **FAIL** — no segment improves. **The learned dense descriptor line closes**, now on a test where
  the objective matched the evaluation, and the remaining bottleneck is configuration, not
  point-level semantics.

Secondary, reported but gating nothing: `W3` vs `CSE_C11` on the same subsets — did changing the
objective help *at all*, independent of whether it cleared proximity?

## 6. Mechanism checks — mandatory; the first three void the run

1. **Split disjointness (VOIDS).** Assert `set(train) ∩ set(eval) = ∅` and that no eval specimen
   index appears in any training batch. A leak would make any PASS meaningless.
2. **Checkpoint loaded (VOIDS).** The trained W3 checkpoint reloaded with `strict=True`, 0 missing
   / 0 unexpected, plus a named tensor compared element-wise against the raw file. Same guard as
   W2 — F1 produced a believable result from a network that never loaded.
3. **Warm-start actually transferred (VOIDS).** Assert the number of backbone tensors copied from
   C11 is > 0 and that a named backbone tensor differs from fresh init.
4. **Chance control.** `W3` embeddings permuted **independently per specimen** must land at the
   `RANDOM` floor. (W2's first attempt used one shared permutation, which preserves matchability
   and tests nothing; that error is not repeated.)
5. **Overfitting check.** Report the same metric on *training* specimens alongside held-out. A large
   train/held-out gap means the network memorised specimen identity rather than learning
   correspondence, and is reported as such even if the held-out number passes.
6. **Training actually converged.** Report the loss curve and in-batch retrieval accuracy. A flat
   loss means the arm tested nothing.

## 7. What each outcome licenses — fixed now

- **PASS** → one thing only: a pre-registered fitter experiment, feeding this correspondence to D1
  and scoring on the *fitter's* metric. It is **not** a fitter claim. Retrieval gains have failed to
  reach the fitter four times (C11, C12, F2, and the mechanism-check series); a fifth is the
  default expectation, not a surprise.
- **PARTIAL** → report the segments and stop. No fitter run, no scaling up.
- **FAIL** → the learned dense descriptor line closes on a fair test. Combined with W1 (handcrafted)
  and W2 (learned, mismatched objective), all three descriptor families will have lost to a rigid
  transform, and the evidence points at configuration rather than point-level semantics.

**Not claimed under any outcome:** anything about real scans. Synthetic, exact correspondence by
construction, oracle part label on both sides, no scan noise, partiality, or topology defects.

## 8. Budget

Single GPU, ≤ 6 h wall. If training has not converged in that budget the arm is reported as
**inconclusive on budget**, not as a FAIL — an untrained network failing is not evidence about
trained ones.

# Pre-registration — W3b: W3 run to convergence. The bar is UNCHANGED.

**Written and committed BEFORE the run**, and specifically before seeing any W3b number.

Date: 2026-08-30.

---

## 1. Why this run exists

W3's registered mechanism check 6 (convergence) **failed**: in-part top-1 went 0.0483 → 0.0874 over
30 epochs and was **still climbing at the last epoch**. Per W3's pre-registration §8 — *"If training
has not converged in that budget the arm is reported as inconclusive on budget, not as a FAIL"* —
`tr`/`fe`/`ti` are currently **inconclusive**, not failed. `co` **passed** (+16.4%, CI
[+12.6%, +19.8%]).

W3b resolves that ambiguity by removing the cause: more epochs. That is the registered remedy for a
failed convergence check, and it is the only change.

## 2. The bar does not move — stated explicitly because this is the hazard

The endpoint, the metric, the splits, the eval specimens, the arms, and the pass thresholds are
**identical to W3's**:

> `W3` median normalised 3D error must beat the **subset-matched** `XYZ_RIGID` on **all four** of
> `co`/`tr`/`fe`/`ti` individually, each with **≥15% relative** reduction and sign test **p < 0.05**.

- Train 0–3699, model-selection val 3700–3799, **eval 3988–3999** — the same 12 specimens W1, W2 and
  W3 used.
- Same seed (0), same density (4096 verts), same 8 subsets × 20 pairs, same six mechanism checks.
- **Only `--epochs` changes: 30 → 150.**

Extending training is not moving a goalpost, but the record must show the bar was fixed *before* the
longer run rather than after seeing which way it went. That is the whole reason this file exists and
is committed before submission.

## 3. Reading — fixed now

- **PASS** (all four) → licenses one thing: a separately pre-registered fitter experiment scored on
  the fitter's metric. Still not a fitter claim.
- **PARTIAL** (e.g. `co` only, as in W3) → report and stop. If `co` alone passes again *at
  convergence*, that is no longer a budget artefact — it becomes the finding: a learned descriptor
  helps exactly where rigid proximity is weak, and nowhere else.
- **FAIL** → the learned dense descriptor line closes on a fair, converged test. Combined with W1
  and W2, all descriptor families will have lost to a rigid transform on the segments where geometry
  is strong.
- **Still not converged at 150 epochs** → reported as inconclusive again, and I stop extending
  rather than fishing for an epoch count that passes. **This is the last extension.**

## 4. Mechanism checks

Unchanged from W3 — all six, three of them voiding (split disjointness, strict checkpoint load with
byte verification, warm-start transfer), plus the per-specimen-independent chance control, the
train-vs-held-out overfitting readout, and the convergence readout that triggered this run.

Convergence is judged on the **loss and in-part top-1 curves**, reported in full. A run whose curve
is still rising at epoch 149 is not converged and is said so.

## 5. Not claimed

Anything about real scans, and anything about the fitter. Synthetic, exact correspondence by
construction, oracle part label on both sides. Retrieval gains have failed to reach the fitter four
times.
